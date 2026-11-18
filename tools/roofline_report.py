#!/usr/bin/env python3
"""Roofline / occupancy report for a .vkopbin (or ONNX) model.

Answers three questions for every REAL operator the model dispatches -- i.e.
every node that is not a host-side view or a fused-away decoration:

  1. work:       FLOPs, and the arithmetic intensity that puts it on a roofline
  2. traffic:    the minimum bytes the kernel must move (read + write)
  3. occupancy:  the theoretical occupancy of the workgroup geometry it
                 launches with, from the device's SM budget and the kernel's
                 actual per-workgroup footprint

Design notes, in the order the hard parts appear:

* Which kernel.  A MatMul's route (split-K GEMV vs tiled GEMM vs coopmat vs the
  SwiGLU fold ...) is decided by core/KernelSelect.cpp at run time, from
  (shape, weight kind, device traits, VKOP_* knobs).  This tool does NOT
  re-implement that table: tools/roofline_oracle.cpp compiles the real
  selector (it #includes core/KernelSelect.cpp, unmodified) behind a stdin/
  stdout JSON line protocol, so every route in the report came out of the same
  code the runtime runs.  The oracle is built on demand with g++ from the
  repo's own sources.  If it cannot be built (no compiler, no Vulkan headers)
  the tool still produces a report, but every MatMul is marked "unknown at
  static time" and its bytes are still exact -- only the kernel attribution
  and the grid are missing.

* Grid and local_size.  local_size comes from the kernel's own SPIR-V
  (OpExecutionMode LocalSize) in the VKSP archive the model carries, and the
  grid from the dispatch formula at the route's site in ops/Matmul.hpp --
  transcribed per route with the source line named, because those formulas are
  not exported by any interface.  They are also checked against the output
  they must cover: a launch whose grid cannot cover its own element count is
  reported as UNDER-DISPATCHED (that is the failure mode a wrong transcription
  has, and it is invisible if the two numbers are read separately).

* Registers.  SPIR-V has no notion of a register file; only the driver's
  compiler knows.  With --probe the tool asks the driver
  (VK_KHR_pipeline_executable_properties, which reports "Max live registers"
  and "Workgroup Memory Size"), one forked process per shader because Intel
  ANV SIGSEGVs on a few of this tree's modules; a module that dies is
  reported as unavailable and the row says "assumed".  Probing CREATES compute
  pipelines (it never dispatches), so it is opt-in: without --probe the tool
  touches no GPU at all and every row states the register premise it used.  There is no useful middle ground: SPIR-V shared memory can be
  read exactly (OpVariable Workgroup + its array type), but every SSA-based
  register proxy tried here landed 3-10x away from what the driver reports, so
  the tool does not print one.  Every occupancy row states which provenance
  its register count has.

* Structural ops.  Reshape/Transpose(flat identity)/Slice/Gather/Split alias
  their input's VkBuffer when the byte range is contiguous and aligned; those
  are reported as VIEW with zero traffic, not as a kernel.  The conditions are
  transcribed from the ops' own code and named per node.  A Transpose whose
  perm is a flat identity and a Reshape are always views; Slice/Gather/Split
  depend on RUNTIME start/step/index values, so they are reported as a
  dispatch with a note that the alias path may replace it.

Usage:
    tools/roofline_report.py MODEL.vkopbin [--device nvidia-ada|intel-arl|...]
        [--decode-m 1] [--prefill-m 512] [--kv-len 1024]
        [--csv out.csv] [--json out.json] [--oracle path/to/roofline_oracle]
        [--probe] [--no-oracle] [--measured-ms 363] [--top N]
    tools/roofline_report.py MODEL.onnx [--onnx-quant int8]   # convert first
    tools/roofline_report.py --self-test                      # check the math

The device defaults to nvidia-ada because that is the b3 card the 27B artifact
is served from; pass --device intel-arl for the dev box.

Reading a 25.6 GB model: the file is mmap'd and only the FlatBuffer metadata
and the embedded VKSP archive are touched.  Weights are never read; their sizes
come from InitializerEntry.size, so the report is O(nodes), not O(bytes).
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict

# ---------------------------------------------------------------------------
# DType / element size
# ---------------------------------------------------------------------------
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "int32": 4,
    "int64": 8,
    "float8e4m3fn": 1,
    "float8e5m2": 1,
    "int4": 0.5,
    "uint4": 0.5,
    "nf4": 0.5,
    "float4e2m1": 0.5,
}


def dtype_bytes(name):
    if not name:
        return 4
    return DTYPE_BYTES.get(name, 4)


# ---------------------------------------------------------------------------
# Device descriptions.  None of these are invented: every field is either a
# constant this repository already carries, a value from the vulkaninfo
# profile JSON checked in next to it, or a vendor-published peak.
#
# `sm` is taken from core/KernelSelect.cpp's sm_estimate() through the oracle
# when the oracle is available; the copy here is only the fallback so a report
# still prints without it (and is then labelled as such).
#
# Peak throughput is computed from the SM's own structure
# (2 flop/clock/lane x lanes-per-SM x clock x SM) so it is internally
# consistent with the occupancy model that divides the same SM resources.
# A cooperative-matrix / tensor-core rate is NOT modelled -- see Roofline.
# ---------------------------------------------------------------------------
DEVICES = {
    "nvidia-ada": {
        # RTX 4000 Ada Generation (AD104): 48 SM, 128 FP32 lanes/SM,
        # 1536 resident threads/SM (48 warps), 64K 32-bit registers/SM,
        # 48 KiB shared-memory window, 20 GB GDDR6 at 360 GB/s.
        "name": "NVIDIA RTX 4000 Ada",
        "vendor_id": 0x10DE,
        "sm": 48,
        "cores_per_sm": 128,
        "clock_ghz": 1.74,
        "subgroup_size": 32,
        "max_invocations_per_wg": 1024,
        # 48 KiB is the default shared-memory window per SM on Ada (the
        # driver can raise it to 100 KiB for one block; the model uses the
        # conservative default).  maxComputeSharedMemorySize on b3 is 48 KiB.
        "shared_per_sm": 48 * 1024,
        "shared_per_wg": 48 * 1024,
        # Register file: 65536 32-bit registers per SM. Handed out per warp in
        # 256-register granules, so a warp's cost is
        # ceil(regs*32/256)*256 registers.
        "reg_file_bytes_per_sm": 64 * 1024 * 4,
        "reg_granule_regs": 256,
        # 1536 resident threads/SM = 48 warps (the AD10x limit is 48 warps).
        "threads_per_sm": 1536,
        # Intel-style GRF accounting is unused on NVIDIA.
        "eu_per_sm": 0,
        "threads_per_eu": 0,
        "grf_bytes": 0,
        "bandwidth_gbs": 360.0,
        "l2_bytes": 48 * 1024 * 1024,
        # Peak rates: 128 FP32 lanes/SM x 2 (FMA) x 1.74 GHz x 48 SM.  fp16
        # and int8 are the same ALUs at 2x / 4x throughput, which is what the
        # tool compares against -- it does NOT claim to model a tensor core.
        "flops_fp32": 2 * 48 * 128 * 1.74e9,
        "flops_fp16": 2 * 48 * 128 * 1.74e9 * 2,
        "flops_int8": 2 * 48 * 128 * 1.74e9 * 4,
        "source": "AD104 published figures (48 SM x 128 FP32 lanes, 1.74 GHz, "
                  "48 warps/SM, 360 GB/s)",
    },
    "intel-arl": {
        # Intel Graphics (ARL): 8 Xe-cores (sm_estimate()'s 8), each with
        # 16 EU (4 XVE x 4 EU), 8 hardware threads per EU, 128 GRF x 32 B per
        # thread, 64 KiB shared memory per Xe-core.
        "name": "Intel Graphics (ARL), Mesa 25.2.8",
        "vendor_id": 0x8086,
        "sm": 8,
        "cores_per_sm": 128,
        "clock_ghz": 1.65,
        "subgroup_size": 32,
        "max_invocations_per_wg": 1024,
        "shared_per_sm": 64 * 1024,
        "shared_per_wg": 64 * 1024,
        # 8 threads/EU x 16 EU x (128 GRF x 32 B). Registers are counted in
        # 8-register (32 B) granules.
        "reg_file_bytes_per_sm": 8 * 16 * 128 * 32,
        "reg_granule_regs": 8,
        "threads_per_sm": 0,   # derived: eu_per_sm * threads_per_eu * SIMD
        "eu_per_sm": 16,
        "threads_per_eu": 8,
        "grf_bytes": 32,
        "bandwidth_gbs": 90.0,
        "l2_bytes": 8 * 1024 * 1024,
        "flops_fp32": 2 * 8 * 128 * 1.65e9,
        "flops_fp16": 2 * 8 * 128 * 1.65e9 * 2,
        "flops_int8": 2 * 8 * 128 * 1.65e9 * 4,
        "source": "VP_VULKANINFO_Intel(R)_Graphics_(ARL)_25_2_8.json "
                  "(maxComputeSharedMemorySize 65536, "
                  "maxComputeWorkGroupInvocations 1024, subgroupSize 32) + "
                  "Xe register-file structure (8 threads/EU, 128 GRF x 32 B)",
    },
    "nvidia-generic": {
        "name": "Generic native NVIDIA dGPU (no coopmat assumed)",
        "vendor_id": 0x10DE,
        "sm": 48,
        "cores_per_sm": 128,
        "clock_ghz": 1.74,
        "subgroup_size": 32,
        "max_invocations_per_wg": 1024,
        "shared_per_sm": 48 * 1024,
        "shared_per_wg": 48 * 1024,
        "reg_file_bytes_per_sm": 64 * 1024 * 4,
        "reg_granule_regs": 256,
        "threads_per_sm": 1536,
        "eu_per_sm": 0,
        "threads_per_eu": 0,
        "grf_bytes": 0,
        "bandwidth_gbs": 360.0,
        "l2_bytes": 48 * 1024 * 1024,
        "flops_fp32": 2 * 48 * 128 * 1.74e9,
        "flops_fp16": 2 * 48 * 128 * 1.74e9 * 2,
        "flops_int8": 2 * 48 * 128 * 1.74e9 * 4,
        "source": "the nvidia-ada resource model without its coopmat tiles",
    },
    "generic": {
        "name": "Unknown device (conservative defaults)",
        "vendor_id": 0,
        "sm": 16,
        "cores_per_sm": 128,
        "clock_ghz": 1.5,
        "subgroup_size": 32,
        "max_invocations_per_wg": 1024,
        "shared_per_sm": 32 * 1024,
        "shared_per_wg": 32 * 1024,
        "reg_file_bytes_per_sm": 64 * 1024 * 4,
        "reg_granule_regs": 256,
        "threads_per_sm": 1024,
        "eu_per_sm": 0,
        "threads_per_eu": 0,
        "grf_bytes": 0,
        "bandwidth_gbs": 100.0,
        "l2_bytes": 4 * 1024 * 1024,
        "flops_fp32": 2 * 16 * 128 * 1.5e9,
        "flops_fp16": 2 * 16 * 128 * 1.5e9 * 2,
        "flops_int8": 2 * 16 * 128 * 1.5e9 * 4,
        "source": "sm_estimate()'s conservative default",
    },
}

SM_ESTIMATE_FALLBACK = {0x8086: 8, 0x10DE: 48, 0x1002: 32}


# ---------------------------------------------------------------------------
# VKSP archive parsing (tools/spv_to_archive.py's format)
# ---------------------------------------------------------------------------
VKSP_MAGIC = b"VKSP"
VKSP_HEADER = 16
VKSP_ENTRY = 72
VKSP_NAME_MAX = 48
SPIRV_MAGIC = 0x07230203


class ShaderArchive:
    """Zero-copy view over a VKSP archive inside an mmap'd model file."""

    def __init__(self, buf, base, size):
        self.buf = buf
        self.base = base
        self.size = size
        self.entries = {}
        if size < VKSP_HEADER or bytes(buf[base:base + 4]) != VKSP_MAGIC:
            return
        version, count = struct.unpack_from("<II", buf, base + 4)
        if version != 1:
            return
        for i in range(count):
            row = base + VKSP_HEADER + i * VKSP_ENTRY
            if row + VKSP_ENTRY > base + size:
                break
            name = bytes(buf[row:row + VKSP_NAME_MAX]).split(b"\x00", 1)[0]
            off, sz, _crc = struct.unpack_from("<QII", buf, row + 48)
            abs_off = base + off
            if off + sz > size or abs_off + sz > len(buf):
                continue
            self.entries[name.decode("utf-8", "replace")] = (abs_off, sz)

    def __contains__(self, key):
        return key in self.entries

    def keys(self):
        return self.entries.keys()

    def get(self, key):
        e = self.entries.get(key)
        if not e:
            return None
        off, sz = e
        return bytes(self.buf[off:off + sz])


# ---------------------------------------------------------------------------
# SPIR-V static analysis.
#
# What SPIR-V *can* tell us:
#   - OpExecutionMode LocalSize (the workgroup geometry the kernel launches
#     with; deterministic, no guessing)
#   - OpVariable Workgroup + its OpTypeArray/OpTypeVector size chain (the
#     static shared-memory footprint; exact for a static array)
#   - a count of SSA values that are live across the shader's back edge -- a
#     LOWER BOUND on the register pressure, not the register count.  The
#     driver's allocator coalesces, spills and re-schedules, so this number is
#     an estimate and is labelled as such everywhere it appears.
#
# What it can NOT tell us: the register count, the SIMD width the driver picks,
# or the occupancy.  Those come from VK_KHR_pipeline_executable_properties when
# the tool was allowed to probe the GPU, and from a documented model otherwise.
# ---------------------------------------------------------------------------
# Opcodes that produce NO result id.  Everything else in a function body has
# its result type in word 1 and its result id in word 2, which is the shape the
# liveness proxy below relies on.  The list is the SPIR-V spec's no-result
# instructions (spirv/unified1/spirv.h); anything missing from it would only
# make the proxy count an id it should not, never crash.
_OPS_NO_RESULT = frozenset({
    16, 17, 18, 19, 22, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    43, 44, 45, 46, 48, 49, 50, 53, 54, 55, 56, 57, 59, 63, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
    259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
    272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
    285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297,
    298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
    311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
    324, 325, 326, 327, 328, 330, 331, 332, 333, 334, 335, 336, 337,
    338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
    351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
    364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
    377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
    390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
    403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
    416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
    429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
    442, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
    456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468,
    469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481,
    482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
    495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
    508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
    521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
    534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
    547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
    560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
    573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585,
    586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598,
    599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611,
    612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624,
    625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637,
    638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650,
    651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663,
    664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676,
    677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689,
})



def _spirv_static(spv_bytes):
    """Static facts only: {local, shared_static, shared_dynamic}.

    Registers are NOT here on purpose -- see the note at the end of the body.
    """
    info = {
        "local": None,
        "shared_static": 0,
        "shared_dynamic": 0,
        "instruction_count": 0,
    }
    if len(spv_bytes) < 20:
        return info
    words = struct.unpack_from("<%dI" % (len(spv_bytes) // 4), spv_bytes, 0)
    if words[0] != SPIRV_MAGIC:
        return info

    types = {}
    consts = {}
    wg_vars = []          # (result_type, result_id, storage_class)
    phi_operands = []     # operand ids of every OpPhi
    block_uses = []       # per function (in declaration order): {blk: [ids]}
    func_id = []          # function declaration order -> result id
    func_call = []        # function declaration order -> [callee result ids]
    cur_func = -1
    cur_block = -1
    entry_func = None
    entry_func_n = None
    # Workgroup variables are MODULE-scope in SPIR-V even when the GLSL
    # declaration sits inside a #if branch, so "which variables does this
    # launch use" is only answerable through the call graph: entry function
    # plus every function it transitively calls.
    func_uses_var = []    # per function: set of variable result ids referenced
    def_blk = {}          # (function, result id) -> block index defined in
    label_id = {}         # (function, label result id) -> block index
    block_terms = {}      # (function, block) -> [target label ids]
    var_func = {}         # module-scope variable -> declaring "function" (-1)
    i = 5
    n = len(words)
    while i < n:
        w = words[i]
        op = w & 0xFFFF
        wc = w >> 16
        if wc == 0:
            break
        # The opcodes below are SpvOp* / SpvExecutionMode values from
        # spirv/unified1/spirv.h.  (OpTypeFloat is 22 and OpTypeInt is 21 --
        # the reverse of the order they are usually written in.)
        if op == 16:  # OpExecutionMode
            if words[i + 2] == 17:  # SpvExecutionModeLocalSize
                info["local"] = [words[i + 3], words[i + 4], words[i + 5]]
        elif op == 15:  # SpvOpEntryPoint: op, EXEC_MODEL, FUNCTION, ...
            if entry_func is None:
                entry_func = words[i + 2]
        elif op == 54:  # SpvOpFunction: op, TYPE, RESULT, CONTROL, FUNCTION_TYPE
            cur_func += 1
            cur_block = -1
            block_uses.append({})
            func_id.append(words[i + 2])
            func_call.append([])
            func_uses_var.append(set())
            if entry_func is not None and words[i + 2] == entry_func:
                entry_func_n = cur_func
        elif op == 56:  # SpvOpFunctionEnd
            cur_func = -1
            cur_block = -1
        elif op == 248:  # SpvOpLabel
            cur_block += 1
            block_uses[cur_func][cur_block] = []
            label_id[(cur_func, words[i + 1])] = cur_block
        elif op == 43:  # SpvOpConstant: op, TYPE, RESULT, value...
            consts[words[i + 2]] = words[i + 3]
        elif op == 22:  # SpvOpTypeFloat
            types[words[i + 1]] = ("float", words[i + 2])
        elif op == 21:  # SpvOpTypeInt
            types[words[i + 1]] = ("int", words[i + 2], words[i + 3])
        elif op == 23:  # SpvOpTypeVector
            types[words[i + 1]] = ("vec", words[i + 2], words[i + 3])
        elif op == 28:  # SpvOpTypeArray
            types[words[i + 1]] = ("arr", words[i + 2], words[i + 3])
        elif op == 32:  # SpvOpTypePointer
            types[words[i + 1]] = ("ptr", words[i + 2], words[i + 3])
        elif op == 59:  # SpvOpVariable: op, TYPE, RESULT, STORAGE, ...
            wg_vars.append((words[i + 1], words[i + 2], words[i + 3]))
            var_func[words[i + 2]] = cur_func
            if cur_func >= 0 and words[i + 3] == 4:
                func_uses_var[cur_func].add(words[i + 2])
        elif op == 57:  # SpvOpFunctionCall: op, TYPE, RESULT, FUNCTION, args
            if cur_func >= 0:
                func_call[cur_func].append(words[i + 3])
        elif op == 245:  # SpvOpPhi: op, TYPE, RESULT, (value, parent)*
            phi_operands.extend(words[i + 3:i + wc:2])
        elif op in (249, 250):  # OpBranch / OpBranchConditional
            # OpBranch: word 1 is the target label.  OpBranchConditional:
            # word 1 = condition, words 2/3 = true/false label, word 4+ =
            # branch weights (optional literal, never a label id above it).
            if cur_func >= 0 and cur_block >= 0:
                if op == 249:
                    tgts = [words[i + 1]]
                else:
                    tgts = [words[i + 2], words[i + 3]]
                block_terms.setdefault((cur_func, cur_block), []).extend(tgts)
        elif op == 251:  # OpSwitch: op, selector, default, (literal, label)*
            if cur_func >= 0 and cur_block >= 0:
                tgts = [words[i + 2]] + [words[i + 4 + 2 * k]
                                         for k in range((wc - 3) // 2)]
                block_terms.setdefault((cur_func, cur_block), []).extend(tgts)

        # Every operand of an in-function instruction is a reference; they are
        # collected per basic block for the live-value proxy below.  Word 1 is
        # the result TYPE and word 2 the result ID for every opcode that has
        # one, so the operand scan starts at 3 there and at 1 otherwise; the
        # two are kept distinct so "defined here" and "used here" do not
        # collapse into each other.
        if cur_func >= 0 and cur_block >= 0:
            info["instruction_count"] += 1
            has_result = op not in _OPS_NO_RESULT
            start = 3 if has_result else 1
            for a in words[i + start:i + wc]:
                if a != 0:
                    block_uses[cur_func][cur_block].append(a)
                    if var_func.get(a) == -1:
                        func_uses_var[cur_func].add(a)
            if has_result:
                def_blk[(cur_func, words[i + 2])] = cur_block
        i += wc

    # Branch targets are label result ids; map them to block indices.
    for key, tgts in list(block_terms.items()):
        f, b = key
        block_terms[key] = [label_id.get((f, t)) for t in tgts
                            if label_id.get((f, t)) is not None]

    def tsize(tid):
        t = types.get(tid)
        if not t:
            return None
        if t[0] in ("int", "float"):
            return t[1] // 8
        if t[0] == "vec":
            b = tsize(t[1])
            return None if b is None else b * t[2]
        if t[0] == "arr":
            b = tsize(t[1])
            if b is None:
                return None
            cnt = consts.get(t[2])
            if cnt is None:
                return None
            return b * cnt
        return None

    # Shared memory: sum only the Workgroup variables of the ENTRY POINT's own
    # function.  A module compiled from a source with #if branches carries
    # every branch's helpers; the launch runs one entry point, so only its
    # function's statics count.  A variable in no function (none in practice)
    # is counted as well, so the sum errs high rather than low.
    # Reachable set from the entry function (call graph, bounded depth).
    reach = set()
    if entry_func_n is not None:
        work = [entry_func_n]
        while work:
            f = work.pop()
            if f in reach:
                continue
            reach.add(f)
            for callee in func_call[f]:
                for j, fid in enumerate(func_id):
                    if fid == callee:
                        work.append(j)
    live_vars = set()
    for f in reach:
        live_vars |= func_uses_var[f]
    if not reach:
        live_vars = {v for (_t, v, sc) in wg_vars if sc == 4}

    total = 0
    for ty, vid, storage in wg_vars:
        if storage != 4:  # Workgroup
            continue
        if vid not in live_vars:
            continue
        t = types.get(ty)
        if not t or t[0] != "ptr":
            continue
        sz = tsize(t[2])
        if sz is None:
            info["shared_dynamic"] += 1
        else:
            total += sz
    info["shared_static"] = total
    # ---------------------------------------------------------------------
    # A register-pressure PROXY, deliberately labelled as such.
    #
    # SPIR-V has no register file, no allocation and no liveness summary.  What
    # can be computed is: for each basic block, the number of DISTINCT
    # non-constant SSA ids it references (whether defined in it or hoisted out
    # of it).  Every one of those must be addressable somewhere in that block,
    # so the maximum across blocks is a lower bound on the peak number of
    # simultaneously-live values -- a real floor, but a weak one (a value
    # contributes to every later block of its live range, and the number of
    # values whose live ranges actually overlap is what a register allocator
    # cares about).
    #
    # Measured on this tree: the bound lands at 15-40 for kernels the Intel
    # driver reports at 22-83 registers, i.e. it under-counts by 2-4x.  That is
    # why a static row's occupancy is explicitly an UPPER bound.
    # Registers are deliberately NOT estimated here.
    #
    # SPIR-V has no register file, no allocation and no coherent notion of
    # "live value" that survives a driver's scheduler, so every quantity this
    # parser can compute (SSA count, live-in set size, instruction count) is
    # off by 3-10x against a real compiler's output -- measured against the
    # Intel driver on this tree: a naive SSA-count proxy gives 350-790 where
    # the driver reports 22-93.  Reporting such a number would be worse than
    # reporting none, because it looks like a measurement.
    #
    # The occupancy path therefore either uses the driver's own count (the
    # VK_KHR_pipeline_executable_properties probe) or states the register
    # assumption it used.  See Analysis.occupancy.
    info["phi_operands"] = len(set(phi_operands))
    return info


# ---------------------------------------------------------------------------
# Driver-side probe: VK_KHR_pipeline_executable_properties.
#
# This is the ONLY way to get a real register count.  It compiles each shader
# into a compute pipeline on the named device and asks the driver for
# "Max live registers" / "Workgroup Memory Size".  The probe is a small C++
# helper compiled on demand (tools/roofline_probe.cpp).  It is opt-in
# (--probe) because creating pipelines on a GPU is a side effect the analysis
# should not have by default; when it does not run, every occupancy row states
# the register premise it used instead.
# ---------------------------------------------------------------------------
PROBE_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "roofline_probe.cpp")


def probe_binary_path():
    """Build (once) and return the probe helper, or None."""
    src = PROBE_SOURCE
    if not os.path.exists(src):
        return None
    out = os.path.join(tempfile.gettempdir(), "vkop_roofline_probe")
    try:
        if (not os.path.exists(out) or
                os.path.getmtime(out) < os.path.getmtime(src)):
            cxx = shutil.which("g++") or shutil.which("clang++")
            if not cxx:
                return None
            r = subprocess.run([cxx, "-std=c++17", "-O1", "-o", out, src,
                                "-lvulkan"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return None
        return out
    except OSError:
        return None


def run_probe(spv_paths, device_index=None):
    """Return ({spv_path: {stat: value}}, {name, vendor}) -- empty when
    probing is impossible."""
    binary = probe_binary_path()
    if not binary or not spv_paths:
        return {}, {}
    cmd = [binary]
    if device_index is not None:
        cmd += ["--device-index", str(device_index)]
    try:
        r = subprocess.run(cmd, input="\n".join(spv_paths) + "\n",
                           capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return {}, {}
    out = {}
    devinfo = {}
    cur = None
    for line in r.stdout.splitlines():
        if line.startswith("DEVICE "):
            devinfo["name"] = line[7:].strip()
        elif line.startswith("VENDOR "):
            try:
                devinfo["vendor"] = int(line[7:].strip(), 0)
            except ValueError:
                pass
        elif line.startswith("SHADER "):
            cur = line[7:].strip()
            out.setdefault(cur, {})
        elif line.startswith("STAT ") and cur:
            rest = line[5:].strip()
            if "=" not in rest:
                continue
            k, v = rest.split("=", 1)
            try:
                out[cur][k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out, devinfo


# ---------------------------------------------------------------------------
# FlatBuffer reader for the VKOP schema (model/vkop_model.fbs).
#
# Hand-rolled on purpose: the model is mmap'd and up to 25 GB, and the Python
# flatbuffers bindings that ship with the repo materialise Vector() bytes when
# asked for the shader archive.  Only the root table, the node/initializer
# vectors and the VKSP blob view are read; nothing else is touched.
# ---------------------------------------------------------------------------
class FBTable:
    __slots__ = ("buf", "pos")

    def __init__(self, buf, pos):
        self.buf = buf
        self.pos = pos

    def field(self, vt_slot):
        """vt_slot is the schema's VT_* constant, i.e. the vtable byte offset
        (4 + 2*field_index)."""
        buf, pos = self.buf, self.pos
        vt = pos - struct.unpack_from("<i", buf, pos)[0]
        vtsize = struct.unpack_from("<H", buf, vt)[0]
        if vt_slot + 2 > vtsize:
            return None
        rel = struct.unpack_from("<H", buf, vt + vt_slot)[0]
        return None if rel == 0 else pos + rel

    def _ind(self, p):
        return p + struct.unpack_from("<I", self.buf, p)[0]

    def i32(self, idx, default=0):
        p = self.field(idx)
        return default if p is None else struct.unpack_from("<i", self.buf, p)[0]

    def u8(self, idx, default=0):
        p = self.field(idx)
        return default if p is None else self.buf[p]

    def u32(self, idx, default=0):
        p = self.field(idx)
        return default if p is None else struct.unpack_from("<I", self.buf, p)[0]

    def u64(self, idx, default=0):
        p = self.field(idx)
        return default if p is None else struct.unpack_from("<Q", self.buf, p)[0]

    def string(self, vt_slot):
        p = self.field(vt_slot)
        return "" if p is None else self._string_at(p)

    def _string_at(self, p):
        v = self._ind(p)
        n = struct.unpack_from("<I", self.buf, v)[0]
        return bytes(self.buf[v + 4:v + 4 + n]).decode("utf-8", "replace")

    def table(self, idx):
        p = self.field(idx)
        return None if p is None else FBTable(self.buf, self._ind(p))

    def _vector(self, idx):
        p = self.field(idx)
        if p is None:
            return None, 0
        v = self._ind(p)
        n = struct.unpack_from("<I", self.buf, v)[0]
        return v + 4, n

    def tables(self, idx):
        start, n = self._vector(idx)
        if start is None:
            return []
        return [FBTable(self.buf, self._ind(start + 4 * i)) for i in range(n)]

    def ints(self, idx):
        start, n = self._vector(idx)
        if start is None:
            return []
        return list(struct.unpack_from("<%di" % n, self.buf, start))

    def uints(self, idx):
        start, n = self._vector(idx)
        if start is None:
            return []
        return list(struct.unpack_from("<%dI" % n, self.buf, start))

    def strings(self, vt_slot):
        start, n = self._vector(vt_slot)
        if start is None:
            return []
        return [self._string_at(start + 4 * i) for i in range(n)]

    def byte_vector_view(self, idx):
        """(absolute offset, length) of a [ubyte] vector, without copying."""
        start, n = self._vector(idx)
        if start is None:
            return None, 0
        return start, n


# Model field ids from model/generated/vkop_model_generated.h
M_VERSION = 6
M_INPUTS, M_OUTPUTS, M_NODES, M_INITIALIZERS = 8, 10, 12, 14
M_SHADER_ARCHIVE = 34
M_KERNEL_PLANS = 36

N_OP_TYPE, N_NAME, N_ATTRS, N_INPUTS, N_OUTPUTS, N_PLAN = 4, 6, 8, 10, 12, 18
S_NAME, S_DIMS, S_DTYPE = 4, 6, 8
A_KEY, A_TYPE, A_SVAL, A_IVAL, A_FVAL, A_INTS, A_FLOATS, A_BVAL = (
    4, 6, 8, 10, 12, 14, 16, 18)
I_NAME, I_DTYPE, I_DIMS, I_OFFSET, I_SIZE = 4, 6, 8, 10, 12
I_MOE_GROUP, I_MOE_EXPERT_ID = 14, 16
P_OP_TYPE, P_IMAGE, P_IMAGE_FP16, P_BUFFER, P_BUFFER_FP16, P_VARIANTS = (
    4, 6, 8, 10, 12, 14)

class VkopModel:
    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self.fd = os.open(path, os.O_RDONLY)
        import mmap
        self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        ident = bytes(self.mm[4:8])
        if ident != b"VKOP":
            raise SystemExit(f"{path}: not a VKOP model (identifier {ident!r})")
        self.root = FBTable(self.mm, struct.unpack_from("<I", self.mm, 0)[0])
        version = self.root.u8(M_VERSION, 3)
        if version != 3:
            raise SystemExit(
                f"{path}: vkopbin version {version}, expected 3")
        self._load()

    def _load(self):
        r = self.root
        self.inputs = [self._shape(t) for t in r.tables(M_INPUTS)]
        self.outputs = [self._shape(t) for t in r.tables(M_OUTPUTS)]
        self.nodes = []
        for t in r.tables(M_NODES):
            self.nodes.append({
                "op_type": t.string(N_OP_TYPE),
                "name": t.string(N_NAME),
                "attrs": self._attrs(t),
                "inputs": [self._shape(s) for s in t.tables(N_INPUTS)],
                "outputs": [self._shape(s) for s in t.tables(N_OUTPUTS)],
                "plan": t.i32(N_PLAN, -1),
            })
        self.initializers = {}
        self.init_moe = {}
        for t in r.tables(M_INITIALIZERS):
            name = t.string(I_NAME)
            self.initializers[name] = {
                "dtype": t.string(I_DTYPE),
                "dims": t.uints(I_DIMS),
                "size": t.u64(I_SIZE),
                "offset": t.u64(I_OFFSET),
                "moe_group": t.string(I_MOE_GROUP),
                "moe_expert_id": t.i32(I_MOE_EXPERT_ID, -1),
            }
        self.plans = {}
        for t in r.tables(M_KERNEL_PLANS):
            self.plans[t.string(P_OP_TYPE)] = {
                "buffer": t.string(P_BUFFER),
                "buffer_fp16": t.string(P_BUFFER_FP16),
                "image": t.string(P_IMAGE),
                "image_fp16": t.string(P_IMAGE_FP16),
                "variants": t.strings(P_VARIANTS),
            }
        blob_off, blob_len = r.byte_vector_view(M_SHADER_ARCHIVE)
        self.archive = (ShaderArchive(self.mm, blob_off, blob_len)
                        if blob_off else ShaderArchive(self.mm, 0, 0))
        # External-blob mode: >800 MB models append the weight blob after the
        # FlatBuffer and store its start in the last 8 bytes (model/pypi/
        # onnx2vkop/inspect.py's ModelView documents the trailer).
        self.external_blob_offset = None
        self.initializer_bytes = 0
        for e in self.initializers.values():
            self.initializer_bytes += e["size"]
        self.initializer_bytes_unique = self._unique_weight_bytes()

    def _unique_weight_bytes(self):
        """Bytes of distinct initializer payloads (a weight consumed by two
        nodes is one buffer, not two)."""
        seen = set()
        total = 0
        for e in self.initializers.values():
            key = (e["offset"], e["size"])
            if e["size"] == 0 or key in seen:
                continue
            seen.add(key)
            total += e["size"]
        return total

    @staticmethod
    def _shape(t):
        return {
            "name": t.string(S_NAME),
            "dims": t.ints(S_DIMS),
            "dtype": t.string(S_DTYPE),
        }

    @staticmethod
    def _attrs(t):
        out = {}
        for a in t.tables(N_ATTRS):
            key = a.string(A_KEY)
            ty = a.u8(A_TYPE, 0)
            if ty == 1:
                out[key] = _i64(a)
            elif ty == 2:
                out[key] = _f32(a)
            elif ty == 3:
                out[key] = a.uints(A_INTS)
            elif ty == 4:
                out[key] = _floats(a)
            elif ty == 6:
                out[key] = bool(a.u8(A_BVAL, 0))
            else:
                out[key] = a.string(A_SVAL)
        return out

    def close(self):
        try:
            self.mm.close()
        finally:
            os.close(self.fd)


def _i64(t):
    p = t.field(A_IVAL)
    return 0 if p is None else struct.unpack_from("<q", t.buf, p)[0]


def _f32(t):
    p = t.field(A_FVAL)
    return 0.0 if p is None else struct.unpack_from("<f", t.buf, p)[0]


def _floats(t):
    start, n = t._vector(A_FLOATS)
    if start is None:
        return []
    return list(struct.unpack_from("<%df" % n, t.buf, start))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The two helpers this tool builds on demand.  Both are compiled from the
# repo's own sources, never vendored: roofline_oracle.cpp #includes the real
# core/KernelSelect.cpp, and roofline_probe.cpp talks to the driver.
ORACLE_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "roofline_oracle.cpp")


def up_div(a, b):
    return -(-a // b) if b else 0


def prod(xs):
    r = 1
    for x in xs:
        r *= x
    return r


class RoutingOracle:
    """Batched MatMul route queries against the real selector."""

    def __init__(self, override=None, device="nvidia-ada"):
        self.device = device
        self.binary = override or self._build()
        self.available = self.binary is not None

    def _build(self):
        if not os.path.exists(ORACLE_SOURCE):
            return None
        out = os.path.join(tempfile.gettempdir(), "vkop_roofline_oracle")
        deps = [ORACLE_SOURCE,
                os.path.join(REPO_ROOT, "core", "KernelSelect.cpp"),
                os.path.join(REPO_ROOT, "core", "KernelSelect.hpp"),
                os.path.join(REPO_ROOT, "vulkan", "VulkanLib.cpp")]
        try:
            if (os.path.exists(out) and
                    all(os.path.getmtime(out) >= os.path.getmtime(d)
                        for d in deps if os.path.exists(d))):
                return out
            cxx = shutil.which("g++") or shutil.which("clang++")
            if not cxx:
                return None
            r = subprocess.run(
                [cxx, "-std=c++17", "-O2", "-I", REPO_ROOT,
                 "-I", os.path.join(REPO_ROOT, "model", "generated"),
                 "-o", out, ORACLE_SOURCE,
                 os.path.join(REPO_ROOT, "vulkan", "VulkanLib.cpp"),
                 "-lvulkan", "-ldl", "-lpthread"],
                capture_output=True, text=True)
            if r.returncode != 0:
                return None
            return out
        except OSError:
            return None

    def query(self, requests):
        """requests: list of dicts.  Returns the same list with route fields."""
        if not self.available or not requests:
            return [dict(r, route=None, oracle_error="oracle unavailable")
                    for r in requests]
        lines = []
        for i, r in enumerate(requests):
            payload = {
                "qid": i,
                "batch": r.get("batch", 1),
                "m": r.get("m", 1),
                "k": r.get("k", 0),
                "n": r.get("n", 0),
                "n_up": r.get("n_up", -1),
                "fp16": bool(r.get("fp16", True)),
                "transB": bool(r.get("transB", False)),
                "weight": r.get("weight", "fp16"),
                "group_size": r.get("group_size", 0),
                "epi_bias": bool(r.get("epi_bias", False)),
                "attn_softmax": bool(r.get("attn_softmax", False)),
                "down_act": bool(r.get("down_act", False)),
                "device": self.device,
            }
            lines.append(json.dumps(payload))
        try:
            proc = subprocess.run([self.binary],
                                  input="\n".join(lines) + "\n",
                                  capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as e:
            return [dict(r, route=None, oracle_error=str(e)) for r in requests]
        got = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            got[obj.get("qid", -1)] = obj
        out = []
        for i, r in enumerate(requests):
            ans = got.get(i)
            if ans is None:
                out.append(dict(r, route=None,
                                oracle_error="no answer from oracle"))
            else:
                out.append(dict(r, **{k: v for k, v in ans.items()
                                      if k != "qid"}))
        return out


# ---------------------------------------------------------------------------
# Kernel launch model: one entry per (route, kernel key) the tool can attribute.
#
# local_size is NOT hard-coded: it is read from the SPIR-V at report time
# (`ls` below is only the documented fallback used when the archive lacks the
# key).  `grid` is a callable mirroring the dispatch site; every one names the
# source line it transcribes.
# ---------------------------------------------------------------------------
class Launch:
    __slots__ = ("key", "grid", "ls", "note", "dispatches", "items")

    def __init__(self, key, grid, ls=None, note="", dispatches=1, items=None):
        self.key = key
        self.grid = grid
        self.ls = ls
        self.note = note
        self.dispatches = dispatches
        # How many elements (or packed words) this dispatch must cover.  Used
        # only by the consistency check: a kernel here launches one thread per
        # item, so blocks*local_size >= items and the slack must be under two
        # workgroups.  None = the op's coverage is data dependent.
        self.items = items


def matmul_launches(plan, m, n, k, batch, n_up, weight_bytes, out_bytes,
                    act_bytes, weight_kind="fp16"):
    """Map a selector answer to the kernels it dispatches.

    Returns (launches, flops, bytes_read, bytes_written, classification_notes).
    """
    route = plan.get("route") or ""
    q = plan.get("ksplit_q") or 16
    ks_n = plan.get("ks_n") or 0
    bm = plan.get("bm") or 0
    bn = plan.get("bn") or 0
    flops = 2.0 * batch * m * n * k
    launches = []
    notes = []

    if route == "ksplit_w8wide512":
        launches.append(Launch("buffer_matmul_ksplit512_w8_wide_fp16",
                               lambda: (up_div(n // 16, q), batch * m, 1),
                               (16, 16), "ops/Matmul.hpp:2437 (512 lanes)",
                               items=batch * m * up_div(n, 16)))
    elif route == "ksplit_w8wide":
        launches.append(Launch("buffer_matmul_w8_wide_fp16",
                               lambda: (up_div(n // 16, q), batch * m, 1),
                               (16, 16),
                               "ops/Matmul.hpp:2479 UP_DIV(n/16,q), batch*m",
                               items=batch * m * up_div(n, 16)))
    elif route == "ksplit_wide512":
        launches.append(Launch("buffer_matmul_ksplit512_fp16",
                               lambda: (up_div(n // 8, q), batch * m, 1),
                               (16, 16), "ops/Matmul.hpp:2485 (512 lanes)",
                               items=batch * m * up_div(n, 8)))
    elif route == "ksplit_wide":
        launches.append(Launch("buffer_matmul_w8_wide_fp16",
                               lambda: (up_div(n // 8, q), batch * m, 1),
                               (16, 16), "ops/Matmul.hpp:2509 UP_DIV(n/8,q)",
                               items=batch * m * up_div(n, 8)))
    elif route == "ksplit512":
        launches.append(Launch("buffer_matmul_ksplit512_fp16",
                               lambda: (up_div(n // 4, q), batch * m, 1),
                               (16, 16), "ops/Matmul.hpp:2495 (512 lanes)",
                               items=batch * m * up_div(n, 4)))
    elif route == "ksplit":
        launches.append(Launch("buffer_matmul_fp16",
                               lambda: (up_div(n // 4, q), batch * m, 1),
                               (16, 16),
                               "ops/Matmul.hpp:2521 UP_DIV(n/4,q), batch*m",
                               items=batch * m * up_div(n, 4)))
    elif route == "tiled_ks":
        tiles = batch * up_div(n, 64) * up_div(m, 64)
        launches.append(Launch(
            "buffer_matmul_tiled_ks_split_fp16",
            lambda: (up_div(n, 64), up_div(m, 64), batch * max(ks_n, 1)),
            (16, 16), "ops/Matmul.hpp:2406 deposit, z = batch*ks_n",
            items=tiles * max(ks_n, 1)))
        launches.append(Launch(
            "buffer_matmul_tiled_ks_fold_fp16",
            lambda: (up_div(n, 64), up_div(m, 64), batch),
            (16, 16), "ops/Matmul.hpp:2411 fold pass", items=tiles))
    elif route == "tiled":
        launches.append(Launch("buffer_matmul_fp16",
                               lambda: (up_div(n, 64), up_div(m, 64), batch),
                               (16, 16),
                               "ops/Matmul.hpp:2423 64x64 tiles, z = batch",
                               items=batch * up_div(n, 64) * up_div(m, 64)))
    elif route == "coopmat":
        if bm <= 0 or bn <= 0:
            bm, bn = 32, 64
        key = ("buffer_matmul_coop16" if bm >= 32
               else "buffer_matmul_coop")
        launches.append(Launch(key, lambda: (up_div(n, bn), up_div(m, bm),
                                             batch), (16, 16),
                               "ops/Matmul.hpp:1832 UP_DIV(n,bn), UP_DIV(m,bm)",
                               items=batch * up_div(n, bn) * up_div(m, bm)))
        if not plan.get("coop_self_pack", True):
            nwords = up_div(batch * m * n, 2)
            launches.append(Launch("buffer_matmul_pack",
                                   lambda: (up_div(nwords, 256), 1, 1),
                                   (256, 1), "ops/Matmul.hpp:1850 pack pass"))
    elif route == "coopmat_sklast":
        # ops/Matmul.hpp:1897-1934: main pass grid (UP_DIV(n,bn), UP_DIV(m,bm),
        # batch) with uPC.ksplit = cs serial K passes inside each workgroup,
        # then the pack pass.  No grid.z split despite the route name -- the
        # cs depth is a serial loop in the kernel.
        if bm <= 0 or bn <= 0:
            bm, bn = 32, 64
        cs = max(1, plan.get("coop_split") or 1)
        launches.append(Launch(
            "buffer_matmul_coop16_sklast_fp32",
            lambda: (up_div(n, bn), up_div(m, bm), batch), (16, 16),
            f"ops/Matmul.hpp:1925 UP_DIV(n,bn),UP_DIV(m,bm); cs={cs} K-passes",
            items=batch * up_div(n, bn) * up_div(m, bm)))
        nwords = up_div(batch * m * n, 2)
        launches.append(Launch(
            "buffer_matmul_pack", lambda: (up_div(nwords, 256), 1, 1),
            (256, 1), "ops/Matmul.hpp:1934 pack pass", items=nwords))
    elif route == "coopmat_i8":
        if bm <= 0 or bn <= 0:
            bm, bn = 32, 64
        launches.append(Launch("buffer_matmul_coop_i8_16x16x32",
                               lambda: (up_div(n, bn), up_div(m, bm), batch),
                               (16, 16),
                               "ops/Matmul.hpp:1640 UP_DIV(n,bn),UP_DIV(m,bm)",
                               items=batch * up_div(n, bn) * up_div(m, bm)))
        nwords = up_div(batch * m * n, 2)
        launches.append(Launch("buffer_matmul_pack",
                               lambda: (up_div(nwords, 256), 1, 1), (256, 1),
                               "ops/Matmul.hpp:1657 pack pass"))
    elif route == "down_act_gemv":
        # Deposit / down / reduce, three dispatches of ONE module.  The grid
        # is the M==1 wide form; M>1 chunks rows and dispatches per chunk, so
        # the total dispatch count is ceil(M/128)*3.
        gx = up_div(n_up, 512)
        rows = max(1, batch * m)
        chunk = 128
        chunks = up_div(rows, chunk)
        launches.append(Launch(
            "buffer_matmul_down_act_fp16",
            lambda: (gx, up_div(min(rows, chunk), 16), 1), (32, 16),
            "ops/Matmul.hpp:2338 deposit (gx=UP_DIV(n_up,512), 16-row tiles)",
            dispatches=chunks))
        launches.append(Launch(
            "buffer_matmul_down_act_fp16",
            lambda: (gx, up_div(min(rows, chunk), 16), 1), (32, 16),
            "ops/Matmul.hpp:2349 down pass", dispatches=chunks))
        launches.append(Launch(
            "buffer_matmul_down_act_fp16",
            lambda: (up_div(n, 2048), min(rows, chunk), 1), (32, 16),
            "ops/Matmul.hpp:2355 reduce pass", dispatches=chunks))
    elif route == "fused_fp16":
        # ops/Matmul.hpp:2527: gx = UP_DIV(n / ((w8_quad||w4_quad)?4:2), 16).
        # w8_quad/w4_quad are only true on the kQuad4 route, so this one packs
        # two columns per lane (one half2 word).
        launches.append(Launch("buffer_matmul_fp16",
                               lambda: (up_div(n // 2, 16), up_div(batch * m, 16),
                                        1), (16, 16),
                               "ops/Matmul.hpp:2527 fused even-N reduce, "
                               "2 columns/lane"))
    elif route == "quad4":
        cols = 4 if weight_kind in ("int8", "fp8", "int4", "nvfp4") else 2
        launches.append(Launch("buffer_matmul_fp16",
                               lambda: (up_div(n // cols, 16),
                                        up_div(batch * m, 16), 1), (16, 16),
                               "ops/Matmul.hpp:2527 column-quad / quad-word"))
    else:  # reduce / reduce_pack / attn_softmax / unknown
        if route == "attn_softmax":
            launches.append(Launch("buffer_matmul_fp16",
                                   lambda: (up_div(n, 16), up_div(batch * m, 16), 1),
                                   (16, 16),
                                   "ops/Matmul.hpp:2555 attn_softmax reduce"))
        else:
            launches.append(Launch("buffer_matmul_fp16",
                                   lambda: (up_div(n, 16), up_div(batch * m, 16), 1),
                                   (16, 16),
                                   "ops/Matmul.hpp:2555 reduce, one thread/elem"))
        if route == "reduce_pack":
            nwords = up_div(batch * m * n, 2)
            launches.append(Launch("buffer_matmul_pack",
                                   lambda: (up_div(nwords, 256), 1, 1), (256, 1),
                                   "ops/Matmul.hpp:2570 odd-N pack"))
    return launches, flops, notes


# ---------------------------------------------------------------------------
# Op semantics: FLOPs, traffic, and the launch geometry for one node.
#
# Every entry below is a transcription of the op's own execute() (named by
# file:line), not a generic ONNX cost model: vkop dispatches its OWN fused
# kernels, so e.g. RMSNorm is one workgroup per row (not the 9 ONNX nodes),
# and a Reshape emits nothing at all.
# ---------------------------------------------------------------------------
ACT_FLOPS = {
    # unary, per element
    "Sigmoid": 4, "Tanh": 5, "Erf": 10, "Sqrt": 1, "Neg": 1, "Relu": 1,
    "Floor": 1, "Exp": 2, "Cos": 6, "Sin": 6, "Atan": 8, "Softplus": 8,
    "Cast": 0, "Pow": 1,
}


class NodeMetric:
    __slots__ = ("op_type", "name", "kind", "flops", "bytes_read",
                 "bytes_written", "weight_bytes", "launches", "notes",
                 "reg_source", "dispatch_count", "route", "shape_note",
                 "occupancies", "best_occupancy", "classification")

    def __init__(self, op_type, name):
        self.op_type = op_type
        self.name = name
        self.kind = "dispatch"
        self.flops = 0.0
        self.bytes_read = 0
        self.bytes_written = 0
        self.weight_bytes = 0
        self.launches = []      # [(key, grid, ls_fallback, note, dispatches)]
        self.notes = []
        self.reg_source = "static-bound"
        self.dispatch_count = 0
        self.route = ""
        self.shape_note = ""
        self.occupancies = []
        self.best_occupancy = None
        self.classification = ""

    @property
    def total_bytes(self):
        return self.bytes_read + self.bytes_written

    @property
    def intensity(self):
        return (self.flops / self.total_bytes) if self.total_bytes else 0.0

    def weight_kind(self):
        for tok in (self.shape_note or "").split():
            if tok.startswith("w="):
                return tok[2:]
        return ""


class Analysis:
    def __init__(self, model, device_name, oracle, probe, decode_m=1,
                 prefill_m=0, kv_len=1024):
        self.model = model
        self.dev = DEVICES[device_name]
        self.device_name = device_name
        self.oracle = oracle
        self.probe = probe
        self.decode_m = decode_m
        self.prefill_m = prefill_m
        self.kv_len = kv_len
        self.spv_cache = {}
        self.probe_device = {}
        # The register count used when the driver probe did not run for a
        # kernel.  It is NOT a measurement and the report says so on every row
        # that uses it.  When the probe DID run on this device, the median of
        # its own measurements is the least arbitrary value available, so it
        # is used instead of a constant.
        self.assumed_regs = 40
        self.assumed_regs_basis = (
            "unmeasured constant (SPIR-V carries no register count and no "
            "driver probe ran for this device); treat every occupancy figure "
            "that uses it as indicative")
        self.matmul_plans = {}
        self.metrics = []
        self.weight_seen = set()
        self.total_weight_bytes = 0
        self.total_weight_bytes_unique = model.initializer_bytes_unique

    # -- shape resolution -------------------------------------------------
    def resolve(self, shape, m, kv=False):
        """Replace dynamic sentinels for the analysis point.

        The convention (validated against every data tensor in the 2B and 27B
        exports): the LAST dynamic axis is the token axis and takes `m`; any
        earlier dynamic axis is the batch and takes 1.  KV-cache tensors
        ([..., kv, D]) take the cache length on that last axis instead.
        """
        dims = list(shape.get("dims") or [])
        out = [d for d in dims]
        last_dyn = -1
        for i, d in enumerate(dims):
            if d < 0:
                last_dyn = i
        for i, d in enumerate(dims):
            if d < 0:
                out[i] = (self.kv_len if kv else 1) if i == last_dyn else 1
        if last_dyn >= 0 and not kv:
            out[last_dyn] = m
        elif last_dyn >= 0 and kv:
            out[last_dyn] = self.kv_len
        return out

    def nbytes(self, shape, m, kv=False, dtype=None):
        d = dtype or shape.get("dtype") or "float32"
        dims = self.resolve(shape, m, kv)
        n = 1
        for x in dims:
            n *= max(x, 1)
        return n, int(n * dtype_bytes(d)), dims

    # -- SPIR-V facts ------------------------------------------------------
    def spv_info(self, key):
        if key in self.spv_cache:
            return self.spv_cache[key]
        info = {"local": None, "shared_static": 0, "shared_dynamic": 0,
                "found": False}
        blob = self.model.archive.get(key)
        if blob:
            info.update(_spirv_static(blob))
            info["found"] = True
        self.spv_cache[key] = info
        return info

    def driver_stats(self, key):
        """Driver-reported per-thread resources, or None.

        Only ever populated when --probe ran and the device accepted the
        pipeline; otherwise the caller states the register assumption it used.
        """
        if not self.probe:
            return None
        st = self.probe.get(key)
        if st is None:
            alt = _key_alternates(self.model, key)
            st = self.probe.get(alt) if alt else None
        return st

    # -- MatMul ------------------------------------------------------------
    def matmul(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        ins = node["inputs"]
        outs = node["outputs"]
        a_shape = ins[0]
        b_shape = ins[1] if len(ins) > 1 else {"dims": [], "dtype": ""}
        w_dtype = b_shape.get("dtype") or ""
        a_dtype = a_shape.get("dtype") or ""
        o_dtype = outs[0].get("dtype") or "" if outs else ""
        wkind = ("int8" if w_dtype == "int8" else
                 "fp8" if w_dtype.startswith("float8") else
                 "int4" if w_dtype in ("int4", "uint4") else
                 "nvfp4" if w_dtype in ("nf4", "float4e2m1") else "fp16")
        wbytes_elem = dtype_bytes(w_dtype or "float16")
        abytes_elem = dtype_bytes(a_dtype or "float16")
        obytes_elem = dtype_bytes(o_dtype or a_dtype or "float16")

        adims = self.resolve(a_shape, m)
        # ONNX MatMul: A [..., M, K], B [..., K, N] (or [..., N, K] @ transB=1)
        transB = bool(node["attrs"].get("transB", 0))
        k = adims[-1] if adims else 0
        a_m = adims[-2] if len(adims) >= 2 else 1
        batch = 1
        for d in adims[:-2]:
            batch *= max(d, 1)
        bdims = list(b_shape.get("dims") or [])
        if not bdims:
            # A SwiGLU-folded node's weight ShapeRefs are written with empty
            # dims (the converter rewrote the inputs and dropped them), so the
            # only place the real [K, N] lives is the initializer entry.
            e = self.model.initializers.get(b_shape.get("name") or "")
            if e and e["dims"]:
                bdims = [int(x) for x in e["dims"]]
                mt.notes.append("weight dims recovered from the initializer "
                                "entry (ShapeRef was empty)")
        if transB:
            n = bdims[-2] if len(bdims) >= 2 else 1
            k_b = bdims[-1] if bdims else 0
        else:
            n = bdims[-1] if bdims else 1
            k_b = bdims[-2] if len(bdims) >= 2 else 0
        if k_b and k and k != k_b:
            mt.notes.append(f"K mismatch: A contracts {k}, B has {k_b}")
        # The SwiGLU fold: this node's k is x's width, the down GEMM's
        # contraction is Wd's row count (ops/Matmul.hpp:1946-1975).
        down_act = bool(node["attrs"].get("down_act", 0))
        n_up = -1
        if down_act and len(ins) >= 4:
            wd_dims = list(ins[1].get("dims") or [])
            if not wd_dims:
                e = self.model.initializers.get(ins[1].get("name") or "")
                if e:
                    wd_dims = [int(x) for x in e["dims"]]
            n_up = wd_dims[0] if wd_dims else -1
        weight_nbytes = int(abs(k) * abs(n or 0) * wbytes_elem) if k and n else 0
        # A SwiGLU-folded node carries THREE weights and does THREE GEMMs:
        #   gate = x[K_out] . Wg[K_out, N_up]
        #   up   = x[K_out] . Wu[K_out, N_up]
        #   down = silu(gate)*up [N_up] . Wd[N_up, N]
        # This node's k is K_out (x's width), so Wd's own contraction -- and
        # its byte count -- is n_up, not k (ops/Matmul.hpp:1946-1975).
        fold_flops = 0.0
        if down_act and len(ins) >= 4 and n_up > 0:
            gate_up = 2.0 * batch * a_m * abs(k) * n_up
            down = 2.0 * batch * a_m * n_up * abs(n or 0)
            fold_flops = 2.0 * gate_up + down
            weight_nbytes = int(abs(n_up) * abs(n or 0) * wbytes_elem)
            for i in (2, 3):
                d = list(ins[i].get("dims") or [])
                if not d:
                    e = self.model.initializers.get(ins[i].get("name") or "")
                    if e:
                        d = [int(x) for x in e["dims"]]
                if d:
                    weight_nbytes += int(abs(prod(d)) * wbytes_elem)

        req = {
            "batch": batch, "m": a_m, "k": abs(k), "n": abs(n or 0),
            "n_up": n_up, "fp16": abytes_elem == 2 or a_dtype == "float16",
            "transB": transB, "weight": wkind,
            "group_size": 0, "epi_bias": bool(node["attrs"].get("epi_bias", 0)),
            "attn_softmax": bool(node["attrs"].get("attn_softmax", 0)),
            "down_act": down_act,
        }
        plan = self.oracle.query([req])[0]
        mt.route = plan.get("route") or ""
        mt.shape_note = (f"batch={batch} m={a_m} n={n} k={abs(k)} "
                         f"w={wkind} transB={int(transB)}")

        launches, flops, _ = matmul_launches(
            plan, a_m, abs(n or 0), abs(k), batch, n_up, weight_nbytes,
            int(a_m * batch * abs(n or 0) * obytes_elem),
            int(a_m * batch * abs(k) * abytes_elem), weight_kind=wkind)
        mt.flops = flops if fold_flops == 0.0 else fold_flops
        mt.bytes_read = int(a_m * batch * abs(k) * abytes_elem) + weight_nbytes
        mt.bytes_written = int(a_m * batch * abs(n or 0) * obytes_elem)
        mt.weight_bytes = weight_nbytes
        # A split-K GEMV issues one weight pass PER output row (grid.y =
        # batch*m): the L2 may absorb it, but the bytes leave the cache if the
        # weight is larger than it.  Reported, not added to bytes_read, so the
        # roofline denominator stays the minimum traffic.
        route = plan.get("route") or ""
        if route.startswith("ksplit") and batch * a_m > 1:
            mt.notes.append(
                f"split-K GEMV re-reads the weight {batch * a_m}x "
                f"(grid.y = batch*m); {weight_nbytes * batch * a_m / 1e9:.2f} GB "
                f"issued vs {weight_nbytes / 1e9:.2f} GB unique")
        mt.launches = launches
        mt.dispatch_count = sum(l.dispatches for l in launches)
        if plan.get("route") is None:
            mt.notes.append(
                "route UNKNOWN at static time (" + str(plan.get("oracle_error"))
                + "): the bytes and FLOPs below are exact, but no kernel, grid "
                  "or occupancy is attributed; every branch the selector could "
                  "take is listed in the tool's route table")
        if plan.get("reason"):
            mt.notes.append("selector: " + plan["reason"])
        return mt

    # -- generic elementwise ----------------------------------------------
    def elementwise(self, node, m, n_inputs=None, reads_per_elem=None):
        mt = NodeMetric(node["op_type"], node["name"])
        outs = node["outputs"]
        ndata, nbytes_out, _odims = self.nbytes(outs[0], m)
        mt.bytes_written = nbytes_out
        if node["op_type"] == "FusedElemwise":
            # The chain's inputs are broadcast to the output shape; a scalar
            # input costs 4 bytes, not N.  ops/attributes encode the program.
            prog = node["attrs"].get("ops") or []
            nops = len(prog) // 4
            flops_per_elem = 0.0
            for i in range(nops):
                code = prog[4 * i]
                # codes mirror ops/FusedElemwise.hpp's FusedOp
                flops_per_elem += {
                    1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 1, 7: 4, 8: 1, 9: 2,
                    10: 5, 11: 0, 12: 0, 13: 10, 14: 5,
                }.get(code, 1)
            mt.flops = flops_per_elem * ndata
            for s in node["inputs"]:
                nn, nb, _d = self.nbytes(s, m)
                mt.bytes_read += nb
            mt.launches = [Launch(
                "buffer_fused_elemwise_fp16",
                lambda: (up_div((ndata + 1) // 2 if shader_fp16(node)
                                else ndata, 256), 1, 1), (256, 1),
                "ops/FusedElemwise.hpp:349 UP_DIV(total/2, 256)",
                items=(ndata + 1) // 2 if shader_fp16(node) else ndata)]
        else:
            per = ACT_FLOPS.get(node["op_type"], 1)
            mt.flops = per * ndata
            if node["op_type"] in ("Add", "Sub", "Mul", "Div", "Max", "Min",
                                   "Where", "Equal", "Mod", "PRelu"):
                for s in node["inputs"]:
                    nn, nb, _d = self.nbytes(s, m)
                    mt.bytes_read += nb
                key = {"Add": "buffer_add_fp16", "Sub": "buffer_sub_fp16",
                       "Mul": "buffer_mul_fp16", "Div": "buffer_div_fp16",
                       "Max": "buffer_max_fp16", "Min": "buffer_min_fp16",
                       "Where": "buffer_where", "Equal": "buffer_equal",
                       "Mod": "buffer_mod_fp16",
                       "PRelu": "buffer_prelu_fp16"}.get(node["op_type"],
                                                        "buffer_add_fp16")
            else:
                mt.bytes_read = nbytes_out  # unary: same element count
                key = {"Sigmoid": "buffer_sigmoid_fp16",
                       "Softplus": "buffer_softplus_fp16",
                       "Pow": "buffer_pow_fp16", "Cos": "buffer_cos_fp16",
                       "Sin": "buffer_sin_fp16", "Sqrt": "buffer_sqrt_fp16",
                       "Neg": "buffer_neg_fp16",
                       "Cast": "buffer_cast"}.get(node["op_type"],
                                                  "buffer_add_fp16")
            nthreads = ndata // 2 if shader_fp16(node) else ndata
            mt.launches = [Launch(key, lambda: (up_div(nthreads, 256), 1, 1),
                                  (256, 1),
                                  f"ops/{node['op_type']}: UP_DIV(total, 256)",
                                  items=nthreads)]
        mt.dispatch_count = 1
        return mt

    # -- RMSNorm -----------------------------------------------------------
    def rmsnorm(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        x = node["inputs"][0]
        ndata, nb, dims = self.nbytes(x, m)
        inner = dims[-1] if dims else 0
        outer = max(1, ndata // inner) if inner else 0
        mt.bytes_read = nb + inner * dtype_bytes(
            node["inputs"][1].get("dtype") or "float16")
        mt.bytes_written = nb
        mt.flops = 4.0 * ndata
        mt.launches = [Launch("buffer_rmsnorm_fp16",
                              lambda: (outer, 1, 1), (256, 1),
                              "ops/RMSNorm.hpp:163 one WG per row")]
        mt.dispatch_count = 1
        return mt

    def rotary(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        ndata, nb, dims = self.nbytes(node["outputs"][0], m)
        mt.bytes_read = nb
        for s in node["inputs"][1:]:
            _nn, b, _d = self.nbytes(s, m)
            mt.bytes_read += b
        mt.bytes_written = nb
        mt.flops = 6.0 * ndata
        nthreads = (ndata + 1) // 2 if shader_fp16(node) else ndata
        mt.launches = [Launch("buffer_rotary_fp16",
                              lambda: (up_div(nthreads, 256), 1, 1), (256, 1),
                              "ops/RotaryEmbedding.hpp:272 UP_DIV(total/2,256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        return mt

    def attention(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        at = node["attrs"]
        heads = int(at.get("num_heads", 1) or 1)
        kvh = int(at.get("num_kv_heads", heads) or heads)
        q = node["inputs"][0]
        _qn, qbytes, qdims = self.nbytes(q, m)
        # Q's layout is [B, H, S, D] on both exports (heads on axis 1, the
        # token axis on axis 2), or [S, H, D] for the rank-3 test form.
        d = qdims[-1] if qdims else 1
        if len(qdims) >= 4:
            batch = max(qdims[0], 1)
            seq = max(qdims[2], 1)
        elif len(qdims) == 3:
            batch, seq = 1, max(qdims[0], 1)
        else:
            batch, seq = 1, 1
        kv_len = self.kv_len
        cache_dtype = "float16"
        if len(node["inputs"]) > 3:
            cache_dtype = node["inputs"][3].get("dtype") or "float16"
        cb = dtype_bytes(cache_dtype)
        # QK^T (2*B*H*S*kv*D) + PV (2*B*H*S*kv*D) + softmax (~5*B*H*S*kv)
        mt.flops = 4.0 * batch * heads * seq * kv_len * d + \
            5.0 * batch * heads * seq * kv_len
        mt.bytes_read = qbytes
        # K and V as this node sees them: the fresh tokens only (the history
        # is already in the cache buffer).
        mt.bytes_read += 2 * batch * kvh * seq * d * cb
        mt.bytes_read += 2 * batch * kvh * kv_len * d * cb   # cache in
        mt.bytes_written = qbytes                              # out
        mt.bytes_written += 2 * batch * kvh * kv_len * d * cb  # present
        grid = batch * heads * seq
        row_elems = 2 * kvh * d
        words = row_elems * kv_len // max(1, 4 // cb)
        grid += up_div(words, 256)
        mt.launches = [Launch("buffer_attention_fp16",
                              lambda: (grid, 1, 1), (256, 1),
                              "ops/Attention.hpp:821 one WG per query row + "
                              "cache-copy WGs (grid is data-dependent)")]
        mt.notes.append(f"kv_len={kv_len} is an analysis input (--kv-len), not "
                        f"a model constant")
        mt.dispatch_count = 1
        return mt

    def linear_attention(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        at = node["attrs"]
        heads = int(at.get("heads", 1) or 1)
        dk = int(at.get("dk", 1) or 1)
        dv = int(at.get("dv", 1) or 1)
        _qn, _qb, qdims = self.nbytes(node["inputs"][0], m)
        # ops/LinearAttention.hpp:206 documents both forms: the exporter's
        # rank-4 [B, S, H, D] (a token's heads adjacent) and the rank-3
        # [H, S, D] the tests use.
        if len(qdims) >= 4:
            batch, seq = max(qdims[0], 1), max(qdims[1], 1)
        elif len(qdims) == 3:
            batch, seq = 1, max(qdims[1], 1)
        else:
            batch, seq = 1, 1
        mt.flops = 4.0 * batch * heads * seq * dk * dv
        for i in (0, 1, 2, 3, 4):
            _n, b, _ = self.nbytes(node["inputs"][i], m)
            mt.bytes_read += b
        _n, b, _ = self.nbytes(node["inputs"][5], m, dtype="float32")
        mt.bytes_read += b
        _n, ob, _ = self.nbytes(node["outputs"][0], m)
        mt.bytes_written += ob
        if len(node["outputs"]) > 1:
            _n, sb, _ = self.nbytes(node["outputs"][1], m, dtype="float32")
            mt.bytes_written += sb
        mt.launches = [Launch("buffer_linear_attention_fp16",
                              lambda: (batch * heads, 1, 1), (128, 1),
                              "ops/LinearAttention.hpp:207 one WG per head")]
        mt.dispatch_count = 1
        return mt

    def conv1d(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        at = node["attrs"]
        kernel = int(at.get("kernel", 1) or 1)
        x = node["inputs"][0]
        _xn, xb, xdims = self.nbytes(x, m)
        channels = xdims[0] if xdims else 1
        seq = xdims[-1] if len(xdims) >= 2 else 1
        mt.flops = 2.0 * channels * seq * kernel
        mt.bytes_read = xb
        _n, wb, _ = self.nbytes(node["inputs"][1], m)
        mt.bytes_read += wb
        if len(node["inputs"]) > 2:
            _n, sb, _ = self.nbytes(node["inputs"][2], m, dtype="float32")
            mt.bytes_read += sb
        _n, ob, _ = self.nbytes(node["outputs"][0], m)
        mt.bytes_written = ob
        if len(node["outputs"]) > 1:
            _n, so, _ = self.nbytes(node["outputs"][1], m, dtype="float32")
            mt.bytes_written += so
        threads = (channels * seq + 1) // 2
        mt.launches = [
            Launch("buffer_conv1d_fp16",
                   lambda: (up_div(threads, 256), 1, 1), (256, 1),
                   "ops/Conv1d.hpp:172 conv pass", items=threads),
            Launch("buffer_conv1d_fp16",
                   lambda: (up_div(channels, 256), 1, 1), (256, 1),
                   "ops/Conv1d.hpp:180 state roll", items=channels),
        ]
        mt.dispatch_count = 2
        return mt

    # -- structural ops: alias or dispatch? -------------------------------
    def reshape(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        mt.kind = "view"
        mt.notes.append("ops/Reshape.hpp:519 aliases the input VkBuffer when "
                        "the input is GPU-resident (no dispatch, no bytes)")
        return mt

    def squeeze(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        mt.kind = "view"
        mt.notes.append("ops/SqueezeUnsqueeze.hpp:210 alias_storage_buffer "
                        "(metadata-only view)")
        return mt

    def transpose(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        perm = node["attrs"].get("perm")
        inshape = self.resolve(node["inputs"][0], m)
        if perm and _flat_identity_perm(inshape, perm):
            mt.kind = "view"
            mt.notes.append("ops/Transpose.hpp:334 flat_identity_perm -> alias "
                            "(bytes are identical under this perm)")
            return mt
        # Real gather dispatch.
        ndata, nb, _odims = self.nbytes(node["outputs"][0], m)
        mt.bytes_read = nb
        mt.bytes_written = nb
        mt.flops = 0
        nthreads = (ndata + 1) // 2 if shader_fp16(node) else ndata
        key = "buffer_transpose_fp16"
        info = self.model.archive
        if key not in info and "buffer_transpose" in info:
            key = "buffer_transpose"
        mt.launches = [Launch(key, lambda: (up_div(nthreads, 256), 1, 1),
                              (256, 1),
                              "ops/Transpose.hpp:315 UP_DIV(total,256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        mt.notes.append("perm is not a flat identity -> real gather dispatch")
        return mt

    def slice(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        # Static contiguity is unknowable here (starts/steps are runtime
        # tensors), so both branches are reported.
        ndata, nb, _ = self.nbytes(node["outputs"][0], m)
        mt.bytes_read = nb
        mt.bytes_written = nb
        nthreads = (ndata + 1) // 2 if shader_fp16(node) else ndata
        mt.launches = [Launch("buffer_slice_fp16",
                              lambda: (up_div(nthreads, 256), 1, 1), (256, 1),
                              "ops/Slice.hpp:426 UP_DIV(total[/2],256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        mt.notes.append("zero-copy alias (ops/Slice.hpp:369) fires only when "
                        "the runtime start/step make a contiguous run; the "
                        "bytes below are the dispatch fallback")
        return mt

    def split(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        at = node["attrs"]
        inshape = self.resolve(node["inputs"][0], m)
        axis = int(at.get("axis", 0) or 0)
        if axis < 0:
            axis += len(inshape)
        prefix = 1
        for d in inshape[:axis]:
            prefix *= max(d, 1)
        nouts = len(node["outputs"])
        if prefix == 1:
            mt.kind = "view"
            mt.notes.append(
                f"ops/Split.hpp:257 prefix==1 on axis {axis} -> every output "
                f"aliases a block of the input ({nouts} outputs, no dispatch)")
            return mt
        ndata, nb, _ = self.nbytes(node["outputs"][0], m)
        mt.bytes_read = nb * nouts
        mt.bytes_written = nb * nouts
        fp16 = shader_fp16(node)
        mt.launches = [Launch("buffer_split_fp16",
                              lambda: (up_div(((ndata + 1) // 2 if fp16
                                               else ndata), 256), 1, 1),
                              (256, 1),
                              "ops/Split.hpp:344 one dispatch per output",
                              dispatches=nouts,
                              items=((ndata + 1) // 2 if fp16 else ndata))]
        mt.dispatch_count = nouts
        mt.notes.append(f"axis {axis} has a non-unit prefix -> real gather")
        return mt

    def gather(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        ndata, nb, _odims = self.nbytes(node["outputs"][0], m)
        mt.bytes_read = nb
        mt.bytes_written = nb
        nthreads = (ndata + 1) // 2 if shader_fp16(node) else ndata
        mt.launches = [Launch("buffer_gather_fp16",
                              lambda: (up_div(nthreads, 256), 1, 1), (256, 1),
                              "ops/Gather.hpp:563 UP_DIV(total,256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        mt.notes.append("the zero-copy path (ops/Gather.hpp:478) needs host-"
                        "authoritative indices and a contiguous run; indices "
                        "here come from the graph, so the dispatch is assumed")
        return mt

    def concat(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        ndata, nb, _ = self.nbytes(node["outputs"][0], m)
        mt.bytes_read = 0
        for s in node["inputs"]:
            _n, b, _d = self.nbytes(s, m)
            mt.bytes_read += b
        mt.bytes_written = nb
        n = len(node["inputs"])
        fp16 = shader_fp16(node)
        mt.launches = [Launch("buffer_concat_fp16",
                              lambda: (up_div(((ndata + 1) // 2 if fp16
                                               else ndata), 256), 1, 1),
                              (256, 1),
                              "ops/Concat.hpp:470 one dispatch per input",
                              dispatches=n,
                              items=((ndata + 1) // 2 if fp16 else ndata))]
        mt.dispatch_count = n
        return mt

    def expand(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        ndata, nb, _ = self.nbytes(node["outputs"][0], m)
        _n, ib, _ = self.nbytes(node["inputs"][0], m)
        mt.bytes_read = ib
        mt.bytes_written = nb
        nthreads = (ndata + 1) // 2 if shader_fp16(node) else ndata
        mt.launches = [Launch("buffer_expand_fp16",
                              lambda: (up_div(nthreads, 256), 1, 1), (256, 1),
                              "ops/Expand.hpp:362 UP_DIV(total[/2],256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        return mt

    def reduce(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        at = node["attrs"]
        _ni, ib, _ = self.nbytes(node["inputs"][0], m)
        _no, ob, _ = self.nbytes(node["outputs"][0], m)
        mt.bytes_read = ib
        mt.bytes_written = ob
        mt.flops = 2.0 * _ni  # one add per reduced element (mean also divides)
        nthreads = (_no + 1) // 2 if shader_fp16(node) else _no
        mt.launches = [Launch("buffer_reduce_fp16",
                              lambda: (up_div(nthreads, 256), 1, 1), (256, 1),
                              "ops/Reduce.hpp:249 UP_DIV(out_total[/2],256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        return mt

    def scatter(self, node, m):
        mt = NodeMetric(node["op_type"], node["name"])
        _no, ob, _ = self.nbytes(node["outputs"][0], m)
        mt.bytes_written = ob
        for s in node["inputs"]:
            _n, b, _d = self.nbytes(s, m)
            mt.bytes_read += b
        nthreads = (_no + 1) // 2 if shader_fp16(node) else _no
        mt.launches = [Launch("buffer_scatter_elements_fp16",
                              lambda: (up_div(nthreads, 256), 1, 1), (256, 1),
                              "ops/ScatterElements.hpp:220 UP_DIV(n[/2],256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        return mt

    def conv2d(self, node, m):
        """ops/Conv2d.hpp:695-709 -- one thread per output element."""
        mt = NodeMetric(node["op_type"], node["name"])
        x = node["inputs"][0]
        w = node["inputs"][1]
        _xn, xb, xdims = self.nbytes(x, m)
        _wn, wb, wdims = self.nbytes(w, m)
        _on, ob, _odims = self.nbytes(node["outputs"][0], m)
        groups = int(node["attrs"].get("group", 1) or 1)
        kh = wdims[-2] if len(wdims) >= 2 else 1
        kw = wdims[-1] if wdims else 1
        cin_per_group = (wdims[1] if len(wdims) >= 2 else 1)
        out_elems = _on
        mt.flops = 2.0 * out_elems * cin_per_group * kh * kw
        mt.bytes_read = xb + wb
        if len(node["inputs"]) > 2 and node["inputs"][2].get("dims"):
            _bn, bb, _ = self.nbytes(node["inputs"][2], m)
            mt.bytes_read += bb
        mt.bytes_written = ob
        nthreads = (out_elems + 1) // 2 if shader_fp16(node) else out_elems
        mt.launches = [Launch("buffer_conv2d_fp16",
                              lambda: (up_div(nthreads, 256), 1, 1), (256, 1),
                              "ops/Conv2d.hpp:709 UP_DIV(items,256)",
                              items=nthreads)]
        mt.dispatch_count = 1
        mt.notes.append(f"groups={groups}, kernel {kh}x{kw}")
        return mt

    def host_op(self, node, m, why):
        mt = NodeMetric(node["op_type"], node["name"])
        mt.kind = "host"
        mt.notes.append(why)
        return mt

    # -- driver -----------------------------------------------------------
    def run(self, m):
        """Analyse every node at one analysis point; returns [NodeMetric]."""
        out = []
        for node in self.model.nodes:
            op = node["op_type"]
            if op == "MatMul":
                mt = self.matmul(node, m)
            elif op in ("Add", "Sub", "Mul", "Div", "Max", "Min", "Where",
                        "Equal", "Mod", "PRelu", "FusedElemwise", "Sigmoid",
                        "Softplus", "Pow", "Cos", "Sin", "Sqrt", "Neg", "Cast",
                        "Tanh", "Erf", "Relu", "Floor", "Exp", "Atan"):
                mt = self.elementwise(node, m)
            elif op in ("RMSNorm", "LayerNorm"):
                mt = self.rmsnorm(node, m)
            elif op == "RotaryEmbedding":
                mt = self.rotary(node, m)
            elif op == "Attention":
                mt = self.attention(node, m)
            elif op == "LinearAttention":
                mt = self.linear_attention(node, m)
            elif op == "Conv1d":
                mt = self.conv1d(node, m)
            elif op in ("Conv2d", "Conv"):
                mt = self.conv2d(node, m)
            elif op == "Reshape":
                mt = self.reshape(node, m)
            elif op in ("Squeeze", "Unsqueeze"):
                mt = self.squeeze(node, m)
            elif op == "Transpose":
                mt = self.transpose(node, m)
            elif op == "Slice":
                mt = self.slice(node, m)
            elif op == "Split":
                mt = self.split(node, m)
            elif op == "Gather":
                mt = self.gather(node, m)
            elif op == "Concat":
                mt = self.concat(node, m)
            elif op == "Expand":
                mt = self.expand(node, m)
            elif op == "Reduce":
                mt = self.reduce(node, m)
            elif op == "ScatterElements":
                mt = self.scatter(node, m)
            elif op == "NonZero":
                mt = self.host_op(node, m,
                                  "ops/NonZero.hpp: host scan of a tiny mask, "
                                  "no GPU dispatch")
            elif op == "Shape":
                mt = self.host_op(node, m, "shape metadata, host-evaluated")
            else:
                mt = self.host_op(node, m, "no kernel model in this tool: "
                                           "counted as host-only")
            out.append(mt)

        # Unique weight traffic for this analysis point: the same initializer
        # bound by several nodes is one buffer and one DRAM read at m=1.
        seen = set()
        wb = 0
        for node, mt in zip(self.model.nodes, out):
            if node["op_type"] != "MatMul" or len(node["inputs"]) < 2:
                continue
            nm = node["inputs"][1]["name"]
            e = self.model.initializers.get(nm)
            if e and e["size"] and nm not in seen:
                seen.add(nm)
                wb += e["size"]
            if node["attrs"].get("down_act") and len(node["inputs"]) >= 4:
                for i in (2, 3):
                    nm2 = node["inputs"][i]["name"]
                    e2 = self.model.initializers.get(nm2)
                    if e2 and e2["size"] and nm2 not in seen:
                        seen.add(nm2)
                        wb += e2["size"]
        self.weight_bytes_at_m = wb
        return out

    # -- occupancy --------------------------------------------------------
    def occupancy(self, key, ls, grid):
        """Theoretical occupancy for one launch.

        The model is the device's own resource accounting, not a heuristic:

          workgroups resident <= min over the four hard limits
            registers : reg_file_bytes(SM) / regs_per_thread rounded to the
                        device's allocation granule, divided by the wg size
            slots     : resident-threads(SM) / wg size
            block cap : maxComputeWorkGroupInvocations / wg size
            shared    : shared-memory(SM) / shared bytes per workgroup

        A launch always gets at least one workgroup resident (the driver spills
        or allocates what the pipeline declared), so the result is floored at
        1 -- a static number that claims 0 would be a contradiction, not a
        finding.

        `reg_source` distinguishes the three possible provenances of the
        register count, and the report prints it on the row:
          driver  -- VK_KHR_pipeline_executable_properties reported it
          assumed -- no driver number was available; a stated constant is used
        """
        info = self.spv_info(key)
        stats = self.driver_stats(key)
        if ls is None:
            ls = info.get("local") or None
        if ls is None:
            return None
        ls = list(ls) + [1] * (3 - len(ls))
        wg = int(ls[0]) * int(ls[1]) * int(ls[2])
        if wg <= 0:
            return None
        dev = self.dev
        shared = info["shared_static"]
        reg_per_thread = None
        reg_source = "assumed"
        simd = None
        if stats:
            if "Max live registers" in stats:
                reg_per_thread = int(stats["Max live registers"])
                reg_source = "driver"
            if "Workgroup Memory Size" in stats:
                shared = int(stats["Workgroup Memory Size"])
            if "Max dispatch width" in stats:
                simd = int(stats["Max dispatch width"])
        if reg_per_thread is None:
            reg_per_thread = self.assumed_regs
            reg_source = "assumed"

        # Registers are allocated in granules.  NVIDIA: one allocation unit is
        # 256 registers for a whole warp (so a warp costs
        # ceil(regs*32/256) * 256 registers).  Intel: one unit is 8 registers
        # (32 B) PER THREAD.
        if dev["vendor_id"] == 0x10DE:
            per_warp = (up_div(reg_per_thread * dev["subgroup_size"],
                               dev["reg_granule_regs"]) * dev["reg_granule_regs"])
            reg_bytes_per_thread = (per_warp / dev["subgroup_size"]) * 4
        else:
            reg_bytes_per_thread = (up_div(reg_per_thread, dev["reg_granule_regs"])
                                    * dev["reg_granule_regs"]) * 4

        reg_limit = int(dev["reg_file_bytes_per_sm"] //
                        max(1, reg_bytes_per_thread * wg))
        # maxComputeWorkGroupInvocations caps the SIZE of one workgroup, not
        # how many fit on an SM, so it is not a residency limit; the residency
        # caps below are.  Kept in the dict only so the report can show that
        # the launch respects it.
        block_limit = 1 << 30
        # Resident threads per SM.  NVIDIA: a fixed 1536.  Intel Xe: 
        # EU/SM * 8 hardware threads/EU * SIMD width the driver compiled for.
        if dev["threads_per_sm"]:
            threads_per_sm = dev["threads_per_sm"]
        else:
            threads_per_sm = (dev["eu_per_sm"] * dev["threads_per_eu"]
                              * (simd or 8))
        slot_limit = threads_per_sm // wg
        shared_limit = (int(dev["shared_per_sm"] // shared)
                        if shared > 0 else 1 << 30)
        max_wg = max(1, min(reg_limit, block_limit, slot_limit, shared_limit))
        resident_threads = max_wg * wg
        return {
            "key": key,
            "local_size": (int(ls[0]), int(ls[1]), int(ls[2])),
            "wg_size": wg,
            "shared_bytes": shared,
            "shared_dynamic_arrays": info.get("shared_dynamic", 0),
            "regs_per_thread": int(reg_per_thread),
            "reg_source": reg_source,
            "simd_width": simd,
            "limits": {"regs": reg_limit, "threads": slot_limit,
                       "shared": shared_limit},
            "threads_per_sm": threads_per_sm,
            "max_workgroups_per_sm": max_wg,
            "occupancy_pct": 100.0 * resident_threads / threads_per_sm,
            "blocks": grid,
            "waves": (grid / (max_wg * dev["sm"])) if max_wg else float("inf"),
        }


def o_dtype16(node):
    """Output dtype is fp16 (used only where the kernel is built from it)."""
    outs = node.get("outputs") or []
    return bool(outs) and (outs[0].get("dtype") in ("float16", "bfloat16"))


def in_dtype16(node):
    ins = node.get("inputs") or []
    return bool(ins) and (ins[0].get("dtype") in ("float16", "bfloat16"))


# ops::OpType values core/runtime.cpp:op_shader_fp16 keys the build on.  Every
# op listed there builds the half kernel from its DATA INPUT's dtype; only
# FUSED_ELEMWISE builds from the OUTPUT's (a fused chain may start fp32 and
# end fp16).  Everything not in the list keeps the runtime's global precision,
# which the .vkopbin's own dtypes make moot -- those ops are modelled from
# their input dtype, which is what the runtime effectively sees.
_BUILD_FROM_OUTPUT = {"FusedElemwise"}


def shader_fp16(node):
    if node.get("op_type") in _BUILD_FROM_OUTPUT:
        return o_dtype16(node)
    return in_dtype16(node)


def _flat_identity_perm(shape, perm):
    """ops/Transpose.hpp:61, transcribed."""
    rank = len(shape)
    perm = list(perm)
    for i in range(len(perm), rank):
        perm.append(i)
    if len(perm) != rank:
        return False
    in_stride = [1] * rank
    out_stride = [1] * rank
    for d in range(rank - 2, -1, -1):
        in_stride[d] = in_stride[d + 1] * max(shape[d + 1], 1)
    for d in range(rank - 2, -1, -1):
        out_stride[d] = out_stride[d + 1] * max(shape[perm[d + 1]], 1)
    for i in range(rank):
        if shape[perm[i]] == 1:
            continue
        if out_stride[i] != in_stride[perm[i]]:
            return False
    return True


# ---------------------------------------------------------------------------
# Roofline classification and rendering
# ---------------------------------------------------------------------------
class Roofline:
    def __init__(self, dev, device_name, sm):
        self.dev = dev
        self.device_name = device_name
        self.sm = sm
        self.peak = {
            "fp32": dev["flops_fp32"],
            "fp16": dev["flops_fp16"],
            "int8": dev["flops_int8"],
        }
        self.bw = dev["bandwidth_gbs"] * 1e9
        self.ridge = {k: v / self.bw for k, v in self.peak.items()}

    def peak_for(self, op_type, route="", weight=""):
        if weight == "int8":
            return "int8"
        if op_type == "MatMul":
            if weight in ("int8", "fp8", "int4", "nvfp4"):
                # A weight-only GEMV does the 'multiply' in fp32 registers; the
                # int8 peak applies only to a real int8 MMA (coop_i8).
                return "int8" if route == "coopmat_i8" else "fp32"
            return "fp16" if route.startswith(("coop", "ksplit_w8wide",
                                               "ksplit_wide", "fused", "tiled",
                                               "down_act")) else "fp32"
        if op_type in ("RMSNorm", "RotaryEmbedding", "Attention",
                       "LinearAttention", "Conv1d", "FusedElemwise"):
            return "fp16"
        return "fp32"

    def classify(self, metric, peak_kind):
        """memory-bound / compute-bound / latency-bound.

        Latency-bound is a GEOMETRY statement and is tested first: a launch
        that cannot fill the device (fewer resident workgroups than SMs, i.e.
        under one wave) has a meaningless achieved intensity -- most of the
        GPU is idle by construction.  Otherwise the kernel is placed against
        the machine balance: intensity vs the device's ridge point.
        """
        loc = metric.best_occupancy
        if loc is not None:
            # Latency-bound = there are fewer workgroups than the device has
            # SMs, so at least one SM gets no work at all and the kernel's
            # runtime is fixed by launch and memory latency rather than by
            # bandwidth or FLOPs.  A stricter "under one full wave" was tried
            # and rejected: it labelled a 622 MB weight stream with 75 blocks
            # on 48 SMs as latency-bound, which is not what a reader means --
            # a launch that saturates DRAM with 1.5 blocks per SM is memory
            # bound, just at partial occupancy.
            if loc["blocks"] < self.sm:
                return "latency-bound"
        intensity = metric.intensity
        ridge = self.ridge[peak_kind]
        return "memory-bound" if intensity < ridge else "compute-bound"


def fmt_bytes(n):
    if n >= 1e12:
        return f"{n / 1e12:.2f} TB"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} kB"
    return f"{n:.0f} B"


def fmt_flops(n):
    if n >= 1e12:
        return f"{n / 1e12:.2f} TF"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GF"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MF"
    if n >= 1e3:
        return f"{n / 1e3:.1f} kF"
    return f"{n:.0f} F"


def table(headers, rows, align=None):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    align = align or ["l"] * len(headers)
    out = []
    hdr = "  ".join((str(h).ljust(widths[i]) if align[i] == "l"
                     else str(h).rjust(widths[i]))
                    for i, h in enumerate(headers))
    out.append(hdr)
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append("  ".join((str(c).ljust(widths[i]) if align[i] == "l"
                              else str(c).rjust(widths[i]))
                             for i, c in enumerate(r)))
    return "\n".join(out)


def build_report(analysis, metrics, roofline, args):
    dev = analysis.dev
    lines = []
    A = lines.append

    total_flops = sum(m.flops for m in metrics)
    total_bytes = sum(m.total_bytes for m in metrics if m.kind == "dispatch")
    total_dispatch = sum(1 for m in metrics if m.kind == "dispatch")
    n_views = sum(1 for m in metrics if m.kind == "view")
    n_host = sum(1 for m in metrics if m.kind == "host")

    A("=" * 78)
    A(f"ROOFLINE REPORT  {os.path.basename(analysis.model.path)}")
    A("=" * 78)
    A(f"device            : {dev['name']}  (preset '{analysis.device_name}')")
    A(f"  SM / lanes      : {roofline.sm} SM x {dev['cores_per_sm']} lanes"
      f" @ {dev['clock_ghz']:.2f} GHz")
    A(f"  peak fp32       : {roofline.peak['fp32'] / 1e12:.2f} TFLOP/s")
    A(f"  peak fp16       : {roofline.peak['fp16'] / 1e12:.2f} TFLOP/s "
      f"(packed half2 ALUs; coopmat MMA not modelled separately)")
    A(f"  peak int8       : {roofline.peak['int8'] / 1e12:.2f} TOP/s "
      f"(packed int8x4 ALUs)")
    A(f"  DRAM bandwidth  : {dev['bandwidth_gbs']:.0f} GB/s")
    A(f"  ridge points    : fp32 {roofline.ridge['fp32']:.1f}  "
      f"fp16 {roofline.ridge['fp16']:.1f}  "
      f"int8 {roofline.ridge['int8']:.1f}  FLOP/Byte")
    A(f"  device figures  : {dev['source']}")
    A("")
    A(f"model             : {len(analysis.model.nodes)} nodes, "
      f"{len(analysis.model.initializers)} initializers, "
      f"{fmt_bytes(analysis.model.initializer_bytes_unique)} unique weights")
    A(f"analysis point    : m = {args.decode_m} (decode)"
      + (f", m = {args.prefill_m} (prefill)" if args.prefill_m else ""))
    A(f"kv length (attn)  : {analysis.kv_len}")
    A(f"embedded shaders  : {len(analysis.model.archive.keys())} VKSP entries")
    A(f"routing oracle    : "
      + ("core/KernelSelect.cpp, live" if analysis.oracle.available
         else "UNAVAILABLE -- every MatMul route is reported as unknown"))
    n_probed = len(analysis.probe)
    A(f"register source   : " + (
        f"driver-reported for {n_probed} kernels via "
        f"VK_KHR_pipeline_executable_properties; the rest use "
        f"{analysis.assumed_regs} registers -- {analysis.assumed_regs_basis}"
        if n_probed else
        f"NO DRIVER PROBE. Every occupancy row uses {analysis.assumed_regs} "
        f"registers: {analysis.assumed_regs_basis}. Occupancy figures are "
        f"therefore INDICATIVE. Run with --probe on the target GPU for "
        f"measured register counts."))
    A("")
    A(f"dispatch nodes    : {total_dispatch}")
    A(f"zero-copy views   : {n_views}  (no kernel, no bytes)")
    A(f"host-only / other : {n_host}")
    A("")
    A(f"TOTAL FLOPs       : {fmt_flops(total_flops)}")
    A(f"TOTAL bytes       : {fmt_bytes(total_bytes)}")
    A(f"overall intensity : {total_flops / total_bytes if total_bytes else 0:.2f}"
      f" FLOP/Byte")
    A("")

    # -- per-op-type table -------------------------------------------------
    A("-" * 78)
    A("1. BY OP TYPE")
    A("-" * 78)
    agg = defaultdict(lambda: {"n": 0, "flops": 0.0, "bytes": 0, "wbytes": 0})
    for m in metrics:
        a = agg[m.op_type]
        a["n"] += 1
        a["flops"] += m.flops
        if m.kind == "dispatch":
            a["bytes"] += m.total_bytes
        a["wbytes"] += m.weight_bytes
    rows = []
    for op, a in sorted(agg.items(), key=lambda kv: -kv[1]["bytes"]):
        inten = a["flops"] / a["bytes"] if a["bytes"] else 0.0
        rows.append([op, a["n"], fmt_flops(a["flops"]), fmt_bytes(a["bytes"]),
                     f"{inten:.2f}", fmt_bytes(a["wbytes"])])
    A(table(["op_type", "calls", "FLOPs", "bytes", "FLOP/B", "weights"], rows,
            ["l", "r", "r", "r", "r", "r"]))
    A("")

    # -- roofline buckets --------------------------------------------------
    A("-" * 78)
    A("2. ROOFLINE CLASSIFICATION (dispatch nodes, sorted by bytes)")
    A("-" * 78)
    # Roofline buckets.  "latency-bound" is a GEOMETRY verdict: fewer
    # workgroups than SMs means part of the device gets no work, so the time is
    # fixed by launch and memory latency and the intensity is meaningless.
    # Everything else is placed by intensity against the device's ridge point
    # for the precision the kernel actually runs at.
    buckets = defaultdict(lambda: {"n": 0, "bytes": 0, "flops": 0.0})
    for m in metrics:
        if m.kind != "dispatch":
            continue
        if m.total_bytes == 0 and m.flops == 0:
            continue
        pk = roofline.peak_for(m.op_type, m.route, _weight_of(m))
        cls = roofline.classify(m, pk)
        m.classification = cls
        b = buckets[cls]
        b["n"] += 1
        b["bytes"] += m.total_bytes
        b["flops"] += m.flops
    rows = []
    for cls in ("memory-bound", "compute-bound", "latency-bound"):
        b = buckets.get(cls)
        if not b:
            continue
        rows.append([cls, b["n"], fmt_bytes(b["bytes"]), fmt_flops(b["flops"]),
                     f"{b['flops'] / b['bytes'] if b['bytes'] else 0:.2f}"])
    A(table(["class", "nodes", "bytes", "FLOPs", "FLOP/B"], rows,
            ["l", "r", "r", "r", "r"]))
    A("")
    A(f"  Heaviest {args.top} dispatch nodes by bytes:")
    top = sorted([m for m in metrics if m.kind == "dispatch"],
                 key=lambda m: -m.total_bytes)[:max(1, args.top)]
    rows = []
    for m in top:
        rows.append([m.name[:44], m.op_type, m.route or "-",
                     fmt_bytes(m.total_bytes),
                     f"{m.intensity:.2f}",
                     getattr(m, "classification", "-")])
    A(table(["node", "op", "route", "bytes", "FLOP/B", "class"], rows,
            ["l", "l", "l", "r", "r", "l"]))
    A("")

    # -- occupancy ---------------------------------------------------------
    A("-" * 78)
    A("3. THEORETICAL OCCUPANCY (per distinct kernel, from its real launch)")
    A("-" * 78)
    occ_rows = defaultdict(lambda: None)
    for m in metrics:
        for loc in getattr(m, "occupancies", []):
            k = (loc["key"], loc["local_size"], loc["wg_size"],
                 loc["shared_bytes"], loc["regs_per_thread"],
                 loc["max_workgroups_per_sm"])
            if occ_rows[k] is None:
                occ_rows[k] = {"loc": loc, "uses": 0, "blocks": loc["blocks"]}
            occ_rows[k]["uses"] += 1
    rows = []
    for k, v in sorted(occ_rows.items(), key=lambda kv: kv[1]["loc"]["key"]):
        loc = v["loc"]
        ls = "x".join(str(x) for x in loc["local_size"])
        rows.append([loc["key"][:40], ls, loc["wg_size"],
                     fmt_bytes(loc["shared_bytes"]),
                     f"{loc['regs_per_thread']}"
                     + ("" if loc["reg_source"] == "driver" else "~"),
                     loc["max_workgroups_per_sm"],
                     f"{loc['occupancy_pct']:.0f}%",
                     v["blocks"] if v["blocks"] < 1e9 else "~",
                     "driver" if loc["reg_source"] == "driver"
                     else "assumed"])
    A(table(["kernel", "local", "wg", "shared", "regs", "WG/SM", "occ",
             "blocks", "reg src"], rows,
            ["l", "l", "r", "r", "r", "r", "r", "r", "l"]))
    A("")
    A("  occ  = resident threads / SM thread budget (100% = all slots busy)")
    A("  WG/SM = resident workgroups per SM = min(register limit, thread")
    A("          slots, shared-memory limit); floored at 1 because a created")
    A("          pipeline always gets one block resident.")
    A("  shared = workgroup shared memory.  With a driver number it is the")
    A("           compiled kernel's actual usage; without one it is the sum of")
    A("           the entry point's reachable Workgroup arrays, which can")
    A("           OVER-count a module that carries several #if branches")
    A("           (measured: 32768 static vs 32000 driver for matmul_fp16).")
    A("  reg src 'driver' = read from the driver via")
    A("          VK_KHR_pipeline_executable_properties (a real measurement);")
    A("          'assumed' (regs shown with '~') = no driver number was")
    A("          available and the premise in the header was used.  SPIR-V")
    A("          cannot supply a register count: only the driver's compiler")
    A("          knows it, and every SSA-based proxy tried here was 3-10x off")
    A("          the real value, so none is reported.")
    A("")

    # -- suspicious --------------------------------------------------------
    A("-" * 78)
    A("4. SUSPICIOUS / NOTEWORTHY")
    A("-" * 78)
    findings = []

    # Very low intensity with large byte volume, aggregated per (op, route)
    # so a 498-node MatMul does not fill the whole section with one shape.
    low = defaultdict(lambda: {"n": 0, "bytes": 0, "flops": 0.0, "name": "",
                               "route": ""})
    for m in metrics:
        if m.kind != "dispatch" or m.total_bytes < 1e7:
            continue
        k = (m.op_type, m.route)
        a = low[k]
        a["n"] += 1
        a["bytes"] += m.total_bytes
        a["flops"] += m.flops
        if not a["name"]:
            a["name"] = m.name
            a["route"] = m.route
    for (op, route), a in sorted(low.items(),
                                 key=lambda kv: -kv[1]["bytes"])[:8]:
        inten = a["flops"] / a["bytes"] if a["bytes"] else 0
        pk = roofline.peak_for(op, route, "")
        ridge = roofline.ridge[pk]
        if inten < 0.5 * ridge:
            findings.append(
                f"LOW INTENSITY  {op}"
                f"{'/' + route if route else ''} x{a['n']} "
                f"({a['name'][:32]}...): {fmt_bytes(a['bytes'])} total for "
                f"{fmt_flops(a['flops'])} -> {inten:.2f} FLOP/B vs a "
                f"{ridge:.1f} ridge -> memory-bound, no compute headroom to "
                f"win")

    # Device fill: a launch with fewer blocks than the device has SMs cannot
    # occupy it no matter how good the kernel is.
    fill = defaultdict(lambda: {"n": 0, "blocks": 0, "name": "", "occs": []})
    for m in metrics:
        for loc in getattr(m, "occupancies", []):
            if loc["blocks"] <= 0:
                findings.append(
                    f"EMPTY DISPATCH  {m.name[:40]} ({loc['key']}) computed a "
                    f"0-workgroup grid -- the shape at this analysis point is "
                    f"empty (a zero dim), so the node costs a submit and no "
                    f"work")
                continue
            if loc["blocks"] < roofline.sm:
                k = loc["key"]
                fill[k]["n"] += 1
                fill[k]["blocks"] += loc["blocks"]
                fill[k]["name"] = m.name
    for k, v in sort_dict(fill, key=lambda kv: -kv[1]["n"]).items():
        per = v["blocks"] / max(1, v["n"])
        findings.append(
            f"LOW DEVICE FILL  {k}: {v['n']} launch(es) averaging {per:.0f} "
            f"workgroups each, under the {roofline.sm} SMs -- a single launch "
            f"cannot fill the device, so its time is fixed by latency, not by "
            f"bandwidth or FLOPs (e.g. {v['name'][:40]})")

    # occupancy too low, per kernel
    occ_bad = defaultdict(lambda: {"n": 0, "pct": 100.0, "key": "",
                                   "wg": 0, "wgsm": 0})
    for m in metrics:
        for loc in getattr(m, "occupancies", []):
            if loc["blocks"] <= 0:
                continue
            a = occ_bad[loc["key"]]
            a["n"] += 1
            a["pct"] = min(a["pct"], loc["occupancy_pct"])
            a["wg"] = loc["wg_size"]
            a["wgsm"] = loc["max_workgroups_per_sm"]
    for k, v in sort_dict(occ_bad, key=lambda kv: kv[1]["pct"]).items():
        if v["pct"] < 50.0:
            findings.append(
                f"LOW OCCUPANCY  {k}: wg={v['wg']}, {v['wgsm']} resident "
                f"workgroup(s)/SM -> {v['pct']:.0f}% of the thread slots "
                f"({v['n']} node(s))")

    # zero-copy views that might still cost something
    for m in metrics:
        if m.kind == "view" and m.op_type in ("Slice", "Gather", "Split"):
            continue
    views = [m for m in metrics if m.kind == "view"]
    if views:
        by_op = defaultdict(int)
        for m in views:
            by_op[m.op_type] += 1
        findings.append(
            "ZERO-COPY VIEWS  "
            + ", ".join(f"{k}:{v}" for k, v in sorted(by_op.items()))
            + "  -- no dispatch, no DRAM traffic; they still cost one host-side"
              " buffer alias each and a descriptor rebind, so they show up in"
              " record time (b3: ~1 ms/round for ~160 of them), not in GPU time")

    # split-K weight re-read, aggregated: at decode m=1 the grid.y factor is 1
    # so no node triggers it, but a prefill or a batched shape does, and the
    # per-node form would print once per layer.
    reread = [m for m in metrics
              if any(n.startswith("split-K GEMV re-reads") for n in m.notes)]
    if reread:
        extra = 0
        for m in reread:
            for n in m.notes:
                if n.startswith("split-K GEMV re-reads"):
                    try:
                        extra += float(n.split("; ")[1].split(" GB")[0]) * 1e9
                    except (IndexError, ValueError):
                        pass
        findings.append(
            f"WEIGHT RE-READ  {len(reread)} split-K GEMV node(s) issue one "
            f"weight pass per output row; at this analysis point that is "
            f"{fmt_bytes(extra)} of issued weight traffic on top of the "
            f"{fmt_bytes(sum(m.weight_bytes for m in reread))} unique. The L2 "
            f"absorbs it only while the weight stays under the cache; above "
            f"that the extra bytes are real DRAM traffic and the node's "
            f"effective intensity is lower than the table shows")

    # Weight residency: at m=1 every weight is read exactly once, so the
    # question is how much of the model's weights the working set is, and
    # whether the L2 can hold the hot ones.
    mm_w = [m for m in metrics if m.op_type == "MatMul" and m.weight_bytes]
    if mm_w and args.decode_m <= 1:
        total_w = sum(m.weight_bytes for m in mm_w)
        big = [m for m in mm_w if m.weight_bytes > dev["l2_bytes"]]
        if big:
            findings.append(
                f"L2 CANNOT HOLD  {len(big)} weight(s) exceed the "
                f"{fmt_bytes(dev['l2_bytes'])} L2 individually; the biggest is "
                f"{fmt_bytes(max(m.weight_bytes for m in big))}. A weight-only "
                f"GEMV streams each of them once per row, so they are pure DRAM "
                f"traffic with no reuse to exploit")

    # lm_head style: a single node dominating
    if metrics:
        if total_bytes:
            for m in metrics:
                if m.kind == "dispatch" and m.total_bytes > 0.15 * total_bytes:
                    findings.append(
                        f"DOMINANT NODE  {m.name[:40]} ({m.op_type}) carries "
                        f"{100.0 * m.total_bytes / total_bytes:.1f}% of all "
                        f"bytes ({fmt_bytes(m.total_bytes)})")
    for f in dedupe(findings, limit=40):
        A("  * " + f)
    A("")

    # -- cross-checks ------------------------------------------------------
    A("-" * 78)
    A("5. CROSS-CHECKS (does the accounting close?)")
    A("-" * 78)
    mm = [m for m in metrics if m.op_type == "MatMul" and m.kind == "dispatch"]
    mm_bytes = sum(m.total_bytes for m in mm)
    mm_weights = sum(m.weight_bytes for m in mm)
    uniq = analysis.model.initializer_bytes_unique
    if mm:
        delta = 100.0 * (mm_weights - uniq) / uniq if uniq else 0.0
        A(f"  MatMul weight bytes  : {fmt_bytes(mm_weights)}")
        A(f"  distinct initializer : {fmt_bytes(uniq)}  "
          f"({delta:+.2f}% vs the MatMul sum)")
        A(f"    -> {len(mm)} MatMul nodes; a shared weight counted once per")
        A(f"       node it feeds, and the sums agree to a fraction of a")
        A(f"       percent on an int8 weight-only model, which is the check")
        A(f"       that the per-node byte model is not double-counting.")
        A(f"  MatMul total traffic : {fmt_bytes(mm_bytes)} "
          f"(weights + activations + outputs)")
        floor_ms = mm_bytes / (roofline.bw) * 1e3
        A(f"  DRAM floor at m={args.decode_m}   : {floor_ms:.1f} ms "
          f"at {dev['bandwidth_gbs']:.0f} GB/s")
        if args.measured_ms:
            eff = mm_bytes / (args.measured_ms * 1e-3) / 1e9
            A(f"  measured MatMul GPU  : {args.measured_ms:.1f} ms "
              f"-> {eff:.1f} GB/s effective, "
              f"{100.0 * floor_ms / args.measured_ms:.0f}% of the DRAM floor")
        else:
            A("  (pass --measured-ms to compare against a profiled round)")
    A(f"  sum of per-node bytes : {fmt_bytes(total_bytes)}")
    A(f"  weights at this point : {fmt_bytes(analysis.weight_bytes_at_m)}"
      f"  ({100.0 * analysis.weight_bytes_at_m / uniq:.1f}% of the file)")
    A("")

    # -- notes -------------------------------------------------------------
    A("-" * 78)
    A("6. ASSUMPTIONS AND LIMITS")
    A("-" * 78)
    A("  * bytes are the MINIMUM a correct kernel must move (read inputs +")
    A("    weight once + write output).  A split-K GEMV re-reads the weight")
    A("    once per output row; those extra bytes are reported per node, not")
    A("    added to the denominator, because the cached copies are not DRAM.")
    A("  * FLOPs count a multiply-add as 2.  Elementwise chains use a")
    A("    per-primitive cost model, not the ISA's actual instruction count.")
    A("  * coopmat MMA throughput is not modelled: a 16x16x16 tensor-core")
    A("    issue rate is not derivable from the SM's vector rate, so a coop")
    A("    node's flops are compared against the fp16 VECTOR peak, which")
    A("    understates the compute headroom it really has.")
    A("  * zeros / sparsity / cache reuse are not modelled.")
    A("  * a dynamic dim is substituted at the analysis point; a model with")
    A("    more than one symbolic data axis is analysed at the known parts")
    A("    times 1 for the others.")
    A("")
    return "\n".join(lines)


def sort_dict(d, key=None):
    return dict(sorted(d.items(), key=key) if key else sorted(d.items()))


def dedupe(findings, limit=40):
    """Collapse runs that differ only by node name, keeping the count.

    A 498-MatMul model produces the same finding once per layer; printing 40
    identical lines hides the ones that matter.  The key is everything before
    the first parenthesised node name.
    """
    seen = {}
    order = []
    for f in findings:
        # Drop the node name (either the leading token or a parenthesised
        # one) so the same finding on 48 layers collapses to one row.
        k = f.split(":", 1)[0] if f.startswith(("LOW DEVICE FILL",
                                                "LOW OCCUPANCY",
                                                "EMPTY DISPATCH")) else f
        k = re.sub(r"\b[A-Za-z_][A-Za-z0-9_./-]*_\d{3,}\b", "<node>", k)
        k = re.sub(r"\([^()]*\.\.\.\)", "", k)
        k = re.sub(r"\([^()]*_\d+[^()]*\)", "()", k)
        if k not in seen:
            seen[k] = [1, f]
            order.append(k)
        else:
            seen[k][0] += 1
    out = []
    for k in order:
        n, f = seen[k]
        out.append(f if n == 1 else f" [x{n}] {f}")
    return out[:limit]


def _weight_of(m):
    for tok in (m.shape_note or "").split():
        if tok.startswith("w="):
            return tok[2:]
    return ""


def metrics_to_rows(metrics):
    rows = []
    for m in metrics:
        loc = m.best_occupancy
        rows.append({
            "node": m.name,
            "op_type": m.op_type,
            "kind": m.kind,
            "route": m.route,
            "flops": m.flops,
            "bytes_read": m.bytes_read,
            "bytes_written": m.bytes_written,
            "weight_bytes": m.weight_bytes,
            "total_bytes": m.total_bytes,
            "intensity": m.intensity,
            "dispatch_count": m.dispatch_count,
            "classification": getattr(m, "classification", ""),
            "kernel": loc["key"] if loc else "",
            "local_size": ("x".join(str(x) for x in loc["local_size"])
                           if loc else ""),
            "wg_size": loc["wg_size"] if loc else "",
            "shared_bytes": loc["shared_bytes"] if loc else "",
            "regs_per_thread": loc["regs_per_thread"] if loc else "",
            "reg_source": loc["reg_source"] if loc else "",
            "occupancy_pct": round(loc["occupancy_pct"], 1) if loc else "",
            "blocks": loc["blocks"] if loc else "",
            "shape": m.shape_note,
        })
    return rows


def check_launch_consistency(loc, launch):
    """Compare the dispatch's lane count with the items it must cover.

    Only UNDER-dispatch is checked, and only against the number of items the
    kernel must cover.  Over-dispatch is normal here (a tiled or masked kernel
    pads to whole tiles, and the column-packing GEMVs give each lane several
    outputs), so flagging it would bury the one case that is a real bug: a
    transcribed grid formula that cannot even cover its own output.  Getting
    that wrong is easy to miss, because a too-small grid and a plausible
    local_size look fine in isolation -- the kernel simply writes part of the
    result.
    """
    if not loc or loc["blocks"] <= 0 or not launch.items:
        return None
    lanes = loc["blocks"] * loc["wg_size"]
    if lanes < launch.items:
        return (f"UNDER-DISPATCHED: grid covers {lanes} lanes for "
                f"{launch.items} items ({100.0 * lanes / launch.items:.0f}%) "
                f"-- the transcribed grid formula for this route cannot cover "
                f"its own output")
    return None


def attach_occupancy(analysis, metrics):
    """Fill m.occupancies / m.best_occupancy for every launch of every metric."""
    for m in metrics:
        m.occupancies = []
        m.best_occupancy = None
        if m.kind != "dispatch":
            continue
        for launch in m.launches:
            key = launch.key
            # Fall back to the op's own plan key if the guessed route key is
            # not in the archive (an older artifact): report the miss instead
            # of silently pricing a kernel that does not exist.
            if key not in analysis.model.archive:
                alt = _key_alternates(analysis.model, key)
                if alt:
                    key = alt
                else:
                    m.notes.append(f"kernel {launch.key} is not in the "
                                   f"archive (older artifact?)")
            info = analysis.spv_info(key)
            ls = info.get("local") or launch.ls
            grid = launch.grid()[0] * launch.grid()[1] * launch.grid()[2]
            if launch.dispatches > 1:
                grid *= launch.dispatches
            loc = analysis.occupancy(key, ls, grid)
            if loc is None:
                continue
            loc["launch_note"] = launch.note
            loc["dispatches"] = launch.dispatches
            warn = check_launch_consistency(loc, launch)
            if warn:
                m.notes.append(warn)
            m.occupancies.append(loc)
        if m.occupancies:
            # "best" = the one that can fill the device most; this is what the
            # classification uses (a route's non-bottleneck passes must not
            # drag its class down).
            m.best_occupancy = max(m.occupancies,
                                   key=lambda l: l["max_workgroups_per_sm"]
                                   * l["wg_size"])


def _key_alternates(model, key):
    """The archive key an older artifact would carry for the same kernel."""
    cands = [key]
    if key.endswith("_fp16"):
        cands.append(key[:-5])
    else:
        cands.append(key + "_fp16")
    for c in cands:
        if c in model.archive:
            return c
    # The op's kernel plan names its primary key.
    return None


def convert_onnx(args):
    """Convert an .onnx through the repo's own converter.

    The analysis must describe the graph the RUNTIME runs, and that graph is
    what onnx2vkop produces (fusion, QDQ folding, kernel plans, the embedded
    shader subset).  Re-deriving any of that here would be a second
    implementation of the converter -- exactly what this tool refuses to do for
    the kernel selector.  So an .onnx input is converted, once, into a temp
    .vkopbin that is then analysed like any other.
    """
    out = args.onnx_output or os.path.join(
        tempfile.mkdtemp(prefix="vkop_roofline_onnx_"),
        os.path.basename(args.model)[:-5] + ".vkopbin")
    cmd = [sys.executable, "-m", "onnx2vkop.cli",
           "-i", args.model, "-o", out]
    if args.onnx_quant:
        cmd += ["-q", args.onnx_quant]
    print(f"[roofline] converting {args.model} -> {out}", file=sys.stderr)
    print("[roofline] (the converter is the repo's own; the analysis then sees "
          "exactly the fused graph and kernel plans the runtime would)",
          file=sys.stderr)
    env = dict(os.environ)
    # onnx2vkop is an editable install of model/pypi; make it importable even
    # when the caller's interpreter has not installed it.
    pypi = os.path.join(REPO_ROOT, "model", "pypi")
    env["PYTHONPATH"] = pypi + (os.pathsep + env["PYTHONPATH"]
                                if env.get("PYTHONPATH") else "")
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise SystemExit("onnx2vkop failed; see the output above")
    print(f"[roofline] converted: {os.path.getsize(out)} bytes",
          file=sys.stderr)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="roofline_report.py",
        description="Per-operator FLOPs / bytes / roofline / occupancy report "
                    "for a .vkopbin model.")
    p.add_argument("model", nargs="?",
                   help="path to a .vkopbin, or an .onnx (converted first with "
                        "the repo's own onnx2vkop, so the analysis sees the "
                        "same graph and kernel plans the runtime would)")
    p.add_argument("--device", default="nvidia-ada",
                   choices=sorted(DEVICES), help="device preset")
    p.add_argument("--decode-m", type=int, default=1,
                   help="rows per node at the decode analysis point")
    p.add_argument("--prefill-m", type=int, default=0,
                   help="also analyse this many rows (prefill); 0 = skip")
    p.add_argument("--kv-len", type=int, default=1024,
                   help="KV-cache length the Attention nodes see (default 1024)")
    p.add_argument("--oracle", default=None,
                   help="path to a prebuilt roofline_oracle binary")
    p.add_argument("--no-oracle", action="store_true",
                   help="skip building/using the KernelSelect oracle")
    p.add_argument("--probe", action="store_true",
                   help="ask the driver for real register counts via "
                        "VK_KHR_pipeline_executable_properties (creates "
                        "compute pipelines; never dispatches). Off by default "
                        "so the analysis never touches a GPU unless asked")
    p.add_argument("--probe-device-index", type=int, default=None,
                   help="Vulkan device index the probe should use")
    p.add_argument("--csv", default=None, help="write per-node CSV here")
    p.add_argument("--json", default=None, help="write per-node JSON here")
    p.add_argument("--top", type=int, default=15,
                   help="rows in the per-node 'heaviest' table")
    p.add_argument("--quiet", action="store_true", help="suppress the report")
    p.add_argument("--self-test", action="store_true",
                   help="run the tool's own arithmetic checks and exit")
    p.add_argument("--onnx-output", default=None,
                   help="where to put the converted .vkopbin for an .onnx "
                        "input (default: a temp file)")
    p.add_argument("--onnx-quant", default=None,
                   help="pass -q <kind> to the converter (fp16, int8, ...)")
    p.add_argument("--measured-ms", type=float, default=0.0,
                   help="a profiled MatMul GPU time for one forward, to "
                        "compare the computed DRAM floor against")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.model:
        p.error("a model path is required (or pass --self-test)")

    if args.model.lower().endswith(".onnx"):
        args.model = convert_onnx(args)

    model = VkopModel(args.model)
    try:
        oracle = RoutingOracle(
            override=None if args.no_oracle else args.oracle,
            device=args.device)
        if args.no_oracle:
            oracle.available = False

        probe = {}
        probe_dev = {}
        if args.probe:
            keys = set()
            # Only the kernels this model can actually launch: probing the
            # whole archive would compile 150 pipelines for a 20-node graph.
            for plan in model.plans.values():
                for k in (plan["buffer"], plan["buffer_fp16"]):
                    if k:
                        keys.add(k)
                for k in plan["variants"]:
                    keys.add(k)
            keys &= set(model.archive.keys())
            paths = []
            tmpdir = tempfile.mkdtemp(prefix="vkop_roofline_spv_")
            for k in sorted(keys):
                blob = model.archive.get(k)
                if not blob:
                    continue
                path = os.path.join(tmpdir, k + ".spv")
                with open(path, "wb") as f:
                    f.write(blob)
                paths.append(path)
            stats, probe_dev = run_probe(paths, args.probe_device_index)
            shutil.rmtree(tmpdir, ignore_errors=True)
            preset_vendor = DEVICES[args.device]["vendor_id"]
            got_vendor = probe_dev.get("vendor")
            if stats and preset_vendor and got_vendor and \
                    got_vendor != preset_vendor:
                # Mixing a probe from a different GPU into this device's
                # resource model would report another card's register counts as
                # this one's.  Refuse, and say why.
                print(f"[roofline] probe ran on {probe_dev.get('name')} "
                      f"(vendor 0x{got_vendor:x}) but --device {args.device} "
                      f"is vendor 0x{preset_vendor:x}; ignoring the probe. "
                      f"Pass --probe-device-index, or drop --probe.",
                      file=sys.stderr)
                probe_dev = {}
            else:
                for path, st in stats.items():
                    probe[os.path.basename(path)[:-4]] = st
                probe_dev["probed"] = len(probe)

        analysis = Analysis(model, args.device, oracle, probe,
                            decode_m=args.decode_m, prefill_m=args.prefill_m,
                            kv_len=args.kv_len)
        analysis.probe_device = probe_dev
        if probe:
            vals = sorted(int(st["Max live registers"]) for st in probe.values()
                          if "Max live registers" in st)
            if vals:
                analysis.assumed_regs = vals[len(vals) // 2]
                analysis.assumed_regs_basis = (
                    f"median of the {len(vals)} kernels this device's driver "
                    f"did report ({vals[0]}-{vals[-1]} registers); a kernel the "
                    f"driver could not report uses this median")
        sm = None
        for k in list(model.archive.keys())[:0]:
            pass
        # The oracle reports the SM count straight out of sm_estimate(); fall
        # back to the vendor table only when it is unavailable.
        sm = SM_ESTIMATE_FALLBACK.get(DEVICES[args.device]["vendor_id"],
                                      DEVICES[args.device]["sm"])
        metrics = analysis.run(args.decode_m)
        attach_occupancy(analysis, metrics)
        # Ask the oracle once for a trivial query so the reported sm is the
        # one the selector itself used.
        probe_q = oracle.query([{"batch": 1, "m": 1, "k": 128, "n": 128,
                                 "weight": "fp16", "fp16": True}])[0]
        if probe_q.get("sm"):
            sm = probe_q["sm"]
        roofline = Roofline(DEVICES[args.device], args.device, sm)

        if not args.quiet:
            print(build_report(analysis, metrics, roofline, args))

        if args.prefill_m:
            pass  # second pass handled below

        rows = metrics_to_rows(metrics)
        if args.csv:
            import csv as _csv
            with open(args.csv, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"[roofline] wrote {args.csv}", file=sys.stderr)
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"device": args.device, "sm": sm,
                           "m": args.decode_m, "kv_len": args.kv_len,
                           "rows": rows}, f, indent=1)
            print(f"[roofline] wrote {args.json}", file=sys.stderr)

        if args.prefill_m:
            m2 = Analysis(model, args.device, oracle, probe,
                          decode_m=args.prefill_m, prefill_m=0,
                          kv_len=args.kv_len)
            metrics2 = m2.run(args.prefill_m)
            attach_occupancy(m2, metrics2)
            print()
            print("=" * 78)
            print(f"PREFILL ANALYSIS POINT  m = {args.prefill_m}")
            print("=" * 78)
            tot_f = sum(x.flops for x in metrics2)
            tot_b = sum(x.total_bytes for x in metrics2 if x.kind == "dispatch")
            print(f"TOTAL FLOPs : {fmt_flops(tot_f)}")
            print(f"TOTAL bytes : {fmt_bytes(tot_b)}")
            print(f"intensity   : {tot_f / tot_b if tot_b else 0:.2f} FLOP/Byte")
            agg = defaultdict(lambda: [0, 0.0, 0])
            for x in metrics2:
                if x.kind != "dispatch":
                    continue
                a = agg[x.op_type]
                a[0] += 1
                a[1] += x.flops
                a[2] += x.total_bytes
            rows2 = []
            for op, (cnt, fl, by) in sorted(agg.items(),
                                            key=lambda kv: -kv[1][2])[:12]:
                rows2.append([op, cnt, fmt_flops(fl), fmt_bytes(by),
                              f"{fl / by if by else 0:.2f}"])
            print(table(["op_type", "calls", "FLOPs", "bytes", "FLOP/B"],
                        rows2, ["l", "r", "r", "r", "r"]))
            print()
            # Routing changes with m -- this is the whole point of a second
            # analysis point, so it is stated rather than left implicit.
            rt = defaultdict(lambda: [0, 0.0, 0])
            for x in metrics2:
                if x.op_type != "MatMul":
                    continue
                a = rt[x.route or "(unknown)"]
                a[0] += 1
                a[1] += x.flops
                a[2] += x.total_bytes
            rrows = [[r, v[0], fmt_flops(v[1]), fmt_bytes(v[2])]
                     for r, v in sorted(rt.items(), key=lambda kv: -kv[1][2])]
            print("MatMul routes at this point:")
            print(table(["route", "nodes", "FLOPs", "bytes"], rrows,
                        ["l", "r", "r", "r"]))
            if args.json:
                with open(args.json, "r") as f:
                    blob = json.load(f)
                blob["prefill"] = {"m": args.prefill_m,
                                   "rows": metrics_to_rows(metrics2)}
                with open(args.json, "w") as f:
                    json.dump(blob, f, indent=1)
    finally:
        model.close()
    return 0




# ===========================================================================
# Self-test
#
# `roofline_report.py --self-test` runs the arithmetic that the report depends
# on against hand-computed answers, so a regression in the byte/FLOP model or
# in the SPIR-V parser fails loudly instead of quietly producing a plausible
# table.  It needs no GPU, no model file and no oracle.
# ===========================================================================
def _make_self_test_spv():
    """A minimal, well-formed compute module built programmatically.

    Written as words rather than a hex blob so the header/operand order is
    self-evident and cannot drift by a byte.  It declares LocalSize 64 2 1 and
    one Workgroup array<float, 100>, which is exactly what the SPIR-V reader
    must recover.
    """
    import struct as _s
    W = []
    # header: magic, version 1.3, generator 0, bound, schema 0
    W += [0x07230203, 0x00010300, 0, 100, 0]
    def inst(op, *ops):
        W.append(((len(ops) + 1) << 16) | op)
        W.extend(ops)
    inst(17, 1)                      # OpCapability Shader
    inst(14, 0, 1)                   # OpMemoryModel Logical GLSL450
    inst(15, 5, 1, 0x6e69616d, 0)    # OpEntryPoint GLCompute %1 "main" (no iface)
    inst(16, 1, 17, 64, 2, 1)        # OpExecutionMode %1 LocalSize 64 2 1
    inst(22, 2, 32)                  # %2 = OpTypeFloat 32
    inst(21, 3, 32, 0)               # %3 = OpTypeInt 32 0
    inst(43, 3, 4, 100)              # %4 = OpConstant %3 100
    inst(28, 5, 2, 4)                # %5 = OpTypeArray %2 %4
    inst(32, 6, 4, 5)                # %6 = OpTypePointer Workgroup %5
    inst(59, 6, 7, 4)                # %7 = OpVariable %6 Workgroup (module scope)
    inst(19, 9)                      # %9 = OpTypeVoid
    inst(33, 10, 9)                  # %10 = OpTypeFunction %9
    inst(32, 12, 4, 2)               # %12 = OpTypePointer Workgroup %2
    inst(43, 3, 11, 0)               # %11 = OpConstant %3 0
    inst(54, 9, 1, 0, 10)            # %1 = OpFunction %9 None %10
    inst(248, 8)                     # %8 = OpLabel
    # Reference the module-scope variable from the ENTRY function, which is
    # what makes it part of this launch's shared-memory footprint.
    inst(65, 12, 13, 7, 11)          # %13 = OpAccessChain %12 %7 %11
    inst(61, 2, 14, 13)              # %14 = OpLoad %2 %13
    inst(253)                        # OpReturn
    inst(56)                         # OpFunctionEnd
    return _s.pack("<%dI" % len(W), *W)


_SELF_TEST_SPV = _make_self_test_spv()


def self_test():
    import traceback
    failures = []

    def check(name, got, want, tol=1e-9):
        ok = (abs(got - want) <= tol * max(1.0, abs(want))
              if isinstance(want, (int, float)) and isinstance(got, (int, float))
              else got == want)
        if not ok:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # --- SPIR-V static facts ------------------------------------------------
    info = _spirv_static(_SELF_TEST_SPV)
    check("spirv local_size", info["local"], [64, 2, 1])
    check("spirv shared_static", info["shared_static"], 400)

    # --- FlatBuffer reader --------------------------------------------------
    if os.path.exists(os.path.join(REPO_ROOT, "model", "add_conv_model.vkopbin")):
        p = os.path.join(REPO_ROOT, "model", "add_conv_model.vkopbin")
        mdl = VkopModel(p)
        check("fb node count", len(mdl.nodes), 2)
        check("fb first op", mdl.nodes[0]["op_type"], "Add")
        check("fb initializers", len(mdl.initializers), 3)
        mdl.close()

    # --- byte / FLOP arithmetic (hand-computed) -----------------------------
    a = Analysis.__new__(Analysis)
    a.kv_len = 1

    # matmul_launches: a ksplit GEMV at m=1 must read exactly the weight once.
    plan = {"route": "ksplit", "ksplit_q": 16, "wide": False, "ks_n": 0}
    wbytes = 2048 * 2048         # int8 weight
    acts = 2048 * 2              # one fp16 row
    outs = 2048 * 2
    launches, flops, _ = matmul_launches(plan, 1, 2048, 2048, 1, -1, wbytes,
                                         outs, acts)
    check("matmul flops", flops, 2.0 * 2048 * 2048)
    check("matmul grid.x", launches[0].grid()[0], up_div(2048 // 4, 16))
    check("matmul grid.y", launches[0].grid()[1], 1)

    # A tiled route with a K split dispatches two passes.
    plan = {"route": "tiled_ks", "ksplit_q": 16, "wide": False, "ks_n": 4}
    launches, _f, _ = matmul_launches(plan, 64, 128, 256, 1, -1, 0, 0, 0)
    check("tiled_ks passes", len(launches), 2)
    check("tiled_ks deposit z", launches[0].grid()[2], 4)
    check("tiled_ks fold z", launches[1].grid()[2], 1)

    # Folded SwiGLU: three GEMMs, not one.
    launches, flops, _ = matmul_launches(
        {"route": "down_act_gemv", "ksplit_q": 16, "wide": False, "ks_n": 0},
        1, 1024, 2048, 1, 4096, 0, 0, 0)
    check("fold passes", len(launches), 3)
    check("fold grid.x", launches[0].grid()[0], up_div(4096, 512))

    # --- resolve(): the dynamic axis convention ----------------------------
    class M:
        pass
    an = Analysis.__new__(Analysis)
    an.kv_len = 512
    check("resolve decode rows",
          an.resolve({"dims": [-1, -1, 2048]}, 1), [1, 1, 2048])
    check("resolve prefill rows",
          an.resolve({"dims": [-1, -1, 2048]}, 256), [1, 256, 2048])
    check("resolve kv axis",
          an.resolve({"dims": [-1, 2, 8, -1, 128]}, 1, kv=True),
          [1, 2, 8, 512, 128])

    # --- flat-identity perm (ops/Transpose.hpp:61) -------------------------
    check("perm identity [1,H,1,D]",
          _flat_identity_perm([1, 16, 1, 128], [0, 2, 1, 3]), True)
    check("perm real transpose",
          _flat_identity_perm([1, 16, 7, 128], [0, 2, 1, 3]), False)

    # --- occupancy model ----------------------------------------------------
    an.dev = DEVICES["intel-arl"]
    an.probe = {}
    an.assumed_regs = 40
    an.spv_cache = {"k": {"local": [64, 2, 1], "shared_static": 400,
                          "shared_dynamic": 0, "found": True}}
    an.driver_stats = lambda key: {"Max live registers": 64.0,
                                   "Workgroup Memory Size": 400.0,
                                   "Max dispatch width": 8.0}
    loc = an.occupancy("k", None, 128)
    check("occ wg size", loc["wg_size"], 128)
    check("occ reg source", loc["reg_source"], "driver")
    # 64 regs at an 8-register (32 B) granule = 64 regs = 256 B/thread.
    # ARL register file = 8 threads/EU x 16 EU x 128 GRF x 32 B = 524288 B.
    # 524288 / (256 B * 128 threads/WG) = 16 workgroups by registers; the
    # shared limit is 65536 / 400 = 163, and the thread-slot limit is
    # 16*8*8/128 = 8.  min(16, 8, 163) = 8.
    check("occ reg limit", loc["limits"]["regs"], 16)
    check("occ shared limit", loc["limits"]["shared"], (64 * 1024) // 400)
    check("occ max wg", loc["max_workgroups_per_sm"], 8)
    check("occ pct", loc["occupancy_pct"],
          100.0 * 8 * 128 / (16 * 8 * 8))

    # A device with no driver number must say so.
    an.driver_stats = lambda key: None
    loc = an.occupancy("k", None, 1)
    check("occ assumed", loc["reg_source"], "assumed")

    # --- formatting ---------------------------------------------------------
    check("fmt GB", fmt_bytes(25_640_000_000), "25.64 GB")
    check("fmt GF", fmt_flops(51_240_000_000), "51.24 GF")

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"self-test OK ({len(_SELF_TEST_SPV)} B synthetic module + "
          f"hand-computed arithmetic)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
