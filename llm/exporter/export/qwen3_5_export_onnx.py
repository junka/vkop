"""导出 Qwen3.5 系列（`Qwen3_5ForConditionalGeneration` 的文本塔）到 ONNX。

与 qwen3_export_onnx.py（纯文本 Qwen3）的区别：

  · **混合层**：每 4 层里 3 层是 linear attention（gated delta net），
    1 层是普通 full attention（`layer_types` 决定）。两种状态完全不同：
      - full_attention  -> KV cache  (B, 2, NKV, kv, HD)   每层一个
      - linear_attention-> conv state (C, K-1) + 递推矩阵 (Hv, Dk, Dv)
  · **q_proj 输出 2 倍宽**：前一半 query，后一半 gate（`attn_output_gate`），
    attention 输出要乘 `sigmoid(gate)`。
  · **3D MRoPE**（不是 Qwen3 的 2D RoPE），`partial_rotary_factor=0.25`，
    只有前 1/4 的 head_dim 参与旋转。
  · **零中心化 RMSNorm**：HF 是 `out * (1 + weight)`，本脚本导出前把每个
    RMSNorm 的 weight 预加 1，图里就退化成标准的 `out * w`。
  · **两个自定义算子**：`Conv1d` 与 `LinearAttention`。递推在序列维上是
    严格串行的，展开成普通 ONNX op 会把 seq 钉死在导出时的长度（prefill
    2048 步意味着上万个节点），所以这两位以自定义节点发出，由运行时的
    ops/Conv1d.hpp / ops/LinearAttention.hpp 承接。

I/O 契约（llm.onnx）：

  入：inputs_embeds        (B, q, HIDDEN)              fp16
      position_ids         (3, B, q)                   int64   3D MRoPE
      attention_bias       (B, 1, q, kv)               fp16    full_attention 层用
      past_key_values_{i}  (B, 2, NKV, kv, HD)         fp16    i ∈ full layers
      conv_state_{i}       (1, C_conv, K-1)            fp32    i ∈ linear layers
      recurrent_state_{i}  (1, Hv, Dk, Dv)             fp32    i ∈ linear layers
  出：logits               (B, q, VOCAB)               fp16
      present_key_values_{i} / conv_state_out_{i} / recurrent_state_out_{i}

用法:
  MODEL_PATH=~/models/Qwen3.5-27B python3 export/qwen3_5_export_onnx.py
  EXPORT_PATH=llm3_5.onnx MODEL_PATH=... python3 export/qwen3_5_export_onnx.py
"""

import os

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = os.environ.get("MODEL_PATH") or "Qwen/Qwen3.5-27B"
EXPORT_PATH = os.environ.get("EXPORT_PATH") or "llm.onnx"
OPSET = 17

# Prefill dummy length: the graph's seq axis is dynamic (the two serial ops take
# their extent from the tensor), so this only has to exercise each layer type.
SEQ_LEN = int(os.environ.get("SEQ_LEN", "16"))

print(f"[load] model = {MODEL_PATH}")

from transformers import AutoModelForCausalLM  # noqa: E402

# The wrapper lives in onnx2vkop/hf_builders/qwen3_5.py so this script and
# `onnx2vkop --hf-checkpoint` export the same graph from one implementation.
import sys  # noqa: E402
_PYPI = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", "..", "model", "pypi"))
if _PYPI not in sys.path:
    sys.path.insert(0, _PYPI)
from onnx2vkop.hf_builders import qwen3_5 as q35  # noqa: E402

# Two ways to get the module tree the exporter traces:
#
#   STREAM_EXPORT=1  build it on the META device -- parameters carry shapes and
#                    dtypes but no storage. The graph is traced from that, and
#                    the weights are spliced in afterwards straight from the
#                    safetensors shards (see onnx2vkop.stream_export). Peak is
#                    the graph, not the checkpoint: a 27B export fits in well
#                    under a gigabyte instead of ~50 GB.
#
#   default          load the real weights. `low_cpu_mem_usage` avoids holding
#                    two copies during load, which is what pushed 5 GB into
#                    swap on the 27B model, but the weights are still resident
#                    for the trace.
STREAM_EXPORT = os.environ.get("STREAM_EXPORT", "") not in ("", "0")

if STREAM_EXPORT:
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(
        MODEL_PATH, local_files_only=os.path.isdir(
            os.path.expanduser(MODEL_PATH)), trust_remote_code=True)
    with torch.device("meta"):
        # torch_dtype is REQUIRED, not cosmetic. from_config defaults to fp32,
        # while the eager path loads the checkpoint at fp16. A fp32 meta model
        # traces fp32 arithmetic, so the fp16 graph inputs get Cast(1) before
        # every use: measured 75 Casts against the eager export's 43, and the
        # added Cast on attention_bias changed its dtype at the Attention
        # boundary, which the runtime then rejected.
        model = AutoModelForCausalLM.from_config(
            _cfg, trust_remote_code=True, torch_dtype=torch.float16)
    print("[load] model built on the META device (no weight storage)")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
model.eval()

# ---------------------------------------------------------------------------
# 架构参数：文本塔的 config，全部动态读取
# ---------------------------------------------------------------------------
full_cfg = model.config
tcfg = getattr(full_cfg, "text_config", full_cfg)

NLAYERS = tcfg.num_hidden_layers
HIDDEN = tcfg.hidden_size
NUM_HEADS = tcfg.num_attention_heads
NUM_KV_HEADS = tcfg.num_key_value_heads
HEAD_DIM = tcfg.head_dim or HIDDEN // NUM_HEADS
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
VOCAB = tcfg.vocab_size
EPS = tcfg.rms_norm_eps
SCALING = HEAD_DIM ** -0.5

LAYER_TYPES = list(tcfg.layer_types)
FULL_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
LIN_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention"]

LIN_HEADS_K = tcfg.linear_num_key_heads
LIN_HEADS_V = tcfg.linear_num_value_heads
LIN_DK = tcfg.linear_key_head_dim
LIN_DV = tcfg.linear_value_head_dim
CONV_K = tcfg.linear_conv_kernel_dim
KEY_DIM = LIN_HEADS_K * LIN_DK
VALUE_DIM = LIN_HEADS_V * LIN_DV
CONV_DIM = KEY_DIM * 2 + VALUE_DIM

ROTARY_DIM = int(HEAD_DIM * tcfg.rope_parameters.get("partial_rotary_factor", 1.0))
HEAD_REP = LIN_HEADS_V // LIN_HEADS_K

print(f"[config] layers={NLAYERS} (full={len(FULL_LAYERS)}, "
      f"linear={len(LIN_LAYERS)}) hidden={HIDDEN} heads={NUM_HEADS} "
      f"kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM} "
      f"vocab={VOCAB}")
print(f"[config] linear: k_heads={LIN_HEADS_K} v_heads={LIN_HEADS_V} "
      f"dk={LIN_DK} dv={LIN_DV} conv_k={CONV_K} conv_dim={CONV_DIM} "
      f"head_rep={HEAD_REP}")

# Wrapper: hybrid full/linear-attention layer loop, zero-centred RMSNorm fold,
# static interleaved MRoPE, Conv1d/LinearAttention custom ops -- see
# onnx2vkop/hf_builders/qwen3_5.py. make_wrapper folds `1 + w` into the norms
# of THIS model (a no-op on meta, hence the addend map for the splice).
wrapper = q35.make_wrapper(model)
_folded_norm_addends = wrapper.folded
print(f"[norm] pre-folded {len(_folded_norm_addends)} zero-centered RMSNorm "
      f"weight(s) to (1+w)")

# Prefill dummy inputs; every one on the model's own device (meta under
# STREAM_EXPORT -- a stray CPU tensor breaks the meta trace).
_dev = torch.device("meta") if STREAM_EXPORT else torch.device("cpu")
inputs, input_names, output_names, dynamic_axes = q35.llm_io(
    wrapper.d, torch.float16, seq=SEQ_LEN, device=_dev)
S = SEQ_LEN

# Numeric self-check against HF's own forward (small models only; SELF_CHECK=1).
if os.environ.get("SELF_CHECK", "") == "1":
    print("[self-check] running HF reference forward ...")
    q35.self_check(model, wrapper, inputs)

print(f"[export] {NLAYERS} layers, {len(input_names)} inputs / "
      f"{len(output_names)} outputs, opset={OPSET}, dummy S={S} ...")

if STREAM_EXPORT:
    # Graph only, then splice the weights in from the shards. See
    # onnx2vkop/stream_export.py for why export_params=False is the only path
    # that never reads a payload.
    from onnx2vkop import stream_export as SX

    with torch.no_grad():
        graph, params, _ = SX.exporter_utils._model_to_graph(
            wrapper, inputs, verbose=False,
            input_names=input_names, output_names=output_names,
            operator_export_type=SX.OperatorExportTypes.ONNX,
            # Off on purpose. With folding on, the exporter inlines an
            # nn.Linear's weight into a MatMul constant -- and under meta it
            # cannot read that weight, so it surfaces as an ANONYMOUS graph
            # input (`onnx::MatMul_915`) instead of the named parameter, which
            # the splice cannot resolve. Folding also has nothing to gain here:
            # the weights are meta, so no fold is computable.
            do_constant_folding=False,
            # REQUIRED here, not only on the write below. torch applies
            # dynamic_axes twice -- once while building the graph (this call)
            # and once while emitting it -- and skipping the first makes the
            # meta trace diverge STRUCTURALLY from the eager export (827 vs
            # 831 nodes: 14 extra Cast, 13 missing Identity, all on the
            # attention mask's path). See onnx2vkop/stream_export.py.
            dynamic_axes=dynamic_axes)
        raw, _em, _ext, _nm = graph._export_onnx(
            {}, OPSET, dynamic_axes, False, SX.OperatorExportTypes.ONNX, True,
            False, {}, True, "", {})

    proto = onnx.ModelProto()
    proto.ParseFromString(raw)
    print(f"[stream] graph: {len(proto.graph.node)} node(s), "
          f"{len(proto.graph.initializer)} initializer(s)", flush=True)

    index = SX.SafetensorsIndex(MODEL_PATH)
    shapes = {k: list(v.shape) for k, v in params.items()}
    # Keep the real graph inputs (activations, KV/conv/recurrent state) and
    # splice everything the checkpoint can supply.
    # The graph is fp16 (that is what the meta model and the dummy inputs are),
    # so an fp32 checkpoint's payloads must be re-encoded on the way out. The
    # eager path gets this for free from from_pretrained(torch_dtype=fp16);
    # skipping it here leaves fp32 weights under fp16 activations and the model
    # produces all-NaN logits.
    resolved = SX.splice_initializers(
        proto, shapes, index,
        weights_file=os.path.basename(EXPORT_PATH).replace(".onnx", ".weights.bin"),
        target_dtype="F16",
        # The zero-centred RMSNorm fold above rewrote `weight` on the LIVE
        # model; on the meta model that write landed nowhere, so the graph here
        # is built for `1 + w` while the shard still holds `w`. Reproduce the
        # fold on the payload, or every norm normalises by a near-zero weight
        # and the logits come out all 0.0.
        addend=_folded_norm_addends,
        # The 27B checkpoint carries an `mtp.*` speculative-decoding head whose
        # parameter names AND shapes repeat the text tower's, so a suffix match
        # alone cannot tell `layers.0.mlp.gate_proj.weight` from its namesake.
        # This is the subtree the wrapper was built from.
        prefix="model.language_model.")
    print(f"[stream] {len(resolved)} weight(s) spliced", flush=True)

    # `mrope.M` is a computed BUFFER, not a checkpoint tensor: the exporter's
    # static interleaved-MRoPE builds it from inv_freq. Under meta it has no
    # payload, so it surfaced as a graph input. Its value is a few dozen floats
    # the exporter can compute exactly, so materialize it here rather than
    # leaving a dangling input.
    # Real table even on meta (StaticInterleavedMrope rebuilds the HF rotary
    # on CPU from the same config).
    mrope_M = wrapper.mrope.M_value
    import numpy as np
    from onnx import numpy_helper
    if any(v.name == "mrope.M" for v in proto.graph.input):
        keep = [v for v in proto.graph.input if v.name != "mrope.M"]
        del proto.graph.input[:]
        proto.graph.input.extend(keep)
        init = proto.graph.initializer.add()
        np_helper = numpy_helper.from_array(mrope_M.numpy(), "mrope.M")
        init.CopyFrom(np_helper)

    wfile = EXPORT_PATH.replace(".onnx", ".weights.bin")
    nbytes, offsets = SX.copy_weights(resolved, wfile, target_dtype="F16")
    SX.patch_offsets(proto, offsets)

    # The shape repair deliberately does NOT run here -- it runs after
    # annotate_custom_ops below, once the Conv1d/LinearAttention output shapes
    # are declared. Until then a dim that lives behind one of those nodes
    # cannot be inferred, and those are exactly the state outputs the runtime
    # reads. See the note at the end of onnx2vkop/stream_export.py.
    onnx.save(proto, EXPORT_PATH)
    print(f"[✓] exported (streamed) -> {EXPORT_PATH} "
          f"({len(resolved)} weights, {nbytes/2**30:.1f} GiB external)")
else:
    with torch.no_grad():
        torch.onnx.export(
            wrapper, inputs, EXPORT_PATH,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
            # legacy TorchScript exporter：dynamo 对 RoPE 的 cos/sin 数据依赖会
            # trigger GuardOnDataDependentSymNode（qwen3vl 的脚本同样如此）。
            dynamo=False,
        )
    print(f"[✓] exported -> {EXPORT_PATH}")


# ---------------------------------------------------------------------------
# 给两个自定义节点的输出补 value_info。
#
# torch 的 legacy exporter 不给自定义节点推导形状，所以它们的输出既不在
# value_info 也不在图输出里。onnx2vkop 读不到形状就退化成 `shape: []`，之后
# 的 dtype 传播、形状推断、fp16 域判定全部落空（实测：Conv1d 的输入被判成
# fp16，运行时报 "fp32-only"）。这里按算子的语义把形状显式写出来。
# ---------------------------------------------------------------------------
def annotate_custom_ops(path):
    # load_external_data=False is REQUIRED: the default reads every external
    # weight back into RAM -- a second full copy of a 52 GB model, which the
    # OOM killer ends. The pass only adds value_info entries (see
    # q35.annotate_custom_ops for the shapes and why each dtype).
    m = onnx.load(path, load_external_data=False)
    added = q35.annotate_custom_ops(m, wrapper.d)
    if added:
        onnx.save(m, path)
    print(f"[onnx] annotated {added} custom-op output(s) with value_info")


annotate_custom_ops(EXPORT_PATH)

if STREAM_EXPORT:
    # Last step of the streaming path, and it has to be last: recovering the
    # dims the meta trace could not prove requires the spliced weights AND the
    # custom-op shapes annotated just above. `logits` diverges on its vocab
    # axis (a placeholder where eager has VOCAB) and the state outputs on the
    # ones that live behind Conv1d/LinearAttention; those are exactly what the
    # runtime sizes its buffers from. See the note at the end of
    # onnx2vkop/stream_export.py.
    _m = onnx.load(EXPORT_PATH, load_external_data=False)
    _n = SX.resolve_shape_dims(_m, dynamic_axes)
    if _n:
        onnx.save(_m, EXPORT_PATH)
    print(f"[stream] resolved {_n} dim(s) from the spliced weights")
