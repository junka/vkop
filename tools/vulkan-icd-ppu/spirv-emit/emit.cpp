// SPIR-V -> LLVM IR (LLVM 13 typed-pointer, text) emitter for PPU.
//
// Originally a straight-line prototype; extended to cover the FusedElemwise
// core shader (shaders/buffer/mul.comp): whole-module functions with pointer
// parameters, structured control flow (selection + loops + switch), the
// GLSL.std.450 extended instruction set, and multi-level access chains into
// buffer blocks, push constants and local aggregate variables.
//
// Everything not implemented returns Status::Unsupported with a message naming
// the instruction, so a coverage gap stays visible instead of silently
// producing wrong code. The standalone tool (tools/emit_main.cpp) turns that
// into "unsupported: <what>" on stderr and a non-zero exit.
//
// Build as a library: g++ -std=c++17 -c emit.cpp (needs SPIRV-Tools headers)
// Build the tool:      g++ -std=c++17 -O1 emit.cpp emit_main.cpp -o emit -lSPIRV-Tools

#include "emit.hpp"

#include <spirv-tools/libspirv.h>
#include <spirv/unified1/spirv.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ppu_emit {
namespace {

// SPIRV-Tools hands the callback a transient spv_parsed_instruction_t: both it
// and its `words` array may be overwritten as soon as the callback returns.
// Keeping the pointer would make every remembered instruction alias the last
// one parsed (observed: every opcode read back as Nop).  Copy what we need.
struct Inst {
    std::vector<uint32_t> words;
    std::vector<spv_parsed_operand_t> operands;
    uint16_t opcode = 0;
    uint32_t type_id = 0;
    uint32_t result_id = 0;

    Inst() = default;
    explicit Inst(const spv_parsed_instruction_t *p)
        : words(p->words, p->words + p->num_words),
          operands(p->operands, p->operands + p->num_operands),
          opcode(p->opcode), type_id(p->type_id), result_id(p->result_id) {}

    uint32_t word(uint16_t i) const { return words[operands[i].offset]; }
    uint16_t argc() const { return static_cast<uint16_t>(operands.size()); }
};

struct TypeInfo {
    SpvOp op = SpvOpNop;
    uint32_t width = 0;    // OpTypeInt / OpTypeFloat bit width
    uint32_t elem = 0;     // vector component, pointer pointee, array element
    uint32_t count = 0;    // OpTypeVector component count / OpTypeArray length
    uint32_t storage = 0;  // OpTypePointer storage class
    std::vector<uint32_t> members;  // OpTypeStruct members
};

struct Const {
    uint32_t type_id = 0;
    std::string ty;    // LLVM type of the constant ("float", "<2 x float>", ...)
    std::string val;   // LLVM literal, WITHOUT the type ("2.0...e+00", "{ float 1, ... }")
    std::string text;  // pre-rendered "<llvm type> <literal>" == ty + " " + val
    bool is_int = false;
    int64_t ival = 0;
    void set(const std::string &t, const std::string &v) {
        ty = t;
        val = v;
        text = t + " " + v;
    }
};

struct Param {
    uint32_t id = 0;
    uint32_t type_id = 0;
};

struct Function {
    uint32_t id = 0;
    uint32_t ret_type = 0;
    uint32_t fn_type = 0;
    std::vector<Param> params;
    std::vector<Inst> insts;
};

struct Module {
    std::map<uint32_t, TypeInfo> types;
    std::map<uint32_t, Const> consts;
    std::map<uint32_t, uint32_t> var_type;
    std::map<uint32_t, uint32_t> var_storage;
    std::map<uint32_t, uint32_t> var_binding;
    std::map<uint32_t, uint32_t> var_set;   // DescriptorSet decoration
    std::map<uint32_t, uint32_t> builtin;
    std::map<uint32_t, std::string> names;  // OpName, used only in diagnostics
    std::set<uint32_t> bad_consts;         // constants we could not render
    std::vector<Function> funcs;
    std::vector<uint32_t> module_buffers;  // all buffer/PC vars, binding order
    uint32_t entry_fn = 0;
    std::vector<uint32_t> entry_interface;
    uint32_t local_x = 0, local_y = 0, local_z = 0;
    // Every OpEntryPoint in the module, by its SPIR-V name. A module may carry
    // more than one; the caller picks one and the others are simply not
    // emitted as kernels (they remain ordinary internal functions).
    std::map<std::string, uint32_t> entry_names;
    // LocalSize per entry-function id, so two entry points with different
    // workgroup sizes do not overwrite each other.
    std::map<uint32_t, std::array<uint32_t, 3>> local_sizes;
};

std::string llvm_type(const Module &m, uint32_t id);
const char *storage_name(uint32_t sc);

bool is_buffer_storage(uint32_t sc) {
    return sc == SpvStorageClassStorageBuffer || sc == SpvStorageClassUniform ||
           sc == SpvStorageClassPushConstant ||
           sc == SpvStorageClassPhysicalStorageBuffer;
}

std::string addrspace_of(uint32_t sc) {
    if (sc == SpvStorageClassWorkgroup) return "3";
    if (is_buffer_storage(sc)) return "1";
    return "0";
}

std::string llvm_type(const Module &m, uint32_t id) {
    auto it = m.types.find(id);
    if (it == m.types.end()) return "i32";
    const TypeInfo &t = it->second;
    switch (t.op) {
    case SpvOpTypeVoid: return "void";
    case SpvOpTypeBool: return "i1";
    case SpvOpTypeInt: return t.width == 1 ? "i1" : "i" + std::to_string(t.width);
    case SpvOpTypeFloat:
        if (t.width == 16) return "half";
        if (t.width == 64) return "double";
        return "float";
    case SpvOpTypeVector:
        return "<" + std::to_string(t.count) + " x " + llvm_type(m, t.elem) + ">";
    case SpvOpTypeRuntimeArray:
        return "[0 x " + llvm_type(m, t.elem) + "]";
    case SpvOpTypeArray:
        return "[" + std::to_string(t.count) + " x " + llvm_type(m, t.elem) + "]";
    case SpvOpTypeStruct:
        return "%s" + std::to_string(id);  // named, so GEP can spell it
    case SpvOpTypePointer:
        return llvm_type(m, t.elem) + " addrspace(" + addrspace_of(t.storage) + ")*";
    default: return "i32";
    }
}

uint32_t type_pointee(const Module &m, uint32_t ptr_type_id) {
    auto t = m.types.find(ptr_type_id);
    return t == m.types.end() ? 0 : t->second.elem;
}

uint32_t type_storage(const Module &m, uint32_t ptr_type_id) {
    auto t = m.types.find(ptr_type_id);
    return t == m.types.end() ? 0 : t->second.storage;
}

uint32_t var_ptr_type(const Module &m, uint32_t var) {
    auto v = m.var_type.find(var);
    return v == m.var_type.end() ? 0 : v->second;
}

uint32_t const_type(const Module &m, uint32_t id) {
    auto c = m.consts.find(id);
    return c == m.consts.end() ? 0 : c->second.type_id;
}

// The PPU LLVM 13 asm parser rejects "2.0" for a float constant ("expected value
// token") and rejects the raw hex form ("floating point constant invalid for
// type").  It accepts decimal %e only when the literal is EXACTLY the fp32 value
// -- %.9e is not enough (1.442695041e+00 is rejected; 1.44269502162933350e+00 is
// accepted), so use %.17e, which round-trips fp32 exactly and also preserves
// subnormals, -0.0 and the float max.
std::string render_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    // NaN and infinities have no decimal spelling LLVM accepts ("nan" and
    // "inf" are rejected by the PPU asm parser); the double-width hex form is
    // accepted for a `float` constant as long as the value is exactly
    // representable, which these are.
    if (!std::isfinite(static_cast<double>(f))) {
        if (std::isinf(static_cast<double>(f)))
            return (bits >> 31) ? "0xFFF0000000000000" : "0x7FF0000000000000";
        // Preserve the payload's quiet bit and sign so weird NaNs round-trip.
        char buf[32];
        std::snprintf(buf, sizeof buf, "0x%016llX",
                      static_cast<unsigned long long>(
                          0x7FF8000000000000ull | ((uint64_t)(bits >> 31) << 63) |
                          ((uint64_t)(bits & 0x007FFFFFu) << 29)));
        return buf;
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17e", static_cast<double>(f));
    return buf;
}

// fp16 literals: SPIR-V stores 16 bits zero-extended in the word.  Decode the
// half and print the exact fp32 form LLVM accepts for a `half` constant.
std::string render_half(uint16_t bits) {
    const uint32_t sign = (bits >> 15) & 1u;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t man = bits & 0x3ffu;
    uint32_t f32;
    if (exp == 0) {
        if (man == 0) {
            f32 = sign << 31;  // +/-0
        } else {
            // subnormal: value = mantissa * 2^-24.  Renormalise to fp32.
            int k = 9;
            while (!(man & (1u << k))) k--;
            const uint32_t frac = (man & ((1u << k) - 1u)) << (23 - k);
            f32 = (sign << 31) | ((uint32_t)(k + 103) << 23) | frac;
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7f800000u | (man << 13);  // inf / nan
    } else {
        f32 = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    return render_float(f32);
}

bool operand_is_id(spv_operand_type_t t) {
    return t == SPV_OPERAND_TYPE_ID || t == SPV_OPERAND_TYPE_SCOPE_ID;
}

// --- Pass 1: collect module-level facts -------------------------------------

struct Collector {
    Module m;
    uint32_t cur_fn = 0;
    bool in_body = false;
    // OpSpecConstantOp lives at module scope (before any OpFunction), so the
    // function-body path below never sees it. Keep the raw instructions for
    // the folding pass.
    std::vector<Inst> spec_ops;
};

spv_result_t on_header(void *, spv_endianness_t, uint32_t, uint32_t, uint32_t,
                       uint32_t, uint32_t) {
    return SPV_SUCCESS;
}

spv_result_t on_inst(void *ud, const spv_parsed_instruction_t *ins) {
    auto *c = static_cast<Collector *>(ud);
    Module &m = c->m;
    const auto w = [ins](uint16_t i) { return ins->words[ins->operands[i].offset]; };

    switch (ins->opcode) {
    case SpvOpTypeVoid:
    case SpvOpTypeBool:
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeVector:
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray:
    case SpvOpTypeStruct:
    case SpvOpTypePointer: {
        TypeInfo t;
        t.op = static_cast<SpvOp>(ins->opcode);
        if (ins->opcode == SpvOpTypeInt || ins->opcode == SpvOpTypeFloat) {
            t.width = w(1);
        } else if (ins->opcode == SpvOpTypeVector) {
            t.elem = w(1);
            t.count = w(2);
        } else if (ins->opcode == SpvOpTypeArray) {
            t.elem = w(1);
            auto lit = m.consts.find(w(2));
            t.count = lit != m.consts.end()
                          ? static_cast<uint32_t>(lit->second.ival)
                          : 0;
        } else if (ins->opcode == SpvOpTypeRuntimeArray) {
            t.elem = w(1);
        } else if (ins->opcode == SpvOpTypeStruct) {
            for (uint16_t i = 1; i < ins->num_operands; i++)
                t.members.push_back(w(i));
        } else if (ins->opcode == SpvOpTypePointer) {
            t.storage = w(1);
            t.elem = w(2);
        }
        m.types[ins->result_id] = t;
        return SPV_SUCCESS;
    }
    case SpvOpConstant:
    case SpvOpSpecConstant: {
        // A specialization constant takes its DEFAULT value here. The ICD does
        // not implement vkCreateComputePipelines' pSpecializationInfo (the
        // Vulkan pipeline is created once and the values are not plumbed into
        // the emitter), so a shader specialized at pipeline creation would run
        // with different values than it was compiled with. Compiling the
        // default means the kernels are self-consistent: the guard a spec
        // constant selects is the one the compiled code takes.
        Const k;
        k.type_id = ins->type_id;
        const uint32_t bits = w(2);
        auto tt = m.types.find(ins->type_id);
        if (tt != m.types.end() && tt->second.op == SpvOpTypeFloat) {
            const bool half = tt->second.width == 16;
            k.set(llvm_type(m, ins->type_id),
                  half ? render_half(static_cast<uint16_t>(bits))
                       : render_float(bits));
        } else {
            k.is_int = true;
            k.ival = static_cast<int32_t>(bits);
            k.set(llvm_type(m, ins->type_id), std::to_string(k.ival));
        }
        m.consts[ins->result_id] = k;
        return SPV_SUCCESS;
    }
    case SpvOpConstantTrue:
    case SpvOpConstantFalse: {
        Const k;
        k.type_id = ins->type_id;
        k.is_int = true;
        k.ival = ins->opcode == SpvOpConstantTrue ? -1 : 0;
        k.set("i1", ins->opcode == SpvOpConstantTrue ? "true" : "false");
        m.consts[ins->result_id] = k;
        return SPV_SUCCESS;
    }
    case SpvOpConstantNull: {
        // Only the scalar zero spellings matter here; aggregate nulls never
        // reach a rendered operand in these shaders.
        Const k;
        k.type_id = ins->type_id;
        const std::string ty = llvm_type(m, ins->type_id);
        k.set(ty, ty == "i1" ? "false" : "0");
        m.consts[ins->result_id] = k;
        return SPV_SUCCESS;
    }
    case SpvOpConstantComposite: {
        // A composite constant is a value like any other: an array/vector
        // initialiser that reaches OpStore or a call argument.  Dropping it
        // (the old behaviour) left `%r<id>` undefined, and rendering it as a
        // zero fill would silently corrupt the NF4 quantisation codebook these
        // shaders carry.  Spell out every constituent.
        Const k;
        k.type_id = ins->type_id;
        const std::string ty = llvm_type(m, ins->type_id);
        auto inner = [&](uint32_t id, std::string &out_ty,
                         std::string &out_val) -> bool {
            auto c = m.consts.find(id);
            if (c == m.consts.end()) return false;
            out_ty = c->second.ty;
            out_val = c->second.val;
            return true;
        };
        std::string elems;
        bool ok = ins->num_operands > 2;
        for (uint16_t i = 2; i < ins->num_operands; i++) {
            std::string ety, eval;
            if (!inner(w(i), ety, eval)) { ok = false; break; }
            if (i > 2) elems += ", ";
            elems += ety + " " + eval;
        }
        if (!ok) {
            // A constituent that is not a rendered constant (another composite
            // we could not spell, or an undef): refuse rather than invent a
            // value.
            m.bad_consts.insert(ins->result_id);
            return SPV_SUCCESS;
        }
        // LLVM spells an array constant `[float 1.0, float 2.0]` (brackets,
        // element types spelled) and a struct constant `{ float 1.0, ... }`.
        // Using braces for an array is "constant expression type mismatch".
        auto t = m.types.find(ins->type_id);
        const bool is_array = t != m.types.end() &&
                              (t->second.op == SpvOpTypeArray ||
                               t->second.op == SpvOpTypeRuntimeArray);
        if (!ty.empty() && ty[0] == '<')
            k.set(ty, "<" + elems + ">");  // vector: lane values
        else if (is_array)
            k.set(ty, "[" + elems + "]");
        else
            k.set(ty, "{ " + elems + " }");  // struct constant
        m.consts[ins->result_id] = k;
        return SPV_SUCCESS;
    }
    case SpvOpVariable:
        if (c->cur_fn != 0) {
            if (!m.funcs.empty()) m.funcs.back().insts.emplace_back(ins);
            return SPV_SUCCESS;
        }
        m.var_type[ins->result_id] = ins->type_id;
        m.var_storage[ins->result_id] = w(2);
        return SPV_SUCCESS;
    case SpvOpDecorate:
        if (w(1) == SpvDecorationBuiltIn && ins->num_operands > 2)
            m.builtin[w(0)] = w(2);
        else if (w(1) == SpvDecorationBinding && ins->num_operands > 2)
            m.var_binding[w(0)] = w(2);
        else if (w(1) == SpvDecorationDescriptorSet && ins->num_operands > 2)
            m.var_set[w(0)] = w(2);
        return SPV_SUCCESS;
    case SpvOpName: {
        // Diagnostics only: "load from unmodelled pointer" has to say WHICH
        // variable it stumbled on.
        const char *s = reinterpret_cast<const char *>(
            ins->words + ins->operands[1].offset);
        m.names[w(0)] = s;
        return SPV_SUCCESS;
    }
    case SpvOpEntryPoint: {
        // Operands: execution model, entry-point id, name (literal string),
        // then the interface ids. The name is what an application passes as
        // VkPipelineShaderStageCreateInfo::pName, so it is carried through to
        // the emitted kernel symbol.
        m.entry_fn = w(1);
        const char *nm = reinterpret_cast<const char *>(
            ins->words + ins->operands[2].offset);
        m.entry_names[nm] = w(1);
        m.entry_interface.clear();
        for (uint16_t i = 3; i < ins->num_operands; i++)
            if (operand_is_id(ins->operands[i].type))
                m.entry_interface.push_back(w(i));
        return SPV_SUCCESS;
    }
    case SpvOpExecutionMode:
        if (w(1) == SpvExecutionModeLocalSize) {
            m.local_x = w(2);
            m.local_y = w(3);
            m.local_z = w(4);
            m.local_sizes[w(0)] = {w(2), w(3), w(4)};
        }
        return SPV_SUCCESS;
    case SpvOpFunction:
        c->cur_fn = ins->result_id;
        c->in_body = false;
        m.funcs.push_back(Function{});
        m.funcs.back().id = ins->result_id;
        m.funcs.back().ret_type = ins->type_id;
        m.funcs.back().fn_type = w(3);
        return SPV_SUCCESS;
    case SpvOpFunctionParameter:
        if (c->cur_fn != 0 && !m.funcs.empty())
            m.funcs.back().params.push_back(Param{ins->result_id, ins->type_id});
        return SPV_SUCCESS;
    case SpvOpFunctionEnd:
        c->cur_fn = 0;
        c->in_body = false;
        return SPV_SUCCESS;
    default:
        break;
    }

    if (c->cur_fn != 0 && !m.funcs.empty()) {
        if (ins->opcode == SpvOpLabel) c->in_body = true;
        if (c->in_body) m.funcs.back().insts.emplace_back(ins);
    } else if (ins->opcode == SpvOpSpecConstantOp) {
        c->spec_ops.emplace_back(ins);
    }
    return SPV_SUCCESS;
}

// --- GLSL.std.450 -----------------------------------------------------------
// The names are the spec's; only the ones a vkop elementwise shader reaches are
// given a lowering.  Anything else fails loud with the instruction number.

const char *glsl_ext_name(uint32_t n) {
    switch (n) {
    case 1: return "Round";   case 2: return "RoundEven"; case 3: return "Trunc";
    case 4: return "FAbs";    case 5: return "SAbs";      case 6: return "FSign";
    case 7: return "SSign";   case 8: return "Floor";     case 9: return "Ceil";
    case 10: return "Fract";  case 11: return "Radians";  case 12: return "Degrees";
    case 13: return "Sin";    case 14: return "Cos";      case 15: return "Tan";
    case 16: return "Asin";   case 17: return "Acos";     case 18: return "Atan";
    case 19: return "Sinh";   case 20: return "Cosh";     case 21: return "Tanh";
    case 22: return "Asinh";  case 23: return "Acosh";    case 24: return "Atanh";
    case 25: return "Atan2";  case 26: return "Pow";      case 27: return "Exp";
    case 28: return "Log";    case 29: return "Exp2";     case 30: return "Log2";
    case 31: return "Sqrt";   case 32: return "InverseSqrt";
    case 37: return "FMin";   case 38: return "UMin";     case 39: return "SMin";
    case 40: return "FMax";   case 41: return "UMax";     case 42: return "SMax";
    case 43: return "FClamp"; case 44: return "UClamp";   case 45: return "SClamp";
    case 46: return "FMix";   case 47: return "IMix";     case 48: return "Step";
    case 49: return "SmoothStep"; case 50: return "Fma";
    case 51: return "Frexp";  case 52: return "FrexpStruct"; case 53: return "Ldexp";
    case 54: return "PackSnorm4x8"; case 55: return "PackUnorm4x8";
    case 56: return "PackSnorm2x16"; case 57: return "PackUnorm2x16";
    case 58: return "PackHalf2x16"; case 59: return "PackDouble2x32";
    case 60: return "UnpackSnorm2x16"; case 61: return "UnpackUnorm2x16";
    case 62: return "UnpackHalf2x16"; case 63: return "UnpackSnorm4x8";
    case 64: return "UnpackUnorm4x8"; case 65: return "UnpackDouble2x32";
    case 66: return "Length"; case 67: return "Distance"; case 68: return "Cross";
    case 69: return "Normalize"; case 70: return "FaceForward";
    case 71: return "Reflect"; case 72: return "Refract";
    case 73: return "FindILsb"; case 74: return "FindSMsb"; case 75: return "FindUMsb";
    case 76: return "InterpolateAtCentroid"; case 77: return "InterpolateAtSample";
    case 78: return "InterpolateAtOffset";
    case 79: return "NMin";   case 80: return "NMax";     case 81: return "NClamp";
    default: return nullptr;
    }
}

// --- Pass 2: emit -----------------------------------------------------------

struct Unsupported {
    std::string what;
};

// A pointer value, kept split so GEP can spell the element type and address
// space rather than re-parsing a rendered string.
struct PtrRef {
    std::string elem;
    std::string as = "1";
    std::string name;
    bool gid = false;
    // Non-empty when this pointer is an Input/Workgroup *builtin* rather than
    // memory.  Holds the component suffix ("x"/"y"/"z"), so loading it can
    // resolve to the right sreg read.
    std::string builtin;  // "", "GlobalInvocationId", "WorkgroupId", ...
    std::string comp;     // "x", "y", "z"
    std::string render() const {
        return elem + " addrspace(" + as + ")* " + name;
    }
};

class Emitter {
  public:
    Emitter(const Module &m, const Function &fn, bool entry,
            std::vector<uint32_t> buffers, std::vector<uint32_t> all_buffers)
        : m_(m), fn_(fn), entry_(entry), buffers_(std::move(buffers)),
          all_buffers_(std::move(all_buffers)) {}

    std::string run() {
        // Input variables decorated with an invocation-index builtin are not
        // memory: they resolve to sreg reads.  GlobalInvocationId keeps its
        // historical %gid alias for .x; every other builtin (and .y/.z) takes
        // the generic path in resolve_builtin().
        for (const auto &kv : m_.builtin)
            if (is_index_builtin(kv.second)) builtin_ids_[kv.first] = kv.second;
        emit_signature();
        if (entry_) emit_prologue();
        else emit_helper_prologue();
        emit_phi_allocas();
        emit_body();
        fixup_phi_edges();
        splice_sreg_temps();
        return decl_ + body_;
    }

  private:
    const Module &m_;
    const Function &fn_;
    bool entry_;
    std::vector<uint32_t> buffers_;
    std::vector<uint32_t> all_buffers_;
    std::string decl_;
    std::string body_;

    std::map<uint32_t, PtrRef> ptr_;                 // pointer id -> PtrRef
    std::map<uint32_t, uint32_t> ptype_;             // pointer id -> SPIR-V ptr type id
    std::map<uint32_t, std::string> val_;            // value id -> "%rN"
    std::map<uint32_t, uint32_t> vtype_;             // value id -> SPIR-V type id
    std::map<uint32_t, std::string> base_;           // module var id -> SSA name
    std::map<uint32_t, uint32_t> builtin_ids_;       // Input var id -> SpvBuiltIn
    std::map<std::string, std::string> sreg_;        // sreg name -> SSA temp
    std::map<std::string, std::string> bval_;        // "<builtin>.<comp>" -> SSA
    std::string hoist_;                              // entry-block definitions

    static bool is_index_builtin(uint32_t b) {
        switch (b) {
        case SpvBuiltInGlobalInvocationId:
        case SpvBuiltInLocalInvocationId:
        case SpvBuiltInLocalInvocationIndex:
        case SpvBuiltInWorkgroupId:
        case SpvBuiltInNumWorkgroups:
        case SpvBuiltInSubgroupId:
        case SpvBuiltInSubgroupLocalInvocationId:
        case SpvBuiltInNumSubgroups:
            return true;
        default:
            return false;
        }
    }

    // Read llvm.ppu.read.ptx.sreg.<name> once per function.  The definitions
    // are collected here and spliced into the entry block by run(); issuing
    // them lazily would put the call in whichever block first asked, and a
    // sibling block's use would then be dominated by nothing.
    std::string sreg(const std::string &name) {
        auto it = sreg_.find(name);
        if (it != sreg_.end()) return it->second;
        const std::string t = tmp();
        sreg_temps_.emplace_back(t, name);
        sreg_[name] = t;
        return t;
    }
    std::vector<std::pair<std::string, std::string>> sreg_temps_;
    static constexpr const char *kSregMarker = "  ; __SREG_TEMPS__\n";

    void splice_sreg_temps() {
        std::string block;
        for (const auto &kv : sreg_temps_)
            block += "  " + kv.first + " = call i32 @llvm.ppu.read.ptx.sreg." +
                     kv.second + "()\n";
        // Builtin index values are computed from those sreg reads; keeping them
        // in one hoisted block is what makes them dominate every use.
        block += hoist_;
        const size_t at = body_.find(kSregMarker);
        if (at == std::string::npos) {
            // No prologue (should not happen): prepend so the reads still
            // dominate every use.
            body_ = block + body_;
            return;
        }
        body_.replace(at, std::strlen(kSregMarker), block);
    }

    // --- phi demotion (reg2mem) ---
    // Each OpPhi becomes an entry-block alloca.  The block that owns the phi
    // loads it at its head; every predecessor stores its incoming value just
    // before its terminator.  This is what LLVM's reg2mem does and it avoids
    // materialising real SSA edges (no dominance bookkeeping, no ordering
    // constraint: a predecessor may be emitted after its successor).
    struct Phi {
        uint32_t id = 0;       // result id (also the alloca name suffix)
        uint32_t type_id = 0;
        uint32_t block = 0;    // block that contains the phi (loads here)
        std::vector<std::pair<uint32_t, uint32_t>> edges;  // (value, pred label)
    };
    std::map<uint32_t, Phi> phis_;                // phi result id -> Phi
    std::map<uint32_t, std::vector<uint32_t>> phis_in_block_;  // label -> phi ids
    std::map<uint32_t, std::vector<uint32_t>> edges_from_block_;  // label -> phi ids
    uint32_t cur_block_ = 0;

    uint32_t first_label_ = 0;
    int tmp_ = 0;
    bool terminated_ = false;

    // Basic-block bookkeeping, used to splice the phi-edge stores in front of
    // each predecessor's terminator once every block has been emitted.
    struct BlockRec {
        uint32_t label = 0;
        size_t start = 0;    // offset of the block's first byte in body_
        size_t term = 0;     // offset of the terminator instruction
        bool has_term = false;
    };
    std::vector<BlockRec> block_recs_;
    std::map<uint32_t, size_t> block_by_label_;

    std::string tmp() { return "%t" + std::to_string(tmp_++); }
    static std::string arg_name(uint32_t id) { return "%p" + std::to_string(id); }

    // The kernel keeps the SPIR-V entry-point name. cuModuleGetFunction later
    // looks the kernel up by exactly the VkPipelineShaderStageCreateInfo::pName
    // string, and llc derives the symbol from this LLVM function name, so the
    // two must agree. Names that are not valid bare LLVM identifiers are
    // quoted ("@\"my-kernel\"") -- llc accepts that and still emits the bare
    // symbol, which is what the driver looks up.
    static std::string quote_symbol(const std::string &name) {
        if (name.empty()) return "@main";
        bool bare = !(name[0] >= '0' && name[0] <= '9');
        for (unsigned char c : name)
            if (!(std::isalnum(c) || c == '_' || c == '.' || c == '$')) { bare = false; break; }
        if (bare) return "@" + name;
        std::string q = "@\"";
        for (char c : name) {
            if (c == '"' || c == '\\') q += '\\';
            q += c;
        }
        return q + "\"";
    }

    std::string fn_name() const {
        if (!entry_) return "@f" + std::to_string(fn_.id);
        auto it = m_.entry_names.end();
        for (auto e = m_.entry_names.begin(); e != m_.entry_names.end(); ++e)
            if (e->second == fn_.id) { it = e; break; }
        return quote_symbol(it == m_.entry_names.end() ? std::string("main")
                                                       : it->first);
    }

    // --- phi demotion ---

    // Collect every OpPhi up front (they are the first instructions of their
    // block, but their incoming values may not be emitted yet) and give each
    // one an i1/whatever alloca in the function-entry block.  Emission order of
    // the allocas does not matter: they are addresses, and both the head load
    // and the predecessor stores mention them by name.
    void emit_phi_allocas() {
        uint32_t cb = first_label_;
        for (const Inst &ins : fn_.insts) {
            if (ins.opcode == SpvOpLabel) { cb = ins.result_id; continue; }
            if (ins.opcode != SpvOpPhi) continue;
            Phi p;
            p.id = ins.result_id;
            p.type_id = ins.type_id;
            p.block = cb;
            // Operands: type id, result id, then (value, parent) pairs.
            for (uint16_t i = 2; i + 1 < ins.argc(); i += 2)
                p.edges.emplace_back(ins.word(i), ins.word(i + 1));
            phis_[p.id] = p;
            phis_in_block_[cb].push_back(p.id);
        }
        for (const auto &kv : phis_) {
            const Phi &p = kv.second;
            const std::string ty = llvm_type(m_, p.type_id);
            body_ += "  %phi" + std::to_string(p.id) + " = alloca " + ty +
                     ", align 4\n";
            PtrRef r;
            r.elem = ty;
            r.as = "0";
            r.name = "%phi" + std::to_string(p.id);
            ptr_[p.id] = r;
        }
    }

    // At a block head, load each of that block's phi allocas.  The result is an
    // ordinary SSA value of the phi's id, so downstream uses are unchanged.
    void emit_phi_loads(uint32_t label) {
        auto it = phis_in_block_.find(label);
        if (it == phis_in_block_.end()) return;
        for (uint32_t id : it->second) {
            const std::string ty = llvm_type(m_, phis_[id].type_id);
            val_[id] = "%r" + std::to_string(id);
            vtype_[id] = phis_[id].type_id;
            body_ += "  %r" + std::to_string(id) + " = load " + ty + ", " + ty +
                     "* %phi" + std::to_string(id) + "\n";
        }
    }

    // After the whole body is emitted, splice each phi-edge store in front of
    // its predecessor's terminator.  Done as a single rebuild so the byte
    // offsets recorded during emission stay valid (inserting in place would
    // shift every later offset).
    void fixup_phi_edges() {
        std::map<size_t, std::string> inserts;
        for (const auto &kv : phis_) {
            const Phi &p = kv.second;
            const std::string ty = llvm_type(m_, p.type_id);
            for (const auto &e : p.edges) {
                auto b = block_by_label_.find(e.second);
                if (b == block_by_label_.end()) continue;
                const BlockRec &rec = block_recs_[b->second];
                if (!rec.has_term) continue;
                inserts[rec.term] += "  store " + value_text(e.first, ty.c_str()) +
                                     ", " + ty + "* %phi" + std::to_string(p.id) +
                                     "\n";
            }
        }
        if (inserts.empty()) return;
        std::string out;
        out.reserve(body_.size() + 256 * inserts.size());
        size_t prev = 0;
        for (const auto &kv : inserts) {
            out.append(body_, prev, kv.first - prev);
            out += kv.second;
            prev = kv.first;
        }
        out.append(body_, prev, body_.size() - prev);
        body_.swap(out);
    }

    // Note where the current block's terminator begins, so phi-edge stores can
    // be inserted immediately before it.
    static bool is_terminator(uint16_t op) {
        switch (op) {
        case SpvOpBranch:
        case SpvOpBranchConditional:
        case SpvOpSwitch:
        case SpvOpReturn:
        case SpvOpReturnValue:
        case SpvOpUnreachable:
        case SpvOpKill:
        case SpvOpTerminateInvocation:
            return true;
        default:
            return false;
        }
    }

    void mark_term() {
        if (block_recs_.empty()) return;
        BlockRec &rec = block_recs_.back();
        if (rec.has_term) return;
        rec.term = body_.size();
        rec.has_term = true;
    }

    // An operand with its type always spelled: "T %x", "T* %x", "T 3".  Used
    // where LLVM requires a typed value regardless of what the id holds (call
    // arguments, phi-edge stores).  Pointers already carry their type in
    // PtrRef::render.
    std::string typed_operand(uint32_t id) {
        auto p = ptr_.find(id);
        if (p != ptr_.end()) return p->second.render();
        auto c = m_.consts.find(id);
        if (c != m_.consts.end()) return c->second.text;
        return type_of_value(id) + " " + bare_value(id);
    }

    std::string type_of_value(uint32_t id) const {
        auto v = vtype_.find(id);
        if (v != vtype_.end()) return llvm_type(m_, v->second);
        auto c = m_.consts.find(id);
        if (c != m_.consts.end()) return llvm_type(m_, c->second.type_id);
        return "i32";
    }

    std::string bare_value(uint32_t id) {
        auto v = val_.find(id);
        if (v != val_.end()) return v->second;
        auto p = ptr_.find(id);
        if (p != ptr_.end()) return p->second.render();
        auto c = m_.consts.find(id);
        if (c != m_.consts.end()) return c->second.val;
        if (m_.bad_consts.count(id))
            throw Unsupported{"unrenderable composite constant %" +
                              std::to_string(id)};
        return "%r" + std::to_string(id);
    }

    // A reference operand: "T %x" / "T* %x" / "T const" as appropriate.
    std::string value_text(uint32_t id, const char *want_type = nullptr) {
        auto v = val_.find(id);
        if (v != val_.end())
            return want_type ? std::string(want_type) + " " + v->second
                             : v->second;
        auto p = ptr_.find(id);
        if (p != ptr_.end()) return p->second.render();
        auto c = m_.consts.find(id);
        if (c != m_.consts.end())
            return want_type ? std::string(want_type) + " " + c->second.val
                             : c->second.text;
        return want_type ? std::string(want_type) + " %r" + std::to_string(id)
                         : "%r" + std::to_string(id);
    }

    // Widen an i32 index to i64 once, memoised: PPU GEP array indices must be i64.
    // Widen an i32 index to i64: PPU GEP array indices must be i64.  Deliberately
    // NOT memoised -- a cached temp created in one block would not dominate a
    // later use in a sibling block.
    std::string as_i64(uint32_t id) {
        const std::string t = tmp();
        body_ += "  " + t + " = sext i32 " +
                 (m_.consts.count(id) ? bare_value(id) : value_text(id)) + " to i64\n";
        return t;
    }

    // The element type / address space a base pointer points at.
    uint32_t ptype_of(uint32_t id) const {
        auto p = ptype_.find(id);
        if (p != ptype_.end()) return p->second;
        auto v = m_.var_type.find(id);
        if (v != m_.var_type.end()) return v->second;
        return 0;
    }

    // --- signature / prologue ---

    void emit_signature() {
        if (entry_) {
            decl_ += "define ppu_kernel void " + fn_name() + "(";
            for (size_t i = 0; i < buffers_.size(); i++) {
                if (i) decl_ += ", ";
                const uint32_t b = buffers_[i];
                decl_ += llvm_type(m_, type_pointee(m_, var_ptr_type(m_, b))) +
                         " addrspace(1)* " + arg_name(b);
            }
            decl_ += ") {\n";
        } else {
            decl_ += "define internal " + llvm_type(m_, fn_.ret_type) + " " +
                     fn_name() + "(";
            for (size_t i = 0; i < fn_.params.size(); i++) {
                if (i) decl_ += ", ";
                const Param &p = fn_.params[i];
                const uint32_t pty = p.type_id;
                decl_ += llvm_type(m_, pty) + " %a" + std::to_string(p.id);
                // A parameter is only a *pointer* if its type is one; the
                // conv1d helpers take plain ints by value.
                auto tt = m_.types.find(pty);
                if (tt != m_.types.end() && tt->second.op == SpvOpTypePointer) {
                    PtrRef r;
                    r.elem = llvm_type(m_, type_pointee(m_, pty));
                    r.as = addrspace_of(type_storage(m_, pty));
                    r.name = "%a" + std::to_string(p.id);
                    ptr_[p.id] = r;
                    ptype_[p.id] = pty;
                } else {
                    val_[p.id] = "%a" + std::to_string(p.id);
                    vtype_[p.id] = pty;
                }
            }
            decl_ += ") {\n";
        }
    }

    // gl_GlobalInvocationID.x = ctaid.x * ntid.x + tid.x.  Kernel buffer
    // arguments are published into module-level cells so the (deliberately
    // non-inlined) helper functions can reach the same buffers; the entry block
    // also pre-loads each cell into a temp so later blocks share one dominating
    // definition.
    void emit_prologue() {
        body_ += "entry:\n";
        body_ += "  %tid = call i32 @llvm.ppu.read.ptx.sreg.tid.x()\n";
        body_ += "  %ctaid = call i32 @llvm.ppu.read.ptx.sreg.ctaid.x()\n";
        body_ += "  %ntid = call i32 @llvm.ppu.read.ptx.sreg.ntid.x()\n";
        body_ += "  %lin = mul i32 %ctaid, %ntid\n";
        body_ += "  %gid = add i32 %lin, %tid\n";
        body_ += kSregMarker;
        for (uint32_t b : buffers_) {
            const std::string t =
                llvm_type(m_, type_pointee(m_, var_ptr_type(m_, b)));
            body_ += "  store " + t + " addrspace(1)* " + arg_name(b) + ", " + t +
                     " addrspace(1)** @g" + std::to_string(b) + "\n";
        }
        // Function-scope reads of a buffer variable all go through these temps.
        for (uint32_t id : all_buffers_) {
            auto it = base_.find(id);
            if (it != base_.end()) continue;
            const std::string t = llvm_type(m_, type_pointee(m_, var_ptr_type(m_, id)));
            const std::string tr = tmp();
            body_ += "  " + tr + " = load " + t + " addrspace(1)*, " + t +
                     " addrspace(1)** @g" + std::to_string(id) + "\n";
            base_[id] = tr;
        }
        for (const Inst &ins : fn_.insts)
            if (ins.opcode == SpvOpLabel) { first_label_ = ins.result_id; break; }
    }

    // A helper's first basic block is its function-entry block, so it can carry
    // the same name; parameters are in scope there.  Pre-load each buffer cell so
    // the whole body shares one dominating definition per buffer.
    void emit_helper_prologue() {
        for (const Inst &ins : fn_.insts)
            if (ins.opcode == SpvOpLabel) { first_label_ = ins.result_id; break; }
        body_ += "entry:\n";
        // A helper can read an invocation-index builtin too (attention's
        // helpers index by tid), so it needs the same %tid/%ctaid/%ntid names
        // and the sreg splice point as the entry kernel.
        body_ += "  %tid = call i32 @llvm.ppu.read.ptx.sreg.tid.x()\n";
        body_ += "  %ctaid = call i32 @llvm.ppu.read.ptx.sreg.ctaid.x()\n";
        body_ += "  %ntid = call i32 @llvm.ppu.read.ptx.sreg.ntid.x()\n";
        body_ += "  %lin = mul i32 %ctaid, %ntid\n";
        body_ += "  %gid = add i32 %lin, %tid\n";
        body_ += kSregMarker;
        for (uint32_t id : all_buffers_) {
            auto it = base_.find(id);
            if (it != base_.end()) continue;
            const std::string t = llvm_type(m_, type_pointee(m_, var_ptr_type(m_, id)));
            const std::string tr = tmp();
            body_ += "  " + tr + " = load " + t + " addrspace(1)*, " + t +
                     " addrspace(1)** @g" + std::to_string(id) + "\n";
            base_[id] = tr;
        }
    }

    // --- body ---

    void emit_body() {
        for (const Inst &ins : fn_.insts) {
            if (is_terminator(ins.opcode)) mark_term();
            switch (ins.opcode) {
            case SpvOpLabel: {
                // Record where this block starts so phi-edge stores can be
                // spliced in front of its terminator later.
                block_by_label_[ins.result_id] = block_recs_.size();
                BlockRec rec;
                rec.label = ins.result_id;
                if (ins.result_id == first_label_) {
                    // The function-entry block's "entry:" label was printed by
                    // the prologue; its start is the offset that follows it.
                    rec.start = body_.size();
                } else {
                    rec.start = body_.size();
                    body_ += "L" + std::to_string(ins.result_id) + ":\n";
                }
                cur_block_ = ins.result_id;
                block_recs_.push_back(rec);
                terminated_ = false;
                emit_phi_loads(ins.result_id);
                break;
            }
            case SpvOpPhi:
                // Demoted to an alloca: loads are emitted at the block head,
                // stores at each predecessor's terminator.  Nothing here.
                break;
            case SpvOpVariable: emit_local_var(ins); break;
            case SpvOpLoad: emit_load(ins); break;
            case SpvOpStore: emit_store(ins); break;
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain: emit_access_chain(ins); break;
            case SpvOpCompositeExtract: emit_composite_extract(ins); break;
            case SpvOpCompositeConstruct: emit_composite_construct(ins); break;
            case SpvOpCompositeInsert: emit_composite_insert(ins); break;
            case SpvOpVectorExtractDynamic: emit_vector_extract_dynamic(ins); break;
            case SpvOpVectorShuffle: emit_vector_shuffle(ins); break;
            case SpvOpFMul: emit_binop(ins, "fmul"); break;
            case SpvOpFAdd: emit_binop(ins, "fadd"); break;
            case SpvOpFSub: emit_binop(ins, "fsub"); break;
            // Plain `fdiv` cannot be lowered for PPU: the backend emits a call
            // to the device libcall __ppumathpriv_div_default_f32, which the
            // SDK ships no implementation of, and llc then hangs.  The native
            // path is llvm.ppu.fdiv.fast (a range-checked fast division).
            case SpvOpFDiv: emit_ppu_fdiv(ins); break;
            case SpvOpFNegate: emit_unop(ins, "fneg"); break;
            case SpvOpIAdd: emit_binop(ins, "add"); break;
            case SpvOpISub: emit_binop(ins, "sub"); break;
            case SpvOpIMul: emit_binop(ins, "mul"); break;
            case SpvOpSDiv: emit_binop(ins, "sdiv"); break;
            case SpvOpUDiv: emit_binop(ins, "udiv"); break;
            case SpvOpSRem: emit_binop(ins, "srem"); break;
            case SpvOpSMod: emit_binop(ins, "srem"); break;   // GLSL % is srem
            case SpvOpUMod: emit_binop(ins, "urem"); break;
            case SpvOpBitwiseAnd: emit_binop(ins, "and"); break;
            case SpvOpBitwiseOr: emit_binop(ins, "or"); break;
            case SpvOpBitwiseXor: emit_binop(ins, "xor"); break;
            case SpvOpShiftLeftLogical: emit_binop(ins, "shl"); break;
            case SpvOpShiftRightArithmetic: emit_binop(ins, "ashr"); break;
            case SpvOpShiftRightLogical: emit_binop(ins, "lshr"); break;
            case SpvOpSLessThan: emit_icmp(ins, "slt"); break;
            case SpvOpSLessThanEqual: emit_icmp(ins, "sle"); break;
            case SpvOpSGreaterThan: emit_icmp(ins, "sgt"); break;
            case SpvOpSGreaterThanEqual: emit_icmp(ins, "sge"); break;
            case SpvOpIEqual: emit_icmp(ins, "eq"); break;
            case SpvOpINotEqual: emit_icmp(ins, "ne"); break;
            case SpvOpULessThan: emit_icmp(ins, "ult"); break;
            case SpvOpULessThanEqual: emit_icmp(ins, "ule"); break;
            case SpvOpUGreaterThan: emit_icmp(ins, "ugt"); break;
            case SpvOpUGreaterThanEqual: emit_icmp(ins, "uge"); break;
            case SpvOpFOrdLessThan: emit_fcmp(ins, "olt"); break;
            case SpvOpFOrdGreaterThan: emit_fcmp(ins, "ogt"); break;
            case SpvOpFOrdEqual: emit_fcmp(ins, "oeq"); break;
            case SpvOpFOrdNotEqual: emit_fcmp(ins, "one"); break;
            case SpvOpFOrdLessThanEqual: emit_fcmp(ins, "ole"); break;
            case SpvOpFOrdGreaterThanEqual: emit_fcmp(ins, "oge"); break;
            case SpvOpFUnordLessThan: emit_fcmp(ins, "ult"); break;
            case SpvOpFUnordGreaterThan: emit_fcmp(ins, "ugt"); break;
            case SpvOpFUnordLessThanEqual: emit_fcmp(ins, "ule"); break;
            case SpvOpFUnordGreaterThanEqual: emit_fcmp(ins, "uge"); break;
            case SpvOpFUnordEqual: emit_fcmp(ins, "ueq"); break;
            case SpvOpFUnordNotEqual: emit_fcmp(ins, "une"); break;
            case SpvOpLogicalAnd: emit_binop(ins, "and"); break;
            case SpvOpLogicalOr: emit_binop(ins, "or"); break;
            case SpvOpLogicalNot: emit_logical_not(ins); break;
            case SpvOpSelect: emit_select(ins); break;
            case SpvOpBitcast: emit_bitcast(ins); break;
            case SpvOpUConvert: emit_width_convert(ins); break;
            case SpvOpSConvert: emit_width_convert(ins); break;
            case SpvOpFConvert: emit_fp_convert(ins); break;
            case SpvOpConvertFToU: emit_convert(ins, "fptoui"); break;
            case SpvOpConvertFToS: emit_convert(ins, "fptosi"); break;
            case SpvOpConvertSToF: emit_convert(ins, "sitofp"); break;
            case SpvOpConvertUToF: emit_convert(ins, "uitofp"); break;
            case SpvOpFunctionCall: emit_call(ins); break;
            case SpvOpVectorTimesScalar: emit_vector_times_scalar(ins); break;
            case SpvOpSNegate: emit_binop_zero(ins, "sub"); break;
            case SpvOpBitFieldSExtract: emit_bitfield_sextract(ins); break;
            case SpvOpControlBarrier: emit_control_barrier(ins); break;
            case SpvOpMemoryBarrier: emit_memory_barrier(ins); break;
            case SpvOpAtomicIAdd:
            case SpvOpAtomicISub:
            case SpvOpAtomicAnd:
            case SpvOpAtomicOr:
            case SpvOpAtomicXor:
            case SpvOpAtomicUMax:
            case SpvOpAtomicSMax:
            case SpvOpAtomicUMin:
            case SpvOpAtomicSMin:
            case SpvOpAtomicExchange:
                emit_atomic_rmw(ins);
                break;
            case SpvOpAtomicCompareExchange: emit_atomic_cmpxchg(ins); break;
            case SpvOpAtomicLoad: emit_atomic_load(ins); break;
            case SpvOpAtomicStore: emit_atomic_store(ins); break;
            case SpvOpGroupNonUniformFAdd:
                emit_group_reduce_float(ins, nullptr);  // plain fadd
                break;
            case SpvOpGroupNonUniformFMax:
                emit_group_reduce_float(ins, "maxnum");
                break;
            case SpvOpGroupNonUniformFMin:
                emit_group_reduce_float(ins, "minnum");
                break;
            case SpvOpGroupNonUniformIAdd:
                emit_group_reduce_int(ins, "add", false);
                break;
            case SpvOpGroupNonUniformIMul:
                emit_group_reduce_int(ins, "mul", false);
                break;
            case SpvOpGroupNonUniformSMax:
                emit_group_reduce_int(ins, "smax", true);
                break;
            case SpvOpGroupNonUniformSMin:
                emit_group_reduce_int(ins, "smin", true);
                break;
            case SpvOpGroupNonUniformUMax:
                emit_group_reduce_int(ins, "umax", false);
                break;
            case SpvOpGroupNonUniformUMin:
                emit_group_reduce_int(ins, "umin", false);
                break;
            case SpvOpGroupNonUniformBitwiseAnd:
                emit_group_reduce_int(ins, "and", false);
                break;
            case SpvOpGroupNonUniformBitwiseOr:
                emit_group_reduce_int(ins, "or", false);
                break;
            case SpvOpGroupNonUniformBitwiseXor:
                emit_group_reduce_int(ins, "xor", false);
                break;
            case SpvOpGroupNonUniformElect: emit_group_elect(ins); break;
            case SpvOpExtInst: emit_ext_inst(ins); break;
            case SpvOpBranch: emit_branch(ins); break;
            case SpvOpBranchConditional: emit_branch_conditional(ins); break;
            case SpvOpSwitch: emit_switch(ins); break;
            case SpvOpSelectionMerge:
            case SpvOpLoopMerge:
            case SpvOpNop:
            case SpvOpLine:
            case SpvOpNoLine:
                break;
            case SpvOpReturn:
                body_ += "  ret void\n";
                terminated_ = true;
                break;
            case SpvOpReturnValue:
                body_ += "  ret " + llvm_type(m_, fn_.ret_type) + " " +
                         bare_value(ins.word(0)) + "\n";
                terminated_ = true;
                break;
            case SpvOpUnreachable:
                body_ += "  unreachable\n";
                terminated_ = true;
                break;
            default:
                throw Unsupported{spvOpcodeString(ins.opcode)};
            }
        }
        if (!terminated_) body_ += "  ret void\n";
        body_ += "}\n";
    }

    // --- instructions ---

    void emit_local_var(const Inst &ins) {
        const uint32_t pty = ins.type_id;
        const uint32_t st = type_storage(m_, pty);
        PtrRef r;
        r.elem = llvm_type(m_, type_pointee(m_, pty));
        r.as = addrspace_of(st);
        r.name = "%l" + std::to_string(ins.result_id);
        body_ += "  " + r.name + " = alloca " + r.elem + ", align 4\n";
        ptr_[ins.result_id] = r;
        ptype_[ins.result_id] = pty;
    }

    // Resolve a module-scope (buffer/push-constant) variable to an SSA pointer,
    // loading the cell the entry kernel published.
    std::string mod_base(uint32_t id) {
        auto b = base_.find(id);
        if (b != base_.end()) return b->second;
        const std::string t = llvm_type(m_, type_pointee(m_, var_ptr_type(m_, id)));
        const std::string tr = tmp();
        body_ += "  " + tr + " = load " + t + " addrspace(1)*, " + t +
                 " addrspace(1)** @g" + std::to_string(id) + "\n";
        base_[id] = tr;
        return tr;
    }

    PtrRef base_pointer(uint32_t id) {
        auto p = ptr_.find(id);
        if (p != ptr_.end()) return p->second;
        if (builtin_ids_.count(id)) {
            PtrRef r;
            r.builtin = builtin_name(builtin_ids_[id]);
            return r;
        }
        if (m_.var_storage.count(id)) {
            PtrRef r;
            const uint32_t storage = m_.var_storage.at(id);
            r.elem = llvm_type(m_, type_pointee(m_, var_ptr_type(m_, id)));
            r.as = addrspace_of(storage);
            // Workgroup storage is a module-level addrspace(3) global; the
            // buffer cells (@gN) are only for the bound buffers.
            r.name = storage == SpvStorageClassWorkgroup
                         ? "@w" + std::to_string(id)
                         : mod_base(id);
            return r;
        }
        std::string msg = "access chain base is neither buffer nor local";
        msg += " %" + std::to_string(id);
        auto n = m_.names.find(id);
        if (n != m_.names.end()) msg += " (" + n->second + ")";
        auto t = m_.types.find(ptype_of(id));
        if (t != m_.types.end())
            msg += " ptrtype-storage=" +
                   std::to_string(static_cast<int>(t->second.storage));
        throw Unsupported{msg};
    }

    static const char *builtin_name(uint32_t b) {
        switch (b) {
        case SpvBuiltInGlobalInvocationId: return "GlobalInvocationId";
        case SpvBuiltInLocalInvocationId: return "LocalInvocationId";
        case SpvBuiltInLocalInvocationIndex: return "LocalInvocationIndex";
        case SpvBuiltInWorkgroupId: return "WorkgroupId";
        case SpvBuiltInNumWorkgroups: return "NumWorkgroups";
        case SpvBuiltInSubgroupId: return "SubgroupId";
        case SpvBuiltInSubgroupLocalInvocationId: return "SubgroupLocalInvocationId";
        case SpvBuiltInNumSubgroups: return "NumSubgroups";
        default: return nullptr;
        }
    }

    // A builtin is not storage: reading it compiles to sreg reads and integer
    // arithmetic.  Every definition is hoisted into the entry block through
    // `hoist_` -- computing it lazily in the block that first asks would leave
    // a sibling block's use without a dominating definition.
    //
    // PPU warp size is 32 (measured on device), so subgroup index derives from
    // the Vulkan LocalInvocationIndex definition.
    std::string builtin_value(const std::string &b, const std::string &c) {
        auto key = b + "." + c;
        auto it = bval_.find(key);
        if (it != bval_.end()) return it->second;

        auto emit = [&](const std::string &line, const std::string &res) {
            hoist_ += "  " + res + " = " + line + "\n";
        };
        // sreg reads for one axis component of tid/ctaid/ntid/nctaid.
        auto axis = [&](const char *base, const std::string &comp) {
            if (comp == "x") {
                if (std::strcmp(base, "tid") == 0) return std::string("%tid");
                if (std::strcmp(base, "ctaid") == 0) return std::string("%ctaid");
                if (std::strcmp(base, "ntid") == 0) return std::string("%ntid");
            }
            return sreg(std::string(base) + "." + comp);
        };
        // LocalInvocationIndex = tid.z*(ntid.x*ntid.y) + tid.y*ntid.x + tid.x
        // (Vulkan definition; PPU's tid.x is the intra-warp lane and tid.y the
        // warp, both measured on device).
        auto local_index = [&]() {
            auto k = bval_.find("LocalInvocationIndex.x");
            if (k != bval_.end()) return k->second;
            const std::string nx = axis("ntid", "x"), ny = axis("ntid", "y"),
                              nz = axis("ntid", "z");
            const std::string tx = axis("tid", "x"), ty = axis("tid", "y"),
                              tz = axis("tid", "z");
            const std::string nxy = tmp();
            emit("mul i32 " + nx + ", " + ny, nxy);
            const std::string zpart = tmp();
            emit("mul i32 " + nxy + ", " + tz, zpart);
            const std::string ypart = tmp();
            emit("mul i32 " + ny + ", " + ty, ypart);
            const std::string sum = tmp();
            emit("add i32 " + zpart + ", " + ypart, sum);
            const std::string r2 = tmp();
            emit("add i32 " + sum + ", " + tx, r2);
            bval_["LocalInvocationIndex.x"] = r2;
            return r2;
        };

        std::string v;
        if (b == "GlobalInvocationId" && c == "x") {
            v = "%gid";
        } else if (b == "GlobalInvocationId") {
            // ctaid.c * ntid.c + tid.c
            const std::string a = tmp();
            emit("mul i32 " + axis("ctaid", c) + ", " + axis("ntid", c), a);
            v = tmp();
            emit("add i32 " + a + ", " + axis("tid", c), v);
        } else if (b == "LocalInvocationId") {
            v = axis("tid", c);
        } else if (b == "WorkgroupId") {
            v = axis("ctaid", c);
        } else if (b == "NumWorkgroups") {
            v = axis("nctaid", c);
        } else if (b == "LocalInvocationIndex") {
            v = local_index();
        } else if (b == "SubgroupId") {
            v = tmp();
            emit("lshr i32 " + local_index() + ", 5", v);  // /32, warp size
        } else if (b == "SubgroupLocalInvocationId") {
            v = tmp();
            emit("and i32 " + local_index() + ", 31", v);
        } else if (b == "NumSubgroups") {
            const std::string p = tmp();
            emit("mul i32 " + axis("ntid", "x") + ", " + axis("ntid", "y"), p);
            const std::string n = tmp();
            emit("mul i32 " + p + ", " + axis("ntid", "z"), n);
            v = tmp();
            emit("lshr i32 " + n + ", 5", v);
        } else {
            throw Unsupported{"builtin " + b + "." + c};
        }
        bval_[key] = v;
        return v;
    }


    std::string unmodelled_ptr_msg(const char *what, uint32_t ptr) const {
        std::string s = std::string(what) + " from unmodelled pointer %" +
                        std::to_string(ptr);
        auto n = m_.names.find(ptr);
        if (n != m_.names.end()) s += " (" + n->second + ")";
        auto v = m_.var_storage.find(ptr);
        if (v != m_.var_storage.end())
            s += " storage=" + std::string(storage_name(v->second));
        return s;
    }

    void emit_load(const Inst &ins) {
        // Operands are (type, result, pointer) followed by optional
        // memory-access operands (Aligned/MakePointerVisible/... plus a mask).
        // The pointer is ALWAYS operand 2; indexing from the end would pick up
        // the trailing memory-access mask instead.
        const uint32_t ptr = ins.word(2);
        auto p = ptr_.find(ptr);
        if (p == ptr_.end()) {
            PtrRef r = base_pointer(ptr);
            if (r.builtin.empty() && r.name.empty())
                throw Unsupported{unmodelled_ptr_msg("load", ptr)};
            ptr_[ptr] = r;
            p = ptr_.find(ptr);
        }
        vtype_[ins.result_id] = ins.type_id;
        if (!p->second.builtin.empty()) {
            // A builtin load has no memory behind it: it resolves to a value
            // already hoisted into the entry block.  A whole-vector load (no
            // component selected) is only meaningful for scalar builtins.
            const std::string c = p->second.comp.empty() ? "x" : p->second.comp;
            val_[ins.result_id] = builtin_value(p->second.builtin, c);
            return;
        }
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        body_ += "  " + r + " = load " + llvm_type(m_, ins.type_id) + ", " +
                 p->second.render() + "\n";
    }

    void emit_store(const Inst &ins) {
        const uint32_t ptr = ins.word(0);
        const uint32_t value = ins.word(1);
        auto p = ptr_.find(ptr);
        if (p == ptr_.end()) {
            PtrRef r = base_pointer(ptr);
            if (r.builtin.empty() && r.name.empty())
                throw Unsupported{unmodelled_ptr_msg("store", ptr)};
            ptr_[ptr] = r;
            p = ptr_.find(ptr);
        }
        if (!p->second.builtin.empty())
            throw Unsupported{"store to builtin " + p->second.builtin + "." +
                              p->second.comp};
        // The stored type comes from the destination pointer: OpStore is the one
        // place SPIR-V omits it.
        const std::string ty = p->second.elem;
        body_ += "  store " + value_text(value, ty.c_str()) + ", " +
                 p->second.render() + "\n";
    }

    void emit_binop(const Inst &ins, const char *op) {
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = " + op + " " + value_text(ins.word(2), ty.c_str()) +
                 ", " + bare_value(ins.word(3)) + "\n";
    }

    void emit_unop(const Inst &ins, const char *op) {
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = " + op + " " + ty + " " + bare_value(ins.word(2)) +
                 "\n";
    }

    // Plain `fdiv` cannot be lowered for PPU: the backend emits a call to the
    // device libcall __ppumathpriv_div_default_f32, for which the SDK ships no
    // implementation -- llc then hangs instead of erroring.  The native path is
    // llvm.ppu.fdiv.fast, a range-checked fast division.  fdiv.fast is float
    // only, so half division widens, divides, and narrows; the intermediate
    // is exact enough that the double rounding never changes an fp16 result.
    // Apply a SCALAR float intrinsic lane by lane to vector operands: this
    // backend has llvm.ppu.exp2/log2/sqrt/rcp/sin/cos/tanh and
    // llvm.ppu.fdiv.fast only for scalars ("Intrinsic has incorrect return
    // type!" on <2 x float>).  Extract, call, insert back.  `n_ops` operands of
    // `ty` go in, one value of `ty` comes out.
    std::string scalar_map(const std::string &ty, const std::vector<std::string> &ops,
                           const char *intr) {
        const size_t at = ty.find(" x ");
        const std::string n = at == std::string::npos ? "1" : ty.substr(1, at - 1);
        const unsigned lanes = static_cast<unsigned>(std::strtoul(n.c_str(), nullptr, 10));
        const std::string sty =
            at == std::string::npos ? ty : ty.substr(at + 3, ty.size() - at - 4);
        std::vector<std::string> lanes_v(lanes);
        for (unsigned l = 0; l < lanes; l++) {
            std::vector<std::string> a;
            for (const std::string &op : ops) {
                const std::string e = tmp();
                body_ += "  " + e + " = extractelement " + ty + " " + op +
                         ", i32 " + std::to_string(l) + "\n";
                a.push_back(e);
            }
            const std::string c = tmp();
            std::string s = "  " + c + " = call " + sty + " @" + intr + "(";
            for (size_t i = 0; i < a.size(); i++)
                s += (i ? ", " : "") + sty + " " + a[i];
            body_ += s + ")\n";
            lanes_v[l] = c;
        }
        std::string agg = "undef";
        for (unsigned l = 0; l < lanes; l++) {
            const std::string dst = l + 1 == lanes ? tmp() : tmp();
            body_ += "  " + dst + " = insertelement " + ty + " " + agg + ", " +
                     sty + " " + lanes_v[l] + ", i32 " + std::to_string(l) + "\n";
            agg = dst;
        }
        return agg;
    }

    static bool is_vector(const std::string &ty) {
        return !ty.empty() && ty[0] == '<';
    }

    // Reinterpret an operand as type `ty`.  LLVM binary ops require both
    // operands to spell the result type, so a scalar constant used against a
    // vector needs a splat.  `from` is the operand's own LLVM type ("" when it
    // is already `ty`).
    std::string coerce(const std::string &v, const std::string &from,
                       const std::string &ty) {
        if (!is_vector(ty) || from == ty || from.empty()) return v;
        const size_t at = ty.find(" x ");
        const std::string n = ty.substr(1, at - 1);
        const std::string sty = ty.substr(at + 3, ty.size() - at - 4);
        const std::string one = tmp();
        body_ += "  " + one + " = insertelement " + ty + " undef, " + sty + " " +
                 v + ", i32 0\n";
        const std::string s = tmp();
        body_ += "  " + s + " = shufflevector " + ty + " " + one + ", " + ty +
                 " undef, <" + n + " x i32> zeroinitializer\n";
        return s;
    }

    void emit_ppu_fdiv(const Inst &ins) {
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        const std::string a = bare_value(ins.word(2)), b = bare_value(ins.word(3));
        if (ty == "half") {
            const std::string wa = tmp(), wb = tmp(), d = tmp();
            body_ += "  " + wa + " = fpext half " + a + " to float\n";
            body_ += "  " + wb + " = fpext half " + b + " to float\n";
            body_ += "  " + d + " = call float @llvm.ppu.fdiv.fast(float " + wa +
                     ", float " + wb + ")\n";
            body_ += "  " + r + " = fptrunc float " + d + " to half\n";
            return;
        }
        if (is_vector(ty)) {
            const std::string v = scalar_map(ty, {a, b}, "llvm.ppu.fdiv.fast");
            body_ += "  " + r + " = bitcast " + ty + " " + v + " to " + ty + "\n";
            return;
        }
        body_ += "  " + r + " = call " + ty + " @llvm.ppu.fdiv.fast(" + ty + " " + a +
                 ", " + ty + " " + b + ")\n";
    }

    void emit_icmp(const Inst &ins, const char *pred) {
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = icmp " + pred + " " + type_of_value(ins.word(2)) +
                 " " + bare_value(ins.word(2)) + ", " + bare_value(ins.word(3)) + "\n";
    }

    void emit_fcmp(const Inst &ins, const char *pred) {
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = fcmp " + pred + " " + type_of_value(ins.word(2)) +
                 " " + bare_value(ins.word(2)) + ", " + bare_value(ins.word(3)) + "\n";
    }

    void emit_logical_not(const Inst &ins) {
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = xor i1 " + bare_value(ins.word(2)) +
                 ", true\n";
    }

    void emit_select(const Inst &ins) {
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        const std::string ty = llvm_type(m_, ins.type_id);
        const uint32_t cond = ins.word(2);
        const std::string cty = type_of_value(cond);
        // SPIR-V allows a vector of bools as the condition, meaning a
        // component-wise select.  LLVM's `select` takes either an i1 (scalar)
        // or a <N x i1> whose length matches the value vectors -- and those two
        // cases are structurally identical here once the types line up.
        const std::string c = bare_value(cond);
        body_ += "  " + r + " = select " + cty + " " + c + ", " + ty + " " +
                 bare_value(ins.word(3)) + ", " + ty + " " +
                 bare_value(ins.word(4)) + "\n";
    }

    void emit_convert(const Inst &ins, const char *op) {
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = " + op + " " + type_of_value(ins.word(2)) + " " +
                 bare_value(ins.word(2)) + " to " + llvm_type(m_, ins.type_id) + "\n";
    }

    // uint<->int at the same width: a no-op on LLVM's type system.
    void emit_bitcast(const Inst &ins) {
        const uint32_t src = ins.word(2);
        if (type_of_value(src) == llvm_type(m_, ins.type_id)) {
            val_[ins.result_id] = bare_value(src);
            vtype_[ins.result_id] = ins.type_id;
            return;
        }
        emit_convert(ins, "bitcast");
    }

    // Same-signedness width change: zext / sext / trunc by width comparison.
    void emit_width_convert(const Inst &ins) {
        const uint32_t src = ins.word(2);
        auto st = m_.types.find(vtype_.count(src) ? vtype_[src]
                                                   : const_type(m_, src));
        auto dt = m_.types.find(ins.type_id);
        const uint32_t sw = st == m_.types.end() ? 32 : st->second.width;
        const uint32_t dw = dt == m_.types.end() ? 32 : dt->second.width;
        const char *op = dw < sw ? "trunc" : (ins.opcode == SpvOpSConvert ? "sext"
                                                                         : "zext");
        emit_convert(ins, op);
    }

    void emit_fp_convert(const Inst& ins) {
        const uint32_t src = ins.word(2);
        auto st = m_.types.find(vtype_.count(src) ? vtype_[src]
                                                  : const_type(m_, src));
        auto dt = m_.types.find(ins.type_id);
        const uint32_t sw = st == m_.types.end() ? 32 : st->second.width;
        const uint32_t dw = dt == m_.types.end() ? 32 : dt->second.width;
        emit_convert(ins, dw < sw ? "fptrunc" : "fpext");
    }

    // OpCompositeExtract over a vector takes extractelement; over a struct or
    // array takes extractvalue (dotted path).
    void emit_composite_extract(const Inst &ins) {
        const uint32_t composite = ins.word(2);
        const std::string r = "%r" + std::to_string(ins.result_id);
        const std::string cty = type_of_value(composite);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        if (!cty.empty() && cty[0] == '<') {
            if (ins.argc() != 4) throw Unsupported{"OpCompositeExtract vector path"};
            body_ += "  " + r + " = extractelement " + cty + " " +
                     bare_value(composite) + ", i32 " + std::to_string(ins.word(3)) +
                     "\n";
            return;
        }
        std::string txt = "  " + r + " = extractvalue " + cty + " " +
                          bare_value(composite);
        for (uint16_t i = 3; i < ins.argc(); i++)
            txt += ", " + std::to_string(ins.word(i));
        body_ += txt + "\n";
    }

    // OpCompositeConstruct: vectors build with insertelement, aggregates with
    // insertvalue.  Both take ONE member per instruction, so a multi-member
    // construct becomes a chain; SPIR-V lists all constituents on one
    // instruction.  The insertelement index is a typed `i32 <k>` operand;
    // insertvalue's is a bare constant.
    void emit_composite_construct(const Inst &ins) {
        const std::string ty = llvm_type(m_, ins.type_id);
        const bool vec = !ty.empty() && ty[0] == '<';
        const char *op = vec ? "insertelement" : "insertvalue";
        const std::string res = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = res;
        vtype_[ins.result_id] = ins.type_id;
        std::string agg = "undef";
        for (uint16_t i = 2; i < ins.argc(); i++) {
            const bool last = (i + 1 == ins.argc());
            const std::string dst = last ? res : tmp();
            body_ += "  " + dst + " = " + op + " " + ty + " " + agg + ", " +
                     type_of_value(ins.word(i)) + " " + bare_value(ins.word(i)) +
                     ", " + (vec ? "i32 " : "") + std::to_string(i - 2) + "\n";
            agg = dst;
        }
        if (ins.argc() <= 2) body_ += "  " + res + " = insertvalue " + ty +
                                        " undef, i32 undef, 0\n";
    }

    // OpCompositeInsert: write ONE member of a composite, yielding a new
    // composite with the other members copied from the base. LLVM's
    // insertelement (vector) / insertvalue (struct, array) are the exact match.
    //
    // This is what `o.y = 0.0;` over a vec2 compiles to in the fp16 store tail
    // of the fused-elementwise kernel, so without it that whole kernel fails to
    // lower. A multi-index path (structs of structs) has no direct LLVM form and
    // nothing in the shader set needs one, so it stays unsupported rather than
    // shipping an untested rebuild chain.
    void emit_composite_insert(const Inst &ins) {
        const std::string ty = llvm_type(m_, ins.type_id);
        const bool vec = !ty.empty() && ty[0] == '<';
        const uint32_t obj = ins.word(2), comp = ins.word(3);
        if (ins.argc() != 5)
            throw Unsupported{"OpCompositeInsert with a multi-index path"};
        const std::string res = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = res;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + res + " = " + (vec ? "insertelement " : "insertvalue ") +
                 ty + " " + bare_value(comp) + ", " + type_of_value(obj) + " " +
                 bare_value(obj) + ", " + (vec ? "i32 " : "") +
                 std::to_string(ins.word(4)) + "\n";
    }

    // Extract a lane of a vector with a runtime index: SPIR-V semantics, but
    // LLVM IR's extractelement is the exact match.
    void emit_vector_extract_dynamic(const Inst &ins) {
        const uint32_t v = ins.word(2);
        const uint32_t idx = ins.word(3);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = extractelement " + type_of_value(v) + " " +
                 bare_value(v) + ", i32 " + bare_value(idx) + "\n";
    }

    // OpVectorShuffle: a compile-time lane permutation over two vectors.
    // `shufflevector` is the exact match.  SPIR-V components index v1 then v2
    // (a value >= |v1| selects v2's lane c-|v1|); LLVM's concatenated operand
    // numbering is the same, so the literals carry over unmodified.  0xFFFFFFFF
    // is "don't care" in SPIR-V and `undef` in LLVM, and a mask NARROWER than
    // the operands (e.g. reading only lanes 0..1 of a vec4) is the normal GLSL
    // "swizzle down" form, so the result type comes from the instruction, not
    // from the operands.
    void emit_vector_shuffle(const Inst &ins) {
        const std::string ty = llvm_type(m_, ins.type_id);
        const uint32_t v1 = ins.word(2), v2 = ins.word(3);
        const std::string op1 = type_of_value(v1), op2 = type_of_value(v2);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        // A vector and a scalar (1-wide) operand cannot share a shufflevector,
        // and SPIR-V forbids it too, but stay explicit rather than emit invalid
        // IR: widen nothing, just name both with their own types.
        const size_t at = ty.find(" x ");
        const std::string n = at == std::string::npos ? "1" : ty.substr(1, at - 1);
        std::string mask = "<" + n + " x i32> <";
        for (uint16_t i = 4; i < ins.argc(); i++) {
            const uint32_t c = ins.word(i);
            if (i > 4) mask += ", ";
            mask += (c == 0xFFFFFFFFu) ? "i32 undef" : "i32 " + std::to_string(c);
        }
        mask += ">";
        body_ += "  " + r + " = shufflevector " + op1 + " " + bare_value(v1) +
                 ", " + op2 + " " + bare_value(v2) + ", " + mask + "\n";
    }

    // GEP selector widths on this backend: i32 for struct members and vector
    // components, i64 for array elements.  The leading i64 0 dereferences the
    // base pointer (SPIR-V AccessChain has no such implicit step).
    void emit_access_chain(const Inst &ins) {
        const uint32_t base = ins.word(2);

        if (builtin_ids_.count(base)) {
            // The only access chain a builtin supports is an index into the
            // vector it represents (gl_GlobalInvocationID.x and friends).
            if (ins.argc() != 4)
                throw Unsupported{"access chain with multiple indices into a builtin"};
            const uint32_t idx = ins.word(3);
            static const char *kComp[4] = {"x", "y", "z", "x"};
            PtrRef r;
            r.builtin = builtin_name(builtin_ids_[base]);
            r.comp = kComp[std::min<uint32_t>(cval(idx), 3)];
            ptr_[ins.result_id] = r;
            ptype_[ins.result_id] = ins.type_id;
            return;
        }

        PtrRef bp = base_pointer(base);
        const uint32_t base_pty = ptype_of(base);
        const std::string base_ty = llvm_type(m_, type_pointee(m_, base_pty));

        uint32_t cur = type_pointee(m_, base_pty);
        std::string idx_s;
        for (uint16_t i = 3; i < ins.argc(); i++) {
            const uint32_t idx = ins.word(i);
            auto t = m_.types.find(cur);
            const SpvOp op = t == m_.types.end() ? SpvOpNop : t->second.op;
            if (op == SpvOpTypeStruct) {
                uint32_t m = cval(idx);
                idx_s += ", i32 " + std::to_string(m);
                cur = m < t->second.members.size() ? t->second.members[m] : 0;
            } else if (op == SpvOpTypeArray || op == SpvOpTypeRuntimeArray) {
                idx_s += ", i64 " + as_i64(idx);
                cur = t->second.elem;
            } else if (op == SpvOpTypeVector) {
                // A vector member is addressed either by a compile-time
                // component (the normal `.x`/`.y`) or by a RUNTIME lane index
                // (`v[i]`, and `p[lin & 1] = x` over a vec2). The dynamic case
                // must keep the index: `cval()` returns 0 for a non-constant,
                // so stringifying it pinned every dynamic lane access to
                // component 0 and silently dropped the other half of a vec2
                // store -- the fused-elementwise fp16 tail writes both halves
                // of a word through exactly this form. GEP with a runtime
                // vector index is the exact lowering LLVM/Clang uses.
                if (m_.consts.count(idx))
                    idx_s += ", i32 " + std::to_string(cval(idx));
                else
                    idx_s += ", i32 " + value_text(idx);
                cur = t->second.elem;
            } else {
                idx_s += ", i64 " + as_i64(idx);
            }
        }

        PtrRef r;
        r.elem = llvm_type(m_, cur);
        r.as = bp.as;
        r.name = tmp();
        ptr_[ins.result_id] = r;
        ptype_[ins.result_id] = ins.type_id;
        body_ += "  " + r.name + " = getelementptr " + base_ty + ", " + base_ty +
                 " addrspace(" + bp.as + ")* " + bp.name + ", i64 0" + idx_s + "\n";
    }

    uint32_t cval(uint32_t id) const {
        auto c = m_.consts.find(id);
        return c == m_.consts.end() ? 0 : static_cast<uint32_t>(c->second.ival);
    }

    void emit_call(const Inst &ins) {
        const uint32_t callee = ins.word(2);
        std::string args;
        for (uint16_t i = 3; i < ins.argc(); i++) {
            if (i > 3) args += ", ";
            // Call arguments always carry their type.
            args += typed_operand(ins.word(i));
        }
        const std::string ret = llvm_type(m_, ins.type_id);
        if (ret == "void") {
            body_ += "  call void @f" + std::to_string(callee) + "(" + args + ")\n";
            return;
        }
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = call " + ret + " @f" + std::to_string(callee) + "(" +
                 args + ")\n";
    }

    // OpVectorTimesScalar: a vector times a scalar.  SPIR-V allows the scalar
    // operand to be a vector as well (scalar is operand 3), so build a splat
    // with insertelement+shufflevector -- PPU lowers that to a cheap broadcast.
    void emit_vector_times_scalar(const Inst &ins) {
        const uint32_t vec = ins.word(2);
        const uint32_t scl = ins.word(3);
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;

        const std::string sty = type_of_value(scl);
        std::string operand = bare_value(scl);
        if (!ty.empty() && ty[0] == '<' && (sty.empty() || sty[0] != '<')) {
            // splat: insertelement undef, %s, 0 then shufflevector zeroinitializer
            const size_t at = ty.find(" x ");
            const std::string n = ty.substr(1, at - 1);
            const std::string one = tmp();
            body_ += "  " + one + " = insertelement " + ty + " undef, " +
                     type_of_value(scl) + " " + operand + ", i32 0\n";
            const std::string splat = tmp();
            body_ += "  " + splat + " = shufflevector " + ty + " " + one + ", " +
                     ty + " undef, <" + n + " x i32> zeroinitializer\n";
            operand = splat;
        }
        body_ += "  " + r + " = fmul " + ty + " " + bare_value(vec) + ", " +
                 operand + "\n";
    }

    // OpSNegate: integer negation has no SSA opcode; 0 - x is exact.
    void emit_binop_zero(const Inst &ins, const char *op) {
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = " + op + " " + ty + " 0, " +
                 bare_value(ins.word(2)) + "\n";
    }

    // OpBitFieldSExtract(base, offset, count) -> sign-extended bitfield.
    // llvm.ppu.bfe lowers to the hardware s.bfe.b32 instruction (verified: the
    // intrinsic selects, and the generic shift/and/shl alternative is longer).
    // `offset`/`count` are i32 in SPIR-V (the count may be a vector of ints);
    // a runtime offset is fine because bfe takes it in a register.
    void emit_bitfield_sextract(const Inst &ins) {
        const uint32_t base = ins.word(2), off = ins.word(3), cnt = ins.word(4);
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = call " + ty + " @llvm.ppu.bfe(" + ty + " " +
                 bare_value(base) + ", i32 " + bare_value(off) + ", i32 " +
                 bare_value(cnt) + ")\n";
    }

    // OpControlBarrier(execution_scope, memory_scope, semantics).
    //
    // PPU equivalent, established by compiling __syncthreads() with nvcc -keep
    // and reading the device IR: the SDK's own wrapper emits
    //   call void @llvm.ppu.sync.warp(i32 -1)
    //   call void @llvm.ppu.barrier.sync.cnt(i32 0, i32 0, i32 31)
    // barrier.sync.cnt's args are (barrier_id, ???, lane_mask): arg 0 is the
    // barrier index in the block, arg 1 is passed straight through to the
    // hardware's s.blksyn.defer first field, arg 2 is the participating-lane
    // mask and values above 31 cannot be selected (llc aborts).  `sync.warp`
    // is a warp-local convergence hint that must precede the block barrier.
    //
    // Visibility: s.blksyn is emitted only after llc has waited on outstanding
    // stores (vmem.fence.blk / s.wait vmem_fence_blk appear immediately before
    // it when a store precedes the barrier), and on-device the neighbour-shift
    // test proves a store in one thread is visible to another after the
    // barrier.  So a Workgroup-scope barrier is ordered for shared memory
    // without an extra fence; a non-Workgroup execution scope is not something
    // this backend models and is refused.
    void emit_control_barrier(const Inst &ins) {
        const uint32_t exec_scope = cval(ins.word(0));
        if (exec_scope != 2)  // Workgroup
            throw Unsupported{"ControlBarrier execution scope " +
                              std::to_string(exec_scope) +
                              " (only Workgroup=2 is mapped to blksyn)"};
        body_ += "  call void @llvm.ppu.sync.warp(i32 -1)\n";
        body_ += "  call void @llvm.ppu.barrier.sync.cnt(i32 0, i32 0, i32 31)\n";
    }

    // OpMemoryBarrier(scope, semantics) with no synchronising op.  Only the
    // Workgroup scope is meaningful for a shared-memory kernel; PPU has a real
    // block-scope fence that lowers to vmem.fence.blk.
    void emit_memory_barrier(const Inst &ins) {
        const uint32_t scope = cval(ins.word(0));
        if (scope == 2) {  // Workgroup
            body_ += "  fence syncscope(\"block\") seq_cst\n";
            return;
        }
        throw Unsupported{"MemoryBarrier scope " + std::to_string(scope)};
    }

    // Resolve an atomic pointer operand.  Only buffer/shared pointers exist.
    std::string atomic_ptr(uint32_t id) {
        auto p = ptr_.find(id);
        if (p == ptr_.end()) throw Unsupported{unmodelled_ptr_msg("atomic", id)};
        if (!p->second.builtin.empty())
            throw Unsupported{"atomic on builtin " + p->second.builtin};
        return p->second.render();
    }

    // OpAtomicIAdd & friends -> LLVM atomicrmw.  These lower to the hardware
    // reduction/cas instructions (v.red.add.u32, vmem.atm.cas.b32, ...);
    // the SDK's __ppukernel_atomicAdd_i32 libcall is NOT implemented on this
    // target (llc hangs with "Unimpled"), so atomicrmw is the only path.
    //
    // SPIR-V's result is the value *before* the operation, which is exactly
    // atomicrmw's return value.
    void emit_atomic_rmw(const Inst &ins) {
        // operands: ptr, scope, semantics, value
        const uint32_t ptr = ins.word(2), val = ins.word(5);
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        const char *op = nullptr;
        switch (ins.opcode) {
        case SpvOpAtomicIAdd: op = "add"; break;
        case SpvOpAtomicISub: op = "sub"; break;
        case SpvOpAtomicAnd: op = "and"; break;
        case SpvOpAtomicOr: op = "or"; break;
        case SpvOpAtomicXor: op = "xor"; break;
        case SpvOpAtomicUMax: op = "umax"; break;
        case SpvOpAtomicSMax: op = "max"; break;
        case SpvOpAtomicUMin: op = "umin"; break;
        case SpvOpAtomicSMin: op = "min"; break;
        case SpvOpAtomicExchange: op = "xchg"; break;
        default: throw Unsupported{spvOpcodeString(ins.opcode)};
        }
        body_ += "  " + r + " = atomicrmw " + op + " " + atomic_ptr(ptr) + ", " +
                 ty + " " + bare_value(val) + " seq_cst\n";
    }

    // OpAtomicCompareExchange: result = old; if (old == comparator) memory = value.
    // cmpxchg's aggregate result is {old, ok}; SPIR-V wants only `old`.
    void emit_atomic_cmpxchg(const Inst &ins) {
        // operands: ptr, scope, eq_sem, neq_sem, value, comparator
        const uint32_t ptr = ins.word(2), val = ins.word(6), cmp = ins.word(7);
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string pair = tmp(), r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + pair + " = cmpxchg " + atomic_ptr(ptr) + ", " + ty + " " +
                 bare_value(cmp) + ", " + ty + " " + bare_value(val) +
                 " seq_cst seq_cst\n";
        body_ += "  " + r + " = extractvalue { " + ty + ", i1 } " + pair + ", 0\n";
    }

    void emit_atomic_load(const Inst &ins) {
        const uint32_t ptr = ins.word(2);
        const std::string ty = llvm_type(m_, ins.type_id);
        const std::string r = "%r" + std::to_string(ins.result_id);
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
        body_ += "  " + r + " = load atomic " + ty + ", " + atomic_ptr(ptr) +
                 " seq_cst, align 4\n";
    }

    // --- subgroup (warp) reductions -------------------------------------
    //
    // OpGroupNonUniform{FAdd,FMax,FMin,...} with GroupOperation Reduce. The PPU
    // warp is 32 lanes and `llvm.ppu.shuffle.bfly.pred` is the butterfly (xor)
    // exchange: `v.shuffle.bfly.b32 dst, src, mask, 0x3e0`, where 0x3e0 is the
    // width/offset encoding the SDK's own __shfl_xor_sync32 builds for width
    // 32. Five xor steps (16, 8, 4, 2, 1) leave every lane holding the full
    // warp reduction, which is exactly the Vulkan Reduce contract.
    //
    // Validated on device: a 1024-thread dispatch summing per-warp lanes
    // reproduced the host reference exactly.
    //
    // Scope: only subgroup scope (3) is lowered. A workgroup-scope reduction
    // needs shared-memory staging plus barriers, and the vkop _sg shaders do
    // that themselves (subgroupAdd, then a per-subgroup shared fold), so
    // anything wider fails loud instead of silently reducing 32 lanes.
    static constexpr int kSubgroupWidth = 32;
    static constexpr int kShuffleWidth32 = 0x3e0;  // width-32 lane encoding

    // One butterfly step: returns an i32 SSA temp holding the partner lane's
    // bits. Callers bitcast around it -- PPU shuffles are bitwise.
    std::string shuffle_bfly(const std::string &bits, int offset) {
        const std::string pair = tmp();
        body_ += "  " + pair + " = call { i32, i1 } @llvm.ppu.shuffle.bfly.pred(i32 " +
                 bits + ", i32 " + std::to_string(offset) + ", i32 " +
                 std::to_string(kShuffleWidth32) + ")\n";
        const std::string got = tmp();
        body_ += "  " + got + " = extractvalue { i32, i1 } " + pair + ", 0\n";
        return got;
    }

    // Only GroupOperation Reduce within subgroup scope. Everything else throws
    // with the opcode name, so a coverage gap stays visible.
    void require_group_reduce(const Inst &ins) {
        auto c = m_.consts.find(ins.word(2));
        if (c == m_.consts.end() || !c->second.is_int || c->second.ival != 3) {
            throw Unsupported{
                std::string(spvOpcodeString(ins.opcode)) +
                " at a scope wider than a subgroup (only subgroup-scope Reduce "
                "is lowered; a workgroup reduction must be written in the shader)"};
        }
        if (ins.word(3) != 0 /* Reduce */) {
            throw Unsupported{std::string(spvOpcodeString(ins.opcode)) +
                              " with a GroupOperation other than Reduce"};
        }
    }

    // Scalar float reduction: bitcast to i32, butterfly-xor, bitcast back.
    // `intrinsic_base` is nullptr for a sum (plain fadd) and "maxnum"/"minnum"
    // for the ordered reductions.
    void emit_group_reduce_float(const Inst &ins, const char *intrinsic_base) {
        require_group_reduce(ins);
        const std::string ty = llvm_type(m_, ins.type_id);
        if (ty != "float") {
            throw Unsupported{std::string(spvOpcodeString(ins.opcode)) + " on " + ty +
                              " (only scalar float is lowered)"};
        }
        std::string acc = bare_value(ins.word(4));
        for (int off = kSubgroupWidth / 2; off >= 1; off >>= 1) {
            const std::string bits = tmp();
            body_ += "  " + bits + " = bitcast float " + acc + " to i32\n";
            const std::string other_bits = shuffle_bfly(bits, off);
            const std::string other = tmp();
            body_ += "  " + other + " = bitcast i32 " + other_bits + " to float\n";
            const std::string next = tmp();
            // maxnum/minnum are the intrinsic spellings the elementwise GLSL
            // lowerings already use, so a reduction and an elementwise op agree
            // bit for bit. Sum has no intrinsic -- plain fadd.
            if (intrinsic_base != nullptr) {
                body_ += "  " + next + " = call float @llvm." + intrinsic_base +
                         ".f32(float " + acc + ", float " + other + ")\n";
            } else {
                body_ += "  " + next + " = fadd float " + acc + ", " + other + "\n";
            }
            acc = next;
        }
        // The result id aliases the last accumulator directly: no copy is
        // needed, and every later use resolves through val_.
        val_[ins.result_id] = acc;
        vtype_[ins.result_id] = ins.type_id;
    }

    // Integer reductions. smax/smin/umax/umin have LLVM intrinsics; add/mul/
    // and/or/xor are plain instructions. Lanes ride in the low bits of the
    // 32-bit shuffle, so i16 is widened and narrowed around the tree.
    void emit_group_reduce_int(const Inst &ins, const char *op, bool intrinsic) {
        require_group_reduce(ins);
        const std::string ty = llvm_type(m_, ins.type_id);
        if (ty == "i64") throw Unsupported{"64-bit subgroup reduction (shuffle is 32-bit)"};
        if (ty != "i32" && ty != "i16") {
            throw Unsupported{std::string(spvOpcodeString(ins.opcode)) + " on " + ty};
        }
        std::string acc = bare_value(ins.word(4));
        std::string wide = acc;
        if (ty == "i16") {
            wide = tmp();
            body_ += "  " + wide + " = zext i16 " + acc + " to i32\n";
        }
        for (int off = kSubgroupWidth / 2; off >= 1; off >>= 1) {
            const std::string other = shuffle_bfly(wide, off);
            const std::string next = tmp();
            if (intrinsic) {
                body_ += "  " + next + " = call i32 @llvm." + op + ".i32(i32 " + wide +
                         ", i32 " + other + ")\n";
            } else {
                body_ += "  " + next + " = " + op + " i32 " + wide + ", " + other + "\n";
            }
            wide = next;
        }
        if (ty == "i16") {
            acc = tmp();
            body_ += "  " + acc + " = trunc i32 " + wide + " to i16\n";
        } else {
            acc = wide;
        }
        val_[ins.result_id] = acc;
        vtype_[ins.result_id] = ins.type_id;
    }

    // OpGroupNonUniformElect: true on exactly one lane of the subgroup. Lane 0
    // is the choice Vulkan permits the implementation to make.
    void emit_group_elect(const Inst &ins) {
        require_group_reduce(ins);
        const std::string lane = builtin_value("SubgroupLocalInvocationId", "x");
        const std::string r = "%r" + std::to_string(ins.result_id);
        body_ += "  " + r + " = icmp eq i32 " + lane + ", 0\n";
        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;
    }

    void emit_atomic_store(const Inst &ins) {
        // operands: ptr, scope, semantics, value
        const uint32_t ptr = ins.word(2), val = ins.word(5);
        const std::string ty = llvm_type(m_, const_type(m_, val));
        body_ += "  store atomic " + ty + " " + bare_value(val) + ", " +
                 atomic_ptr(ptr) + " seq_cst, align 4\n";
    }

    // --- GLSL.std.450 ---
    // LLVM IR has no nested expressions, so each step lands in its own %t temp.
    //
    // The fp16 build reaches a few of these with `half` / vector operands.
    // PPU has no half-precision transcendental intrinsic (llvm.ppu.exp2.half is
    // "Cannot select"), so half math is widened to float, computed, and
    // narrowed back.  That is exact for exp2/log2/tanh at fp16 precision.

    void emit_ext_inst(const Inst &ins) {
        const uint32_t ext = ins.word(3);
        const std::string r = "%r" + std::to_string(ins.result_id);
        const std::string ty = llvm_type(m_, ins.type_id);

        // The value type of operand 0 may differ from the result type for the
        // pack/unpack family (they take/produce vectors); use it for the
        // arithmetic lowerings below.
        const uint32_t a0id = ins.argc() > 4 ? ins.word(4) : 0;
        const std::string aty = a0id ? type_of_value(a0id) : ty;

        std::string args[3];
        for (uint16_t i = 4; i < ins.argc() && i < 7; i++)
            args[i - 4] = bare_value(ins.word(i));

        val_[ins.result_id] = r;
        vtype_[ins.result_id] = ins.type_id;

        const bool is_half = ty == "half" || aty == "half" ||
                             ty.find("half") != std::string::npos ||
                             aty.find("half") != std::string::npos;

        if (is_half && aty == ty) {
            emit_half_ext(ins, ext, ty, r, args);
            return;
        }

        // emit a call to `intr` over the operands, result in `res`
        auto step = [&](const std::string &res, const char *intr,
                        const std::vector<std::string> &ops) {
            // The llvm.ppu.* transcendentals are scalar-only; a vector operand
            // has to go lane by lane.
            if (is_vector(ty) && std::strncmp(intr, "llvm.ppu.", 9) == 0) {
                body_ += "  " + res + " = bitcast " + ty + " " +
                         scalar_map(ty, ops, intr) + " to " + ty + "\n";
                return;
            }
            std::string s = "  " + res + " = call " + ty + " @" +
                            intr_name(intr, ty) + "(";
            for (size_t i = 0; i < ops.size(); i++)
                s += (i ? ", " : "") + ty + " " + ops[i];
            body_ += s + ")\n";
        };
        // fmul of `v` (whose own type is `vty`) by a fp literal.  When the
        // result is a vector the literal has to be splatted: LLVM requires both
        // operands to spell the result type.
        auto fmul_lit = [&](const std::string &res, const std::string &v,
                            const std::string &vty, const char *lit) {
            const std::string a = coerce(v, vty, ty);
            const std::string b = coerce(lit, "scalar", ty);
            body_ += "  " + res + " = fmul " + ty + " " + a + ", " + b + "\n";
        };

        switch (ext) {
        case 1:  step(r, "llvm.round.f32", {args[0]}); return;  // Round (half away from zero)
        case 2:  step(r, "llvm.rint.f32", {args[0]}); return;   // RoundEven
        case 4:  step(r, "llvm.fabs.f32", {args[0]}); return;   // FAbs
        case 8:  step(r, "llvm.floor.f32", {args[0]}); return;  // Floor
        case 9:  step(r, "llvm.ceil.f32", {args[0]}); return;   // Ceil
        case 13: {                                              // Sin
            // The PPU sine/cosine take the angle in QUARTER TURNS (units of
            // pi/2), not radians -- measured on device: llvm.ppu.sin(1) is
            // 1.0, llvm.ppu.sin(2) is 0.0, i.e. sin(pi/2 * x). A literal
            // argument hides this because llc folds it with the host sinf,
            // which is why only runtime values (attention rotary tables)
            // exposed it. Scale by 2/pi before the call.
            const std::string t = tmp();
            fmul_lit(t, args[0], aty, "6.36619746685028076e-01");  // 2/pi
            step(r, "llvm.ppu.sin", {t});
            return;
        }
        case 14: {                                              // Cos
            const std::string t = tmp();
            fmul_lit(t, args[0], aty, "6.36619746685028076e-01");  // 2/pi
            step(r, "llvm.ppu.cos", {t});
            return;
        }
        case 21: step(r, "llvm.ppu.tanh", {args[0]}); return;   // Tanh
        case 25: {                                              // Atan2
            // atan2(y,x) = atan(y/x) with quadrant fixups; atan has no PPU
            // native, so refuse rather than approximate silently.
            throw Unsupported{"OpExtInst GLSL.std.450 #25 (Atan2)"};
        }
        case 27: {                                              // Exp (natural)
            // llvm.exp.f32 has no libcall legalisation on PPU; exp2 is native.
            // exp(x) = exp2(x * log2(e)) -- the scaling goes INSIDE exp2, not
            // outside it.  Verified on-device: the outside form gives 8*log2e
            // where expf(3)=20.09.
            const std::string t = tmp();
            fmul_lit(t, args[0], aty, "1.44269502162933350e+00");
            step(r, "llvm.ppu.exp2", {t});
            return;
        }
        case 28: {                                              // Log (natural)
            const std::string t = tmp();
            step(t, "llvm.ppu.log2", {args[0]});
            fmul_lit(r, t, ty, "6.93147182464599609e-01");
            return;
        }
        case 29: step(r, "llvm.ppu.exp2", {args[0]}); return;   // Exp2
        case 30: step(r, "llvm.ppu.log2", {args[0]}); return;   // Log2
        case 31: step(r, "llvm.ppu.sqrt", {args[0]}); return;   // Sqrt
        case 32: {                                              // InverseSqrt
            const std::string t = tmp();
            step(t, "llvm.ppu.sqrt", {args[0]});
            step(r, "llvm.ppu.rcp", {t});
            return;
        }
        case 37: step(r, "llvm.minnum.f32", {args[0], args[1]}); return;  // FMin
        case 40: step(r, "llvm.maxnum.f32", {args[0], args[1]}); return;  // FMax
        // Integer min/max.  llvm.{s,u}{min,max}.iN are plain generic
        // intrinsics and select fine on PPU (llvm.ppu.max.* is the bf16-only
        // hardware op); widths come from the result type.
        case 38: step(r, ("llvm.umin." + ty).c_str(), {args[0], args[1]}); return;  // UMin
        case 39: step(r, ("llvm.smin." + ty).c_str(), {args[0], args[1]}); return;  // SMin
        case 41: step(r, ("llvm.umax." + ty).c_str(), {args[0], args[1]}); return;  // UMax
        case 42: step(r, ("llvm.smax." + ty).c_str(), {args[0], args[1]}); return;  // SMax
        case 44:                                                     // UClamp
        case 45: {                                                   // SClamp
            // Clamp has no intrinsic: GLSL defines it as min(max(x,lo),hi),
            // and that composition selects on PPU for every integer width.
            const char *lo = ext == 45 ? "llvm.smax." : "llvm.umax.";
            const char *hi = ext == 45 ? "llvm.smin." : "llvm.umin.";
            const std::string t = tmp();
            step(t, (std::string(lo) + ty).c_str(), {args[0], args[1]});
            step(r, (std::string(hi) + ty).c_str(), {t, args[2]});
            return;
        }
        case 6: {                                               // FSign
            // No native sign op: 0 compare + 2 selects, exactly GLSL's
            // definition (sign(0)=0, sign(NaN)=0).
            const std::string gt = tmp(), lt = tmp(), s1 = tmp();
            body_ += "  " + gt + " = fcmp ogt " + ty + " " + args[0] +
                     ", 0.00000000000000000e+00\n";
            body_ += "  " + lt + " = fcmp olt " + ty + " " + args[0] +
                     ", 0.00000000000000000e+00\n";
            body_ += "  " + s1 + " = select i1 " + gt + ", " + ty +
                     " 1.00000000000000000e+00, " + ty +
                     " 0.00000000000000000e+00\n";
            body_ += "  " + r + " = select i1 " + lt + ", " + ty +
                     " -1.00000000000000000e+00, " + ty + " " + s1 + "\n";
            return;
        }
        case 26: {                                              // Pow
            // No native pow; x^y = exp2(y * log2(x)).  This is what the
            // hardware rough-pow path does and matches exp2/log2's accuracy.
            const std::string l = tmp(), m = tmp();
            step(l, "llvm.ppu.log2", {args[0]});
            body_ += "  " + m + " = fmul " + ty + " " + args[1] + ", " + l + "\n";
            step(r, "llvm.ppu.exp2", {m});
            return;
        }
        case 43: {                                              // FClamp
            const std::string t = tmp();
            step(t, "llvm.maxnum.f32", {args[0], args[1]});
            step(r, "llvm.minnum.f32", {t, args[2]});
            return;
        }
        case 46: {                                              // FMix = a*(1-c)+b*c
            const std::string t = tmp();
            body_ += "  " + t + " = fsub " + ty + " " + args[1] + ", " + args[0] +
                     "\n";
            step(r, "llvm.fmuladd.f32", {t, args[2], args[0]});
            return;
        }
        case 50: step(r, "llvm.fmuladd.f32", {args[0], args[1], args[2]}); return;
        case 58: {                                              // PackHalf2x16
            // float2 -> uint: narrow to half2, then reinterpret the 32 bits.
            const std::string h = "<2 x half>";
            const std::string t = tmp();
            body_ += "  " + t + " = fptrunc " + aty + " " + args[0] + " to " + h +
                     "\n";
            body_ += "  " + r + " = bitcast " + h + " " + t + " to " + ty + "\n";
            return;
        }
        case 62: {                                              // UnpackHalf2x16
            // uint -> float2: reinterpret as half2, then widen each lane.
            const std::string h = "<2 x half>";
            const std::string t = tmp();
            body_ += "  " + t + " = bitcast " + aty + " " + args[0] + " to " + h +
                     "\n";
            body_ += "  " + r + " = fpext " + h + " " + t + " to " + ty + "\n";
            return;
        }
        default: break;
        }
        const char *nm = glsl_ext_name(ext);
        throw Unsupported{"OpExtInst GLSL.std.450 #" + std::to_string(ext) + " (" +
                          (nm ? nm : "unknown") + ")"};
    }

    // "half" -> "float"; "<N x half>" -> "<N x float>".
    static std::string f32_of(const std::string &t) {
        std::string s = t;
        size_t p = s.find("half");
        if (p != std::string::npos) s.replace(p, 4, "float");
        return s;
    }

    // Intrinsic names carry their signature suffix: llvm.fabs.f32 does not
    // accept a <2 x float> operand ("Intrinsic has incorrect argument type!").
    // The vector forms exist for the generic float intrinsics and select on
    // PPU, so rewrite the suffix to match the value type.  The llvm.ppu.* ones
    // (exp2/log2/sqrt/rcp/sin/cos/tanh) are scalar-only -- llvm.ppu.exp2.v2f32
    // fails ISel -- so those keep their bare name and the caller must have
    // already been handed a scalar.
    static std::string intr_name(const std::string &intr, const std::string &ty) {
        const bool vec = !ty.empty() && ty[0] == '<';
        if (!vec) return intr;
        const size_t p = intr.rfind(".f32");
        if (p == std::string::npos || p + 4 != intr.size()) return intr;
        if (intr.compare(0, 9, "llvm.ppu.") == 0) return intr;
        const size_t at = ty.find(" x ");
        const std::string n = at == std::string::npos ? "" : ty.substr(1, at - 1);
        if (n.empty()) return intr;
        return intr.substr(0, p) + ".v" + n + "f32";
    }

    // Half-precision GLSL.std.450 math: PPU has no f16 transcendental
    // (llvm.ppu.exp2.half selects to nothing), so widen to float, run the f32
    // lowering, and narrow back.  Exact for these ops at fp16 precision.
    void emit_half_ext(const Inst &ins, uint32_t ext, const std::string &ty,
                       const std::string &r, std::string (&args)[3]) {
        const std::string f = f32_of(ty);
        std::string w[3];
        for (int i = 0; i < 3; i++) {
            if (args[i].empty()) continue;
            w[i] = tmp();
            body_ += "  " + w[i] + " = fpext " + ty + " " + args[i] + " to " + f +
                     "\n";
        }
        auto fcall = [&](const std::string &res, const char *intr,
                         std::vector<std::string> ops) {
            std::string s = "  " + res + " = call " + f + " @" +
                            intr_name(intr, f) + "(";
            for (size_t i = 0; i < ops.size(); i++)
                s += (i ? ", " : "") + f + " " + ops[i];
            body_ += s + ")\n";
        };
        const std::string t = tmp();
        switch (ext) {
        case 4:  fcall(t, "llvm.fabs.f32", {w[0]}); break;   // FAbs
        case 8:  fcall(t, "llvm.floor.f32", {w[0]}); break;  // Floor
        case 9:  fcall(t, "llvm.ceil.f32", {w[0]}); break;   // Ceil
        case 13: {                                           // Sin
            // Quarter-turn argument, as in the scalar path: sin(pi/2 * x).
            const std::string s = tmp();
            body_ += "  " + s + " = fmul " + f + " " + w[0] +
                     ", 6.36619746685028076e-01\n";
            fcall(t, "llvm.ppu.sin", {s});
            break;
        }
        case 14: {                                           // Cos
            const std::string s = tmp();
            body_ += "  " + s + " = fmul " + f + " " + w[0] +
                     ", 6.36619746685028076e-01\n";
            fcall(t, "llvm.ppu.cos", {s});
            break;
        }
        case 21: fcall(t, "llvm.ppu.tanh", {w[0]}); break;   // Tanh
        case 27: {                                           // Exp
            const std::string s = tmp();
            body_ += "  " + s + " = fmul " + f + " " + w[0] +
                     ", 1.44269502162933350e+00\n";
            fcall(t, "llvm.ppu.exp2", {s});
            break;
        }
        case 28: {                                           // Log
            const std::string s = tmp();
            fcall(s, "llvm.ppu.log2", {w[0]});
            body_ += "  " + t + " = fmul " + f + " " + s +
                     ", 6.93147182464599609e-01\n";
            break;
        }
        case 29: fcall(t, "llvm.ppu.exp2", {w[0]}); break;   // Exp2
        case 30: fcall(t, "llvm.ppu.log2", {w[0]}); break;   // Log2
        case 31: fcall(t, "llvm.ppu.sqrt", {w[0]}); break;   // Sqrt
        case 37: fcall(t, "llvm.minnum.f32", {w[0], w[1]}); break;  // FMin
        case 40: fcall(t, "llvm.maxnum.f32", {w[0], w[1]}); break;  // FMax
        case 26: {                                           // Pow
            const std::string s = tmp();
            fcall(s, "llvm.ppu.log2", {w[0]});
            const std::string m = tmp();
            body_ += "  " + m + " = fmul " + f + " " + w[1] + ", " + s + "\n";
            fcall(t, "llvm.ppu.exp2", {m});
            break;
        }
        case 6: {                                            // FSign
            const std::string gt = tmp(), lt = tmp(), s1 = tmp();
            body_ += "  " + gt + " = fcmp ogt " + f + " " + w[0] +
                     ", 0.00000000000000000e+00\n";
            body_ += "  " + lt + " = fcmp olt " + f + " " + w[0] +
                     ", 0.00000000000000000e+00\n";
            body_ += "  " + s1 + " = select i1 " + gt + ", " + f +
                     " 1.00000000000000000e+00, " + f +
                     " 0.00000000000000000e+00\n";
            body_ += "  " + t + " = select i1 " + lt + ", " + f +
                     " -1.00000000000000000e+00, " + f + " " + s1 + "\n";
            break;
        }
        case 43: {                                           // FClamp
            const std::string s = tmp();
            fcall(s, "llvm.maxnum.f32", {w[0], w[1]});
            fcall(t, "llvm.minnum.f32", {s, w[2]});
            break;
        }
        case 46: {                                           // FMix
            const std::string s = tmp();
            body_ += "  " + s + " = fsub " + f + " " + w[1] + ", " + w[0] + "\n";
            fcall(t, "llvm.fmuladd.f32", {s, w[2], w[0]});
            break;
        }
        case 50: fcall(t, "llvm.fmuladd.f32", {w[0], w[1], w[2]}); break;
        default: {
            const char *nm = glsl_ext_name(ext);
            throw Unsupported{"OpExtInst GLSL.std.450 #" + std::to_string(ext) +
                              " (" + (nm ? nm : "unknown") + ") on half"};
        }
        }
        body_ += "  " + r + " = fptrunc " + f + " " + t + " to " + ty + "\n";
    }

    // --- control flow ---

    void emit_branch(const Inst &ins) {
        body_ += "  br label %L" + std::to_string(ins.word(0)) + "\n";
        terminated_ = true;
    }

    void emit_branch_conditional(const Inst &ins) {
        body_ += "  br i1 " + bare_value(ins.word(0)) + ", label %L" +
                 std::to_string(ins.word(1)) + ", label %L" +
                 std::to_string(ins.word(2)) + "\n";
        terminated_ = true;
    }

    void emit_switch(const Inst &ins) {
        const std::string ty = type_of_value(ins.word(0));
        std::string s = "  switch " + ty + " " + bare_value(ins.word(0)) +
                        ", label %L" + std::to_string(ins.word(1)) + " [\n";
        for (uint16_t i = 2; i + 1 < ins.argc(); i += 2)
            s += "    i32 " + std::to_string(ins.word(i)) + ", label %L" +
                 std::to_string(ins.word(i + 1)) + "\n";
        s += "  ]\n";
        body_ += s;
        terminated_ = true;
    }
};

const char *storage_name(uint32_t sc) {
    switch (sc) {
    case SpvStorageClassUniformConstant: return "UniformConstant";
    case SpvStorageClassUniform: return "Uniform";
    case SpvStorageClassStorageBuffer: return "StorageBuffer";
    case SpvStorageClassInput: return "Input";
    case SpvStorageClassFunction: return "Function";
    case SpvStorageClassPrivate: return "Private";
    case SpvStorageClassWorkgroup: return "Workgroup";
    case SpvStorageClassPushConstant: return "PushConstant";
    default: return "Other";
    }
}

std::string emit_type_defs(const Module &m) {
    std::string s;
    for (const auto &kv : m.types) {
        if (kv.second.op != SpvOpTypeStruct) continue;
        s += "%s" + std::to_string(kv.first) + " = type { ";
        for (size_t i = 0; i < kv.second.members.size(); i++) {
            if (i) s += ", ";
            s += llvm_type(m, kv.second.members[i]);
        }
        s += " }\n";
    }
    if (!s.empty()) s += "\n";
    return s;
}

const char *kDeclares =
    "declare i32 @llvm.ppu.read.ptx.sreg.tid.x() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.tid.y() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.tid.z() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.ctaid.x() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.ctaid.y() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.ctaid.z() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.ntid.x() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.ntid.y() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.ntid.z() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.nctaid.x() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.nctaid.y() nounwind readnone\n"
    "declare i32 @llvm.ppu.read.ptx.sreg.nctaid.z() nounwind readnone\n"
    "declare void @llvm.ppu.sync.warp(i32) convergent nounwind\n"
    "declare { i32, i1 } @llvm.ppu.shuffle.bfly.pred(i32, i32, i32) convergent nounwind\n"
    "declare void @llvm.ppu.barrier.sync.cnt(i32, i32, i32) convergent nounwind\n"
    "declare i32 @llvm.ppu.bfe(i32, i32, i32) nounwind readnone\n"
    "declare float @llvm.rint.f32(float) nounwind readnone\n"
    "declare <2 x float> @llvm.round.v2f32(<2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.round.v3f32(<3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.round.v4f32(<4 x float>) nounwind readnone\n"
    "declare float @llvm.round.f32(float) nounwind readnone\n"
    "declare <2 x float> @llvm.rint.v2f32(<2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.rint.v3f32(<3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.rint.v4f32(<4 x float>) nounwind readnone\n"
    "declare float @llvm.maxnum.f32(float, float) nounwind readnone\n"
    "declare float @llvm.minnum.f32(float, float) nounwind readnone\n"
    "declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.maxnum.v3f32(<3 x float>, <3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) nounwind readnone\n"
    "declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.minnum.v3f32(<3 x float>, <3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>) nounwind readnone\n"
    "declare <2 x float> @llvm.fabs.v2f32(<2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.fabs.v3f32(<3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.fabs.v4f32(<4 x float>) nounwind readnone\n"
    "declare <2 x float> @llvm.floor.v2f32(<2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.floor.v3f32(<3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.floor.v4f32(<4 x float>) nounwind readnone\n"
    "declare <2 x float> @llvm.ceil.v2f32(<2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.ceil.v3f32(<3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.ceil.v4f32(<4 x float>) nounwind readnone\n"
    "declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone\n"
    "declare <3 x float> @llvm.fmuladd.v3f32(<3 x float>, <3 x float>, <3 x float>) nounwind readnone\n"
    "declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone\n"
    "declare float @llvm.copysign.f32(float, float) nounwind readnone\n"
    "declare i32 @llvm.smax.i32(i32, i32) nounwind readnone\n"
    "declare i16 @llvm.smax.i16(i16, i16) nounwind readnone\n"
    "declare i64 @llvm.smax.i64(i64, i64) nounwind readnone\n"
    "declare i32 @llvm.smin.i32(i32, i32) nounwind readnone\n"
    "declare i16 @llvm.smin.i16(i16, i16) nounwind readnone\n"
    "declare i64 @llvm.smin.i64(i64, i64) nounwind readnone\n"
    "declare i32 @llvm.umax.i32(i32, i32) nounwind readnone\n"
    "declare i16 @llvm.umax.i16(i16, i16) nounwind readnone\n"
    "declare i64 @llvm.umax.i64(i64, i64) nounwind readnone\n"
    "declare i32 @llvm.umin.i32(i32, i32) nounwind readnone\n"
    "declare i16 @llvm.umin.i16(i16, i16) nounwind readnone\n"
    "declare i64 @llvm.umin.i64(i64, i64) nounwind readnone\n"
    "declare float @llvm.fabs.f32(float) nounwind readnone\n"
    "declare float @llvm.floor.f32(float) nounwind readnone\n"
    "declare float @llvm.ceil.f32(float) nounwind readnone\n"
    "declare float @llvm.fmuladd.f32(float, float, float) nounwind readnone\n"
    "declare float @llvm.ppu.exp2(float) nounwind readnone\n"
    "declare float @llvm.ppu.log2(float) nounwind readnone\n"
    "declare float @llvm.ppu.tanh(float) nounwind readnone\n"
    "declare float @llvm.ppu.sin(float) nounwind readnone\n"
    "declare float @llvm.ppu.cos(float) nounwind readnone\n"
    "declare float @llvm.ppu.sqrt(float) nounwind readnone\n"
    "declare float @llvm.ppu.rcp(float) nounwind readnone\n"
    "declare float @llvm.ppu.fdiv.fast(float, float) nounwind readnone\n\n";

}  // namespace

namespace {

// Result of a folded OpSpecConstantOp: an ordinary constant of the same type.
void set_int_const(Module &m, const Inst &ins, int64_t v) {
    Const k;
    k.type_id = ins.type_id;
    k.is_int = true;
    k.ival = v;
    const std::string ty = llvm_type(m, ins.type_id);
    k.set(ty, std::to_string(v));
    m.consts[ins.result_id] = k;
}

void set_bool_const(Module &m, const Inst &ins, bool v) {
    Const k;
    k.type_id = ins.type_id;
    k.is_int = true;
    k.ival = v ? -1 : 0;
    k.set("i1", v ? "true" : "false");
    m.consts[ins.result_id] = k;
}

bool spec_fold_fail(const Module &m, const Inst &ins, std::string &error) {
    error = "OpSpecConstantOp %" + std::to_string(ins.result_id) +
            " has a non-constant operand (type " + llvm_type(m, ins.type_id) + ")";
    return false;
}

// Fold OpSpecConstantOp to its DEFAULT-value result.
//
// The ICD never reads pSpecializationInfo, so every specialization constant in
// a pipeline keeps the default value the SPIR-V author wrote. Folding those
// expressions here therefore reproduces the geometry the application asked for
// *and* keeps the emitter from having to model spec constants at all: after this
// pass the instructions are ordinary constants.
//
// The fused-elementwise kernel uses them for its whole controlled program
// (every `SC_*` word is a spec constant, and the branch guards are
// `SLessThan`/`SGreaterThanEqual`/`LogicalAnd` over them), so without this
// pass the kernel does not lower at all and the artifact that embeds it fails
// to create a pipeline.
//
// Only the operators those shaders actually use are implemented; anything else
// reports its name rather than quietly producing a wrong constant.
bool fold_spec_constant_ops(Module &m, const std::vector<Inst> &spec_ops,
                            std::string &error) {
    for (const Inst &ins : spec_ops) {
        // Operands: type, result id, opcode literal, then the operands of
        // the equivalent ordinary instruction.
        const uint32_t opcode = ins.word(2);
        const auto operand_i64 = [&](uint16_t i, int64_t &out) -> bool {
            const uint32_t id = ins.word(i);
            auto c = m.consts.find(id);
            if (c == m.consts.end()) return false;
            out = c->second.ival;
            return true;
        };
        int64_t a = 0, b = 0;
        switch (opcode) {
        case SpvOpIAdd:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_int_const(m, ins, a + b);
            break;
        case SpvOpISub:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_int_const(m, ins, a - b);
            break;
        case SpvOpIMul:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_int_const(m, ins, a * b);
            break;
        case SpvOpIEqual:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a == b);
            break;
        case SpvOpINotEqual:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a != b);
            break;
        case SpvOpSLessThan:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a < b);
            break;
        case SpvOpSLessThanEqual:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a <= b);
            break;
        case SpvOpSGreaterThan:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a > b);
            break;
        case SpvOpSGreaterThanEqual:
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a >= b);
            break;
        case SpvOpLogicalAnd: {
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a != 0 && b != 0);
            break;
        }
        case SpvOpLogicalOr: {
            if (ins.argc() < 5 || !operand_i64(3, a) || !operand_i64(4, b))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a != 0 || b != 0);
            break;
        }
        case SpvOpLogicalNot:
            if (ins.argc() < 4 || !operand_i64(3, a))
                return spec_fold_fail(m, ins, error);
            set_bool_const(m, ins, a == 0);
            break;
        case SpvOpSelect: {
            if (ins.argc() < 6 || !operand_i64(3, a))
                return spec_fold_fail(m, ins, error);
            const uint32_t take = ins.word(a != 0 ? 4 : 5);
            auto c = m.consts.find(take);
            if (c == m.consts.end()) return spec_fold_fail(m, ins, error);
            m.consts[ins.result_id] = c->second;
            break;
        }
        case SpvOpSNegate:
            if (ins.argc() < 4 || !operand_i64(3, a))
                return spec_fold_fail(m, ins, error);
            set_int_const(m, ins, -a);
            break;
        default:
            error = "OpSpecConstantOp opcode " + std::to_string(opcode) +
                    " is not folded (the ICD uses default specialization "
                    "values; add it to fold_spec_constant_ops)";
            return false;
        }
    }
    return true;
}

}  // namespace

namespace {

// Parse and validate once. Returns Status and fills `m` on success.
Status collect(const uint32_t *words, size_t word_count, Module &m, std::string &error) {
    if (words == nullptr || word_count < 5) {
        error = "module is empty or truncated";
        return Status::InvalidModule;
    }
    spv_context ctx = spvContextCreate(SPV_ENV_VULKAN_1_1);
    spv_diagnostic diag = nullptr;
    if (spvValidateBinary(ctx, words, word_count, &diag) != SPV_SUCCESS) {
        error = std::string("validate: ") +
                (diag && diag->error ? diag->error : "unknown");
        spvDiagnosticDestroy(diag);
        spvContextDestroy(ctx);
        return Status::InvalidModule;
    }
    spvDiagnosticDestroy(diag);

    Collector col;
    if (spvBinaryParse(ctx, &col, words, word_count, on_header, on_inst, nullptr) !=
        SPV_SUCCESS) {
        spvContextDestroy(ctx);
        error = "SPIR-V parse failed";
        return Status::InvalidModule;
    }
    spvContextDestroy(ctx);
    std::vector<Inst> spec_ops = std::move(col.spec_ops);
    m = std::move(col.m);
    if (!fold_spec_constant_ops(m, spec_ops, error)) return Status::Unsupported;
    return Status::Ok;
}

// Kernel argument order: descriptor bindings ascending by (set, binding), push
// constant last. This is exactly the order emit_signature() emits.
std::vector<uint32_t> kernel_buffers(const Module &m) {
    std::vector<uint32_t> bufs;
    for (const auto &kv : m.var_storage)
        if (kv.second == SpvStorageClassStorageBuffer ||
            kv.second == SpvStorageClassUniform ||
            kv.second == SpvStorageClassPushConstant)
            bufs.push_back(kv.first);
    const auto key = [&](uint32_t id) {
        auto s = m.var_set.find(id);
        auto b = m.var_binding.find(id);
        const uint32_t set = s == m.var_set.end() ? 0u : s->second;
        const uint32_t bind = b == m.var_binding.end() ? 0xffffu : b->second;
        return (static_cast<uint64_t>(set) << 32) | bind;
    };
    std::sort(bufs.begin(), bufs.end(),
              [&](uint32_t a, uint32_t b) { return key(a) < key(b); });
    return bufs;
}

}  // namespace

const char *storage_class_name(uint32_t storage_class) {
    return storage_name(storage_class);
}

Status emit_llvm_ir(const uint32_t *spirv, size_t word_count, const char *entry_name,
                    std::string &out_ir, std::string &error) {
    KernelInfo info;
    return emit_llvm_ir(spirv, word_count, entry_name, out_ir, &info, error);
}

Status emit_llvm_ir(const uint32_t *spirv, size_t word_count, const char *entry_name,
                    std::string &out_ir, KernelInfo *out_info, std::string &error) {
    out_ir.clear();
    Module m;
    Status st = collect(spirv, word_count, m, error);
    if (st != Status::Ok) return st;

    if (m.entry_names.empty()) {
        error = "module declares no OpEntryPoint";
        return Status::InvalidModule;
    }

    // Resolve which entry point to emit. A Vulkan application names one with
    // pName; an empty request means "the module's first entry point".
    uint32_t chosen = 0;
    std::string chosen_name;
    if (entry_name != nullptr && *entry_name != '\0') {
        auto it = m.entry_names.find(entry_name);
        if (it == m.entry_names.end()) {
            error = "no entry point named '" + std::string(entry_name) + "' (module has:";
            for (const auto &kv : m.entry_names) error += " '" + kv.first + "'";
            error += ")";
            return Status::InvalidModule;
        }
        chosen = it->second;
        chosen_name = it->first;
    } else {
        chosen = m.entry_fn;
        for (const auto &kv : m.entry_names)
            if (kv.second == chosen) chosen_name = kv.first;
    }

    // Every other OpEntryPoint must not be emitted as a kernel: the PPU backend
    // takes one entry per module. They stay ordinary internal functions.
    // Every other OpEntryPoint was emitted by the loop below as an ordinary
    // function (Emitter's entry_ flag is true only for `chosen`), because the
    // PPU backend takes exactly one kernel per module.
    m.entry_fn = chosen;

    const Function *entry = nullptr;
    for (const Function &fn : m.funcs)
        if (fn.id == chosen) entry = &fn;
    if (entry == nullptr) {
        error = "entry point '" + chosen_name + "' names a function the module does not define";
        return Status::InvalidModule;
    }

    std::vector<uint32_t> bufs = kernel_buffers(m);

    try {
        std::string out = emit_type_defs(m);
        out += kDeclares;
        for (uint32_t id : bufs) {
            const std::string t =
                llvm_type(m, type_pointee(m, var_ptr_type(m, id)));
            out += "@g" + std::to_string(id) + " = internal global " + t +
                   " addrspace(1)* null\n";
        }
        // Workgroup variables are real shared memory: one module-level
        // addrspace(3) global per SPIR-V Workgroup OpVariable, shared by every
        // invocation in the workgroup.  (Validated on device: a barrier makes
        // one thread's store visible to its neighbours.)
        for (const auto &kv : m.var_storage) {
            if (kv.second != SpvStorageClassWorkgroup) continue;
            const std::string t =
                llvm_type(m, type_pointee(m, var_ptr_type(m, kv.first)));
            out += "@w" + std::to_string(kv.first) +
                   " = internal addrspace(3) global " + t + " undef\n";
        }
        out += "\n";
        for (const Function &fn : m.funcs) {
            const bool is_kernel = fn.id == chosen;
            Emitter em(m, fn, is_kernel, is_kernel ? bufs : std::vector<uint32_t>{},
                       bufs);
            out += em.run();
            out += "\n";
        }
        out_ir = std::move(out);
    } catch (const Unsupported &u) {
        error = "unsupported: " + u.what;
        return Status::Unsupported;
    } catch (const std::exception &e) {
        error = std::string("emitter failed: ") + e.what();
        return Status::Unsupported;
    }

    if (out_info != nullptr) {
        KernelInfo info;
        info.entry_name = chosen_name;
        auto ls = m.local_sizes.find(chosen);
        if (ls != m.local_sizes.end()) {
            info.local_size[0] = ls->second[0];
            info.local_size[1] = ls->second[1];
            info.local_size[2] = ls->second[2];
        } else if (m.local_x || m.local_y || m.local_z) {
            info.local_size[0] = m.local_x;
            info.local_size[1] = m.local_y;
            info.local_size[2] = m.local_z;
        }
        for (uint32_t id : bufs) {
            KernelArg a;
            a.storage_class = m.var_storage[id];
            a.push_constant = a.storage_class == SpvStorageClassPushConstant;
            auto s = m.var_set.find(id);
            auto b = m.var_binding.find(id);
            a.set = s == m.var_set.end() ? 0u : s->second;
            a.binding = b == m.var_binding.end() ? 0u : b->second;
            info.args.push_back(a);
        }
        *out_info = std::move(info);
    }
    return Status::Ok;
}

Status emit_report(const uint32_t *spirv, size_t word_count, std::string &out_report,
                   std::string &error) {
    Module m;
    Status st = collect(spirv, word_count, m, error);
    if (st != Status::Ok) return st;
    char line[256];
    std::snprintf(line, sizeof line, "; entry=%u local=(%u,%u,%u)\n", m.entry_fn,
                  m.local_x, m.local_y, m.local_z);
    std::string r = line;
    for (const auto &kv : m.entry_names) {
        std::snprintf(line, sizeof line, "; entry name '%s' -> %%%u\n",
                      kv.first.c_str(), kv.second);
        r += line;
    }
    for (const auto &kv : m.var_storage) {
        std::snprintf(line, sizeof line, "; global var %u (%s)\n", kv.first,
                      storage_name(kv.second));
        r += line;
    }
    for (const Function &fn : m.funcs) {
        std::snprintf(line, sizeof line, "; function %u: %zu params, %zu body instructions\n",
                      fn.id, fn.params.size(), fn.insts.size());
        r += line;
        std::map<std::string, int> hist;
        for (const Inst &ins : fn.insts) hist[spvOpcodeString(ins.opcode)]++;
        for (const auto &kv : hist) {
            std::snprintf(line, sizeof line, ";   %-30s %d\n", kv.first.c_str(),
                          kv.second);
            r += line;
        }
    }
    out_report = std::move(r);
    return Status::Ok;
}

}  // namespace ppu_emit

