// SPIR-V -> LLVM 13 (text, typed pointers) emitter for the PPU backend.
//
// This is the library form of the standalone `emit` tool. It is the same code:
// the tool is a thin main() over these functions, so the two cannot drift.
//
// The emitter is deliberately fail-loud. Anything it cannot lower makes the
// call return Status::Unsupported with a message naming the instruction (and,
// where one exists, the SPIR-V opcode name). It never emits partial IR and it
// never substitutes a plausible-looking wrong lowering.
#ifndef PPU_EMIT_HPP
#define PPU_EMIT_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ppu_emit {

// One kernel parameter, in the order it appears in the emitted function
// signature. Descriptor-backed parameters carry the (set, binding) they came
// from so the caller can map kernel arguments back to Vulkan descriptors; the
// push-constant block is passed as a trailing pointer parameter and has no
// binding.
struct KernelArg {
  uint32_t set = 0;
  uint32_t binding = 0;
  uint32_t storage_class = 0;  // SpvStorageClass* of the source OpVariable
  bool push_constant = false;
};

struct KernelInfo {
  std::string entry_name;               // as requested, and as emitted
  std::vector<KernelArg> args;          // kernel signature order
  uint32_t local_size[3] = {1, 1, 1};   // OpExecutionMode LocalSize
};

enum class Status {
  Ok,
  InvalidModule,  // failed validation/parse, or no such entry point
  Unsupported,    // a valid module using something the emitter cannot lower
};

// Emit LLVM IR for `entry_name`. When `entry_name` is null or empty, the
// module's first entry point is used and its own name is emitted.
//
// On failure `error` explains what and where; `out_ir` and `out_info` are
// unspecified but never partially usable.
Status emit_llvm_ir(const uint32_t *spirv, size_t word_count, const char *entry_name,
                    std::string &out_ir, KernelInfo *out_info, std::string &error);

// Convenience overload without the kernel-shape metadata.
Status emit_llvm_ir(const uint32_t *spirv, size_t word_count, const char *entry_name,
                    std::string &out_ir, std::string &error);

// The `--report` dump the standalone tool prints: entry point, local size,
// module variables and an opcode histogram per function.
Status emit_report(const uint32_t *spirv, size_t word_count, std::string &out_report,
                   std::string &error);

// Human-readable name for an SpvStorageClass, for diagnostics.
const char *storage_class_name(uint32_t storage_class);

}  // namespace ppu_emit

#endif  // PPU_EMIT_HPP
