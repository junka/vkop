// Copyright 2025 @junka
#ifndef CORE_SHADER_ARCHIVE_HPP_
#define CORE_SHADER_ARCHIVE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/load.hpp"

namespace vkop {
namespace core {

// Loads a VKSP archive: either the whole vkop_shaders.bin the CMake build
// produces (tools/spv_to_archive.py), or the per-model subset embedded in a
// .vkopbin (onnx2vkop writes one by default). The archive mirrors the
// folder_name convention CMakeLists.txt already uses for its embedded
// `buffer_<op>_spv` / `image_<op>_spv` C-array symbols, so a runtime lookup
// key is exactly the string an op header would have referenced through its
// `extern unsigned char X_spv[]` block.
//
// The file form is mapped through load::FileMapping, so only the pages an
// actually-loaded shader touches fault into RSS. A model that fires 26
// shaders out of the ~95 in libvkop sees ~26 blob ranges resident, not the
// whole file. The embedded form is a bounded copy out of the vkopbin mmap
// (the mapping is released when LoadModel returns) and is small by
// construction: it holds only the keys that graph can dispatch.
//
// Failure mode: `open()` / `open_from_memory()` return nullptr on any
// structural problem (missing file, bad magic, version mismatch, corrupt
// index). Runtime treats that as "this model has no shaders": the op keeps
// the placeholder array it was built with, and Operator turns that into a
// named error instead of a null pipeline.
class ShaderArchive {
  public:
    static std::unique_ptr<ShaderArchive> open(const std::string &path);

    // Open an archive that lives inside another file's mapping -- the
    // `shader_archive` blob a .vkopbin carries (Runtime::LoadModel). `data`
    // must point at a VKSP header and `size` covers the whole archive; the
    // bytes are copied into an owned buffer, so the caller's mapping (the
    // transient load::VkModel) may be released immediately after. Returns
    // nullptr on a structural problem, exactly like open().
    static std::unique_ptr<ShaderArchive> open_from_memory(const uint8_t *data,
                                                           size_t size);

    // Install a process-wide archive for ops built outside
    // Runtime::LoadModel. Runtime hands its own handle to every op it builds,
    // but the unit-test fixtures, the probes and bench_matmul construct
    // operators straight through ops::create_from_type(), so they never pass
    // through that hook: under VKOP_STRIP_BUILTIN_SHADERS=ON (the default)
    // they would have no SPIR-V at all and die on the stub array. Operator
    // falls back to this when its own pointer is null.
    //
    // Dev-only, and deliberately not driven by an environment variable: this
    // is the build tree's own archive, installed by a header the shipping
    // binaries do not include (include/dev_shaders.hpp). A runtime override
    // would let a profiled run quietly execute kernels other than the ones
    // the model carries, which is exactly what the single-file design is
    // meant to rule out.
    static void set_process_default(const std::string &path);

    // The archive installed above, or nullptr.
    static const ShaderArchive *process_default();

    // Returns (nullptr, 0) on miss. On hit, the returned pointer is a stable
    // address inside the mapping and remains valid for the archive's
    // lifetime. `size` is in bytes and is always 4-byte aligned so a
    // reinterpret_cast<const uint32_t *> is a well-formed SPIR-V module.
    std::pair<const uint32_t *, uint32_t> lookup(const std::string &key) const;

    // Reports how many shader entries the archive contains. Used by
    // VKOP_SHADER_DEBUG diagnostics; not on any hot path.
    size_t entry_count() const { return index_.size(); }

  private:
    ShaderArchive() = default;
    struct Entry {
        uint64_t offset;
        uint32_t size;
    };

    // Parse a VKSP archive at `base`. Shared by both open paths so the
    // mmap'd and the copied-into-memory case validate identically. Returns
    // false (leaving index_ empty) on any structural problem.
    bool parse(const uint8_t *base, size_t total);

    // Only one of these is populated: open() maps a file, open_from_memory()
    // owns a copy of bytes that live in someone else's mapping. base_/size_
    // are what parse() and lookup() actually read.
    load::FileMapping mapping_;
    std::vector<uint8_t> owned_;
    const uint8_t *base_ = nullptr;
    size_t size_ = 0;
    std::unordered_map<std::string, Entry> index_;
};

} // namespace core
} // namespace vkop

#endif // CORE_SHADER_ARCHIVE_HPP_
