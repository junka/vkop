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
// index). Callers (Runtime) treat that as "use the embedded SPIR-V symbols
// of the next source in line", so a deployment that forgot the .bin still
// runs, just without the memory saving.
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

    // The archive named by VKOP_SHADER_ARCHIVE, opened once per process and
    // owned here. Runtime::LoadModel hands its own handle to every op it
    // builds, but ops created straight through ops::create_from_type() — the
    // unit-test fixtures and the example drivers — never pass through that
    // hook, so under VKOP_STRIP_BUILTIN_SHADERS=ON they could not reach any
    // SPIR-V at all and died on the stub array even with a valid archive in
    // the environment. Operator falls back to this when its own pointer is
    // null. Returns nullptr when the env is unset or the file is unusable,
    // which leaves the existing loud throw in place.
    static const ShaderArchive *process_default();

    // Returns (nullptr, 0) on miss. On hit, the returned pointer is a stable
    // address inside the mapping and remains valid for the archive's
    // lifetime. `size` is in bytes and is always 4-byte aligned so a
    // reinterpret_cast<const uint32_t *> is a well-formed SPIR-V module.
    std::pair<const uint32_t *, uint32_t> lookup(const std::string &key) const;

    // Reports how many shader entries the archive contains. Used by
    // VKOP_SHADER_ARCHIVE_DEBUG diagnostics; not on any hot path.
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
