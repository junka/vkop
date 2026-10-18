// Copyright 2025 @junka
#ifndef CORE_SHADER_ARCHIVE_HPP_
#define CORE_SHADER_ARCHIVE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "model/load.hpp"

namespace vkop {
namespace core {

// Loads the vkop_shaders.bin archive produced by tools/spv_to_archive.py at
// build time. The archive mirrors the folder_name convention CMakeLists.txt
// already uses for its embedded `buffer_<op>_spv` / `image_<op>_spv` C-array
// symbols, so a runtime lookup key is exactly the string an op header would
// have referenced through its `extern unsigned char X_spv[]` block.
//
// The archive is mapped through load::FileMapping, so only the pages an
// actually-loaded shader touches fault into RSS. A model that fires 26
// shaders out of the ~95 in libvkop sees ~26 blob ranges resident, not the
// whole file.
//
// Failure mode: `open()` returns nullptr on any structural problem (missing
// file, bad magic, version mismatch, corrupt index). Callers (Runtime) treat
// a null archive as "use the embedded SPIR-V symbols", so a deployment that
// forgot the .bin still runs, just without the memory saving.
class ShaderArchive {
  public:
    static std::unique_ptr<ShaderArchive> open(const std::string &path);

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
    load::FileMapping mapping_;
    std::unordered_map<std::string, Entry> index_;
};

} // namespace core
} // namespace vkop

#endif // CORE_SHADER_ARCHIVE_HPP_
