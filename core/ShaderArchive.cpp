// Copyright 2025 @junka
#include "core/ShaderArchive.hpp"

#include <cstring>

namespace vkop {
namespace core {

namespace {
constexpr uint32_t kHeaderSize = 16;
constexpr uint32_t kEntrySize = 72;
constexpr uint32_t kNameMax = 48;
constexpr uint32_t kVersion = 1;

bool has_spirv_magic(const uint8_t *p, uint32_t size) {
    // SPIR-V modules start with magic number 0x07230203, written
    // little-endian as bytes 03 02 23 07.
    return size >= 4 && p[0] == 0x03 && p[1] == 0x02 && p[2] == 0x23 &&
           p[3] == 0x07;
}
} // namespace

std::unique_ptr<ShaderArchive> ShaderArchive::open(const std::string &path) {
    auto arch = std::unique_ptr<ShaderArchive>(new ShaderArchive());
    if (!arch->mapping_.map_file(path)) {
        return nullptr;
    }
    const uint8_t *base = static_cast<const uint8_t *>(arch->mapping_.data);
    const size_t total = arch->mapping_.size;
    if (total < kHeaderSize) {
        return nullptr;
    }
    if (std::memcmp(base, "VKSP", 4) != 0) {
        return nullptr;
    }
    uint32_t version = 0, count = 0;
    std::memcpy(&version, base + 4, 4);
    std::memcpy(&count, base + 8, 4);
    if (version != kVersion) {
        return nullptr;
    }
    // Reject any file whose entry table would run past the mapped region —
    // a corrupt count would otherwise index off the end during the loop.
    if (static_cast<uint64_t>(count) * kEntrySize > total - kHeaderSize) {
        return nullptr;
    }

    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *row = base + kHeaderSize + i * kEntrySize;
        const char *name_raw = reinterpret_cast<const char *>(row);
        size_t nlen = strnlen(name_raw, kNameMax);
        uint64_t off = 0;
        uint32_t size = 0;
        std::memcpy(&off, row + 48, 8);
        std::memcpy(&size, row + 56, 4);
        // Bounds-check against the mapping before trusting the pointer.
        if (off > total || size > total - off) {
            continue;
        }
        if (!has_spirv_magic(base + off, size)) {
            continue;
        }
        arch->index_.emplace(std::string(name_raw, nlen), Entry{off, size});
    }
    return arch;
}

std::pair<const uint32_t *, uint32_t>
ShaderArchive::lookup(const std::string &key) const {
    auto it = index_.find(key);
    if (it == index_.end()) {
        return {nullptr, 0};
    }
    const uint8_t *base = static_cast<const uint8_t *>(mapping_.data);
    // Blob offsets are 4-byte aligned by the packer, so this cast yields a
    // well-formed uint32_t SPIR-V word stream.
    return {reinterpret_cast<const uint32_t *>(base + it->second.offset),
            it->second.size};
}

} // namespace core
} // namespace vkop
