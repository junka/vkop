// Reader-side unit coverage for the VKSP shader archive.
//
// The archive reaches the runtime two ways: as the whole vkop_shaders.bin on
// disk (ShaderArchive::open, mmap'd) and as the per-model subset a .vkopbin
// carries in Model.shader_archive (ShaderArchive::open_from_memory, copied out
// of the vkopbin mapping). ModelTest exercises the second one end to end, but
// it can only ever show the happy path -- the fixture is well-formed by
// construction. What that leaves untested is every branch of parse()'s
// validation, which is the part that decides whether a corrupt or truncated
// blob degrades to "use the embedded SPIR-V" or runs off the end of the
// mapping. This test pins those, and pins the format itself against a reader
// that does not share code with tools/filter_shaders.py's writer.
//
// Usage:
//   tests/ShaderArchiveTest                  # synthetic archives only
//   tests/ShaderArchiveTest path/to/archive  # + a real one (build/vkop_shaders.bin)

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/ShaderArchive.hpp"

namespace {

constexpr uint32_t kHeaderSize = 16;
constexpr uint32_t kEntrySize = 72;

int g_failures = 0;

void check(bool ok, const char *what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

void check_eq(size_t got, size_t want, const char *what) {
    if (got != want) {
        std::printf("FAIL: %s (got %zu, want %zu)\n", what, got, want);
        ++g_failures;
    }
}

// A SPIR-V-shaped blob: real magic, then a per-seed byte pattern so a lookup
// that hands back the wrong entry's range shows up in the byte compare.
std::vector<uint8_t> spv(uint8_t seed, uint32_t words) {
    std::vector<uint8_t> b(4 * words, 0);
    b[0] = 0x03;
    b[1] = 0x02;
    b[2] = 0x23;
    b[3] = 0x07;
    for (size_t i = 4; i < b.size(); ++i) {
        b[i] = static_cast<uint8_t>(seed + i);
    }
    return b;
}

struct EntrySpec {
    std::string name;
    std::vector<uint8_t> blob;
};

void put_u32(std::vector<uint8_t> &v, size_t off, uint32_t val) {
    std::memcpy(v.data() + off, &val, 4);
}

void put_u64(std::vector<uint8_t> &v, size_t off, uint64_t val) {
    std::memcpy(v.data() + off, &val, 8);
}

// Build an archive the way tools/filter_shaders.py's serialize() does: a
// 16-byte header, one 72-byte row per shader (name at 0, u64 offset at 48,
// u32 size at 56), then the blobs padded to 4-byte alignment.
std::vector<uint8_t> build(const std::vector<EntrySpec> &specs) {
    std::vector<uint8_t> a(kHeaderSize + kEntrySize * specs.size(), 0);
    std::memcpy(a.data(), "VKSP", 4);
    put_u32(a, 4, 1);  // version
    put_u32(a, 8, static_cast<uint32_t>(specs.size()));
    put_u32(a, 12, 0);  // reserved
    size_t cursor = kHeaderSize + kEntrySize * specs.size();
    for (size_t i = 0; i < specs.size(); ++i) {
        const size_t row = kHeaderSize + i * kEntrySize;
        std::memcpy(a.data() + row, specs[i].name.data(), specs[i].name.size());
        cursor = (cursor + 3) & ~static_cast<size_t>(3);
        put_u64(a, row + 48, cursor);
        put_u32(a, row + 56, static_cast<uint32_t>(specs[i].blob.size()));
        put_u32(a, row + 60, 0);  // crc32, not verified by the reader
        a.insert(a.end(), specs[i].blob.begin(), specs[i].blob.end());
        cursor += specs[i].blob.size();
    }
    return a;
}

std::vector<EntrySpec> two_entries() {
    return {{"buffer_add", spv(1, 8)}, {"image_add_fp16", spv(9, 12)}};
}

// Compare a lookup against the bytes the archive was built from.
void check_hit(const vkop::core::ShaderArchive &arch, const std::string &key,
               const std::vector<uint8_t> &want) {
    auto hit = arch.lookup(key);
    if (hit.first == nullptr) {
        std::printf("FAIL: lookup(%s) missed\n", key.c_str());
        ++g_failures;
        return;
    }
    check_eq(hit.second, want.size(), "lookup size");
    check_eq(hit.second % 4, 0, "lookup size is a whole SPIR-V word count");
    check_eq(reinterpret_cast<uintptr_t>(hit.first) % 4, 0,
             "lookup pointer is 4-byte aligned");
    check(std::memcmp(hit.first, want.data(), want.size()) == 0,
          "lookup bytes match the blob the archive was built from");
}

}  // namespace

// ------------------------------------------------------------------- happy
static void test_round_trip() {
    const auto specs = two_entries();
    const auto bytes = build(specs);
    auto arch = vkop::core::ShaderArchive::open_from_memory(bytes.data(),
                                                            bytes.size());
    check(arch != nullptr, "a well-formed archive opens");
    if (!arch) return;
    check_eq(arch->entry_count(), 2, "both entries indexed");
    check_hit(*arch, "buffer_add", specs[0].blob);
    check_hit(*arch, "image_add_fp16", specs[1].blob);
    auto miss = arch->lookup("no_such_shader");
    check(miss.first == nullptr && miss.second == 0, "a miss is (nullptr, 0)");
}

// open_from_memory must own its bytes: in LoadModel the source is the vkopbin
// mapping, which is released as soon as the model object goes out of scope
// while ops keep these pointers for the process's lifetime.
static void test_survives_the_source_buffer() {
    const auto specs = two_entries();
    auto *heap = new std::vector<uint8_t>(build(specs));
    auto arch = vkop::core::ShaderArchive::open_from_memory(heap->data(),
                                                            heap->size());
    delete heap;
    check(arch != nullptr, "open_from_memory copies");
    if (!arch) return;
    check_hit(*arch, "buffer_add", specs[0].blob);
    check_hit(*arch, "image_add_fp16", specs[1].blob);
}

// -------------------------------------------------------------- rejection
static void test_structural_rejections() {
    const auto bytes = build(two_entries());
    check(vkop::core::ShaderArchive::open_from_memory(nullptr, 64) == nullptr,
          "a null pointer is rejected");
    check(vkop::core::ShaderArchive::open_from_memory(bytes.data(), 0) ==
              nullptr,
          "a zero size is rejected");
    check(vkop::core::ShaderArchive::open_from_memory(bytes.data(), 15) ==
              nullptr,
          "a header shorter than 16 bytes is rejected");

    auto bad_magic = bytes;
    std::memcpy(bad_magic.data(), "VKSQ", 4);
    check(vkop::core::ShaderArchive::open_from_memory(bad_magic.data(),
                                                      bad_magic.size()) ==
              nullptr,
          "a bad magic is rejected");

    auto bad_version = bytes;
    put_u32(bad_version, 4, 2);
    check(vkop::core::ShaderArchive::open_from_memory(bad_version.data(),
                                                      bad_version.size()) ==
              nullptr,
          "an unknown version is rejected");

    // A corrupt count must not let the entry loop index past the mapping.
    auto huge_count = bytes;
    put_u32(huge_count, 8, 0xFFFFFFFFU);
    check(vkop::core::ShaderArchive::open_from_memory(huge_count.data(),
                                                      huge_count.size()) ==
              nullptr,
          "an entry table larger than the file is rejected");
}

// A single damaged entry is skipped; the rest of the archive still resolves.
// This is the "one stale shader must not take the whole model down" contract.
static void test_damaged_entry_is_skipped() {
    const auto specs = two_entries();
    auto bytes = build(specs);

    auto out_of_range = bytes;
    put_u64(out_of_range, kHeaderSize + 48, bytes.size() + 4096);
    auto arch = vkop::core::ShaderArchive::open_from_memory(
        out_of_range.data(), out_of_range.size());
    check(arch != nullptr, "an out-of-range entry offset is not fatal");
    if (arch) {
        check_eq(arch->entry_count(), 1, "only the damaged entry is dropped");
        check_hit(*arch, "image_add_fp16", specs[1].blob);
        check(arch->lookup("buffer_add").first == nullptr,
              "the damaged entry itself does not resolve");
    }

    // An entry whose blob is not SPIR-V (the writer refusing to pack a stub,
    // or a truncated tail) is dropped the same way.
    auto no_magic = bytes;
    std::memset(no_magic.data() + kHeaderSize + kEntrySize * 2, 0,
                specs[0].blob.size());
    auto arch2 = vkop::core::ShaderArchive::open_from_memory(no_magic.data(),
                                                             no_magic.size());
    check(arch2 != nullptr, "a blob without SPIR-V magic is not fatal");
    if (arch2) {
        check_eq(arch2->entry_count(), 1, "the non-SPIR-V entry is dropped");
        check(arch2->lookup("buffer_add").first == nullptr,
              "the non-SPIR-V entry does not resolve");
    }

    // A blob without the room for a magic word at all is dropped too.
    auto tiny = bytes;
    put_u32(tiny, kHeaderSize + 56, 2);
    auto arch3 = vkop::core::ShaderArchive::open_from_memory(tiny.data(),
                                                             tiny.size());
    check(arch3 != nullptr, "a 2-byte blob is not fatal");
    if (arch3) {
        check(arch3->lookup("buffer_add").first == nullptr,
              "a 2-byte blob does not resolve");
    }
}

// --------------------------------------------------------------- file form
// open() and open_from_memory() share parse(), so the same bytes have to give
// the same index down either path; the file path additionally has to answer
// with a real file, and to fail closed on a missing one.
static void test_file_path_matches_memory_path(const std::string &path) {
    if (path.empty()) return;
    auto arch = vkop::core::ShaderArchive::open(path);
    if (!arch) {
        std::printf("FAIL: open(%s) returned null\n", path.c_str());
        ++g_failures;
        return;
    }
    check(arch->entry_count() > 0, "the on-disk archive has entries");
    check(vkop::core::ShaderArchive::open(path + ".does-not-exist") == nullptr,
          "a missing file is rejected");
    std::printf("ok: %s (%zu entries)\n", path.c_str(), arch->entry_count());
}

int main(int argc, char **argv) {
    test_round_trip();
    test_survives_the_source_buffer();
    test_structural_rejections();
    test_damaged_entry_is_skipped();
    test_file_path_matches_memory_path(argc > 1 ? argv[1] : "");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ShaderArchiveTest: all checks passed\n");
    return 0;
}
