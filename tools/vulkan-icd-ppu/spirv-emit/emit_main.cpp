// Standalone front end for the SPIR-V -> LLVM IR emitter.
//
// Kept separate from emit.cpp so the emitter can also be linked into the
// Vulkan ICD as a library (tools/vulkan-icd-ppu/src/spirv_compile.cpp). Both
// use exactly the same code; this file only does argument handling and the
// process exit codes the coverage sweeps rely on:
//
//   0  emitted
//   1  invalid module (failed validation, bad file, no such entry point)
//   2  usage error
//   3  unsupported instruction (the error names it; the emitter never emits
//      partial or approximate IR)
//
// Build: g++ -std=c++17 -O1 emit.cpp emit_main.cpp -o emit -lSPIRV-Tools
#include "emit.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool read_file(const char *path, std::vector<uint32_t> &out) {
    std::FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 4 != 0) {
        std::fclose(f);
        return false;
    }
    out.resize(static_cast<size_t>(bytes) / 4);
    const size_t got = std::fread(out.data(), 4, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

int status_exit(ppu_emit::Status st) {
    switch (st) {
        case ppu_emit::Status::Ok: return 0;
        case ppu_emit::Status::InvalidModule: return 1;
        case ppu_emit::Status::Unsupported: return 3;
    }
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: emit <file.spv> [--report] [--entry <name>]\n");
        return 2;
    }
    std::vector<uint32_t> words;
    if (!read_file(argv[1], words)) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }

    bool report = false;
    const char *entry = nullptr;
    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--report") == 0) {
            report = true;
        } else if (std::strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
            entry = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }

    std::string error;
    if (report) {
        std::string text;
        ppu_emit::Status st = ppu_emit::emit_report(words.data(), words.size(), text, error);
        if (st != ppu_emit::Status::Ok) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return status_exit(st);
        }
        std::fputs(text.c_str(), stdout);
        return 0;
    }

    std::string ir;
    ppu_emit::Status st = ppu_emit::emit_llvm_ir(words.data(), words.size(), entry, ir, error);
    if (st != ppu_emit::Status::Ok) {
        // The message already says "unsupported: ..." when that is why.
        std::fprintf(stderr, "%s\n", error.c_str());
        return status_exit(st);
    }
    std::fputs(ir.c_str(), stdout);
    return 0;
}
