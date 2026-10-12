// junka @ 2026
// get_rope_index C++ verification harness.
//
// Loads input_ids / mm_token_type_ids / attention_mask / grid_thw dumped by
// dump_rope_ref.py, runs the C++ get_rope_index port, and diffs the resulting
// (3, B, L) int64 position_ids byte-for-byte against the HF reference
// pos_ids.bin. Also compares rope_delta.
//
// Usage:
//   rope_probe <rope_ref_dir>     # reads <dir>/{input_ids,mtt,am,pos_ids}.bin + meta.txt
//
// Exit 0 = exact match, 1 = load error, 2 = value mismatch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <map>

#include "image_preproc.hpp"

using vkop::export_::get_rope_index;

static std::vector<uint8_t> load_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v(sz);
    f.read(reinterpret_cast<char *>(v.data()), sz);
    return v;
}

// Parse "key=value" lines from meta.txt.
static std::map<std::string, std::string> parse_meta(const std::string &dir) {
    std::map<std::string, std::string> m;
    std::ifstream f(dir + "/meta.txt");
    if (!f) return m;
    std::string line;
    while (std::getline(f, line)) {
        auto p = line.find('=');
        if (p == std::string::npos) continue;
        m[line.substr(0, p)] = line.substr(p + 1);
    }
    return m;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <rope_ref_dir>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];

    auto meta = parse_meta(dir);
    int L = std::atoi(meta["L"].c_str());
    int B = std::atoi(meta["B"].c_str());
    int n_img = std::atoi(meta["n_img"].c_str());
    if (L <= 0 || B <= 0) {
        std::fprintf(stderr, "bad meta L=%d B=%d\n", L, B);
        return 1;
    }
    // grid_thw_<i>=T,H,W
    std::vector<int> grid_thw;
    for (int i = 0; i < n_img; ++i) {
        std::string key = "grid_thw_" + std::to_string(i);
        auto it = meta.find(key);
        if (it == meta.end()) {
            std::fprintf(stderr, "missing %s\n", key.c_str());
            return 1;
        }
        std::stringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ',')) grid_thw.push_back(std::atoi(tok.c_str()));
    }

    auto ids_raw = load_file(dir + "/input_ids.bin");
    auto mtt_raw = load_file(dir + "/mtt.bin");
    auto am_raw  = load_file(dir + "/am.bin");
    auto ref_raw = load_file(dir + "/pos_ids.bin");
    if (ids_raw.empty() || mtt_raw.empty() || am_raw.empty() || ref_raw.empty()) {
        std::fprintf(stderr, "missing input bins in %s\n", dir.c_str());
        return 1;
    }
    const int64_t *input_ids = reinterpret_cast<const int64_t *>(ids_raw.data());
    const int32_t *mtt = reinterpret_cast<const int32_t *>(mtt_raw.data());
    const int8_t  *am  = reinterpret_cast<const int8_t *>(am_raw.data());
    const int64_t *ref = reinterpret_cast<const int64_t *>(ref_raw.data());

    auto res = get_rope_index(input_ids, mtt, am, grid_thw.data(), n_img, B, L);

    // Diff pos_ids.
    int n = 3 * B * L;
    int mism = 0;
    int64_t max_abs = 0;
    int first_mism = -1;
    for (int i = 0; i < n; ++i) {
        if (res.pos_ids[i] != ref[i]) {
            if (mism == 0) first_mism = i;
            ++mism;
            int64_t d = res.pos_ids[i] - ref[i];
            if (d < 0) d = -d;
            if (d > max_abs) max_abs = d;
        }
    }
    // Diff rope_delta (parse from meta: rope_delta=[[x]] or [[x, y, ...]]).
    std::vector<int64_t> ref_delta;
    {
        std::string s = meta["rope_delta"];
        // strip everything but digits, '-', ','.
        std::string clean;
        for (char c : s) if (std::isdigit((unsigned char)c) || c == '-' || c == ',')
            clean += c;
        std::stringstream ss(clean);
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) ref_delta.push_back(std::atoll(tok.c_str()));
    }

    std::printf("L=%d B=%d n_img=%d grid_thw=[", L, B, n_img);
    for (size_t i = 0; i < grid_thw.size(); ++i) {
        std::printf("%d%s", grid_thw[i], (i + 1) % 3 == 0 ? " " : ",");
    }
    std::printf("]\n");
    std::printf("pos_ids: %d elems, mismatches=%d (max_abs=%lld)\n",
                n, mism, (long long)max_abs);
    if (first_mism >= 0) {
        int axis = first_mism / (B * L);
        int rem = first_mism % (B * L);
        int b = rem / L, seq = rem % L;
        std::printf("  first mismatch: axis=%d b=%d seq=%d cpp=%lld ref=%lld\n",
                    axis, b, seq, (long long)res.pos_ids[first_mism],
                    (long long)ref[first_mism]);
    }
    std::printf("rope_delta: cpp=[");
    for (int b = 0; b < B; ++b) std::printf("%lld%s", (long long)res.rope_delta[b], b + 1 < B ? "," : "");
    std::printf("] ref=[");
    for (size_t b = 0; b < ref_delta.size(); ++b) std::printf("%lld%s", (long long)ref_delta[b], b + 1 < ref_delta.size() ? "," : "");
    std::printf("]\n");

    bool delta_ok = (ref_delta.size() == (size_t)B);
    for (int b = 0; delta_ok && b < B; ++b)
        if (res.rope_delta[b] != ref_delta[b]) delta_ok = false;

    if (mism == 0 && delta_ok) {
        std::printf("RESULT: EXACT MATCH\n");
        return 0;
    }
    std::printf("RESULT: %s\n", mism == 0 ? "pos_ids OK, delta differs"
                                          : "MISMATCH");
    return 2;
}
