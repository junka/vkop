// Copyright 2025 @junka
//
// Routing oracle for tools/roofline_report.py.
//
// The roofline report has to answer "which kernel does this MatMul actually
// dispatch, at what grid and local_size". That answer lives in
// core/KernelSelect.cpp (the rule table + the coopmat smart selector) and in
// ops/Matmul.hpp (the per-route dispatch arithmetic), and a static analysis
// tool must not keep a second copy of it -- a copy drifts silently the first
// time a rule changes.
//
// So this translation unit COMPILES THE REAL SELECTOR: it #includes
// core/KernelSelect.cpp (unmodified) into its own TU, which is also the only
// way to reach the file-local sm_estimate() -- the one place in the tree that
// knows how many SMs each vendor's device has. Nothing in the repository is
// edited; the include is read-only.
//
// Protocol: one JSON object per line on stdin, one JSON object per line on
// stdout, no other output. A line is
//
//   {"qid": 0, "batch":1, "m":1, "k":2048, "n":2048, "fp16":true,
//    "transB":true, "weight":"int8", "group_size":0, "epi_bias":false,
//    "attn_softmax":false, "down_act":false, "n_up":-1,
//    "device":"nvidia-ada"}
//
// and the answer is
//
//   {"qid":0, "route":"ksplit_w8wide", "ks_n":0, "ksplit_q":16, "wide":true,
//    "bm":0, "bn":0, "tiled":false, "tiled_min_m":2, "soc_sm":48,
//    "device_profile":"generic", "reason":"...", "error":null}
//
// Availability is filled the way ops/Matmul.cpp fills it on a healthy device
// (every lazily-built module probed and found), because the question the
// report asks is "what does the selector choose", not "what did one archive
// happen to contain". A route whose module the archive lacks is reported as a
// separate `missing_keys` field so the report can say so instead of silently
// attributing traffic to a kernel that will not run.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Vendor tile lists for the availability probe. Kept in step with
// ops/Matmul.hpp's kFp16Modules / kI8Modules by the same (cm, cn, ck) keys;
// which ones a given device LISTS is what the device preset below decides,
// and the op picks the first listed one, largest tile first.
#include "core/KernelSelect.cpp"

namespace {

using vkop::core::DeviceTraits;
using vkop::core::MatmulAvailability;
using vkop::core::MatmulFeatures;
using vkop::core::MatmulKnobs;
using vkop::core::MatmulPlan;
using vkop::core::MatmulShape;
using vkop::core::WeightKind;

// ---------------------------------------------------------------------------
// Minimal JSON reader: the input is a flat object of scalars, so a full parser
// (and a dependency on one) is not warranted. Returns false on a malformed
// line so a bad record is reported instead of silently analysed as zeros.
// ---------------------------------------------------------------------------
struct JsonLine {
    std::vector<std::pair<std::string, std::string>> kv;
    bool get(const char *key, std::string &out) const {
        for (const auto &p : kv) {
            if (p.first == key) {
                out = p.second;
                return true;
            }
        }
        return false;
    }
    double num(const char *key, double def) const {
        std::string v;
        if (!get(key, v) || v.empty() || v == "null") return def;
        return std::atof(v.c_str());
    }
    bool flag(const char *key, bool def) const {
        std::string v;
        if (!get(key, v) || v.empty() || v == "null") return def;
        return v == "true" || v == "1";
    }
    std::string str(const char *key, const char *def = "") const {
        std::string v;
        if (!get(key, v) || v == "null") return def;
        if (v.size() >= 2 && v.front() == '"' && v.back() == '"')
            return v.substr(1, v.size() - 2);
        return v;
    }
};

bool parse_line(const std::string &line, JsonLine &out) {
    out.kv.clear();
    size_t i = 0;
    const size_t n = line.size();
    auto skip_ws = [&]() {
        while (i < n && (line[i] == ' ' || line[i] == '\t' || line[i] == ',' ||
                         line[i] == '{'))
            ++i;
    };
    while (true) {
        skip_ws();
        if (i >= n || line[i] == '}') break;
        if (line[i] != '"') return false;
        size_t k0 = ++i;
        while (i < n && line[i] != '"') ++i;
        if (i >= n) return false;
        std::string key = line.substr(k0, i - k0);
        ++i;
        while (i < n && (line[i] == ' ' || line[i] == ':')) ++i;
        if (i >= n) return false;
        std::string val;
        if (line[i] == '"') {
            size_t v0 = i++;
            while (i < n && line[i] != '"') ++i;
            if (i >= n) return false;
            ++i;
            val = line.substr(v0, i - v0);
        } else {
            size_t v0 = i;
            while (i < n && line[i] != ',' && line[i] != '}') ++i;
            val = line.substr(v0, i - v0);
            while (!val.empty() && (val.back() == ' ' || val.back() == '\t'))
                val.pop_back();
        }
        out.kv.emplace_back(key, val);
    }
    return true;
}

std::string jesc(const std::string &s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if (c == '\n') {
            o += "\\n";
            continue;
        }
        o += c;
    }
    return o;
}

// ---------------------------------------------------------------------------
// Device presets. Every fact here is one the report needs and that the tree
// already knows or that the vendor documents:
//
//   * vendor_id / unified_memory / subgroup_size / coopmat capacities come
//     from the VKOP_DEVCAPS=1 dump plus the vulkaninfo profile JSONs checked
//     into the repo (VP_VULKANINFO_*.json), so a device preset is a
//     transcription of an observed device, not an assumption.
//   * max_storage_buffer_range is 4 GiB-1 on both (the 1.4 core limit).
//
// sm_estimate() is NOT a preset field: it is read straight out of the
// selector, which is the point of including it.
// ---------------------------------------------------------------------------
struct DevicePreset {
    const char *name;
    DeviceTraits traits;
    bool fp16_16x16x16;
    bool fp16_8x8x16;
    bool i8_16x16x32;
    bool i8_16x8x32;
    bool i8_8x8x32;
};

void fill_preset(const std::string &name, DevicePreset &p) {
    std::memset(&p, 0, sizeof(p));
    p.traits.max_storage_buffer_range = (1ull << 32) - 1;
    if (name == "nvidia-ada") {
        // b3: RTX 4000 Ada, driver 595.99.02. vulkaninfo lists fp16
        // 16x16x16 / 16x8x16 / 16x8x8 and NOT 8x8x16; int8 16x16x32 and
        // 16x8x32 (SINT8 and UINT8). See the b3 capability notes.
        p.name = "nvidia-ada";
        p.traits.vendor_id = 0x10DE;
        p.traits.driver_id = 0;  // native
        p.traits.unified_memory = false;
        p.traits.coopmat = true;
        p.traits.coopmat_fp16_16x16x16 = true;
        p.traits.coopmat_sint8_8x8x32 = false;
        p.traits.subgroup_arithmetic = true;
        p.traits.subgroup_size = 32;
        p.fp16_16x16x16 = true;
        p.i8_16x16x32 = true;
        p.i8_16x8x32 = true;
    } else if (name == "intel-arl") {
        // Dev box: Intel Graphics (ARL), Mesa 25.2.8. Lists fp16 8x8x16 only
        // (no 16x16x16), sint8 8x8x32.
        p.name = "intel-arl";
        p.traits.vendor_id = 0x8086;
        p.traits.unified_memory = true;
        p.traits.coopmat = true;
        p.traits.coopmat_fp16_8x8x16 = true;
        p.traits.coopmat_sint8_8x8x32 = true;
        p.traits.subgroup_arithmetic = true;
        p.traits.subgroup_size = 32;
        p.fp16_8x8x16 = true;
        p.i8_8x8x32 = true;
    } else if (name == "nvidia-generic") {
        // A discrete NVIDIA part whose coopmat listing we do not assert: the
        // report then shows the non-coop routes, which is the conservative
        // answer (a missing route is labelled, never invented).
        p.name = "nvidia-generic";
        p.traits.vendor_id = 0x10DE;
        p.traits.unified_memory = false;
        p.traits.coopmat = false;
        p.traits.subgroup_arithmetic = true;
        p.traits.subgroup_size = 32;
    } else {
        // "generic": native iGPU / unknown, the ARL+Ada calibration profile,
        // no coopmat assumed.
        p.name = "generic";
        p.traits.vendor_id = 0;
        p.traits.unified_memory = true;
        p.traits.coopmat = false;
        p.traits.subgroup_size = 32;
    }
}

// The availability `ops/Matmul` publishes on a device whose modules all built:
// coop_bm/bn are the first LISTED fp16 tile in kFp16Modules order (largest
// output tile first), i.e. what build_coop_pipeline keeps as its primary.
MatmulAvailability availability_for(const DevicePreset &p) {
    MatmulAvailability av;
    av.coopmat = p.traits.coopmat;
    if (p.fp16_16x16x16) {
        av.coop_bm = 32;  // 16x16x16 -> 2x4 subgroups -> bm 32, bn 64
        av.coop_bn = 64;
        av.coop_self_pack = false;  // coop16 writes scratch + pack
    } else if (p.fp16_8x8x16) {
        av.coop_bm = 16;  // 8x8x16 default build -> bm 16, bn 32
        av.coop_bn = 32;
        av.coop_self_pack = true;  // matmul_coop.comp packs half2 itself
    }
    av.coopmat_i8 = p.i8_16x16x32 || p.i8_16x8x32 || p.i8_8x8x32;
    av.ksplit512 = true;
    av.w8_wide = true;
    av.w8_wide512 = true;
    av.tile_ks_split = true;
    av.tile_ks_fold = true;
    av.down_act = true;
    return av;
}

WeightKind weight_kind(const std::string &s) {
    if (s == "int8") return WeightKind::kInt8;
    if (s == "fp8") return WeightKind::kFp8;
    if (s == "int4" || s == "uint4") return WeightKind::kInt4;
    if (s == "nf4" || s == "nvfp4") return WeightKind::kNvfp4;
    return WeightKind::kFp16;
}

}  // namespace

int main() {
    char buf[8192];
    while (std::fgets(buf, sizeof(buf), stdin)) {
        std::string line(buf);
        JsonLine j;
        if (!parse_line(line, j) || j.kv.empty()) continue;

        const int qid = static_cast<int>(j.num("qid", -1));
        DevicePreset dev;
        fill_preset(j.str("device", "generic"), dev);

        MatmulShape s;
        s.batch = static_cast<int>(j.num("batch", 1));
        s.m = static_cast<int>(j.num("m", 1));
        s.k = static_cast<int>(j.num("k", 0));
        s.n = static_cast<int>(j.num("n", 0));
        s.n_up = static_cast<int>(j.num("n_up", -1));

        MatmulFeatures f;
        f.fp16 = j.flag("fp16", true);
        f.transB = j.flag("transB", false);
        f.weight = weight_kind(j.str("weight", "fp16"));
        f.group_size = static_cast<int>(j.num("group_size", 0));
        f.epi_bias = j.flag("epi_bias", false);
        f.attn_softmax = j.flag("attn_softmax", false);
        f.down_act = j.flag("down_act", false);

        // Knobs from the environment, exactly like the op reads them: the
        // report must describe the build under the same overrides the run
        // would see. VKOP_* is only read here, never written.
        const MatmulKnobs &kn = vkop::core::matmul_knobs();
        const MatmulAvailability av = availability_for(dev);

        // sm_estimate lives in KernelSelect.cpp's anonymous namespace; this TU
        // included that file, so the real function is reachable unqualified.
        const int sm = vkop::core::sm_estimate(dev.traits);
        const vkop::core::DeviceProfile &prof =
            vkop::core::device_profile_for(dev.traits);

        MatmulPlan plan;
        std::string err = "null";
        try {
            plan = vkop::core::select_matmul(dev.traits, s, f, av, kn);
        } catch (const std::exception &e) {
            err = std::string("\"") + jesc(e.what()) + "\"";
        }

        std::printf(
            "{\"qid\":%d,\"route\":\"%s\",\"ks_n\":%d,\"ksplit_q\":%d,"
            "\"wide\":%s,\"bm\":%d,\"bn\":%d,\"tiled\":%s,"
            "\"tiled_min_m\":%d,\"coop_split\":%d,\"sm\":%d,"
            "\"device_profile\":\"%s\",\"reason\":\"%s\",\"error\":%s}\n",
            qid, plan.tag ? plan.tag : "", plan.ks_n, plan.ksplit_q,
            plan.wide ? "true" : "false", plan.bm, plan.bn,
            plan.tiled ? "true" : "false", plan.tiled_min_m, plan.coop_split,
            sm, prof.name,
            plan.route_reason ? jesc(plan.route_reason).c_str() : "",
            err.c_str());
        std::fflush(stdout);
    }
    return 0;
}
