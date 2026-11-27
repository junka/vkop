// junka @ 2026
// llm_serve —— llm.vkopbin 的行协议推理服务端（供 Python 的 HTTP 层驱动）。
//
// 为什么是行协议而不是 HTTP：JSON、Anthropic schema、SSE、tool_use 解析都在
// Python 侧做（那里有标准库，改起来不用重编）。这里只负责「一段 ChatML 文本进，
// 一段 token 出」，于是协议只需要 role/text 对，text 用 base64 传，彻底避开
// 转义与编码问题。
//
// 协议（每行一帧，从 --rpc-in 读，往 --rpc-out 写）：
//   请求                                    响应
//   HELLO                              →  READY <abi> <max_kv>
//   COUNT <id> <nmsg>                    → COUNTED <id> <tokens>
//     <role>\n<text-b64>\n  × nmsg
//   GEN <id> <max_new> <nmsg>            → TOKENS_IN <id> <n>
//     <role>\n<text-b64>\n  × nmsg       → DELTA <id> <text-b64>   (0..N 次)
//     [temperature top_k top_p]          → DONE <id> <stop_reason> <n_out>
//       三个采样参数可省略；负值＝客户端没传，用后端默认（贪心）。
//   出错                                 → ERROR <id> <msg-b64>
//   STOP <id>                          → （无独立响应；本轮以 DONE 收尾）
//     生成期间唯一可发的行：客户端已从流里解析出可用的 tool call，
//     不必再为剩下的 max_new 付 GPU 时间。decode 每步之间轮询一次。
//   RESET                              → OK
//   BYE                                → BYE
//
// stop_reason: "stop"（命中停止符或收到 STOP）/ "length"（max_new 用尽）。
//
// 与 llm_chat 的关系：架构推断、embed 查表、causal bias、输入填充、GPU argmax
// 和 decode 输入写入（DecodeInputs：host-imported 零拷贝，不可用时一次 batch
// 上传）都在 llm_common.hpp 里共用，数值路径逐位同构。仍**故意不带**滑窗环。
// GPU TopK 后处理（--sample-topk K，默认 64）：贪心读回第 0 个，采样只读回
// K 个候选（Sampler::pick_topk，分布精确，候选定不下来的步才读整行）；
// VKOP_GPU_ARGMAX=0 回退 host 扫描；零拷贝
// VKOP_DECODE_ZEROCOPY=0 关闭，hybrid（linear-attention）模型默认不开，
// 需 =1 显式打开。跨请求的 KV 前缀复用保留：下一段的 ids 以上一段覆盖的序列
// 为前缀时只算新增的 token —— Claude Code 每轮都把整段历史重发，省下的正是
// 「整个上下文重新 prefill」。
//
// Chunked prefill (--prefill-chunk) and the on-disk prefix cache (--kv-warm /
// --kv-dump, see dump_kv()) are the two performance layers that did get wired
// up here, because without them an agent's ~11k token prompt is unusable: the
// chunk keeps one dispatch's logits buffer inside VRAM, the file keeps the
// prefix it produced across processes.
//
// Build: `make llm_serve`（ENABLE_LLM_CHAT 打开时一起构建）。

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <poll.h>
#endif

#include <fcntl.h>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanLib.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "core/WeightPager.hpp"
#include "tokenizer.hpp"
#include "kv_cache.hpp"
#include "model/load.hpp"
#include "llm_common.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanBuffer;
using vkop::VulkanCommandBuffer;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::core::ElemKind;
using vkop::core::elem_name;
using vkop::export_::KVCache;
using vkop::export_::LinearState;
using qwen::Tokenizer;
using vkop::llm::causal_bias;
using vkop::llm::embed_lookup;
using vkop::llm::fill_fp16_input;
using vkop::llm::fill_i64_input;
using vkop::llm::upload_input;

namespace {

// ---------------------------------------------------------- platform I/O ----
// ucrt supplies read/write/open/close but neither fsync nor a rename that
// replaces an existing file, and the KV dump publishes through both.
// ucrt's _read/_write also take an unsigned count where POSIX takes size_t;
// keep each platform's own type so a transfer is never narrowed.
#ifdef _WIN32
using io_count_t = unsigned;
inline int os_fsync(int fd) { return _commit(fd); }
// FlushFileBuffers needs write access, unlike fsync on a read-only fd.
inline int os_open_for_fsync(const char *p) { return _open(p, _O_WRONLY); }
inline int os_replace(const char *from, const char *to) {
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
}
// True when a read would not block. The CRT has no non-blocking read on a pipe
// fd, but PeekNamedPipe reports the bytes waiting in the anonymous pipe the
// parent handed us.
inline bool os_readable(int fd) {
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    DWORD avail = 0;
    if (h == INVALID_HANDLE_VALUE) return false;
    return PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr) && avail > 0;
}
#else
using io_count_t = std::size_t;
inline int os_fsync(int fd) { return ::fsync(fd); }
inline int os_open_for_fsync(const char *p) { return ::open(p, O_RDONLY); }
inline int os_replace(const char *from, const char *to) {
    return std::rename(from, to);
}
inline bool os_readable(int fd) {
    struct pollfd p = {fd, POLLIN, 0};
    return ::poll(&p, 1, 0) > 0 && (p.revents & POLLIN) != 0;
}
#endif

// ---------------------------------------------------------------- base64 ----
const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string b64_encode(const std::string &in) {
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    std::size_t i = 0;
    while (i + 2 < in.size()) {
        uint32_t v = (static_cast<uint8_t>(in[i]) << 16) |
                     (static_cast<uint8_t>(in[i + 1]) << 8) |
                     static_cast<uint8_t>(in[i + 2]);
        out.push_back(kB64[(v >> 18) & 63]);
        out.push_back(kB64[(v >> 12) & 63]);
        out.push_back(kB64[(v >> 6) & 63]);
        out.push_back(kB64[v & 63]);
        i += 3;
    }
    const std::size_t left = in.size() - i;
    if (left == 1) {
        uint32_t v = static_cast<uint8_t>(in[i]) << 16;
        out.push_back(kB64[(v >> 18) & 63]);
        out.push_back(kB64[(v >> 12) & 63]);
        out.push_back('=');
        out.push_back('=');
    } else if (left == 2) {
        uint32_t v = (static_cast<uint8_t>(in[i]) << 16) |
                     (static_cast<uint8_t>(in[i + 1]) << 8);
        out.push_back(kB64[(v >> 18) & 63]);
        out.push_back(kB64[(v >> 12) & 63]);
        out.push_back(kB64[(v >> 6) & 63]);
        out.push_back('=');
    }
    return out;
}

std::string b64_decode(const std::string &in) {
    static int8_t rev[256];
    static bool init = false;
    if (!init) {
        std::memset(rev, -1, sizeof(rev));
        for (int i = 0; i < 64; ++i) rev[static_cast<uint8_t>(kB64[i])] = i;
        init = true;
    }
    std::string out;
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (unsigned char ch : in) {
        const int v = rev[ch];
        if (v < 0) break;  // '=' or garbage terminates the group
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// ----------------------------------------------------------- line reader ----
// Buffered read() over a raw fd: the protocol is line based and the responses
// must not interleave with the model's chatty stdout, so we never mix FILE*.
class LineReader {
  public:
    explicit LineReader(int fd) : fd_(fd) {}
    bool getline(std::string &out) {
        out.clear();
        for (;;) {
            const std::size_t nl = buf_.find('\n', pos_);
            if (nl != std::string::npos) {
                out.assign(buf_, pos_, nl - pos_);
                pos_ = nl + 1;
                if (!out.empty() && out.back() == '\r') out.pop_back();
                return true;
            }
            if (pos_ > 0) {
                buf_.erase(0, pos_);
                pos_ = 0;
            }
            char tmp[65536];
            const auto n = ::read(fd_, tmp, static_cast<io_count_t>(sizeof(tmp)));
            if (n <= 0) return false;
            buf_.append(tmp, static_cast<std::size_t>(n));
        }
    }

    // True once a STOP line has arrived, without blocking. Any other line is
    // left buffered for getline(): the main loop still has to see it. Called
    // between decode steps, where a blocking read would stall the round until
    // the client sent something it has no reason to send.
    bool poll_stop() {
        for (;;) {
            const std::size_t nl = buf_.find('\n', pos_);
            if (nl != std::string::npos) {
                std::string line(buf_, pos_, nl - pos_);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line != "STOP" && line.compare(0, 5, "STOP ") != 0)
                    return false;
                pos_ = nl + 1;
                return true;
            }
            if (pos_ > 0) {
                buf_.erase(0, pos_);
                pos_ = 0;
            }
            if (!os_readable(fd_)) return false;
            char tmp[4096];
            const auto n = ::read(fd_, tmp, static_cast<io_count_t>(sizeof(tmp)));
            if (n <= 0) return false;
            buf_.append(tmp, static_cast<std::size_t>(n));
        }
    }

  private:
    int fd_;
    std::string buf_;
    std::size_t pos_ = 0;
};

// The architecture, embedding and input helpers are shared with llm_chat
// (llm_common.hpp); only the server's own pieces live below.
using vkop::llm::ModelArch;
using vkop::llm::fp16_to_f32;
using vkop::llm::half_to_f32;
using vkop::llm::is_half16;

// ------------------------------------------------------------- sampling ----
using vkop::llm::SampleParams;
using vkop::llm::Sampler;

// Candidates the GPU TopK keeps per row. 1 is plain argmax; larger lets
// non-greedy requests sample from K values instead of reading back the whole
// vocabulary row (see Sampler::pick_topk). --sample-topk / VKOP_SAMPLE_TOPK.
constexpr int kDefaultSampleTopK = 64;

// The next token from the last logits row. `topk_k` is the k of the TopK
// post-process registered at load (0: none, 1: argmax only).
//
// Greedy reads back one id per row from the GPU TopK. Sampling with k > 1
// reads back the K candidates and lets Sampler::pick_topk decide, which only
// reads the full 151936-wide row on the steps it cannot settle exactly from
// the candidates. VKOP_LOGIT_TOP (debug: print the head of the distribution)
// always takes the full row.
int sample_last_token(const std::shared_ptr<Runtime> &rt,
                      const std::shared_ptr<vkop::VulkanCommandPool> &cp,
                      int fallback_vocab, int topk_k, Sampler &sampler,
                      const std::vector<uint32_t> &history,
                      const SampleParams &sp) {
    const bool logit_top = std::getenv("VKOP_LOGIT_TOP") != nullptr;
    const bool greedy = sp.temperature <= 0.0f && sp.repetition_penalty == 1.0f;
    // The full row decodes by the output's kind: fp16, or bf16 when the
    // artifact left `logits` bf16 (onnx2vkop --act bf16 keeps it fp16).
    ElemKind lk = ElemKind::kFloat16;
    if (auto l = rt->GetOutput("logits")) lk = l->elem_kind();
    sampler.set_logits_kind(lk);
    auto idx_out = topk_k > 0 ? rt->GetOutput("post_TopK1") : nullptr;
    if (idx_out && !logit_top && greedy) {
        auto ti = as_tensor<int>(idx_out);
        ti->copyToCPU(cp);
        // (rows, k) sorted descending: the last row's first entry. With k == 1
        // that is the last element, as before.
        const auto shape = ti->getShape();
        const int k = shape.empty() ? 1 : shape.back();
        return ti->at(ti->num_elements() - k);
    }

    // The full row, read lazily: the candidate path below may not need it.
    std::shared_ptr<vkop::core::Tensor<uint16_t>> lg;
    int vocab = 0, q = 1;
    auto full_row = [&]() -> const uint16_t * {
        if (!lg) {
            auto logits = rt->GetOutput("logits");
            if (!logits) throw std::runtime_error("no 'logits' output");
            lg = as_tensor<uint16_t>(logits);
            lg->copyToCPU(cp);
            const auto shape = lg->getShape();
            vocab = shape.empty() ? 0 : shape.back();
            q = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
            if (vocab <= 0 || q <= 0) {
                vocab = fallback_vocab;
                q = vocab > 0 ? lg->num_elements() / vocab : 1;
            }
        }
        return reinterpret_cast<const uint16_t *>(lg->data().data()) +
               static_cast<std::ptrdiff_t>(q - 1) * vocab;
    };

    if (idx_out && topk_k > 1 && !logit_top) {
        auto vt = as_tensor<uint16_t>(rt->GetOutput("post_TopK0"));
        auto ti = as_tensor<int>(idx_out);
        vt->copyToCPU(cp);
        ti->copyToCPU(cp);
        const auto shape = ti->getShape();
        const int k = shape.empty() ? 1 : shape.back();
        const size_t off = static_cast<size_t>(ti->num_elements() - k);
        const uint16_t *vals =
            reinterpret_cast<const uint16_t *>(vt->data().data()) + off;
        const int *ids = ti->data().data() + off;
        const int v = fallback_vocab;
        const int tok = sampler.pick_topk(vals, ids, k, v, history, sp,
                                          full_row);
        // VKOP_TOPK_VERIFY: the candidates must be exactly the row's k
        // largest values (as a multiset) -- the premise the exactness
        // argument rests on.
        if (std::getenv("VKOP_TOPK_VERIFY")) {
            const uint16_t *row = full_row();
            std::vector<float> all(static_cast<size_t>(vocab));
            for (int i = 0; i < vocab; ++i) all[i] = half_to_f32(row[i], lk);
            std::partial_sort(all.begin(), all.begin() + k, all.end(),
                              std::greater<float>());
            std::vector<float> got(static_cast<size_t>(k));
            for (int i = 0; i < k; ++i) got[i] = fp16_to_f32(vals[i]);
            std::sort(got.begin(), got.end(), std::greater<float>());
            bool ok = true;
            for (int i = 0; i < k; ++i) {
                ok = ok && got[i] == all[i] && ids[i] >= 0 && ids[i] < vocab &&
                     half_to_f32(row[ids[i]], lk) == fp16_to_f32(vals[i]);
            }
            std::fprintf(stderr, "[topk-verify] %s\n", ok ? "ok" : "MISMATCH");
        }
        return tok;
    }

    const uint16_t *row = full_row();
    if (logit_top) {
        // The head of the distribution. A model producing plausible text has
        // a broad top-5; one whose state or mask is wrong tends to pin a
        // single token (often EOS) far above the rest.
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < vocab; ++i) {
            const float v = half_to_f32(row[i], lk);
            if (top.size() == 5 && v <= top.back().first) continue;
            top.emplace_back(v, i);
            std::sort(top.begin(), top.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
            if (top.size() > 5) top.pop_back();
        }
        std::printf("[logit] q=%d vocab=%d top:", q, vocab);
        for (const auto &e : top) std::printf(" %d(%.3f)", e.second, e.first);
        std::printf("\n");
        std::fflush(stdout);
    }
    return sampler.pick(row, vocab, history, sp);
}

// ------------------------------------------------------------- the server ----
class InferenceServer {
  public:
    InferenceServer(const std::string &model, const std::string &embed,
                    const std::string &tok_path, int max_kv,
                    const std::string &kv_kind, int prefill_chunk,
                    const std::string &kv_warm = "",
                    float repetition_penalty = 1.0f, uint64_t seed = 0,
                    int sample_topk = kDefaultSampleTopK,
                    int context_length = 0, double kv_budget_mb = 0,
                    std::string page_policy = "")
        : max_kv_(max_kv), prefill_chunk_(std::max(1, prefill_chunk)),
          model_path_(model), page_policy_(std::move(page_policy)) {
        sampler_ = Sampler(seed);
        seed_ = seed;
        defaults_.repetition_penalty = repetition_penalty;
        Logger::getInstance().setLevel(LOG_INFO);
        const auto &devs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        if (devs.empty()) throw std::runtime_error("no vulkan device");
        unsigned idx = 0;
        if (const char *s = std::getenv("VKOP_DEVICE")) idx = (unsigned)atoi(s);
        if (idx >= devs.size())
            throw std::runtime_error("VKOP_DEVICE out of range");
        dev_ = std::make_shared<VulkanDevice>(devs[idx]);
        cmdpool_ = std::make_shared<vkop::VulkanCommandPool>(dev_);
        std::printf("[serve] GPU: %s\n", dev_->getDeviceName().c_str());

        tok_.load(tok_path);
        rt_ = std::make_shared<Runtime>(cmdpool_, model, /*precision=*/1);
        rt_->set_backend_buffer(true);
        // ---- layer paging, decided before LoadModel ----
        //
        // The plan must reach the Runtime BEFORE LoadModel decides what to
        // upload. WeightPager's declaration half reads only the artifact file
        // (no runtime, no GPU), so the bands are derivable here even though the
        // tensors do not exist yet.
        //
        // What paging buys, measured on b3: the 27B int8 artifact's 23.87 GiB of
        // pageable weights fit a 1.19 GiB shared arena. What it costs: every
        // page-in is a host read plus a PCIe copy (23 GB/s measured), and the
        // seam has to settle the device before each one. That is why the bands
        // worth keeping resident are picked LARGEST FIRST: the lm_head band is
        // 1212.5 MiB against 170 MiB for a per-layer band, so keeping the head
        // resident takes 7x less budget out of the arena than anything else.
        if (!page_policy_.empty()) {
            std::shared_ptr<Runtime> no_rt;
            // A real pool even for the declaration-only phase: this same object
            // is the one attached to the seam later, and attach_to refuses a
            // pager it cannot submit through.
            auto pager = std::make_unique<vkop::core::WeightPager>(no_rt, cmdpool_);
            vkop::load::VkModel artifact(model);
            const size_t nbands = pager->add_bands_from_graph(artifact, model);
            (void)nbands;
            auto bands = pager->group_bytes();
            std::sort(bands.begin(), bands.end(),
                      [](const auto &a, const auto &b) {
                          return a.second > b.second;
                      });
            // `--page-weights float:<MiB>` keeps the biggest bands resident
            // until their total reaches that budget; the rest are paged.
            // `all` pages everything pageable, `none` disables paging.
            size_t budget = 0;
            bool do_paging = true;
            if (page_policy_.rfind("float:", 0) == 0) {
                budget = static_cast<size_t>(
                    std::atof(page_policy_.c_str() + 6) * 1048576.0);
            } else if (page_policy_ == "all") {
                budget = 0;
            } else {
                do_paging = false;
            }
            if (do_paging) {
                size_t kept = 0, kept_n = 0;
                for (const auto &b : bands) {
                    if (kept + b.second <= budget) {
                        kept += b.second;
                        ++kept_n;
                        pager->drop_group(b.first);
                        continue;
                    }
                    break;
                }
                if (pager->size() == 0) {
                    std::printf("[serve] paging: every band fits the %.1f MiB "
                                "budget, nothing to page\n", budget / 1048576.0);
                    pager.reset();
                } else {
                    vkop::core::PagedArenaPlan plan;
                    plan.names = pager->names();
                    // Bands reuse one region; the weights INSIDE a band do not.
                    // A band is paged in by a single call and every one of its
                    // weights is read by the same level, so flattening them to
                    // one offset silently kept only the last weight of each
                    // band -- measured as a 3-way k/q/v band serving v_proj's
                    // bytes to k and q, which answered "The main and most
                    // famous capital of France" and turned to noise once enough
                    // multi-weight bands were paged.
                    plan.share_slots = true;
                    plan.slot_of = pager->slot_groups();
                    rt_->set_paged_initializers(std::move(plan));
                    std::printf("[serve] paging: %zu band(s) paged (%zu weight(s)), "
                                "%zu largest band(s) (%.1f MiB) kept resident\n",
                                pager->groups().size(), pager->size(), kept_n,
                                kept / 1048576.0);
                    pager_ = std::move(pager);
                }
                std::fflush(stdout);
            }
        }
        std::printf("[serve] === LoadModel ===\n");
        rt_->LoadModel();
        std::printf("[serve] === LoadModel done ===\n");
        // The seam needs the runtime, so attach after LoadModel: the pager
        // resolves each weight's buffer on every page-in rather than caching it.
        if (pager_) {
            // The callback must not capture anything by reference: attach_to
            // stores it in the runtime's level hook, so it outlives this block
            // and runs once per level inside Run(). Capturing local counters
            // here is a dangling write on every level -- measured as a SIGSEGV
            // after the last level, and, under a different stack layout, as a
            // corrupted `prefill L=...`.
            pager_->attach_to(rt_, [this](const vkop::core::PageStats &st) {
                ++page_calls_;
                page_bytes_ += st.bytes;
                page_host_ms_ += st.host_ms;
                page_device_ms_ += st.device_ms;
                if (std::getenv("VKOP_SERVE_VERBOSE"))
                    std::printf("[serve] page-in %zu B (host %.1f + dev %.1f ms)\n",
                                st.bytes, st.host_ms, st.device_ms);
            });
            std::printf("[serve] paging attached: %zu band(s), arena %.1f MiB\n",
                        pager_->groups().size(),
                        rt_->paged_arena() ? rt_->paged_arena()->getSize() / 1048576.0
                                           : 0.0);
            std::fflush(stdout);
        }
        topk_k_ = vkop::llm::register_gpu_topk(rt_, "serve", sample_topk);
        arch_ = vkop::llm::infer_arch(rt_, "serve");
        if (kv_kind != "auto") {
            // The cache dtype belongs to the artifact: the converter wires
            // QuantizeLinear/DequantizeLinear around past/present and the
            // attention shader is picked from that declaration, so overriding
            // the kind here cannot change what gets decoded -- it only hands the
            // kernel a differently-sized buffer. llm_chat warns and continues;
            // a server silently answering from a misread cache is worse, and a
            // 4-bit override on an fp16 graph died building the plan.
            const ElemKind want = parse_kind(kv_kind, arch_.cache_kind);
            if (want != arch_.cache_kind) {
                std::string msg = "--kv-kind " + kv_kind + " but " + model +
                                  " declares a " + elem_name(arch_.cache_kind) +
                                  " cache";
                // make_kv_variant.py only knows how to build the fp8 arm, so
                // name it only for the request it can actually serve.
                if (want == ElemKind::kFloat8E4M3FN &&
                    arch_.cache_kind == ElemKind::kFloat16)
                    msg += "; export one with llm/exporter/make_kv_variant.py --fp8";
                else
                    msg += " (the cache dtype comes from the artifact, not this flag)";
                throw std::runtime_error(msg);
            }
        }
        std::printf("[serve] arch hidden=%d nlayers=%d nkv=%d hd=%d pos=%dD "
                    "deepstack=%d image_mask=%d kv=%s\n",
                    arch_.hidden, arch_.nlayers, arch_.nkv, arch_.hd,
                    arch_.position_ids_dims, (int)arch_.has_deepstack,
                    (int)arch_.has_image_pad_mask, elem_name(arch_.cache_kind));

        auto loaded = vkop::llm::load_embed_table(embed, arch_.hidden, "serve");
        embed_table_ = std::move(loaded.first);
        vocab_ = loaded.second;
        std::printf("[serve] embed vocab=%d hidden=%d\n", vocab_, arch_.hidden);

        max_kv_ = resolve_window_(max_kv_, context_length, kv_budget_mb);
        kv_ = std::make_unique<KVCache>(rt_, cmdpool_, arch_.nlayers, arch_.nkv,
                                        arch_.hd, max_kv_, arch_.cache_kind,
                                        arch_.kv_layers);
        // Hybrid decoders (Qwen3.5) also carry per-layer conv/recurrent state
        // for their linear-attention layers. Dense ones have none and this
        // stays empty.
        lin_state_ = std::make_unique<LinearState>(rt_, cmdpool_);
        if (lin_state_->present()) {
            lin_state_->reset();
            lin_state_->upload();
            std::printf("[serve] linear-attention state: %zu pair(s)\n",
                        lin_state_->pair_count());
            std::fflush(stdout);
        }
        for (int32_t id : tok_.stop_token_ids())
            stop_ids_.push_back(static_cast<uint32_t>(id));
        // Turn boundaries the chat template's own stop list leaves out. A
        // model that writes a new turn header (<|im_start|>), the document
        // end, or a tool *result* has finished its turn: without these the
        // reply carried a literal "<|im_start|>user\n<|im_start|>assistant"
        // text block to Claude Code (seen in a real session), and a
        // hallucinated <tool_response> would be streamed as prose.
        for (const char *lit : {"<|im_start|>", "<|endoftext|>",
                                "<tool_response>", "<|user|>", "<|system|>"}) {
            const int32_t id = tok_.special_token_id(lit);
            if (id >= 0 && !is_stop(static_cast<uint32_t>(id)))
                stop_ids_.push_back(static_cast<uint32_t>(id));
        }
        // Special tokens that may reach the client: the markers the HTTP layer
        // parses. Every other special id is dropped from the text (see
        // emits_text()).
        for (const char *lit : {"<tool_call>", "</tool_call>", "<think>",
                                "</think>"}) {
            const int32_t id = tok_.special_token_id(lit);
            if (id >= 0) text_specials_.push_back(static_cast<uint32_t>(id));
        }
        dec_in_ = std::make_unique<vkop::llm::DecodeInputs>(rt_, dev_, cmdpool_,
                                                            arch_);
        if (!tok_.chat_template().empty())
            std::printf("[serve] chat template loaded\n");
        else
            std::printf("[serve] WARNING: tokenizer has no chat template\n");

        if (!kv_warm.empty()) {
            try {
                load_kv(kv_warm);
            } catch (const std::exception &e) {
                // A cache file is only a shortcut, so an unusable one must not
                // stop the server: say it loudly and prefill from scratch. The
                // answers are the same either way -- refusing to start over a
                // reconverted model would be worse than a slow first round.
                std::fprintf(stderr, "[serve] %s -- serving cold\n", e.what());
                std::fflush(stderr);
            }
        }
    }

    const ModelArch &arch() const { return arch_; }
    int vocab() const { return vocab_; }

    static bool turn_eq(const qwen::ChatMessage &a, const qwen::ChatMessage &b) {
        if (a.role != b.role || a.contents.size() != b.contents.size())
            return false;
        for (std::size_t i = 0; i < a.contents.size(); ++i) {
            if (a.contents[i].type != b.contents[i].type ||
                a.contents[i].content != b.contents[i].content)
                return false;
        }
        return true;
    }

    static std::vector<qwen::ChatMessage> to_chat(
        const std::vector<std::pair<std::string, std::string>> &msgs) {
        std::vector<qwen::ChatMessage> chat;
        chat.reserve(msgs.size());
        for (const auto &m : msgs) {
            qwen::ChatMessage cm;
            cm.role = m.first;
            cm.contents.push_back({/*type=*/"text", m.second});
            chat.push_back(std::move(cm));
        }
        return chat;
    }

    // Render the request's role/text pairs through the tokenizer's chat
    // template and return the token ids. The callers pass assistant turns as
    // their literal text (the HTTP layer flattens tool_use into the
    // <tool_call> text the template itself would emit), so nothing here has to
    // know about tools.
    //
    // Why the previous round's ids are spliced back instead of re-tokenizing:
    // BBPE decode -> encode is not an identity. The word-start space marker is
    // dropped by decode, so re-encoding an assistant reply yields ids that
    // differ from the ones the model actually wrote. That is not a correctness
    // problem on its own (the text is the same), but it makes the prefix
    // comparison fail every round, and every round then re-prefills the whole
    // 11k-token system prompt. llm_chat sidesteps this by keeping token ids in
    // Conversation; a stateless HTTP API cannot, so the ids of the last round
    // are kept here and only the appended turns are encoded.
    //
    // Pure with respect to the splice cache: `begin_round()` is what records
    // the turn list, so a COUNT request between two GENs cannot disturb it.
    std::vector<uint32_t> render(
        const std::vector<std::pair<std::string, std::string>> &msgs) const {
        std::vector<qwen::ChatMessage> chat = to_chat(msgs);
        // Splice only when every previous turn is still present unchanged:
        // then the previous round's id stream (prompt + generated tokens + the
        // turn terminator) is a literal prefix of this prompt. Anything else —
        // an edited or trimmed history — must be rendered whole, because the
        // ids past the divergence point are not derivable from the text.
        if (!prev_turns_.empty() && prev_turns_.size() <= chat.size()) {
            bool same = true;
            for (std::size_t i = 0; i < prev_turns_.size() && same; ++i)
                same = turn_eq(prev_turns_[i], chat[i]);
            if (same) {
                std::vector<uint32_t> ids = prev_prefix_ids_;
                const std::vector<qwen::ChatMessage> rest(
                    chat.begin() + static_cast<std::ptrdiff_t>(prev_turns_.size()),
                    chat.end());
                const std::vector<uint32_t> more = tok_.encode(
                    tok_.apply_chat_template(rest, /*add_generation_prompt=*/true));
                ids.insert(ids.end(), more.begin(), more.end());
                return ids;
            }
        }
        const std::string text =
            tok_.apply_chat_template(chat, /*add_generation_prompt=*/true);
        if (text.empty())
            throw std::runtime_error(
                "chat template produced an empty prompt (tokenizer.bin has no "
                "template?)");
        return tok_.encode(text);
    }

    // Render and remember the turn list, so the next request can splice.
    std::vector<uint32_t> begin_round(
        const std::vector<std::pair<std::string, std::string>> &msgs) {
        last_render_turns_ = to_chat(msgs);
        return render(msgs);
    }

    // Capture what the next request will splice onto. Called after generation,
    // once covered_ids_ holds the prompt plus every token decode fed back.
    //
    // `assistant_text` is the reply exactly as it was streamed to the client:
    // the client echoes that text back as the assistant turn, so it is the key
    // the next render compares against. It deliberately does not take part in
    // re-tokenization — the ids come from prev_prefix_ids_.
    //
    // The template closes an assistant turn with the turn terminator
    // (<|im_end|>\n), which the generation loop never consumed -- it stops *on*
    // the stop token. Appending the terminator's ids therefore reproduces the
    // template's rendering of this turn exactly (both sides are special-token
    // boundaries, so no BPE merge can cross the seam). Doing it this way keeps
    // the model's own ids: re-encoding the reply text would not reproduce them.
    void record_turn(const std::string &assistant_text) {
        if (last_render_turns_.empty()) return;
        prev_turns_ = last_render_turns_;
        prev_turns_.push_back(
            {"assistant", {{/*type=*/"text", assistant_text}}});
        prev_prefix_ids_ = covered_ids_;
        const std::vector<uint32_t> tail =
            tok_.encode(tok_.turn_tail_literal());
        prev_prefix_ids_.insert(prev_prefix_ids_.end(), tail.begin(),
                                tail.end());
    }

    // ---- the prefix cache on disk ------------------------------------------
    // What a finished round leaves in the KV buffer, written out so a *new*
    // process can start a conversation in the middle of it. The engine half
    // already exists and is unchanged: covered_ids_ + KVCache::resume_from()
    // skip the rows that are already computed, and they cannot tell whether
    // those rows came from this process's prefill or from a file.
    //
    // Worth doing only because an agent's stable prefix is enormous: Claude
    // Code sends ~11k tokens of system prompt and tool schemas before any user
    // text, which is ~400 s of prefill here against the seconds it takes to
    // read 630 MB of fp8 rows back off disk.
    //
    // The file carries (model path, model bytes) as a fingerprint. The stored
    // ids prove the *text* matched; only the artifact identity proves the cache
    // bytes came from this graph, so a reconverted or swapped model is refused
    // at startup -- a slow first round instead of answers read out of another
    // model's cache.
    static constexpr char kKVMagic[8] = {'V', 'K', 'O', 'P', 'K', 'V', 'C', '1'};

    // Values in one cache row: K and V stacked on dim 1, then the kv heads.
    size_t kv_row_vals() const { return 2ull * arch_.nkv * arch_.hd; }

    size_t kv_elem_bytes() const {
        return is_half16(arch_.cache_kind) ? 2 : 1;
    }

    // Bytes one cache row occupies in the file for one layer. Zero means "no
    // layout": a 4-bit container packs two values per byte, so its rows are not
    // `values * 1 byte` (kv_cache.hpp), and the server has never served one.
    size_t kv_row_bytes() const {
        if (is_half16(arch_.cache_kind) ||
            arch_.cache_kind == ElemKind::kFloat8E4M3FN ||
            arch_.cache_kind == ElemKind::kFloat8E5M2)
            return kv_row_vals() * kv_elem_bytes();
        return 0;
    }

    // ---- the KV window ------------------------------------------------------
    // `requested` <= 0 is "auto": the model's native context (--context-length,
    // the checkpoint's max_position_embeddings) capped by what the cache may
    // take of device memory. An explicit --max-kv is honoured but clamped to
    // the native context -- rows past it are positions the model was never
    // trained on. The result is a multiple of 1024 rows so the attention tiles
    // (256) and prefill chunks divide it.
    //
    // Memory: --kv-budget-mb when given; otherwise 25% of what the device-local
    // heap still has after the model loaded (VK_EXT_memory_budget), or 30% of
    // the heap size without that extension. On a UMA Mac the heap is system
    // RAM, so the cap is what keeps a 256k-token model from asking for 30 GB.
    size_t kv_bytes_per_row_all_layers_() const {
        const size_t layers =
            arch_.kv_layers.empty() ? static_cast<size_t>(arch_.nlayers)
                                    : arch_.kv_layers.size();
        return kv_row_vals() * kv_elem_bytes() * layers;
    }

    double kv_memory_budget_bytes_(double budget_mb) const {
        if (budget_mb > 0) return budget_mb * 1048576.0;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{};
        bp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp{};
        mp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        const bool has_budget = dev_->is_support_memory_budget();
        if (has_budget) mp.pNext = &bp;
        // Through the project loader (vulkan/VulkanLib.hpp): the binary does
        // not link libvulkan, so the global prototype has no definition.
        vkop::VulkanLib::getVulkanLib().vkGetPhysicalDeviceMemoryProperties2(
            dev_->getPhysicalDevice(), &mp);
        const auto &mem = mp.memoryProperties;
        uint32_t heap = 0;
        for (uint32_t h = 0; h < mem.memoryHeapCount; ++h) {
            if ((mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                mem.memoryHeaps[h].size > mem.memoryHeaps[heap].size)
                heap = h;
        }
        const double size = static_cast<double>(mem.memoryHeaps[heap].size);
        if (has_budget && bp.heapBudget[heap] > bp.heapUsage[heap]) {
            const double free_b =
                static_cast<double>(bp.heapBudget[heap] - bp.heapUsage[heap]);
            std::printf("[serve] device heap %.1f GB, budget %.1f GB, in use "
                        "%.1f GB\n", size / 1e9, bp.heapBudget[heap] / 1e9,
                        bp.heapUsage[heap] / 1e9);
            return 0.25 * free_b;
        }
        return 0.3 * size;
    }

    int resolve_window_(int requested, int context_length, double budget_mb) {
        constexpr int kAlign = 1024;
        const size_t row_bytes = kv_bytes_per_row_all_layers_();
        int window = requested;
        if (requested <= 0) {
            const double budget = kv_memory_budget_bytes_(budget_mb);
            const double fit = row_bytes ? budget / static_cast<double>(row_bytes) : 0;
            int64_t rows = static_cast<int64_t>(fit);
            if (context_length > 0 && rows > context_length) rows = context_length;
            rows = (rows / kAlign) * kAlign;
            if (rows < kAlign) {
                throw std::runtime_error(
                    "--max-kv auto: the KV budget fits only " +
                    std::to_string(static_cast<int64_t>(fit)) +
                    " rows; pass --max-kv or --kv-budget-mb");
            }
            window = static_cast<int>(rows);
            std::printf("[serve] --max-kv auto: %d rows (model context %s, "
                        "budget fits %lld, %.2f GB of cache)\n",
                        window,
                        context_length > 0 ? std::to_string(context_length).c_str()
                                           : "unknown",
                        static_cast<long long>(fit),
                        static_cast<double>(window) * row_bytes / 1e9);
        } else if (context_length > 0 && requested > context_length) {
            window = context_length;
            std::printf("[serve] --max-kv %d exceeds the model's %d-token "
                        "context; clamped to %d\n",
                        requested, context_length, window);
        }
        std::fflush(stdout);
        return window;
    }

    // The window this server really holds (after auto / clamping). Reported in
    // the READY line so the HTTP layer trims against the same number.
    int max_kv() const { return max_kv_; }

    // ---- the row pitch ------------------------------------------------------
    // The kernel does not address cache rows by the tensor's logical length.
    // Attention::row_capacity() takes the stride from the bound buffer's SIZE,
    // so row j of a K/V block lives at j * cap * hd with cap = the allocated
    // rows (--max-kv), and the rows a round leaves behind are scattered with
    // that stride. That is what makes appending O(S) -- growing kv_len never
    // moves a row already written -- and it is also why the file cannot be the
    // tensor's first rows * row_bytes bytes: read that way the cache would be
    // head 0's first rows and nothing else, i.e. noise. pack_rows_ /
    // unpack_rows_ walk the 2 * nkv blocks against the real pitch.
    int kv_pitch_(int layer) {
        const std::string name = arch_.kv_in(layer);
        auto t = rt_->GetInput(name);
        if (!t) throw std::runtime_error("--kv-cache: no " + name);
        size_t bytes = 0;
        if (is_half16(arch_.cache_kind))
            bytes = as_tensor<uint16_t>(t)->as_storage_buffer(dev_)->getSize();
        else
            bytes = as_tensor<int8_t>(t)->as_storage_buffer(dev_)->getSize();
        const int pitch = static_cast<int>(bytes / (kv_row_vals() * kv_elem_bytes()));
        if (pitch < 1)
            throw std::runtime_error("--kv-cache: " + name + " has no cache buffer");
        return pitch;
    }

    // The layer's preallocated cache buffer (past and present share it).
    std::shared_ptr<VulkanBuffer> kv_buffer_(int layer) {
        auto t = rt_->GetInput(arch_.kv_in(layer));
        if (!t) throw std::runtime_error("--kv-cache: no " + arch_.kv_in(layer));
        if (is_half16(arch_.cache_kind))
            return as_tensor<uint16_t>(t)->as_storage_buffer(dev_);
        return as_tensor<int8_t>(t)->as_storage_buffer(dev_);
    }

    // A mapped staging buffer for one layer's live rows (rows * row_bytes).
    std::shared_ptr<VulkanBuffer> kv_staging_(size_t bytes) {
        auto st = std::make_shared<VulkanBuffer>(
            dev_, bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!st->getMappedMemory())
            throw std::runtime_error("--kv-cache: staging buffer not mappable");
        return st;
    }

    // Move layer `layer`'s first `rows` rows between the pitched device buffer
    // and `staging`, packed block-major like the file: 2*nkv ranged copies of
    // rows*hd elements each. Only the live rows travel. The previous form read
    // back / uploaded the WHOLE pitched layer through a host vector, so dump
    // and warm-load cost grew with --max-kv rather than with the prefix: fine
    // at 16k rows, ~19 GB of traffic once the window follows the model's
    // native context.
    void transfer_rows_(int layer, uint32_t rows, int pitch,
                        const std::shared_ptr<VulkanBuffer> &staging,
                        bool to_device) {
        auto buf = kv_buffer_(layer);
        const VkDeviceSize eb = static_cast<VkDeviceSize>(kv_elem_bytes());
        const VkDeviceSize dev_blk = static_cast<VkDeviceSize>(pitch) * arch_.hd * eb;
        const VkDeviceSize file_blk = static_cast<VkDeviceSize>(rows) * arch_.hd * eb;
        auto cmd = std::make_shared<VulkanCommandBuffer>(cmdpool_);
        cmd->begin();
        if (!to_device) {
            // The fused Attention wrote these rows; make them visible to the
            // transfer (the buffer's own access tracking does not see shader
            // writes).
            buf->shaderWriteBarrier(cmd->get());
        }
        for (int b = 0; b < 2 * arch_.nkv; ++b) {
            if (to_device)
                buf->copyStageBufferToBuffer(cmd->get(), staging->getBuffer(),
                                             b * file_blk, file_blk, b * dev_blk);
            else
                buf->copyBufferToStageBuffer(cmd->get(), staging->getBuffer(),
                                             b * file_blk, file_blk, b * dev_blk);
        }
        if (to_device) buf->readBarrier(cmd->get());
        cmd->end();
        cmd->submit(dev_->getComputeQueue());
        cmd->wait();
    }

    size_t artifact_bytes() const {
        std::ifstream f(model_path_, std::ios::ate | std::ios::binary);
        return f ? static_cast<size_t>(f.tellg()) : 0;
    }

    // Deletes `path` when the scope that owns it ends, unless `keep` was set
    // once the file is complete. A throw anywhere inside dump_kv -- the pitch
    // guard, a readback failure -- otherwise leaves a .tmp that a later `ls`
    // would mistake for a cache.
    struct TmpFile {
        std::string path;
        bool keep = false;
        ~TmpFile() {
            if (keep) return;
            // ENOENT is fine: `out` may have failed to create the file at all.
            if (::remove(path.c_str()) != 0 && errno != ENOENT)
                std::fprintf(stderr, "[serve] could not remove %s\n",
                             path.c_str());
            std::fflush(stderr);
        }
    };

    // Write the covered rows to `path` (via .tmp + rename, so a crash halfway
    // cannot leave a short file that a later start would trust).
    void dump_kv(const std::string &path) {
        const size_t row_bytes = kv_row_bytes();
        if (row_bytes == 0)
            throw std::runtime_error(
                "--kv-dump: no file layout for a " +
                std::string(elem_name(arch_.cache_kind)) + " cache");
        const uint32_t rows = static_cast<uint32_t>(covered_ids_.size());
        if (rows == 0)
            throw std::runtime_error("--kv-dump: the KV window holds nothing yet");

        const auto t0 = std::chrono::steady_clock::now();
        const std::string tmp = path + ".tmp";
        TmpFile guard{tmp};
        std::ofstream out(tmp, std::ios::binary);
        if (!out) throw std::runtime_error("--kv-dump: cannot open " + tmp);
        out.write(kKVMagic, sizeof(kKVMagic));
        const uint32_t hdr[5] = {rows, static_cast<uint32_t>(arch_.nlayers),
                                static_cast<uint32_t>(arch_.nkv),
                                static_cast<uint32_t>(arch_.hd),
                                static_cast<uint32_t>(arch_.cache_kind)};
        const uint64_t model_bytes = static_cast<uint64_t>(artifact_bytes());
        out.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
        out.write(reinterpret_cast<const char *>(&model_bytes),
                  sizeof(model_bytes));
        const uint32_t plen = static_cast<uint32_t>(model_path_.size());
        out.write(reinterpret_cast<const char *>(&plen), sizeof(plen));
        out.write(model_path_.data(), plen);
        out.write(reinterpret_cast<const char *>(covered_ids_.data()),
                  static_cast<std::streamsize>(rows) * 4);

        const size_t layer_bytes = static_cast<size_t>(rows) * row_bytes;
        auto staging = kv_staging_(layer_bytes);
        const char *mapped = static_cast<const char *>(staging->getMappedMemory());
        for (int i = 0; i < arch_.nlayers; ++i) {
            const int pitch = kv_pitch_(i);
            if (static_cast<uint32_t>(pitch) < rows)
                throw std::runtime_error(
                    "--kv-dump: layer " + std::to_string(i) + " holds " +
                    std::to_string(pitch) + " rows but the conversation covered " +
                    std::to_string(rows));
            transfer_rows_(i, rows, pitch, staging, /*to_device=*/false);
            out.write(mapped, static_cast<std::streamsize>(layer_bytes));
        }
        out.flush();
        if (!out) throw std::runtime_error("--kv-dump: write failed: " + tmp);
        out.close();
        // Rename only publishes bytes the OS already has: without the fsync a
        // crash could leave a complete-looking, partly zero-filled file that a
        // later start trusts. `path` keeps the previous good cache either way.
        const int fd = os_open_for_fsync(tmp.c_str());
        if (fd < 0 || os_fsync(fd) != 0) {
            const int saved = errno;
            if (fd >= 0) ::close(fd);
            throw std::runtime_error(std::string("--kv-dump: fsync failed for ") +
                                     tmp + ": " + std::strerror(saved));
        }
        ::close(fd);
        if (os_replace(tmp.c_str(), path.c_str()) != 0)
            throw std::runtime_error("--kv-dump: cannot move " + tmp + " onto " +
                                     path);
        guard.keep = true;
        // The readbacks flipped each past tensor to host residency; say again
        // what is still true -- the rows are on the GPU, untouched.
        kv_->resume_from(static_cast<int>(rows));
        kv_file_rows_ = rows;
        const double ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        std::printf("[serve] --kv-dump %s: %u rows, %.0f MB in %.0f ms\n",
                    path.c_str(), rows,
                    static_cast<double>(layer_bytes) * arch_.nlayers / 1048576.0,
                    ms);
        std::fflush(stdout);
    }

    // Load rows written by dump_kv and present them as this round's history.
    // A missing file is the normal cold start and returns quietly; anything
    // else that is wrong with the cache throws.
    void load_kv(const std::string &path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::printf("[serve] --kv-warm: no %s, starting cold\n", path.c_str());
            std::fflush(stdout);
            return;
        }
        const auto fail = [&path](const std::string &why) {
            return std::runtime_error("--kv-warm " + path + ": " + why);
        };
        char magic[sizeof(kKVMagic)];
        in.read(magic, sizeof(magic));
        if (!in || std::memcmp(magic, kKVMagic, sizeof(magic)) != 0)
            throw fail("not a KV cache file");
        uint32_t hdr[5] = {};
        in.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
        uint64_t model_bytes = 0;
        in.read(reinterpret_cast<char *>(&model_bytes), sizeof(model_bytes));
        uint32_t plen = 0;
        in.read(reinterpret_cast<char *>(&plen), sizeof(plen));
        // Never size a container from an unchecked field: a corrupt header with
        // a 4 GB plen would throw bad_alloc out of a path whose contract is
        // "refuse the cache, serve cold".
        if (plen > 4096) throw fail("corrupt header: model path length " +
                                    std::to_string(plen));
        std::string mpath(plen, '\0');
        if (plen) in.read(&mpath[0], plen);
        const uint32_t rows = hdr[0];
        const int nlayers = static_cast<int>(hdr[1]);
        if (!in) throw fail("truncated header");
        if (rows == 0) throw fail("the file holds no rows");
        if (rows > static_cast<uint32_t>(max_kv_))
            throw fail(std::to_string(rows) + " rows exceed this --max-kv " +
                       std::to_string(max_kv_));
        if (nlayers != arch_.nlayers || hdr[2] != static_cast<uint32_t>(arch_.nkv) ||
            hdr[3] != static_cast<uint32_t>(arch_.hd) ||
            hdr[4] != static_cast<uint32_t>(arch_.cache_kind))
            throw fail("was written for " + std::to_string(nlayers) + " layers / " +
                       std::to_string(hdr[2]) + " kv heads / head " +
                       std::to_string(hdr[3]) + " / cache " +
                       elem_name(static_cast<ElemKind>(hdr[4])) +
                       ", this graph is different");
        if (mpath != model_path_ || model_bytes != artifact_bytes())
            throw fail("belongs to " + mpath + " (" +
                       std::to_string(model_bytes) +
                       " bytes); re-dump it, this artifact has changed");

        const size_t row_bytes = kv_row_bytes();
        if (row_bytes == 0)
            throw fail("no file layout for a " +
                       std::string(elem_name(arch_.cache_kind)) + " cache");

        std::vector<uint32_t> ids(rows);
        in.read(reinterpret_cast<char *>(ids.data()),
                static_cast<std::streamsize>(rows) * 4);
        if (!in) throw fail("truncated id stream");
        const size_t layer_bytes = static_cast<size_t>(rows) * row_bytes;
        auto staging = kv_staging_(layer_bytes);
        char *mapped = static_cast<char *>(staging->getMappedMemory());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < nlayers; ++i) {
            in.read(mapped, static_cast<std::streamsize>(layer_bytes));
            if (!in) throw fail("truncated cache payload");
            const int pitch = kv_pitch_(i);
            if (static_cast<uint32_t>(pitch) < rows)
                throw fail("wants " + std::to_string(rows) +
                           " rows but layer " + std::to_string(i) +
                           " has room for " + std::to_string(pitch) +
                           " (--max-kv)");
            transfer_rows_(i, rows, pitch, staging, /*to_device=*/true);
        }
        covered_ids_ = std::move(ids);
        kv_->resume_from(static_cast<int>(rows));
        kv_file_rows_ = rows;
        const double ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        std::printf("[serve] --kv-warm: %u rows (%.0f MB) resumed in %.0f ms\n",
                    rows, static_cast<double>(layer_bytes) * nlayers / 1048576.0,
                    ms);
        std::fflush(stdout);
    }

    // ---- when the file gets written ----------------------------------------
    // The dump only pays for its I/O when it captures work a *future* process
    // would redo, so it fires twice: after the first completed round (that is
    // the agent's ~11k token stable prefix) and once more at shutdown if this
    // conversation reached further than the file does. Rewriting after every
    // round would move the whole 630 MB cache to save the few hundred tokens a
    // turn adds -- the in-process covered_ids_ prefix reuse already covers
    // those, and needs no disk.
    //
    // Neither trigger may fail a request: the answer is already streamed, and
    // a cache file is a convenience. So the caller catches, and a failed write
    // just leaves the previous file in place (dump_kv writes .tmp and renames).
    void dump_after_first_round() {
        if (!kv_dump_pending_) return;
        // A completed dump retires the trigger. If it throws -- disk full, a
        // failed readback -- the flag stays up so the next round, and the exit
        // dump below, get another chance.
        dump_if_longer_();
        kv_dump_pending_ = false;
    }
    void dump_on_exit() { dump_if_longer_(); }

    // Turns the on-disk cache on. `path` is rewritten in place, so it should
    // not live on a ramdisk and should have rows * 28 layers * row bytes free.
    void set_kv_dump_path(const std::string &path) { kv_dump_path_ = path; }

    // Backend-wide defaults; a request overrides temperature/top_k/top_p only
    // when the client actually sent them.
    const SampleParams &default_params() const { return defaults_; }

    // Generate from an already rendered prompt. `out_ids` receives every
    // generated token (the ones actually fed back included). Returns
    // "stop" / "length".
    //
    // on_token fires only for tokens that belong to the text: a stop token is
    // recorded in out_ids but never handed to the decoder, so the caller can
    // decode with skip_special=false and still not see <|im_end|> in the
    // output. That flag matters because <tool_call> / </tool_call> are
    // registered special tokens (151657/151658) — skipping them would strip
    // the markers and leave a bare JSON object behind, which the HTTP layer
    // would report as prose instead of a tool call.
    //
    // `should_stop` is polled between decode steps. A client that has already
    // parsed a usable tool call out of the stream must not have to pay for the
    // rest of max_new: at hundreds of ms per token that is minutes of GPU time
    // for text it will throw away.
    std::string generate(const std::vector<uint32_t> &ids, int max_new,
                         std::vector<uint32_t> &out_ids,
                         const std::function<void(uint32_t)> &on_token,
                         const SampleParams &sp,
                         const std::function<bool()> &should_stop = nullptr) {
        // Rewind the RNG every round: with a fixed --seed the same prompt and
        // the same parameters must give the same reply, which a process-wide
        // stream advancing across requests would not.
        sampler_ = Sampler(seed_);
        // The window is a hard bound on prompt + generation together. A client
        // asking for max_tokens=32000 against an 8k window is not an error —
        // clamp to what actually fits and report "length" if we hit the wall,
        // which is exactly what the API's stop_reason means. Only a prompt
        // that fills the window on its own is a genuine failure.
        // One row of headroom: decode writes a KV row per generated token, and
        // the last one lands at index max_kv_-1. Letting prompt+max_new equal
        // max_kv_ exactly would ask the final write to go one row past the
        // preallocated buffer (KVCache::resume_from rejects that, but only
        // after the prompt is already through prefill).
        const int room = max_kv_ - 1 - static_cast<int>(ids.size());
        if (room <= 0)
            throw std::runtime_error(
                "context_overflow: prompt is " + std::to_string(ids.size()) +
                " tokens, which fills the " + std::to_string(max_kv_) +
                "-row KV window; raise --max-kv or shorten the conversation");
        if (max_new > room) {
            std::printf("[serve] max_tokens %d clamped to %d (window %d, prompt %zu)\n",
                        max_new, room, max_kv_, ids.size());
            std::fflush(stdout);
            max_new = room;
        }

        // Phase boundary: the previous request's decode left per-op shape
        // caches keyed to q_len=1 shapes. Reset them before this round's
        // ResizeInput/fillToCPU, exactly as llm_chat does.
        rt_->invalidate_replay();

        const int L = static_cast<int>(ids.size());

        // ---- prefix reuse ----
        // covered_ids_ is the sequence the KV buffer actually holds (prompt +
        // the tokens decode fed back, or the rows --kv-warm loaded). The
        // reusable part is the *longest common prefix*, not "is the whole
        // covered stream a prefix": a warm file or a new conversation usually
        // shares only the agent's stable opening -- system prompt, then tool
        // schemas -- and diverges at the first user turn. Causality makes any
        // shared run reusable (row i's KV depends only on ids[0..i], and the
        // graph is fed absolute positions A+i, so a reused row sits at the same
        // position it was computed at), and the divergence point is simply
        // where the recompute starts. Rows past it are stale bytes from
        // something else; this round overwrites them and attention never reads
        // beyond past_len.
        //
        // At least one token must stay outside the prefix: with nothing left to
        // compute there would be no Run() to produce the first logits row.
        // Same rule as llm_chat (llm_common.hpp); the server has no image
        // spans, so it is the plain longest common prefix.
        size_t match = static_cast<size_t>(
            vkop::llm::reusable_prefix(covered_ids_, ids));
        // VKOP_SERVE_PREFIX_DEBUG: why the prefix stopped matching. A short
        // reuse is otherwise indistinguishable from "the client sent a
        // different prompt" -- which is what it was blamed on once -- so dump
        // the ids and their pieces on both sides of the divergence.
        if (const char *pd = std::getenv("VKOP_SERVE_PREFIX_DEBUG")) {
            if (pd[0] != '\0' && pd[0] != '0') {
                const size_t lo = match > 40 ? match - 40 : 0;
                const size_t hi = std::min(match + 40,
                                           std::max(covered_ids_.size(),
                                                    ids.size()));
                std::fprintf(stderr,
                             "[prefix] covered=%zu incoming=%zu match=%zu\n",
                             covered_ids_.size(), ids.size(), match);
                for (size_t i = lo; i < hi; ++i) {
                    const bool have_a = i < covered_ids_.size();
                    const bool have_b = i < ids.size();
                    const uint32_t a = have_a ? covered_ids_[i] : 0;
                    const uint32_t b = have_b ? ids[i] : 0;
                    const std::string pa =
                        have_a ? tok_.id_to_piece(a, false) : std::string("<end>");
                    const std::string pb =
                        have_b ? tok_.id_to_piece(b, false) : std::string("<end>");
                    const char *mark = (i == match) ? " <-- diverges" : "";
                    std::fprintf(
                        stderr, "[prefix] %6zu %s %6u %-28s | %6u %-28s%s\n",
                        i, (have_a && have_b && a == b) ? "=" : "!", a,
                        pa.c_str(), b, pb.c_str(), mark);
                }
                std::fflush(stderr);
            }
        }
        // A hybrid decoder's conv/recurrent state is a lossy accumulator: after
        // a round it corresponds to the END of covered_ids_, not to any position
        // inside it. So reusing P rows of KV while the state sits elsewhere
        // makes the linear layers see a different history from the attention
        // layers. When the reuse is only partial the state has to be rebuilt,
        // which means recomputing from 0.
        //
        // VKOP_NO_PREFIX_REUSE=1 forces it unconditionally, for A/B runs.
        if ((lin_state_->present() && match < covered_ids_.size()) ||
            std::getenv("VKOP_NO_PREFIX_REUSE")) {
            match = 0;
        }
        const int P = static_cast<int>(match);
        const int S = L - P;
        // VKOP_PREFIX_TRACE: the reuse decision, AFTER any override, with the
        // numbers the round will actually use. The per-id dump above runs
        // before the override and so cannot answer "did disabling reuse take".
        if (std::getenv("VKOP_PREFIX_TRACE")) {
            std::fprintf(stderr,
                         "[ptrace] covered=%zu incoming=%zu -> P=%d S=%d "
                         "state=%d reuse_off=%d\n",
                         covered_ids_.size(), ids.size(), P, S,
                         lin_state_->present() ? 1 : 0,
                         std::getenv("VKOP_NO_PREFIX_REUSE") ? 1 : 0);
            std::fflush(stderr);
        }
        // Reusing less than the whole covered stream means this round's
        // prefill writes over rows covered_ids_ still claims. Until the round
        // finishes, that claim is not trustworthy (see invalidate_prefix_state).
        prefix_trusted_ = match >= covered_ids_.size();

        // ---- prefill ----
        // Chunked, because the graph's logits output is (1, q, vocab) and
        // q == prompt length would allocate q*vocab*2 bytes: at 11k prompt
        // tokens and a 151936-entry vocab that is 3.3 GB for a row nobody
        // reads except the last one. Each chunk writes its own KV rows and
        // only the final chunk's logits are used for the first token. This is
        // the same device llm_chat's VKOP_PREFILL_CHUNK uses; here it is not
        // optional, because an agent's system prompt is long by construction.
        const int C_max = std::max(1, prefill_chunk_);
        const int n_chunks = (S + C_max - 1) / C_max;
        int past_len = P;
        double last_run_ms = 0;
        // Whole-prefill wall clock, host work included. last_run_ms is the
        // LAST chunk's Run() only, so printing it alone as "the prefill"
        // understates a multi-chunk prompt by however many chunks there were.
        // Both are printed because they answer different questions: this one
        // is what the client waits for before the first token, last_run_ms is
        // the per-dispatch cost at the end of the context.
        const auto prefill_t0 = std::chrono::steady_clock::now();
        for (int blk = 0; blk < n_chunks; ++blk) {
            const int A = P + blk * C_max;                     // absolute start
            const int C = std::min(C_max, P + S - A);

            const std::vector<uint32_t> chunk_ids(ids.begin() + A,
                                                  ids.begin() + A + C);
            auto emb = embed_lookup(embed_table_, chunk_ids, vocab_, arch_.hidden);
            fill_fp16_input(rt_, "inputs_embeds",
                            {1u, static_cast<uint32_t>(C),
                             static_cast<uint32_t>(arch_.hidden)},
                            emb.data());

            if (arch_.position_ids_dims == 3) {
                std::vector<int64_t> pos3(static_cast<size_t>(3) * C);
                for (int c = 0; c < 3; ++c)
                    for (int i = 0; i < C; ++i)
                        pos3[static_cast<size_t>(c) * C + i] = A + i;
                fill_i64_input(rt_, "position_ids",
                               {3u, 1u, static_cast<uint32_t>(C)}, pos3.data());
            } else {
                std::vector<int64_t> pos(static_cast<size_t>(C));
                for (int i = 0; i < C; ++i) pos[i] = A + i;
                fill_i64_input(rt_, "position_ids",
                               {1u, static_cast<uint32_t>(C)}, pos.data());
            }

            auto ab = causal_bias(C, past_len + C);
            fill_fp16_input(rt_, "attention_bias",
                            {1u, 1u, static_cast<uint32_t>(C),
                             static_cast<uint32_t>(past_len + C)},
                            ab.data());

            if (arch_.has_deepstack) {
                for (int d = 0; d < 3; ++d)
                    fill_fp16_input(rt_, "deepstack_embeds_" + std::to_string(d),
                                    dec_in_->ds_shape(), dec_in_->ds_zero().data());
            }
            if (arch_.has_image_pad_mask) {
                std::vector<int8_t> mask(static_cast<size_t>(C), 0);
                rt_->ResizeInput("image_pad_mask", {1u, static_cast<uint32_t>(C)});
                auto t = rt_->GetInput("image_pad_mask");
                auto tg = as_tensor<int8_t>(t);
                if (tg->num_elements() > 0) tg->fillToCPU(mask.data());
            }

            if (past_len > 0) {
                kv_->resume_from(past_len);   // rows already in the buffer
            } else {
                kv_->reset_for_prefill();
                kv_->upload();
                // A fresh conversation starts from a zero linear-attention
                // state, exactly as it starts from an empty KV cache. Reusing
                // a prefix (past_len > 0) keeps whatever the earlier rounds
                // left, which is the point of the reuse.
                if (lin_state_->present()) {
                    lin_state_->reset();
                    lin_state_->upload();
                    if (std::getenv("VKOP_STATE_DUMP"))
                        lin_state_->dump_debug("after reset");
                }
            }
            if (lin_state_->present()) lin_state_->bind_for_round();

            upload_input(cmdpool_, rt_->GetInput("inputs_embeds"));
            upload_input(cmdpool_, rt_->GetInput("position_ids"));
            upload_input(cmdpool_, rt_->GetInput("attention_bias"));
            if (arch_.has_deepstack) {
                for (int d = 0; d < 3; ++d)
                    upload_input(cmdpool_,
                                 rt_->GetInput("deepstack_embeds_" + std::to_string(d)));
            }
            if (arch_.has_image_pad_mask)
                upload_input(cmdpool_, rt_->GetInput("image_pad_mask"));

            // Only the LAST chunk's logits are read (the first generated token
            // comes from it; every earlier chunk writes KV rows nobody samples
            // from). Declaring that lets Run() drop the LM head and the GPU
            // TopK on this chunk -- together the two largest single ops in a
            // prefill chunk (~39 ms + ~27 ms at q=256), so a 28-chunk 8k prompt
            // stops paying them 27 times. The skip is derived from the graph's
            // own dependencies in Runtime::compute_live_levels, so an output
            // this model does not have is simply not skipped.
            //
            // VKOP_SKIP_DEAD_LOGITS=0 is the A/B arm and the escape hatch: it
            // restores the old "every chunk computes logits" behaviour.
            static const bool skip_dead_logits = [] {
                const char *e = std::getenv("VKOP_SKIP_DEAD_LOGITS");
                return !(e && e[0] == '0');
            }();
            if (skip_dead_logits && blk + 1 < n_chunks) {
                rt_->set_unused_outputs({"logits", "post_TopK0", "post_TopK1"});
            }
            last_run_ms = rt_->Run();
            rt_->invalidate_replay();
            dev_->wait_all_done();
            past_len = kv_->feedback();
            // The state swap must wait for the GPU: the buffer this round read
            // is the one the NEXT round writes over.
            if (lin_state_->present()) lin_state_->advance();
        }
        if (lin_state_->present() && std::getenv("VKOP_STATE_DUMP"))
            lin_state_->dump_debug("after prefill");

        out_ids.clear();
        // Unique generated ids, in first-seen order, for the repetition
        // penalty. The marker array is only allocated when the penalty is on.
        std::vector<uint32_t> penalised;
        std::vector<uint8_t> seen;
        if (sp.repetition_penalty != 1.0f && vocab_ > 0)
            seen.assign(static_cast<size_t>(vocab_), 0);
        const auto note = [&](uint32_t id) {
            if (!seen.empty() && id < seen.size() && !seen[id]) {
                seen[id] = 1;
                penalised.push_back(id);
            }
        };
        int next_id = sample_last_token(rt_, cmdpool_, vocab_, topk_k_, sampler_,
                                        penalised, sp);
        out_ids.push_back(static_cast<uint32_t>(next_id));
        note(static_cast<uint32_t>(next_id));
        if (!is_stop(static_cast<uint32_t>(next_id)))
            on_token(static_cast<uint32_t>(next_id));

        const double prefill_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefill_t0).count();
        std::printf("[serve] prefill L=%d (reuse %d, %d chunk(s)) %.0fms "
                    "total, last chunk %.0fms -> %d\n",
                    L, P, n_chunks, prefill_ms, last_run_ms, next_id);
        if (pager_) {
            // Paging cost is the other half of the trade: the arena freed the
            // VRAM, and this is what it charged per prefill.
            std::printf("[serve] paging: %zu page-in(s), %.1f MiB moved "
                        "(host %.0f ms + device %.0f ms)\n",
                        page_calls_, page_bytes_ / 1048576.0, page_host_ms_,
                        page_device_ms_);
            page_calls_ = page_bytes_ = 0;
            page_host_ms_ = page_device_ms_ = 0.0;
        }
        std::fflush(stdout);

        // ---- decode ----
        std::string reason = "length";
        if (is_stop(static_cast<uint32_t>(next_id))) {
            reason = "stop";
        } else {
            // Host-imported decode inputs (llm_chat's zero-copy path): each
            // step writes its embedding/position straight into device-visible
            // memory instead of a staging submit+wait. Re-installed every
            // round because a prefill wider than the import's capacity drops
            // it. Hybrid (linear-attention) models keep the batched upload
            // until that combination has been checked token-for-token;
            // VKOP_DECODE_ZEROCOPY=1 opts them in.
            const char *zc_env = std::getenv("VKOP_DECODE_ZEROCOPY");
            const bool zc_forced = zc_env && std::strcmp(zc_env, "1") == 0;
            if (!lin_state_->present() || zc_forced)
                dec_in_->install(max_kv_);
            for (int step = 1; step < max_new; ++step) {
                // Checked here rather than after on_token: the previous step
                // already fed its token back and streamed it, so out_ids still
                // holds exactly one token the KV cache has not consumed.
                if (should_stop && should_stop()) {
                    reason = "stop";
                    break;
                }
                auto cur = embed_lookup(embed_table_,
                                        {static_cast<uint32_t>(next_id)},
                                        vocab_, arch_.hidden);
                // Imported inputs are written in place; the rest go up in one
                // batched submit (same bytes, same staging path as before).
                dec_in_->step(cur.data(), past_len, past_len + 1);

                if (lin_state_->present()) lin_state_->bind_for_round();
                const double ms = rt_->Run();
                next_id = sample_last_token(rt_, cmdpool_, vocab_, topk_k_, sampler_,
                                            penalised, sp);
                out_ids.push_back(static_cast<uint32_t>(next_id));
                note(static_cast<uint32_t>(next_id));
                past_len = kv_->feedback();
                // sample_last_token reads logits back, which is a host
                // round-trip, so the GPU is idle here and the swap is safe.
                if (lin_state_->present()) lin_state_->advance();
                if (std::getenv("VKOP_SERVE_VERBOSE")) {
                    std::printf("[serve] r%d %.0fms past=%d -> %d\n", step, ms,
                                past_len, next_id);
                    std::fflush(stdout);
                }
                if (is_stop(static_cast<uint32_t>(next_id))) {
                    reason = "stop";
                    break;
                }
                on_token(static_cast<uint32_t>(next_id));
            }
        }

        // What the KV cache now covers: the prompt plus every token decode
        // fed back. The stop token was generated but never consumed, so it is
        // not part of it — matching kv_->feedback()'s row count.
        if (sampler_.fast_picks + sampler_.full_picks > 0 &&
            (sp.temperature > 0.0f || sp.repetition_penalty != 1.0f)) {
            std::printf("[serve] sampling: %d step(s) from the GPU top-%d, %d "
                        "needed the full row\n", sampler_.fast_picks, topk_k_,
                        sampler_.full_picks);
            std::fflush(stdout);
        }
        covered_ids_ = ids;
        const size_t fed = out_ids.size() > 0 ? out_ids.size() - 1 : 0;
        covered_ids_.insert(covered_ids_.end(), out_ids.begin(),
                            out_ids.begin() + static_cast<std::ptrdiff_t>(fed));
        prefix_trusted_ = true;
        return reason;
    }

    void reset() {
        invalidate_prefix_state();
        rt_->invalidate_replay();
    }

    // A round that threw after its prefill started leaves cache rows that
    // belong to neither the old stream nor the new one: longest-common-prefix
    // reuse means the prefill had already overwritten [P, covered) when the
    // failure happened. Keeping covered_ids_ would let the next round splice
    // onto bytes that are no longer there, so the whole conversation claim goes
    // with it and the following round re-prefills from row 0. The cache FILE is
    // not touched -- its rows still are what its own id stream produced.
    void invalidate_prefix_state() {
        covered_ids_.clear();
        prev_turns_.clear();
        prev_prefix_ids_.clear();
        last_render_turns_.clear();
        prefix_trusted_ = true;
    }

    // False between "this round may have overwritten the rows covered_ids_
    // claims" and "the round finished and restated them". See the catch in
    // main(): a failed round must not leave a prefix claim behind.
    bool prefix_trusted() const { return prefix_trusted_; }

    Tokenizer &tok() { return tok_; }

  private:
    // Skip the write when the KV really does not cover more than the file
    // already does: a round served entirely from --kv-warm leaves both lengths
    // equal, and re-serialising the same rows would be pure disk traffic.
    void dump_if_longer_() {
        if (kv_dump_path_.empty()) return;
        if (covered_ids_.size() <= kv_file_rows_) return;
        dump_kv(kv_dump_path_);
    }

    static ElemKind parse_kind(const std::string &s, ElemKind fallback) {
        if (s == "fp16") return ElemKind::kFloat16;
        if (s == "bf16") return ElemKind::kBFloat16;
        if (s == "fp8e4m3") return ElemKind::kFloat8E4M3FN;
        if (s == "fp8e5m2") return ElemKind::kFloat8E5M2;
        if (s == "int4") return ElemKind::kInt4;
        if (s == "nf4") return ElemKind::kNF4;
        if (s == "auto") return fallback;
        throw std::runtime_error("unknown --kv-kind " + s);
    }

    bool is_stop(uint32_t id) const {
        return std::find(stop_ids_.begin(), stop_ids_.end(), id) !=
               stop_ids_.end();
    }

  public:
    // Whether a generated (non-stop) token belongs in the reply text. Decoding
    // runs with skip_special=false so the tool-call markers survive, which
    // also let every other control token (<|vision_start|>, <|image_pad|>,
    // ...) through as literal text. Only the allow-listed markers pass now.
    bool emits_text(uint32_t id) const {
        if (!tok_.id_to_piece(id, /*skip_special=*/true).empty()) return true;
        if (tok_.id_to_piece(id, /*skip_special=*/false).empty()) return true;
        return std::find(text_specials_.begin(), text_specials_.end(), id) !=
               text_specials_.end();
    }

  private:

    int max_kv_;
    // k of the GPU TopK post-process (0: none, 1: argmax only).
    int topk_k_ = 0;
    // Prompt tokens per prefill dispatch. Bounds the graph's (1, q, vocab)
    // logits output, which is q*vocab*2 bytes: at 2048 tokens and a 152k vocab
    // that is ~620 MB per dispatch, at 8192 it is ~2.5 GB. Larger is faster
    // (every chunk re-runs lm_head's [q,2048]x[2048,151936] MatMul), but the
    // buffer has to actually fit; VKOP_SERVE_CHUNK / --prefill-chunk override.
    // 8192 splits an 11k agent prompt into two dispatches instead of six.
    int prefill_chunk_ = vkop::llm::kDefaultPrefillChunk;
    // The artifact this server loaded, kept because a cache file is only valid
    // for it: a .vkopbin reconverted with different shader keys, or with a
    // different --kv-kind, holds KV bytes that mean something else.
    std::string model_path_;
    // On-disk prefix cache (--kv-dump); empty means the feature is off.
    std::string kv_dump_path_;
    // Rows the cache file currently holds, and whether the first round has
    // been written yet (see dump_after_first_round()).
    size_t kv_file_rows_ = 0;
    bool kv_dump_pending_ = true;
    std::shared_ptr<VulkanDevice> dev_;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool_;
    std::shared_ptr<Runtime> rt_;
    // Layer paging (--page-weights). Declared AFTER rt_ on purpose: the pager
    // dereferences the runtime on every page-in, so it must be destroyed first.
    std::unique_ptr<vkop::core::WeightPager> pager_;
    // "", "none", "all" or "float:<MiB>" -- see the plan in the constructor.
    std::string page_policy_;
    // Page-in accounting, written from the level hook (so members, not locals).
    size_t page_calls_ = 0;
    size_t page_bytes_ = 0;
    double page_host_ms_ = 0.0;
    double page_device_ms_ = 0.0;
    Tokenizer tok_;
    ModelArch arch_;
    std::vector<uint16_t> embed_table_;
    int vocab_ = 0;
    Sampler sampler_;
    uint64_t seed_ = 0;
    SampleParams defaults_;
    std::unique_ptr<KVCache> kv_;
    // Conv/recurrent state of a hybrid decoder's linear-attention layers. Null
    // for a dense model -- nothing in the dense path touches it.
    std::unique_ptr<LinearState> lin_state_;
    std::vector<uint32_t> stop_ids_;
    std::vector<uint32_t> text_specials_;
    std::vector<uint32_t> covered_ids_;
    // False while a round is overwriting rows covered_ids_ still claims.
    bool prefix_trusted_ = true;
    // The turn list of the last request and the exact id stream its KV holds
    // (prompt + the tokens decode fed back + the assistant turn's terminator).
    // The next request splices onto this instead of re-tokenizing, because
    // re-encoding an assistant reply does not reproduce the ids the model
    // wrote: BBPE's word-start marker is lossy through decode, so a round trip
    // changes the token boundaries and the prefix comparison then fails every
    // round -- which is exactly what re-prefilling 11k tokens every time costs.
    std::vector<qwen::ChatMessage> prev_turns_;
    std::vector<uint32_t> prev_prefix_ids_;
    // Remember the turn list this render started from. The id stream itself is
    // captured after generation, in record_turn(), because only then is it
    // known how many tokens the model actually wrote.
    std::vector<qwen::ChatMessage> last_render_turns_;
    // Per-step decode input writer (zero-copy where the device allows).
    std::unique_ptr<vkop::llm::DecodeInputs> dec_in_;
};

void emit(int fd, const std::string &line) {
    std::string s = line;
    s.push_back('\n');
    const char *p = s.data();
    size_t left = s.size();
    while (left > 0) {
        const auto n = ::write(fd, p, static_cast<io_count_t>(left));
        if (n <= 0) return;
        p += n;
        left -= static_cast<size_t>(n);
    }
}

const char *kUsage =
    "usage: %s <llm.vkopbin> <embed_tokens.bin> <tokenizer.bin> "
    "[--max-kv N|auto] [--context-length N] [--kv-budget-mb MB] [--kv-kind {auto,fp16,fp8e4m3,fp8e5m2,int4,nf4}] "
    "[--prefill-chunk N] [--kv-warm FILE] [--kv-dump FILE] "
    "[--page-weights {all|float:MiB|none}] "
    "[--repetition-penalty F] [--seed N] [--sample-topk K] "
    "[--rpc-in FD] [--rpc-out FD]\n"
    "\n"
    "  --page-weights  stream the single-consuming-level weights from the\n"
    "                  artifact through one shared VRAM arena instead of\n"
    "                  keeping them all resident (needs the disk read to be\n"
    "                  fast: every token re-reads every paged band).\n"
    "                  all        page every pageable weight (smallest VRAM)\n"
    "                  float:MiB  keep the largest bands resident up to this\n"
    "                             budget, page the rest\n"
    "                  none       off (default: every weight resident)\n";

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, kUsage, argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string embed_path = argv[2];
    const std::string tok_path = argv[3];
    int max_kv = 8192;   // 0 = auto (see resolve_window_)
    int context_length = 0;
    double kv_budget_mb = 0;
    int prefill_chunk = vkop::llm::kDefaultPrefillChunk;
    int sample_topk = kDefaultSampleTopK;
    if (const char *e = std::getenv("VKOP_SAMPLE_TOPK")) sample_topk = std::atoi(e);
    int rpc_in = 3, rpc_out = 4;
    std::string kv_kind = "auto";
    std::string kv_warm, kv_dump;
    // "" = paging off (the historical behaviour, every weight resident).
    std::string page_policy;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max-kv" && i + 1 < argc) {
            const std::string v = argv[++i];
            max_kv = (v == "auto") ? 0 : std::atoi(v.c_str());
            if (v != "auto" && max_kv <= 0) {
                std::fprintf(stderr, "[serve] --max-kv must be > 0 or auto\n");
                return 1;
            }
        }
        else if (a == "--context-length" && i + 1 < argc)
            context_length = std::atoi(argv[++i]);
        else if (a == "--kv-budget-mb" && i + 1 < argc)
            kv_budget_mb = std::atof(argv[++i]);
        else if (a == "--kv-kind" && i + 1 < argc) kv_kind = argv[++i];
        else if (a == "--page-weights" && i + 1 < argc) page_policy = argv[++i];
        else if (a == "--prefill-chunk" && i + 1 < argc) prefill_chunk = std::atoi(argv[++i]);
        else if (a == "--kv-warm" && i + 1 < argc) kv_warm = argv[++i];
        else if (a == "--kv-dump" && i + 1 < argc) kv_dump = argv[++i];
        else if (a == "--repetition-penalty" && i + 1 < argc)
            repetition_penalty = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--sample-topk" && i + 1 < argc)
            sample_topk = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc)
            seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--rpc-in" && i + 1 < argc) rpc_in = std::atoi(argv[++i]);
        else if (a == "--rpc-out" && i + 1 < argc) rpc_out = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            std::fprintf(stderr, kUsage, argv[0]);
            return 1;
        }
    }

#ifdef _WIN32
    // The parent passes HANDLE values here, not CRT fds: Windows has no
    // pass_fds, so it marks the pipe ends inheritable and sends the handles.
    rpc_in = _open_osfhandle(static_cast<intptr_t>(rpc_in), _O_RDONLY);
    rpc_out = _open_osfhandle(static_cast<intptr_t>(rpc_out), _O_WRONLY);
    if (rpc_in < 0 || rpc_out < 0) {
        std::fprintf(stderr, "[serve] cannot wrap the inherited RPC handles\n");
        return 1;
    }
#endif

    // The fds may be anything the parent handed us; the loop is synchronous,
    // so a single request at a time is the contract.
    std::unique_ptr<InferenceServer> srv;
    try {
        srv = std::make_unique<InferenceServer>(model_path, embed_path, tok_path,
                                                max_kv, kv_kind, prefill_chunk,
                                                kv_warm, repetition_penalty, seed,
                                                sample_topk, context_length,
                                                kv_budget_mb, page_policy);
        srv->set_kv_dump_path(kv_dump);
    } catch (const std::exception &e) {
        emit(rpc_out, std::string("FATAL ") + b64_encode(e.what()));
        std::fprintf(stderr, "[serve] startup failed: %s\n", e.what());
        return 1;
    }
    // READY <abi> <window>: the second field is the KV window actually held
    // (after --max-kv auto / clamping); older HTTP layers read only the first.
    const std::string ready = "READY 3 " + std::to_string(srv->max_kv());
    emit(rpc_out, ready);

    LineReader in(rpc_in);
    std::string line;
    while (in.getline(line)) {
        if (line.empty()) continue;
        // Command word is everything up to the first space.
        const size_t sp = line.find(' ');
        const std::string cmd = line.substr(0, sp);
        std::vector<std::string> args;
        {
            size_t pos = sp == std::string::npos ? line.size() : sp + 1;
            while (pos < line.size()) {
                const size_t nx = line.find(' ', pos);
                args.push_back(line.substr(pos, nx == std::string::npos
                                                    ? std::string::npos
                                                    : nx - pos));
                if (nx == std::string::npos) break;
                pos = nx + 1;
            }
        }
        if (cmd == "BYE") {
            // One last write if this conversation got further than the file:
            // cheap here, and the next session starts that much deeper into its
            // own prefix.
            try {
                srv->dump_on_exit();
            } catch (const std::exception &e) {
                std::fprintf(stderr, "[serve] %s\n", e.what());
                std::fflush(stderr);
            }
            emit(rpc_out, "BYE");
            break;
        }
        if (cmd == "HELLO") { emit(rpc_out, ready); continue; }
        if (cmd == "RESET") {
            try { srv->reset(); emit(rpc_out, "OK"); }
            catch (const std::exception &e) {
                emit(rpc_out, "ERROR 0 " + b64_encode(e.what()));
            }
            continue;
        }
        if (cmd != "GEN" && cmd != "COUNT") {
            emit(rpc_out, "ERROR 0 " + b64_encode("unknown command: " + cmd));
            continue;
        }
        // args: <id> [max_new] <nmsg> [temperature top_k top_p]
        const std::string id = args.empty() ? "0" : args[0];
        int max_new = 64;
        size_t nmsg = 0;
        // Negative means the client never sent the parameter, so the backend
        // default stands -- which is greedy.
        float req_temp = -1.0f, req_top_p = -1.0f;
        int req_top_k = -1;
        if (cmd == "GEN") {
            if (args.size() < 3) {
                emit(rpc_out, "ERROR " + id + " " +
                                  b64_encode("GEN needs <id> <max_new> <nmsg>"));
                continue;
            }
            max_new = std::atoi(args[1].c_str());
            nmsg = static_cast<size_t>(std::atoi(args[2].c_str()));
            if (args.size() > 3)
                req_temp = static_cast<float>(std::atof(args[3].c_str()));
            if (args.size() > 4) req_top_k = std::atoi(args[4].c_str());
            if (args.size() > 5)
                req_top_p = static_cast<float>(std::atof(args[5].c_str()));
        } else {
            if (args.size() < 2) {
                emit(rpc_out, "ERROR " + id + " " +
                                  b64_encode("COUNT needs <id> <nmsg>"));
                continue;
            }
            nmsg = static_cast<size_t>(std::atoi(args[1].c_str()));
        }

        std::vector<std::pair<std::string, std::string>> msgs;
        msgs.reserve(nmsg);
        bool framing_ok = true;
        for (size_t k = 0; k < nmsg; ++k) {
            std::string role, text;
            if (!in.getline(role) || !in.getline(text)) {
                framing_ok = false;
                break;
            }
            msgs.emplace_back(std::move(role), b64_decode(text));
        }
        if (!framing_ok) {
            emit(rpc_out, "ERROR " + id + " " + b64_encode("truncated message block"));
            break;  // the stream is desynchronised; nothing can be trusted after
        }

        try {
            // begin_round, not render: it also remembers the turn list, which
            // is what record_turn() below splices from. COUNT must not touch
            // that state (it is a different request that never generates).
            std::vector<uint32_t> ids =
                cmd == "GEN" ? srv->begin_round(msgs) : srv->render(msgs);
            if (cmd == "COUNT") {
                emit(rpc_out, "COUNTED " + id + " " + std::to_string(ids.size()));
                continue;
            }
            emit(rpc_out, "TOKENS_IN " + id + " " + std::to_string(ids.size()));

            // Streaming deltas: emit_delta turns the accumulated id list into
            // well-formed UTF-8 increments, holding a split codepoint back
            // until its remaining bytes arrive. The callback owns the id
            // stream it feeds the decoder, so generate() only has to say which
            // token it produced.
            std::vector<uint32_t> out_ids;
            std::vector<uint32_t> emitted_ids;
            std::string streamed;   // exactly what the client saw
            qwen::StreamState st;
            auto cb = [&](uint32_t tok) {
                if (!srv->emits_text(tok)) return;
                emitted_ids.push_back(tok);
                // skip_special=false: <tool_call> / </tool_call> are registered
                // special tokens, and they are how the client sees a tool call
                // at all. Stop tokens never reach this callback.
                const std::string d =
                    srv->tok().emit_delta(st, emitted_ids, /*skip_special=*/false);
                if (!d.empty()) {
                    streamed += d;
                    emit(rpc_out, "DELTA " + id + " " + b64_encode(d));
                }
            };
            SampleParams sp = srv->default_params();
            if (req_temp >= 0.0f) sp.temperature = req_temp;
            if (req_top_k >= 0) sp.top_k = req_top_k;
            if (req_top_p >= 0.0f) sp.top_p = req_top_p;
            const std::string reason =
                srv->generate(ids, max_new, out_ids, cb, sp,
                              [&in]() { return in.poll_stop(); });
            const std::string tail = srv->tok().emit_delta_flush(st);
            if (!tail.empty()) {
                streamed += tail;
                emit(rpc_out, "DELTA " + id + " " + b64_encode(tail));
            }
            // Only a completed round leaves a spliceable state: the client
            // will echo this text back as the assistant turn.
            srv->record_turn(streamed);
            emit(rpc_out, "DONE " + id + " " + reason + " " +
                              std::to_string(out_ids.size()));
            // The expensive stable prefix is now in the KV buffer, so this is
            // the round worth writing out (no-op after the first one). Done
            // after DONE: the reply is delivered, and a cache write that fails
            // -- disk full, wrong path -- must not be reported as a generation
            // error.
            try {
                srv->dump_after_first_round();
            } catch (const std::exception &e) {
                std::fprintf(stderr, "[serve] %s\n", e.what());
                std::fflush(stderr);
            }
        } catch (const std::exception &e) {
            emit(rpc_out, "ERROR " + id + " " + b64_encode(e.what()));
            // A round that threw midway may already have rewritten cache rows
            // that covered_ids_ still claims to hold (the divergence point cut
            // the reuse short, and the new ids were not recorded yet). Forgetting
            // the prefix costs the next round a re-prefill; trusting it would
            // answer from rows the model overwrote.
            if (!srv->prefix_trusted()) srv->invalidate_prefix_state();
            // A lost GPU is terminal for this VkDevice: every later request
            // fails the same way while the parent still believes it owns a
            // live backend. Exit instead of answering 500 forever -- the
            // front end then sees a dead child it can report or respawn.
            if (std::strstr(e.what(), "VK_ERROR_DEVICE_LOST") != nullptr) {
                std::fprintf(stderr,
                             "[serve] GPU device lost mid-request; exiting. "
                             "Restart to recover (a wide --prefill-chunk is "
                             "the known trigger on this driver).\n");
                std::fflush(stdout);
                return 4;
            }
        }
        std::fflush(stdout);
    }
    return 0;
}
