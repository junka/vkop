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
//   HELLO                              →  READY <abi>
//   COUNT <id> <nmsg>                    → COUNTED <id> <tokens>
//     <role>\n<text-b64>\n  × nmsg
//   GEN <id> <max_new> <nmsg>            → TOKENS_IN <id> <n>
//     <role>\n<text-b64>\n  × nmsg       → DELTA <id> <text-b64>   (0..N 次)
//                                        → DONE <id> <stop_reason> <n_out>
//   出错                                 → ERROR <id> <msg-b64>
//   RESET                              → OK
//   BYE                                → BYE
//
// stop_reason: "stop"（命中停止符）/ "length"（max_new 用尽）。
//
// 与 llm_chat 的关系：prefill/decode 的数值路径与它逐位同构（同样的
// embed 查表、causal bias、KVCache 语义、present→past 只改逻辑长度），但**故意
// 不带**滑窗环、分块 prefill、host-imported 零拷贝输入、GPU argmax 后处理。
// 那些是性能层，任何一个出问题都会静默错答案；服务端先要正确，快的那部分等
// 行为被验证过再接。跨请求的 KV 前缀复用保留：下一段的 ids 以上一段覆盖的序列
// 为前缀时只算新增的 token —— Claude Code 每轮都把整段历史重发，省下的正是
// 「整个上下文重新 prefill」。
//
// Build: `make llm_serve`（ENABLE_LLM_CHAT 打开时一起构建）。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "tokenizer.hpp"
#include "kv_cache.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::core::ElemKind;
using vkop::core::elem_name;
using vkop::export_::KVCache;
using qwen::Tokenizer;

namespace {

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
            const ssize_t n = ::read(fd_, tmp, sizeof(tmp));
            if (n <= 0) return false;
            buf_.append(tmp, static_cast<std::size_t>(n));
        }
    }

  private:
    int fd_;
    std::string buf_;
    std::size_t pos_ = 0;
};

// --------------------------------------------------------------- model ----
constexpr int LEGACY_HIDDEN = 2048;
constexpr uint16_t FP16_MIN = 0xFBFF;  // -65504, Qwen's causal-mask fill

struct ModelArch {
    int hidden = LEGACY_HIDDEN;
    int nkv = 8;
    int hd = 128;
    int nlayers = 28;
    int position_ids_dims = 2;
    bool has_deepstack = false;
    bool has_image_pad_mask = false;
    ElemKind cache_kind = ElemKind::kFloat16;
};

inline float fp16_to_f32(uint16_t h) { return ITensor::fp16_to_fp32(h); }

ModelArch infer_arch(const std::shared_ptr<Runtime> &rt) {
    ModelArch a;
    int n = 0;
    while (rt->GetInput("past_key_values_" + std::to_string(n))) ++n;
    if (n == 0) throw std::runtime_error("model has no past_key_values_* inputs");
    a.nlayers = n;
    auto pk0 = rt->GetInput("past_key_values_0");
    auto s = pk0->getShape();
    if (s.size() >= 3) a.nkv = static_cast<int>(s[2]);
    if (s.size() >= 5) a.hd = static_cast<int>(s[4]);
    a.cache_kind = pk0->elem_kind();
    if (a.cache_kind != ElemKind::kFloat16 &&
        a.cache_kind != ElemKind::kFloat8E4M3FN &&
        a.cache_kind != ElemKind::kFloat8E5M2 &&
        a.cache_kind != ElemKind::kInt4 && a.cache_kind != ElemKind::kNF4) {
        std::fprintf(stderr, "[serve] cache dtype %s unsupported, using fp16\n",
                     elem_name(a.cache_kind));
        a.cache_kind = ElemKind::kFloat16;
    }
    auto emb = rt->GetInput("inputs_embeds");
    auto es = emb->getShape();
    if (es.size() >= 3) a.hidden = static_cast<int>(es[es.size() - 1]);
    auto pid = rt->GetInput("position_ids");
    a.position_ids_dims = pid ? static_cast<int>(pid->getShape().size()) : 2;
    a.has_deepstack = rt->GetInput("deepstack_embeds_0") != nullptr;
    a.has_image_pad_mask = rt->GetInput("image_pad_mask") != nullptr;
    return a;
}

std::pair<std::vector<uint16_t>, int> load_embed_table(const std::string &path,
                                                       int hidden) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    const size_t row = static_cast<size_t>(hidden) * sizeof(uint16_t);
    if (row == 0) throw std::runtime_error("hidden must be > 0");
    if (bytes % row != 0) {
        std::fprintf(stderr,
                     "[serve] embed table %zu bytes / row %zu -> truncated vocab\n",
                     bytes, row);
    }
    const int vocab = static_cast<int>(bytes / row);
    std::vector<uint16_t> buf(static_cast<size_t>(vocab) * hidden);
    f.read(reinterpret_cast<char *>(buf.data()),
           static_cast<std::streamsize>(buf.size() * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("short read on " + path);
    return {std::move(buf), vocab};
}

void fill_fp16_input(const std::shared_ptr<Runtime> &rt, const std::string &name,
                     const std::vector<uint32_t> &shape, const uint16_t *data) {
    rt->ResizeInput(name, shape);
    auto t = rt->GetInput(name);
    auto tg = as_tensor<uint16_t>(t);
    if (tg->num_elements() > 0) tg->fillToCPU(data);
}

void fill_i64_input(const std::shared_ptr<Runtime> &rt, const std::string &name,
                    const std::vector<uint32_t> &shape, const int64_t *data) {
    rt->ResizeInput(name, shape);
    auto t = rt->GetInput(name);
    auto tg = as_tensor<int64_t>(t);
    if (tg->num_elements() > 0) tg->fillToCPU(data);
}

void upload_input(const std::shared_ptr<vkop::VulkanCommandPool> &cp,
                  const std::shared_ptr<ITensor> &t) {
    if (t->dtype() == typeid(int64_t))
        as_tensor<int64_t>(t)->copyToGPU(cp);
    else if (t->dtype() == typeid(int))
        as_tensor<int>(t)->copyToGPU(cp);
    else if (t->dtype() == typeid(int8_t))
        as_tensor<int8_t>(t)->copyToGPU(cp);
    else if (t->dtype() == typeid(float))
        as_tensor<float>(t)->copyToGPU(cp);
    else
        as_tensor<uint16_t>(t)->copyToGPU(cp);
}

std::vector<uint16_t> causal_bias(int q, int kv) {
    std::vector<uint16_t> m(static_cast<size_t>(q) * kv, 0);
    for (int i = 0; i < q; ++i) {
        const int threshold = i + (kv - q);
        for (int j = threshold + 1; j < kv; ++j)
            m[static_cast<size_t>(i) * kv + j] = FP16_MIN;
    }
    return m;
}

std::vector<uint16_t> embed_lookup(const std::vector<uint16_t> &table,
                                   const std::vector<uint32_t> &ids, int vocab,
                                   int hidden) {
    std::vector<uint16_t> out(ids.size() * hidden, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] >= static_cast<uint32_t>(vocab)) {
            std::fprintf(stderr, "[serve] token %u >= vocab %d (zeroed)\n",
                         ids[i], vocab);
            continue;
        }
        std::memcpy(&out[i * hidden], &table[static_cast<size_t>(ids[i]) * hidden],
                    hidden * sizeof(uint16_t));
    }
    return out;
}

int argmax_last_token(const std::shared_ptr<Runtime> &rt,
                      const std::shared_ptr<vkop::VulkanCommandPool> &cp,
                      int fallback_vocab) {
    auto logits = rt->GetOutput("logits");
    if (!logits) throw std::runtime_error("no 'logits' output");
    auto lg = as_tensor<uint16_t>(logits);
    lg->copyToCPU(cp);
    const uint16_t *p = reinterpret_cast<const uint16_t *>(lg->data().data());
    auto shape = lg->getShape();
    int vocab = shape.empty() ? 0 : shape.back();
    int q = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
    const int total = lg->num_elements();
    if (vocab <= 0 || q <= 0) {
        vocab = fallback_vocab;
        q = vocab > 0 ? total / vocab : 1;
    }
    const uint16_t *row = p + static_cast<std::ptrdiff_t>(q - 1) * vocab;
    int best = 0;
    float bestv = -1e30f;
    for (int i = 0; i < vocab; ++i) {
        const float v = fp16_to_f32(row[i]);
        if (v > bestv) { bestv = v; best = i; }
    }
    return best;
}

// ------------------------------------------------------------- the server ----
class InferenceServer {
  public:
    InferenceServer(const std::string &model, const std::string &embed,
                    const std::string &tok_path, int max_kv,
                    const std::string &kv_kind, int prefill_chunk)
        : max_kv_(max_kv), prefill_chunk_(std::max(1, prefill_chunk)) {
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
        std::printf("[serve] === LoadModel ===\n");
        rt_->LoadModel();
        std::printf("[serve] === LoadModel done ===\n");
        arch_ = infer_arch(rt_);
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

        auto loaded = load_embed_table(embed, arch_.hidden);
        embed_table_ = std::move(loaded.first);
        vocab_ = loaded.second;
        std::printf("[serve] embed vocab=%d hidden=%d\n", vocab_, arch_.hidden);

        kv_ = std::make_unique<KVCache>(rt_, cmdpool_, arch_.nlayers, arch_.nkv,
                                        arch_.hd, max_kv_, arch_.cache_kind);
        for (int32_t id : tok_.stop_token_ids())
            stop_ids_.push_back(static_cast<uint32_t>(id));
        ds_zero_.assign(static_cast<size_t>(arch_.hidden), 0);
        ds_shape_ = {1u, static_cast<uint32_t>(arch_.hidden)};
        if (!tok_.chat_template().empty())
            std::printf("[serve] chat template loaded\n");
        else
            std::printf("[serve] WARNING: tokenizer has no chat template\n");
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
    std::string generate(const std::vector<uint32_t> &ids, int max_new,
                         std::vector<uint32_t> &out_ids,
                         const std::function<void(uint32_t)> &on_token) {
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
        // the tokens decode fed back). Only a strict prefix is reusable; the
        // position of every reused row must be identical, and it is, because
        // the render is a pure concatenation of the same messages.
        int P = 0;
        // Strict prefix AND strictly shorter. A prompt that is exactly what the
        // KV already covers would leave no token to compute, so there would be
        // no Run() to produce a logits row at all; equal-length is therefore
        // excluded here and takes the full-prefill path below.
        if (!covered_ids_.empty() && covered_ids_.size() < ids.size() &&
            std::equal(covered_ids_.begin(), covered_ids_.end(), ids.begin())) {
            P = static_cast<int>(covered_ids_.size());
        }
        const int S = L - P;

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
                                    ds_shape_, ds_zero_.data());
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
            }

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

            last_run_ms = rt_->Run();
            rt_->invalidate_replay();
            dev_->wait_all_done();
            past_len = kv_->feedback();
        }

        out_ids.clear();
        int next_id = argmax_last_token(rt_, cmdpool_, vocab_);
        out_ids.push_back(static_cast<uint32_t>(next_id));
        if (!is_stop(static_cast<uint32_t>(next_id)))
            on_token(static_cast<uint32_t>(next_id));

        std::printf("[serve] prefill L=%d (reuse %d, %d chunk(s)) %.0fms -> %d\n",
                    L, P, n_chunks, last_run_ms, next_id);
        std::fflush(stdout);

        // ---- decode ----
        std::string reason = "length";
        if (is_stop(static_cast<uint32_t>(next_id))) {
            reason = "stop";
        } else {
            std::vector<uint16_t> zeros;
            for (int step = 1; step < max_new; ++step) {
                auto cur = embed_lookup(embed_table_,
                                        {static_cast<uint32_t>(next_id)},
                                        vocab_, arch_.hidden);
                fill_fp16_input(rt_, "inputs_embeds",
                                {1u, 1u, static_cast<uint32_t>(arch_.hidden)},
                                cur.data());

                const int64_t p = past_len;
                if (arch_.position_ids_dims == 3) {
                    int64_t pos[3] = {p, p, p};
                    fill_i64_input(rt_, "position_ids", {3u, 1u, 1u}, pos);
                } else {
                    fill_i64_input(rt_, "position_ids", {1u, 1u}, &p);
                }
                // All-zero full-history mask; only the shape moves.
                if (zeros.size() < static_cast<size_t>(past_len + 1))
                    zeros.assign(static_cast<size_t>(past_len + 1), 0);
                fill_fp16_input(rt_, "attention_bias",
                                {1u, 1u, 1u, static_cast<uint32_t>(past_len + 1)},
                                zeros.data());
                if (arch_.has_deepstack) {
                    for (int d = 0; d < 3; ++d)
                        fill_fp16_input(rt_, "deepstack_embeds_" + std::to_string(d),
                                        ds_shape_, ds_zero_.data());
                }
                if (arch_.has_image_pad_mask) {
                    int8_t zero = 0;
                    rt_->ResizeInput("image_pad_mask", {1u, 1u});
                    auto t = rt_->GetInput("image_pad_mask");
                    auto tg = as_tensor<int8_t>(t);
                    if (tg->num_elements() > 0) tg->fillToCPU(&zero);
                }

                upload_input(cmdpool_, rt_->GetInput("inputs_embeds"));
                upload_input(cmdpool_, rt_->GetInput("position_ids"));
                upload_input(cmdpool_, rt_->GetInput("attention_bias"));
                if (arch_.has_deepstack) {
                    for (int d = 0; d < 3; ++d)
                        upload_input(cmdpool_,
                                     rt_->GetInput("deepstack_embeds_" +
                                                   std::to_string(d)));
                }
                if (arch_.has_image_pad_mask)
                    upload_input(cmdpool_, rt_->GetInput("image_pad_mask"));

                const double ms = rt_->Run();
                next_id = argmax_last_token(rt_, cmdpool_, vocab_);
                out_ids.push_back(static_cast<uint32_t>(next_id));
                past_len = kv_->feedback();
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
        covered_ids_ = ids;
        const size_t fed = out_ids.size() > 0 ? out_ids.size() - 1 : 0;
        covered_ids_.insert(covered_ids_.end(), out_ids.begin(),
                            out_ids.begin() + static_cast<std::ptrdiff_t>(fed));
        return reason;
    }

    void reset() {
        covered_ids_.clear();
        rt_->invalidate_replay();
    }

    Tokenizer &tok() { return tok_; }

  private:
    static ElemKind parse_kind(const std::string &s, ElemKind fallback) {
        if (s == "fp16") return ElemKind::kFloat16;
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

    int max_kv_;
    // Prompt tokens per prefill dispatch. Bounds the graph's (1, q, vocab)
    // logits output, which is q*vocab*2 bytes: at 2048 tokens and a 152k vocab
    // that is ~620 MB per dispatch, at 8192 it is ~2.5 GB. Larger is faster
    // (every chunk re-runs lm_head's [q,2048]x[2048,151936] MatMul), but the
    // buffer has to actually fit; VKOP_SERVE_CHUNK / --prefill-chunk override.
    // 8192 splits an 11k agent prompt into two dispatches instead of six.
    int prefill_chunk_ = 8192;
    std::shared_ptr<VulkanDevice> dev_;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool_;
    std::shared_ptr<Runtime> rt_;
    Tokenizer tok_;
    ModelArch arch_;
    std::vector<uint16_t> embed_table_;
    int vocab_ = 0;
    std::unique_ptr<KVCache> kv_;
    std::vector<uint32_t> stop_ids_;
    std::vector<uint32_t> covered_ids_;
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
    std::vector<uint16_t> ds_zero_;
    std::vector<uint32_t> ds_shape_;
};

void emit(int fd, const std::string &line) {
    std::string s = line;
    s.push_back('\n');
    const char *p = s.data();
    size_t left = s.size();
    while (left > 0) {
        const ssize_t n = ::write(fd, p, left);
        if (n <= 0) return;
        p += n;
        left -= static_cast<size_t>(n);
    }
}

const char *kUsage =
    "usage: %s <llm.vkopbin> <embed_tokens.bin> <tokenizer.bin> "
    "[--max-kv N] [--kv-kind {auto,fp16,fp8e4m3,fp8e5m2,int4,nf4}] "
    "[--prefill-chunk N] [--rpc-in FD] [--rpc-out FD]\n";

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, kUsage, argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string embed_path = argv[2];
    const std::string tok_path = argv[3];
    int max_kv = 8192;
    int prefill_chunk = 8192;
    int rpc_in = 3, rpc_out = 4;
    std::string kv_kind = "auto";
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max-kv" && i + 1 < argc) max_kv = std::atoi(argv[++i]);
        else if (a == "--kv-kind" && i + 1 < argc) kv_kind = argv[++i];
        else if (a == "--prefill-chunk" && i + 1 < argc) prefill_chunk = std::atoi(argv[++i]);
        else if (a == "--rpc-in" && i + 1 < argc) rpc_in = std::atoi(argv[++i]);
        else if (a == "--rpc-out" && i + 1 < argc) rpc_out = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            std::fprintf(stderr, kUsage, argv[0]);
            return 1;
        }
    }
    if (max_kv <= 0) {
        std::fprintf(stderr, "[serve] --max-kv must be > 0\n");
        return 1;
    }

    // The fds may be anything the parent handed us; the loop is synchronous,
    // so a single request at a time is the contract.
    std::unique_ptr<InferenceServer> srv;
    try {
        srv = std::make_unique<InferenceServer>(model_path, embed_path, tok_path,
                                                max_kv, kv_kind, prefill_chunk);
    } catch (const std::exception &e) {
        emit(rpc_out, std::string("FATAL ") + b64_encode(e.what()));
        std::fprintf(stderr, "[serve] startup failed: %s\n", e.what());
        return 1;
    }
    emit(rpc_out, "READY 1");

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
        if (cmd == "BYE") { emit(rpc_out, "BYE"); break; }
        if (cmd == "HELLO") { emit(rpc_out, "READY 1"); continue; }
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
        // args: <id> [max_new] <nmsg>
        const std::string id = args.empty() ? "0" : args[0];
        int max_new = 64;
        size_t nmsg = 0;
        if (cmd == "GEN") {
            if (args.size() < 3) {
                emit(rpc_out, "ERROR " + id + " " +
                                  b64_encode("GEN needs <id> <max_new> <nmsg>"));
                continue;
            }
            max_new = std::atoi(args[1].c_str());
            nmsg = static_cast<size_t>(std::atoi(args[2].c_str()));
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
            const std::string reason = srv->generate(ids, max_new, out_ids, cb);
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
        } catch (const std::exception &e) {
            emit(rpc_out, "ERROR " + id + " " + b64_encode(e.what()));
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
