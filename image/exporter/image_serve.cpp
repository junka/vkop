// junka @ 2026
// image_serve —— Qwen-Image-2.1 的常驻行协议出图服务端（供 Python 的 HTTP 层 /
// 总网关驱动）。和 llm_serve 同一套约定：每行一帧，从 --rpc-in 读、往 --rpc-out
// 写，文本一律 base64，模型自己的日志照旧走 stdout/stderr，不会混进协议。
//
// 协议：
//   请求                                        响应
//   HELLO                                  →  READY <abi> <width> <height> <max_prompt_tokens>
//   COUNT <id> <prompt-b64>                →  COUNTED <id> <tokens> <max_prompt_tokens>
//   GEN <id> <steps> <seed> <n> <prompt-b64> [out_dir-b64]
//                                          →  STAGE <id> <encode|prefill|load|denoise|vae> <seconds>
//                                          →  STEP <id> <image> <step> <steps>      (每个去噪步一次)
//                                          →  IMAGE <id> <image> <seed> <png-path-b64>
//                                          →  DONE <id> <done|stop> <n_images> <metrics-json-b64>
//   出错                                   →  ERROR <id> <msg-b64>
//   STOP <id>                              →  （无独立响应；本轮以 DONE stop 收尾）
//     去噪每步之间轮询一次。已写出的图保留，正在算的那张丢弃。
//   UNLOAD                                 →  OK   释放所有常驻图（网关要把内存让给别的模型）
//   BYE                                    →  BYE
//
// 第 k 张图的种子是 seed+k；同一 prompt/seed/steps 的输出与 image_gen 逐字节相同。
//
// 为什么常驻能省时间，以及省不了什么：四张图（文本塔 13.9 GB、DiT prefill 13 GB、
// DiT decode 13 GB、VAE 0.5 GB）在 36 GB 统一内存里只能一次装一张大的，所以
// "全部常驻" 做不到。能做的是：
//   * prompt 的 DiT 文本 KV（每层 [1,2,heads,prefix,hd]，整张 ~50 MB）按 prompt
//     缓存在主机内存里。同一 prompt 再出图（换种子 / 多张 / 重试）完全不碰
//     文本塔和 prefill，decode 图也不用卸。
//   * decode 和 VAE 在请求之间保持装载（--keep decode，默认）。只有遇到没缓存过的
//     prompt 才先卸掉它们，给文本塔和 prefill 腾地方。
//   * tokenizer、查表 mmap、Vulkan 设备和管线缓存整个进程只建一次。
//
// Build: `make image_serve`（ENABLE_IMAGE_GEN 打开时一起构建）。

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "model/load.hpp"
#include "tokenizer.hpp"

#include "qwen_image.hpp"

using namespace qimg;
using vkop::VulkanInstance;

namespace {

// ------------------------------------------------------------------ I/O ----
const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string b64_encode(const std::string& in) {
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = ((uint8_t)in[i] << 16) | ((uint8_t)in[i + 1] << 8) |
                           (uint8_t)in[i + 2];
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];
        out += kB64[v & 63];
    }
    const size_t left = in.size() - i;
    if (left) {
        uint32_t v = (uint8_t)in[i] << 16;
        if (left == 2) v |= (uint8_t)in[i + 1] << 8;
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += left == 2 ? kB64[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::string b64_decode(const std::string& in) {
    static int8_t rev[256];
    static bool init = false;
    if (!init) {
        std::memset(rev, -1, sizeof(rev));
        for (int i = 0; i < 64; ++i) rev[(uint8_t)kB64[i]] = (int8_t)i;
        init = true;
    }
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    for (unsigned char ch : in) {
        const int v = rev[ch];
        if (v < 0) break;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char)((acc >> bits) & 0xFF));
        }
    }
    return out;
}

void emit(int fd, const std::string& line) {
    const std::string s = line + "\n";
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::write(fd, s.data() + off, s.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return;  // parent went away; the read side will notice too
        }
        off += (size_t)n;
    }
}

// Buffered read() over a raw fd (same as llm_serve's): getline blocks,
// poll_stop never does and only consumes a STOP line.
class LineReader {
public:
    explicit LineReader(int fd) : fd_(fd) {}
    bool getline(std::string& out) {
        for (;;) {
            const size_t nl = buf_.find('\n', pos_);
            if (nl != std::string::npos) {
                out.assign(buf_, pos_, nl - pos_);
                pos_ = nl + 1;
                if (!out.empty() && out.back() == '\r') out.pop_back();
                return true;
            }
            compact();
            if (!fill(true)) return false;
        }
    }
    bool poll_stop() {
        for (;;) {
            const size_t nl = buf_.find('\n', pos_);
            if (nl != std::string::npos) {
                std::string line(buf_, pos_, nl - pos_);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line != "STOP" && line.compare(0, 5, "STOP ") != 0) return false;
                pos_ = nl + 1;
                return true;
            }
            compact();
            if (!fill(false)) return false;
        }
    }

private:
    void compact() {
        if (pos_ > 0) {
            buf_.erase(0, pos_);
            pos_ = 0;
        }
    }
    bool fill(bool block) {
        if (!block) {
            struct pollfd p = {fd_, POLLIN, 0};
            if (::poll(&p, 1, 0) <= 0 || !(p.revents & POLLIN)) return false;
        }
        char tmp[8192];
        const ssize_t n = ::read(fd_, tmp, sizeof(tmp));
        if (n <= 0) return false;
        buf_.append(tmp, (size_t)n);
        return true;
    }
    int fd_;
    std::string buf_;
    size_t pos_ = 0;
};

double secs_since(std::chrono::high_resolution_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t)
        .count();
}

std::vector<int32_t> header_dims(const std::string& path, bool input,
                                 const std::string& name) {
    vkop::load::VkModel m(path);
    for (const auto& s : input ? m.inputs : m.outputs)
        if (s.name == name) return s.dims;
    return {};
}

void fill16(const std::shared_ptr<ITensor>& t, const std::vector<uint16_t>& d,
            const std::shared_ptr<VulkanCommandPool>& cp) {
    auto tt = as_tensor<uint16_t>(t);
    tt->fillToCPU(d.data());
    tt->copyToGPU(cp);
}
void fill32(const std::shared_ptr<ITensor>& t, const std::vector<float>& d,
            const std::shared_ptr<VulkanCommandPool>& cp) {
    auto tt = as_tensor<float>(t);
    tt->fillToCPU(d.data());
    tt->copyToGPU(cp);
}

// --------------------------------------------------------------- server ----
// A request the server rejects before touching the GPU: the resident graphs are
// still good, so the error path must not throw them away.
struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Paths {
    std::string prefill, decode, te, te_table, tok, vae, out_dir = ".";
};

// What the DiT decode needs from one prompt: the per-layer text KV the prefill
// wrote, and how many of the prefix rows are real tokens.
struct PromptKV {
    std::vector<std::vector<uint16_t>> kv;
    std::vector<std::vector<int>> shapes;
    int valid_len = 0;
    size_t bytes() const {
        size_t b = 0;
        for (const auto& l : kv) b += l.size() * sizeof(uint16_t);
        return b;
    }
};

struct GenRequest {
    std::string id, prompt, out_dir;
    int steps = 0, n = 1;
    uint32_t seed = 0;
};

struct GenEvents {
    std::function<void(const char* stage)> stage;
    std::function<void(int image, int step, int steps)> step;
    std::function<void(int image, uint32_t seed, const std::string& path)> image;
    std::function<bool()> stop;
};

class ImageServer {
public:
    ImageServer(Paths p, bool keep_resident, int prompt_cache)
        : p_(std::move(p)), keep_(keep_resident), cache_cap_(std::max(0, prompt_cache)) {
        const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        if (phydevs.empty()) throw std::runtime_error("no vulkan device");
        dev_ = std::make_shared<VulkanDevice>(phydevs[0]);
        if (dev_->getDeviceName().find("llvmpipe") != std::string::npos)
            throw std::runtime_error("no valid vulkan device");
        cp_ = std::make_shared<VulkanCommandPool>(dev_);

        fe_.open(p_.te, p_.te_table, p_.tok);

        // Every shape the four graphs were exported with is read from their
        // headers (mmap'd, no weights touched), and checked against each other
        // here, once, instead of failing mid-request after a 13 GB load.
        const auto pe = header_dims(p_.prefill, true, "prompt_embeds");
        if (pe.size() != 3) throw std::runtime_error(p_.prefill + ": no prompt_embeds input");
        prefix_len_ = pe[1];
        ctx_dim_ = pe[2];
        if (prefix_len_ != fe_.prefix_len() || ctx_dim_ != fe_.ctx_dim())
            throw std::runtime_error(
                "text tower gives (" + std::to_string(fe_.prefix_len()) + "," +
                std::to_string(fe_.ctx_dim()) + "), prefill wants (" +
                std::to_string(prefix_len_) + "," + std::to_string(ctx_dim_) +
                "): not exported as one set");
        const auto tl = header_dims(p_.decode, true, "target_latents");
        if (tl.size() != 3 || tl[1] <= 0 || tl[2] <= 0)
            throw std::runtime_error(p_.decode + ": target_latents must be static (1,L,C)");
        target_len_ = tl[1];
        latent_c_ = tl[2];
        lh_ = lw_ = (int)std::lround(std::sqrt((double)target_len_));
        if (lh_ * lw_ != target_len_)
            throw std::runtime_error("target_len " + std::to_string(target_len_) +
                                     " is not a square latent grid");
        printf("[serve] %dx%d px, latent %dx%dx%d, prompt <= %d tokens, keep=%s, "
               "prompt cache %d\n",
               width(), height(), lh_, lw_, latent_c_, prefix_len_,
               keep_ ? "decode" : "none", cache_cap_);
        fflush(stdout);
    }

    int width() const { return lw_ * 16; }
    int height() const { return lh_ * 16; }
    int max_prompt_tokens() const { return prefix_len_; }
    int count(const std::string& prompt) const { return fe_.count(prompt.c_str()); }

    void unload() {
        if (!decode_rt_ && !vae_rt_) return;
        const auto t = std::chrono::high_resolution_clock::now();
        decode_rt_.reset();
        vae_rt_.reset();
        uploaded_key_.clear();
        dev_->wait_all_done();
        cp_->reset();
        printf("[mem] resident graphs released in %.2fs\n", secs_since(t));
        fflush(stdout);
    }

    // Returns "done" or "stop"; fills `metrics` with a JSON object.
    std::string generate(const GenRequest& r, const GenEvents& ev, std::string& metrics) {
        g_run_start = std::chrono::high_resolution_clock::now();
        double t_encode = 0, t_prefill = 0, t_load = 0, t_denoise = 0, t_vae = 0;
        int steps_run = 0, images = 0;

        int valid_len = 0;
        try {
            fe_.tokenize(r.prompt.c_str(), valid_len);
        } catch (const std::exception& e) {
            throw BadRequest(e.what());  // over-long prompt: nothing loaded yet
        }

        PromptKV* kv = lookup(r.prompt);
        const bool hit = kv != nullptr;
        if (!hit) {
            // The text tower and the prefill each need ~13.9 GB: nothing
            // resident may stay while they load.
            unload();
            ev.stage("encode");
            auto t = std::chrono::high_resolution_clock::now();
            PromptEmbeds pe;
            fe_.encode(r.prompt.c_str(), cp_, dev_, pe);
            t_encode = secs_since(t);
            ev.stage("prefill");
            t = std::chrono::high_resolution_clock::now();
            kv = insert(r.prompt, run_prefill(pe));
            t_prefill = secs_since(t);
        }

        if (!decode_rt_) {
            ev.stage("load");
            const auto t = std::chrono::high_resolution_clock::now();
            decode_rt_ = std::make_shared<Runtime>(cp_, p_.decode.c_str(), /*precision=*/1);
            decode_rt_->set_backend_buffer(true);
            decode_rt_->LoadModel();
            uploaded_key_.clear();
            t_load = secs_since(t);
            printf("[load] decode loaded in %.2fs\n", t_load);
        }
        if (uploaded_key_ != r.prompt) upload_static(*kv), uploaded_key_ = r.prompt;

        const auto sigmas = official_sigmas(r.steps, target_len_);
        const int latent_n = target_len_ * latent_c_;
        std::string reason = "done";
        for (int k = 0; k < r.n && reason == "done"; ++k) {
            const uint32_t seed = r.seed + (uint32_t)k;
            std::vector<float> latent(latent_n);
            generate_random_latent(latent.data(), latent_n, seed);

            ev.stage("denoise");
            auto t = std::chrono::high_resolution_clock::now();
            std::vector<uint16_t> lat16(latent_n);
            for (int step = 0; step < r.steps; ++step) {
                if (ev.stop()) {
                    reason = "stop";
                    break;
                }
                for (int i = 0; i < latent_n; ++i) lat16[i] = ITensor::fp32_to_fp16(latent[i]);
                // Same per-step ResizeInput + fill as image_gen, so the latent
                // trajectory is bit-identical to the CLI's.
                decode_rt_->ResizeInput("target_latents",
                                        {1u, (uint32_t)target_len_, (uint32_t)latent_c_});
                fill16(decode_rt_->GetInput("target_latents"), lat16, cp_);
                decode_rt_->ResizeInput("timestep", {1u});
                fill32(decode_rt_->GetInput("timestep"), {sigmas[step]}, cp_);
                decode_rt_->Run();
                decode_rt_->ReadResult();
                auto st = as_tensor<uint16_t>(decode_rt_->GetOutput("sample"));
                st->copyToCPU(cp_);
                const auto& v = st->data();
                const float d = sigmas[step + 1] - sigmas[step];
                for (int i = 0; i < latent_n; ++i)
                    latent[i] += d * ITensor::fp16_to_fp32(v[i]);
                ++steps_run;
                ev.step(k, step + 1, r.steps);
            }
            t_denoise += secs_since(t);
            if (reason != "done") break;

            ev.stage("vae");
            t = std::chrono::high_resolution_clock::now();
            const std::string path = r.out_dir + "/" + r.id + "-" + std::to_string(k) +
                                     "-s" + std::to_string(seed) + ".png";
            decode_vae(latent, path);
            t_vae += secs_since(t);
            ++images;
            ev.image(k, seed, path);
        }

        if (!keep_) unload();

        char buf[512];
        snprintf(buf, sizeof(buf),
                 "{\"prompt_cache_hit\":%s,\"prompt_tokens\":%d,\"encode_s\":%.3f,"
                 "\"prefill_s\":%.3f,\"load_s\":%.3f,\"denoise_s\":%.3f,"
                 "\"steps_run\":%d,\"s_per_step\":%.3f,\"vae_s\":%.3f,\"total_s\":%.3f}",
                 hit ? "true" : "false", valid_len, t_encode, t_prefill, t_load,
                 t_denoise, steps_run, steps_run ? t_denoise / steps_run : 0.0, t_vae,
                 stage_s());
        metrics = buf;
        printf("[serve] %s: %d image(s), %s\n", reason.c_str(), images, buf);
        fflush(stdout);
        return reason;
    }

    // After a request threw, the resident graphs may hold half-written state.
    void recover() {
        decode_rt_.reset();
        vae_rt_.reset();
        uploaded_key_.clear();
        try {
            dev_->wait_all_done();
            cp_->reset();
        } catch (...) {
        }
    }

private:
    PromptKV* lookup(const std::string& prompt) {
        for (auto it = lru_.begin(); it != lru_.end(); ++it)
            if (it->first == prompt) {
                lru_.splice(lru_.begin(), lru_, it);
                return &lru_.front().second;
            }
        return nullptr;
    }
    PromptKV* insert(const std::string& prompt, PromptKV kv) {
        lru_.emplace_front(prompt, std::move(kv));
        // Capacity 0 still keeps the entry the current request is using.
        while ((int)lru_.size() > std::max(1, cache_cap_)) lru_.pop_back();
        return &lru_.front().second;
    }

    PromptKV run_prefill(const PromptEmbeds& pe) {
        auto t = std::chrono::high_resolution_clock::now();
        auto rt = std::make_shared<Runtime>(cp_, p_.prefill.c_str(), /*precision=*/1);
        rt->set_backend_buffer(true);
        rt->LoadModel();
        printf("[load] prefill loaded in %.2fs\n", secs_since(t));

        const int P = prefix_len_;
        rt->ResizeInput("prompt_embeds", {1u, (uint32_t)P, (uint32_t)ctx_dim_});
        fill16(rt->GetInput("prompt_embeds"), pe.data, cp_);
        const RopeTables rope = build_rope(P, pe.valid_len, lh_, lw_);
        rt->ResizeInput("cos", {(uint32_t)P, (uint32_t)kRopeHeadDim});
        fill16(rt->GetInput("cos"), rope.cos_text, cp_);
        rt->ResizeInput("sin", {(uint32_t)P, (uint32_t)kRopeHeadDim});
        fill16(rt->GetInput("sin"), rope.sin_text, cp_);
        rt->ResizeInput("timestep_zero", {1u});
        fill32(rt->GetInput("timestep_zero"), {0.0f}, cp_);
        std::vector<uint16_t> bias((size_t)P * P, 0);
        const uint16_t neg = ITensor::fp32_to_fp16(-65504.0f);
        for (int i = 0; i < P; ++i)
            for (int j = i + 1; j < P; ++j) bias[(size_t)i * P + j] = neg;
        rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)P, (uint32_t)P});
        fill16(rt->GetInput("attention_bias"), bias, cp_);

        rt->Run();
        rt->ReadResult();

        PromptKV out;
        out.valid_len = pe.valid_len;
        for (int layer = 0;; ++layer) {
            auto o = rt->GetOutput("present_kv_" + std::to_string(layer));
            if (!o) break;
            auto tt = as_tensor<uint16_t>(o);
            tt->copyToCPU(cp_);
            out.kv.emplace_back(tt->data().begin(), tt->data().end());
            out.shapes.push_back(tt->getShape());
        }
        if (out.kv.empty()) throw std::runtime_error("prefill has no present_kv outputs");

        rt.reset();
        dev_->wait_all_done();
        cp_->reset();
        printf("[prefill] %zu KV layers, %.1f MB cached, %.2fs incl. load/release\n",
               out.kv.size(), out.bytes() / 1e6, secs_since(t));
        return out;
    }

    // rope, past_kv and the attention bias only change with the prompt; the
    // decode graph never writes them, so they stay in their buffers across Runs.
    void upload_static(const PromptKV& kv) {
        const int hd = kRopeHeadDim;
        const RopeTables rope = build_rope(prefix_len_, kv.valid_len, lh_, lw_);
        decode_rt_->ResizeInput("cos", {(uint32_t)target_len_, (uint32_t)hd});
        fill16(decode_rt_->GetInput("cos"), rope.cos_img, cp_);
        decode_rt_->ResizeInput("sin", {(uint32_t)target_len_, (uint32_t)hd});
        fill16(decode_rt_->GetInput("sin"), rope.sin_img, cp_);
        for (size_t layer = 0; layer < kv.kv.size(); ++layer) {
            const auto& s = kv.shapes[layer];
            const std::string name = "past_kv_" + std::to_string(layer);
            std::vector<uint32_t> dims(s.begin(), s.end());
            decode_rt_->ResizeInput(name, dims);
            auto t = decode_rt_->GetInput(name);
            if (!t) throw std::runtime_error("decode graph has no input " + name);
            fill16(t, kv.kv[layer], cp_);
        }
        // Text KV columns past valid_len are zero padding; the image rows attend
        // to the text bidirectionally, so those columns must be masked out.
        const int kv_len = prefix_len_ + target_len_;
        std::vector<uint16_t> bias((size_t)target_len_ * kv_len, 0);
        const uint16_t neg = ITensor::fp32_to_fp16(-65504.0f);
        for (int i = 0; i < target_len_; ++i)
            for (int j = kv.valid_len; j < prefix_len_; ++j)
                bias[(size_t)i * kv_len + j] = neg;
        decode_rt_->ResizeInput("attention_bias",
                                {1u, 1u, (uint32_t)target_len_, (uint32_t)kv_len});
        fill16(decode_rt_->GetInput("attention_bias"), bias, cp_);
    }

    void decode_vae(const std::vector<float>& latent, const std::string& path) {
        if (!vae_rt_) {
            const auto t = std::chrono::high_resolution_clock::now();
            vae_rt_ = std::make_shared<Runtime>(cp_, p_.vae.c_str(), /*precision=*/1);
            vae_rt_->set_backend_buffer(true);
            vae_rt_->LoadModel();
            printf("[load] VAE loaded in %.2fs\n", secs_since(t));
        }
        // BLC [1, L, C] -> NCDHW [1, C, 1, H, W], de-normalised per channel.
        std::vector<float> ncdhw((size_t)latent_c_ * target_len_);
        for (int t = 0; t < target_len_; ++t) {
            const int y = t / lw_, x = t % lw_;
            for (int c = 0; c < latent_c_; ++c)
                ncdhw[((size_t)c * lh_ + y) * lw_ + x] =
                    latent[(size_t)t * latent_c_ + c] * kLatentsStd[c] + kLatentsMean[c];
        }
        auto in = vae_rt_->GetInput("latent");
        if (!in) throw std::runtime_error("VAE has no 'latent' input");
        const std::vector<int> shape{1, latent_c_, 1, lh_, lw_};
        if (in->dtype() == typeid(float)) {
            auto tt = as_tensor<float>(in);
            tt->resize(shape);
            tt->fillToCPU(ncdhw.data());
            tt->copyToGPU(cp_);
        } else {
            std::vector<uint16_t> h(ncdhw.size());
            for (size_t i = 0; i < h.size(); ++i) h[i] = ITensor::fp32_to_fp16(ncdhw[i]);
            auto tt = as_tensor<uint16_t>(in);
            tt->resize(shape);
            tt->fillToCPU(h);
            tt->copyToGPU(cp_);
        }
        vae_rt_->Run();
        auto out = vae_rt_->GetOutput("decoded");
        if (!out) throw std::runtime_error("VAE has no 'decoded' output");
        const auto ds = out->getShape();
        if (ds.size() < 4) throw std::runtime_error("VAE output rank < 4");
        std::vector<float> img;
        if (out->dtype() == typeid(float)) {
            auto tt = as_tensor<float>(out);
            tt->copyToCPU(cp_);
            img.assign(tt->data().begin(), tt->data().end());
        } else {
            auto tt = as_tensor<uint16_t>(out);
            tt->copyToCPU(cp_);
            img.resize(tt->num_elements());
            for (size_t i = 0; i < img.size(); ++i)
                img[i] = ITensor::fp16_to_fp32(tt->data()[i]);
        }
        const int h = ds[ds.size() - 2], w = ds[ds.size() - 1];
        if (!save_latent_as_png(img, h, w, path.c_str()))
            throw std::runtime_error("cannot write " + path);
        printf("[png] %s (%dx%d)\n", path.c_str(), w, h);
    }

    Paths p_;
    bool keep_;
    int cache_cap_;
    std::shared_ptr<VulkanDevice> dev_;
    std::shared_ptr<VulkanCommandPool> cp_;
    TextFrontEnd fe_;
    int prefix_len_ = 0, ctx_dim_ = 0, target_len_ = 0, latent_c_ = 0, lh_ = 0, lw_ = 0;
    std::shared_ptr<Runtime> decode_rt_, vae_rt_;
    std::string uploaded_key_;  // prompt whose statics are in decode_rt_'s inputs
    std::list<std::pair<std::string, PromptKV>> lru_;
};

bool safe_id(const std::string& id) {
    if (id.empty() || id.size() > 64) return false;
    for (char c : id)
        if (!(isalnum((unsigned char)c) || c == '-' || c == '_')) return false;
    return true;
}

const char kUsage[] =
    "Usage: %s <dit_prefill.vkopbin> <dit_decode.vkopbin> --rpc-in FD --rpc-out FD\n"
    "          [--text-encoder P] [--embeds P] [--tokenizer P] [--vae P]\n"
    "          [--out-dir DIR] [--steps N] [--keep decode|none] [--prompt-cache N]\n"
    "  --keep decode (default) keeps the decode graph and VAE loaded between\n"
    "  requests; none releases everything after each request.\n";

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, kUsage, argv[0]);
        return 1;
    }
    auto env_or = [](const char* var, const char* def) {
        const char* v = getenv(var);
        return std::string(v && *v ? v : def);
    };
    Paths p;
    p.prefill = argv[1];
    p.decode = argv[2];
    p.te = env_or("TEXT_ENCODER_VKOPBIN", "text_encoder.vkopbin");
    p.te_table = env_or("TEXT_ENCODER_EMBEDS", "text_encoder_embeds.bin");
    p.tok = env_or("TOKENIZER_BIN", "llm/tokenizer/qwen3_vl.bin");
    p.vae = env_or("VAE_VKOPBIN", "vae_decoder_512.vkopbin");
    int rpc_in = -1, rpc_out = -1, default_steps = 40, prompt_cache = 8;
    bool keep = true;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--rpc-in") rpc_in = atoi(next().c_str());
        else if (a == "--rpc-out") rpc_out = atoi(next().c_str());
        else if (a == "--text-encoder") p.te = next();
        else if (a == "--embeds") p.te_table = next();
        else if (a == "--tokenizer") p.tok = next();
        else if (a == "--vae") p.vae = next();
        else if (a == "--out-dir") p.out_dir = next();
        else if (a == "--steps") default_steps = atoi(next().c_str());
        else if (a == "--prompt-cache") prompt_cache = atoi(next().c_str());
        else if (a == "--keep") {
            const std::string v = next();
            if (v != "decode" && v != "none") {
                fprintf(stderr, "--keep takes decode or none\n");
                return 1;
            }
            keep = v == "decode";
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            fprintf(stderr, kUsage, argv[0]);
            return 1;
        }
    }
    if (rpc_in < 0 || rpc_out < 0) {
        fprintf(stderr, "--rpc-in and --rpc-out are required\n");
        return 1;
    }

    std::unique_ptr<ImageServer> srv;
    try {
        srv = std::make_unique<ImageServer>(p, keep, prompt_cache);
    } catch (const std::exception& e) {
        emit(rpc_out, std::string("FATAL ") + b64_encode(e.what()));
        fprintf(stderr, "[serve] startup failed: %s\n", e.what());
        return 1;
    }
    const std::string ready = "READY 1 " + std::to_string(srv->width()) + " " +
                              std::to_string(srv->height()) + " " +
                              std::to_string(srv->max_prompt_tokens());
    emit(rpc_out, ready);

    LineReader in(rpc_in);
    std::string line;
    while (in.getline(line)) {
        if (line.empty()) continue;
        std::vector<std::string> a;
        for (size_t pos = 0; pos <= line.size();) {
            const size_t nx = line.find(' ', pos);
            a.push_back(line.substr(pos, nx == std::string::npos ? std::string::npos
                                                                 : nx - pos));
            if (nx == std::string::npos) break;
            pos = nx + 1;
        }
        const std::string& cmd = a[0];
        const std::string id = a.size() > 1 ? a[1] : "0";
        if (cmd == "BYE") {
            emit(rpc_out, "BYE");
            break;
        }
        if (cmd == "HELLO") {
            emit(rpc_out, ready);
            continue;
        }
        if (cmd == "STOP") continue;  // arrived after its request already ended
        try {
            if (cmd == "UNLOAD") {
                srv->unload();
                emit(rpc_out, "OK");
            } else if (cmd == "COUNT") {
                if (a.size() < 3) throw BadRequest("COUNT needs <id> <prompt-b64>");
                emit(rpc_out, "COUNTED " + id + " " +
                                  std::to_string(srv->count(b64_decode(a[2]))) + " " +
                                  std::to_string(srv->max_prompt_tokens()));
            } else if (cmd == "GEN") {
                if (a.size() < 6)
                    throw BadRequest(
                        "GEN needs <id> <steps> <seed> <n> <prompt-b64> [out_dir-b64]");
                if (!safe_id(id))
                    throw BadRequest("id must be 1-64 chars of [A-Za-z0-9_-]");
                GenRequest r;
                r.id = id;
                r.steps = atoi(a[2].c_str());
                if (r.steps <= 0) r.steps = default_steps;
                r.seed = (uint32_t)strtoul(a[3].c_str(), nullptr, 10);
                r.n = atoi(a[4].c_str());
                r.prompt = b64_decode(a[5]);
                r.out_dir = a.size() > 6 ? b64_decode(a[6]) : p.out_dir;
                if (r.steps > 200) throw BadRequest("steps must be 1..200");
                if (r.n < 1 || r.n > 16) throw BadRequest("n must be 1..16");
                if (r.prompt.empty()) throw BadRequest("empty prompt");
                struct stat st;
                if (stat(r.out_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
                    throw BadRequest("out_dir is not a directory: " + r.out_dir);

                GenEvents ev;
                ev.stage = [&](const char* s) {
                    char b[32];
                    snprintf(b, sizeof(b), "%.2f", stage_s());
                    emit(rpc_out, "STAGE " + id + " " + s + " " + b);
                };
                ev.step = [&](int k, int s, int n) {
                    emit(rpc_out, "STEP " + id + " " + std::to_string(k) + " " +
                                      std::to_string(s) + " " + std::to_string(n));
                };
                ev.image = [&](int k, uint32_t seed, const std::string& path) {
                    emit(rpc_out, "IMAGE " + id + " " + std::to_string(k) + " " +
                                      std::to_string(seed) + " " + b64_encode(path));
                };
                int images = 0;
                auto img_cb = ev.image;
                ev.image = [&, img_cb](int k, uint32_t seed, const std::string& path) {
                    ++images;
                    img_cb(k, seed, path);
                };
                ev.stop = [&]() { return in.poll_stop(); };
                std::string metrics;
                const std::string reason = srv->generate(r, ev, metrics);
                emit(rpc_out, "DONE " + id + " " + reason + " " + std::to_string(images) +
                                  " " + b64_encode(metrics));
            } else {
                throw BadRequest("unknown command: " + cmd);
            }
        } catch (const BadRequest& e) {
            emit(rpc_out, "ERROR " + id + " " + b64_encode(e.what()));
        } catch (const std::exception& e) {
            emit(rpc_out, "ERROR " + id + " " + b64_encode(e.what()));
            fprintf(stderr, "[serve] %s\n", e.what());
            if (strstr(e.what(), "VK_ERROR_DEVICE_LOST")) {
                fprintf(stderr, "[serve] GPU device lost; exiting\n");
                return 4;
            }
            srv->recover();
        }
        fflush(stdout);
    }
    return 0;
}
