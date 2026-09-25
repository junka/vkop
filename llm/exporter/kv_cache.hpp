// junka @ 2026
// KVCache —— 图里那组 past/present KV 张量的生命周期句柄（host 侧壳，不新增
// GPU 结构）。Runtime 只把 KV 当普通输入/输出暴露，「一条序列的 KV 到哪一步了」
// 这件会话级的事原本散在驱动里（预分配、每轮 reset、present→past 回填），这里
// 收拢成一个对象，让驱动只表达「本轮 prefill 从空 KV 开始」「本轮结束把 KV 接上」。
//
// 三件事：
//  1. 预分配：LoadModel 之后立刻把每层 past/present 的 buffer 一次性按 max_kv
//     开好。之后每轮 ResizeInput 只改逻辑形状，make_vkbuff 走 prealloc_keep_
//     复用同一个 VkBuffer，不再每轮重分配几十 MB。
//  2. reset_for_prefill()：把 past 的逻辑 kv_len 归零（整段重新 prefill 的
//     语义），随后 upload() 把「空 past」真正传到 GPU。
//  3. feedback()：present→past 的 device→device 拷贝（全部层记在同一个 command
//     buffer 里，一次 wait），返回新的 past_len 供下一轮算位置/attention_bias。
//
// 前缀续用（跨轮跳过已算 KV）的接口留在这里：需要 feedback() 之外再加一个
// 「保留前缀、只补后面」的入口，并先有 token 前缀校验（Conversation::prefixMatch）。
// 目前图是单段连续 buffer、无 block table，所以还没接。

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "vulkan/VulkanCommandPool.hpp"

namespace vkop {
namespace export_ {

class KVCache {
public:
    KVCache(const std::shared_ptr<core::Runtime>& rt,
            const std::shared_ptr<VulkanCommandPool>& cmdpool, int nlayers,
            int nkv, int hd, int max_kv)
        : rt_(rt), cmdpool_(cmdpool), nlayers_(nlayers), nkv_(nkv), hd_(hd) {
        if (nlayers <= 0 || nkv <= 0 || hd <= 0 || max_kv <= 0) {
            throw std::runtime_error("KVCache: invalid arch parameters");
        }
        // past/present 都按 max_kv 开好，长度增长时不再重分配。
        const std::size_t kv_elems = static_cast<std::size_t>(2) * nkv * max_kv * hd;
        auto dev = cmdpool->getVulkanDevice();
        for (int i = 0; i < nlayers; ++i) {
            auto pin = rt_->GetInput("past_key_values_" + std::to_string(i));
            auto pout = rt_->GetOutput("present_key_values_" + std::to_string(i));
            if (!pin || !pout) {
                throw std::runtime_error("KVCache: missing KV tensor layer " +
                                         std::to_string(i));
            }
            core::as_tensor<uint16_t>(pin)->preallocate_buffer(dev, kv_elems);
            core::as_tensor<uint16_t>(pout)->preallocate_buffer(dev, kv_elems);
        }
    }

    // 每层 past 的逻辑形状成 kv_len=0（buffer 复用，不重新分配）。必须在填其它
    // 输入之前调用，且之后要 upload()。
    void reset_for_prefill() {
        for (int i = 0; i < nlayers_; ++i) {
            rt_->ResizeInput("past_key_values_" + std::to_string(i),
                             {1u, 2u, static_cast<uint32_t>(nkv_), 0u,
                              static_cast<uint32_t>(hd_)});
        }
    }

    // reset_for_prefill() 之后把空 past 传上 GPU。
    void upload() {
        for (int i = 0; i < nlayers_; ++i) {
            auto t = rt_->GetInput("past_key_values_" + std::to_string(i));
            core::as_tensor<uint16_t>(t)->copyToGPU(cmdpool_);
        }
    }

    // present→past：全部层的拷贝记进一个 command buffer，一次 sync。返回新的
    // past_len（下一轮的位置/attention_bias 都以它为准）。
    int feedback() {
        auto dev = cmdpool_->getVulkanDevice();
        // 先推出每层 kv_len 并 ResizeInput past（纯逻辑形状，buffer 走
        // prealloc_keep_ 复用）。必须先做，因为 ResizeInput 会把 converted_
        // 置 false，下面的 as_storage_buffer 才重新把 buffer 标给这次拷贝。
        std::vector<int> kv_lens(nlayers_);
        for (int i = 0; i < nlayers_; ++i) {
            auto pres = core::as_tensor<uint16_t>(
                rt_->GetOutput("present_key_values_" + std::to_string(i)));
            const int kv_len = pres->num_elements() / (2 * nkv_ * hd_);
            kv_lens[i] = kv_len;
            rt_->ResizeInput("past_key_values_" + std::to_string(i),
                             {1u, 2u, static_cast<uint32_t>(nkv_),
                              static_cast<uint32_t>(kv_len),
                              static_cast<uint32_t>(hd_)});
        }
        VulkanCommandBuffer cmd(cmdpool_);
        cmd.begin();
        int past_len = 0;
        for (int i = 0; i < nlayers_; ++i) {
            auto pres = core::as_tensor<uint16_t>(
                rt_->GetOutput("present_key_values_" + std::to_string(i)));
            auto past = core::as_tensor<uint16_t>(
                rt_->GetInput("past_key_values_" + std::to_string(i)));
            auto pres_buf = pres->as_storage_buffer(dev, nullptr);
            auto past_buf = past->as_storage_buffer(dev, nullptr);
            const VkDeviceSize copy_bytes = static_cast<VkDeviceSize>(
                2 * nkv_ * kv_lens[i] * hd_ * sizeof(uint16_t));
            if (copy_bytes == 0) continue;
            pres_buf->transferReadBarrier(cmd.get(), copy_bytes, 0);
            past_buf->copyStageBufferToBuffer(cmd.get(), pres_buf->getBuffer(), 0,
                                             copy_bytes, 0);
            past->toGPU();
            past_len = kv_lens[i];
        }
        cmd.end();
        cmd.submit(dev->getComputeQueue());
        cmd.wait();
        return past_len;
    }

    int nlayers() const { return nlayers_; }
    int nkv() const { return nkv_; }
    int hd() const { return hd_; }

private:
    std::shared_ptr<core::Runtime> rt_;
    std::shared_ptr<VulkanCommandPool> cmdpool_;
    int nlayers_, nkv_, hd_;
};

} // namespace export_
} // namespace vkop
