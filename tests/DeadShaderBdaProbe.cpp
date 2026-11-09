// Copyright 2026 @junka
// BDA probe for the shaders no model reaches: the host-only ops
// (NonZero/Range), the reference argmax kernel, and the disabled coopmat
// split-K / in-workgroup routes. These have no engine dispatch path, so the
// probe drives them directly: one BDA pipeline with a 6-slot/2-slot address
// table at binding 0, the shader's own push constants, and a raw
// vkCmdDispatch. DEAD_BDA_DESC_ONLY=1 runs the same shaders through the
// descriptor builds (N storage bindings) as the A/B baseline.

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "core/ShaderArchive.hpp"
#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/Matmul.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanPipeline.hpp"

using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::VulkanBuffer;
using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::VulkanPipeline;

namespace {

int failures = 0;
bool g_desc_only = false;
std::shared_ptr<VulkanDevice> g_dev;
std::shared_ptr<VulkanCommandPool> g_pool;

struct Step {
    const char *key_bda;
    const char *key_desc;
    std::vector<std::shared_ptr<VulkanBuffer>> bufs;
    const void *pc = nullptr;
    size_t pc_size = 0;
    int gx = 1, gy = 1, gz = 1;
    // Optional execution dependency after this dispatch (split-K deposit ->
    // fold reads the scratch planes).
    std::shared_ptr<VulkanBuffer> barrier_after;
};

const char *key_for(const Step &s) {
    return g_desc_only ? s.key_desc : s.key_bda;
}

// Build one pipeline for `key`, bind the address table (BDA) or the N data
// buffers (descriptor), then dispatch.
bool run_steps(const char *name, std::vector<Step> steps) {
    auto *archive = vkop::core::ShaderArchive::process_default();
    if (!archive) {
        std::printf("FAIL: %s: no shader archive\n", name);
        return false;
    }
    auto cmd = std::make_shared<VulkanCommandBuffer>(g_pool);
    std::vector<std::shared_ptr<VulkanPipeline>> pipes;
    std::vector<std::shared_ptr<VulkanBuffer>> tables;

    cmd->begin();
    for (auto &s : steps) {
        const char *key = key_for(s);
        if (key == nullptr) {
            std::printf("SKIP: %s: no descriptor key for %s\n", name,
                        s.key_bda);
            cmd->end();
            return true;
        }
        auto [spv, len] = archive->lookup(key);
        if (!spv) {
            std::printf("FAIL: %s: archive lacks %s\n", name, key);
            cmd->end();
            return false;
        }
        const int n = static_cast<int>(s.bufs.size());
        std::vector<VkDescriptorType> types(
            g_desc_only ? n : 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        auto pipe = std::make_shared<VulkanPipeline>(
            g_dev->getLogicalDevice(), types, s.pc_size,
            reinterpret_cast<const uint32_t *>(spv), static_cast<int>(len),
            /*update_after_bind=*/false, 0);
        pipes.push_back(pipe);
        VkDescriptorSet ds = pipe->allocDescriptorSets();
        std::vector<VkDescriptorBufferInfo> infos(n);
        std::vector<VkWriteDescriptorSet> writes(g_desc_only ? n : 1);
        if (g_desc_only) {
            for (int i = 0; i < n; ++i) {
                infos[i] =
                    *std::get<VkDescriptorBufferInfo *>(
                        s.bufs[i]->getDescriptorInfo());
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = ds;
                writes[i].dstBinding = static_cast<uint32_t>(i);
                writes[i].dstArrayElement = 0;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos[i];
            }
        } else {
            auto table = std::make_shared<VulkanBuffer>(
                g_dev, n * sizeof(VkDeviceAddress),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            tables.push_back(table);
            std::vector<VkDeviceAddress> addrs(n);
            for (int i = 0; i < n; ++i) {
                if (!s.bufs[i]->hasDeviceAddress()) {
                    std::printf("FAIL: %s: buffer %d has no device address\n",
                                name, i);
                    cmd->end();
                    return false;
                }
                addrs[i] = s.bufs[i]->getDeviceAddress();
            }
            table->updateBuffer(cmd->get(), addrs.data(),
                                n * sizeof(VkDeviceAddress));
            table->readBarrier(cmd->get(), n * sizeof(VkDeviceAddress), 0);
            infos[0] = *std::get<VkDescriptorBufferInfo *>(
                table->getDescriptorInfo());
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = ds;
            writes[0].dstBinding = 0;
            writes[0].dstArrayElement = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &infos[0];
        }
        pipe->updateDescriptorSets(writes);
        cmd->bind(*pipe, ds);
        if (s.pc) {
            cmd->push_constants(*pipe, static_cast<uint32_t>(s.pc_size),
                                s.pc);
        }
        cmd->dispatch(s.gx, s.gy, s.gz);
        if (s.barrier_after) {
            s.barrier_after->shaderWriteBarrier(cmd->get());
        }
    }
    cmd->end();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(g_dev->getComputeQueue(), info);
    cmd->wait();
    g_dev->wait_all_done();
    return true;
}

template <typename T>
std::shared_ptr<VulkanBuffer> upload(const std::shared_ptr<Tensor<T>> &t,
                                     const std::vector<T> &vals) {
    t->fillToCPU(vals);
    auto b = t->as_storage_buffer(g_dev);
    t->copyToGPU(g_pool);
    return b;
}

template <typename T>
std::shared_ptr<VulkanBuffer> new_output(const std::shared_ptr<Tensor<T>> &t) {
    auto b = t->as_storage_buffer(g_dev);
    t->toGPU();
    return b;
}

// Zero an output on the DEVICE (not via fillToCPU: a host-filled tensor makes
// copyToCPU early-return its host copy, hiding whatever the shader wrote).
template <typename T>
std::shared_ptr<VulkanBuffer>
zeroed_output(const std::shared_ptr<Tensor<T>> &t,
              const std::vector<T> &zeros) {
    auto b = t->as_storage_buffer(g_dev);
    t->toGPU();
    VulkanCommandBuffer up(g_pool);
    up.begin();
    b->updateBuffer(up.get(), zeros.data(), zeros.size() * sizeof(T));
    b->readBarrier(up.get(), zeros.size() * sizeof(T), 0);
    up.end();
    std::vector<VkSubmitInfo> info{up.buildSubmitInfo()};
    VulkanCommandBuffer::submit(g_dev->getComputeQueue(), info);
    up.wait();
    return b;
}


// ---- NonZero: single atomicAdd pass, output = [count, idx...] -------------
void test_nonzero() {
    auto in = std::make_shared<Tensor<float>>(std::vector<int>{8});
    auto out = std::make_shared<Tensor<int>>(std::vector<int>{9});
    std::vector<float> vals{0, 1, 0, 2, 3, 0, 0, 4};
    std::vector<int> zeros(9, 0);
    auto in_buf = upload(in, vals);
    auto out_buf = zeroed_output(out, zeros);
    struct PC {
        int total, p0, p1, p2;
    } pc{8, 0, 0, 0};
    Step s{"buffer_nonzero_bda", "buffer_nonzero",
           {out_buf, in_buf}, &pc, sizeof(pc), 1, 1, 1, nullptr};
    bool ok = run_steps("NonZero", {s});
    out->copyToCPU(g_pool);
    const std::vector<int> &o = out->data();
    if (!ok) {
        return;
    }
    if (o[0] != 4) {
        std::printf("NonZero count got %d want 4\n", o[0]);
        failures++;
        return;
    }
    std::vector<int> idx(o.begin() + 1, o.begin() + 1 + o[0]);
    std::sort(idx.begin(), idx.end());
    if (idx != std::vector<int>{1, 3, 4, 7}) {
        std::printf("NonZero indices wrong\n");
        failures++;
    } else {
        std::printf("PASS: NonZero\n");
    }
}

// ---- Range: start/limit/delta as integer buffers, fp32 output --------------
void test_range() {
    auto start = std::make_shared<Tensor<int>>(std::vector<int>{1});
    auto limit = std::make_shared<Tensor<int>>(std::vector<int>{1});
    auto delta = std::make_shared<Tensor<int>>(std::vector<int>{1});
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{5});
    auto sb = upload(start, {1});
    auto lb = upload(limit, {10});
    auto db = upload(delta, {2});
    auto ob = new_output(out);
    struct PC {
        int fp16;
    } pc{0};
    Step s{"buffer_range_bda", "buffer_range", {ob, sb, lb, db}, &pc,
           sizeof(pc), 1, 1, 1, nullptr};
    bool ok = run_steps("Range", {s});
    out->copyToCPU(g_pool);
    if (!ok) {
        return;
    }
    std::vector<float> want{1, 3, 5, 7, 9};
    for (int i = 0; i < 5; ++i) {
        if ((*out)[i] != want[i]) {
            std::printf("Range [%d] got %f want %f\n", i, (*out)[i], want[i]);
            failures++;
            return;
        }
    }
    std::printf("PASS: Range\n");
}

// ---- argmax: one workgroup over the whole vocab ---------------------------
void test_argmax() {
    const int vocab = 256;
    auto logits = std::make_shared<Tensor<float>>(std::vector<int>{vocab});
    std::vector<float> vals(vocab);
    for (int i = 0; i < vocab; ++i) {
        vals[i] = 0.01F * static_cast<float>(i % 17);
    }
    vals[123] = 9.5F; // unique maximum
    auto out = std::make_shared<Tensor<int>>(std::vector<int>{2});
    auto lb = upload(logits, vals);
    auto ob = upload(out, {0, 0});
    struct PC {
        uint32_t base_offset, vocab;
    } pc{0, static_cast<uint32_t>(vocab)};
    Step s{"buffer_argmax_bda", "buffer_argmax", {lb, ob}, &pc, sizeof(pc), 1,
           1, 1, nullptr};
    bool ok = run_steps("ArgMax", {s});
    out->copyToCPU(g_pool);
    if (!ok) {
        return;
    }
    const std::vector<int> &o = out->data();
    float got_v;
    std::memcpy(&got_v, &o[0], sizeof(float));
    if (o[1] != 123 || got_v != 9.5F) {
        std::printf("argmax got idx=%d v=%f want 123/9.5\n", o[1], got_v);
        failures++;
    } else {
        std::printf("PASS: ArgMax\n");
    }
}

// ---- coopmat split-K deposit + fold --------------------------------------
void test_coop_splitk(const char *bda_dep, const char *bda_fold,
                      const char *desc_dep, const char *desc_fold, int M,
                      int N, int K, int bm, int bn, uint32_t tile_m,
                      uint32_t tile_n) {
    const std::vector<int> a_shape{M, K};
    const std::vector<int> b_shape{K, N};
    auto a = std::make_shared<Tensor<uint16_t>>(a_shape);
    auto b = std::make_shared<Tensor<uint16_t>>(b_shape);
    auto out = std::make_shared<Tensor<uint16_t>>(std::vector<int>{M, N});
    std::vector<uint16_t> av(M * K), bv(K * N);
    std::vector<float> af(M * K), bf(K * N);
    for (int i = 0; i < M * K; ++i) {
        af[i] = 0.25F * static_cast<float>((i * 7) % 11 - 5);
        av[i] = ITensor::fp32_to_fp16(af[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        bf[i] = 0.25F * static_cast<float>((i * 5) % 13 - 6);
        bv[i] = ITensor::fp32_to_fp16(bf[i]);
    }
    auto ab = upload(a, av);
    auto bb = upload(b, bv);
    auto ob = new_output(out);

    const int coop = 2;
    auto scratch = std::make_shared<VulkanBuffer>(
        g_dev, static_cast<VkDeviceSize>(coop) * M * N * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto sb = scratch;

    vkop::ops::matmul::GpuMatMulParam pc{};
    pc.M = M;
    pc.N = N;
    pc.K = K;
    pc.C = 1;
    pc.fp32 = 0;
    pc.transB = 0;
    pc.ksplit = coop;

    // scale/global slots are never dereferenced for a fp16 weight; point them
    // at A so the table holds valid addresses.
    std::vector<Step> steps{
        {bda_dep, desc_dep, {ob, ab, bb, sb, ab, ab}, &pc, sizeof(pc),
         (N + bn - 1) / bn, (M + bm - 1) / bm, coop, scratch},
        {bda_fold, desc_fold, {ob, ab, bb, sb, ab, ab}, &pc, sizeof(pc),
         (N + bn - 1) / bn, (M + bm - 1) / bm, 1, nullptr},
    };
    if (!run_steps("coop_splitk", steps)) {
        return;
    }
    out->copyToCPU(g_pool);
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float want = 0.0F;
            for (int k = 0; k < K; ++k) {
                want += af[m * K + k] * bf[k * N + n];
            }
            const float got =
                ITensor::fp16_to_fp32((*out)[m * N + n]);
            if (std::fabs(got - want) >
                std::max(std::fabs(want) * 0.02F, 0.05F)) {
                if (bad < 4) {
                    std::printf("splitk [%d,%d] got %f want %f\n", m, n, got,
                                want);
                }
                ++bad;
            }
        }
    }
    if (bad) {
        std::printf("FAIL: coop splitk (%d bad)\n", bad);
        failures++;
    } else {
        std::printf("PASS: coop splitk %dx%dx%d (tile %ux%u)\n", M, N, K,
                    tile_m, tile_n);
    }
}

// ---- coopmat in-workgroup split-K (single dispatch) ----------------------
void test_coop_inwg(int M, int N, int K, int bm, int bn) {
    auto a = std::make_shared<Tensor<uint16_t>>(std::vector<int>{M, K});
    auto b = std::make_shared<Tensor<uint16_t>>(std::vector<int>{K, N});
    auto out = std::make_shared<Tensor<uint16_t>>(std::vector<int>{M, N});
    std::vector<uint16_t> av(M * K), bv(K * N);
    std::vector<float> af(M * K), bf(K * N);
    for (int i = 0; i < M * K; ++i) {
        af[i] = 0.25F * static_cast<float>((i * 7) % 11 - 5);
        av[i] = ITensor::fp32_to_fp16(af[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        bf[i] = 0.25F * static_cast<float>((i * 5) % 13 - 6);
        bv[i] = ITensor::fp32_to_fp16(bf[i]);
    }
    auto ab = upload(a, av);
    auto bb = upload(b, bv);
    auto ob = new_output(out);
    // binding 3 (scratch) is unused by SPLITK_INWG; point the table slot at A.
    vkop::ops::matmul::GpuMatMulParam pc{};
    pc.M = M;
    pc.N = N;
    pc.K = K;
    pc.C = 1;
    pc.ksplit = 2;
    Step s{"buffer_matmul_coop_inwg_bda", "buffer_matmul_coop_inwg",
           {ob, ab, bb, ab, ab, ab}, &pc, sizeof(pc), (N + bn - 1) / bn,
           (M + bm - 1) / bm, 1, nullptr};
    if (!run_steps("coop_inwg", {s})) {
        return;
    }
    out->copyToCPU(g_pool);
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float want = 0.0F;
            for (int k = 0; k < K; ++k) {
                want += af[m * K + k] * bf[k * N + n];
            }
            const float got = ITensor::fp16_to_fp32((*out)[m * N + n]);
            if (std::fabs(got - want) >
                std::max(std::fabs(want) * 0.02F, 0.05F)) {
                ++bad;
            }
        }
    }
    if (bad) {
        std::printf("FAIL: coop inwg (%d bad)\n", bad);
        failures++;
    } else {
        std::printf("PASS: coop inwg %dx%dx%d\n", M, N, K);
    }
}

// ---- coop16 in-workgroup split-K (single dispatch, fp32 scratch out) -----
// The 16x16x16 twin above: BM=32/BN=64 from the default SG_M=2/SG_N=4 split,
// and with ENABLE_HALF2_EPI off the epilogue stores fp32 [M,N] to binding 3,
// not half2 to binding 0. Slot 0 therefore stays a dummy.
void test_coop16_inwg(int M, int N, int K) {
    auto a = std::make_shared<Tensor<uint16_t>>(std::vector<int>{M, K});
    auto b = std::make_shared<Tensor<uint16_t>>(std::vector<int>{K, N});
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{M, N});
    std::vector<uint16_t> av(M * K), bv(K * N);
    std::vector<float> af(M * K), bf(K * N);
    for (int i = 0; i < M * K; ++i) {
        af[i] = 0.25F * static_cast<float>((i * 7) % 11 - 5);
        av[i] = ITensor::fp32_to_fp16(af[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        bf[i] = 0.25F * static_cast<float>((i * 5) % 13 - 6);
        bv[i] = ITensor::fp32_to_fp16(bf[i]);
    }
    auto ab = upload(a, av);
    auto bb = upload(b, bv);
    auto ob = new_output(out);
    vkop::ops::matmul::GpuMatMulParam pc{};
    pc.M = M;
    pc.N = N;
    pc.K = K;
    pc.C = 1;
    pc.ksplit = 2;
    Step s{"buffer_matmul_coop16_inwg_bda", "buffer_matmul_coop16_inwg",
           {ab, ab, bb, ob, ab, ab}, &pc, sizeof(pc), (N + 63) / 64,
           (M + 31) / 32, 1, nullptr};
    if (!run_steps("coop16_inwg", {s})) {
        return;
    }
    out->copyToCPU(g_pool);
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float want = 0.0F;
            for (int k = 0; k < K; ++k) {
                want += af[m * K + k] * bf[k * N + n];
            }
            const float got = (*out)[m * N + n];
            if (std::fabs(got - want) >
                std::max(std::fabs(want) * 0.02F, 0.05F)) {
                ++bad;
            }
        }
    }
    if (bad) {
        std::printf("FAIL: coop16 inwg (%d bad)\n", bad);
        failures++;
    } else {
        std::printf("PASS: coop16 inwg %dx%dx%d\n", M, N, K);
    }
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    g_desc_only = std::getenv("DEAD_BDA_DESC_ONLY") != nullptr;
    if (g_desc_only) {
        unsetenv("VKOP_BDA");
    } else {
        setenv("VKOP_BDA", "1", 1);
    }
    Logger::getInstance().setLevel(LOG_ERROR);
    if (const char *archive = std::getenv("VKOP_ADA_ARCHIVE")) {
        vkop::core::ShaderArchive::set_process_default(archive);
    } else {
        vkop::dev::seed_shader_archive_from_build_tree();
    }
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("SKIP: no Vulkan device\n");
        return 0;
    }
    g_dev = std::make_shared<VulkanDevice>(phydevs[0]);
    g_pool = std::make_shared<VulkanCommandPool>(g_dev);
    std::printf("GPU: %s, BDA=%d, variant=%s\n", g_dev->getDeviceName().c_str(),
                g_dev->is_support_buffer_device_address() ? 1 : 0,
                g_desc_only ? "descriptor (gate)" : "bda");
    if (!g_dev->is_support_buffer_device_address()) {
        std::printf("SKIP: buffer device address unsupported\n");
        return 0;
    }

    test_nonzero();
    test_range();
    test_argmax();

    // ARL lists the 8x8x16 MMA (BM=16, BN=32); the 16x16x16 modules only run
    // on a device that lists them (Ada: BM=32, BN=64).
    if (g_dev->supports_coopmat_fp16_tile(8, 8, 16)) {
        test_coop_splitk("buffer_matmul_coop_splitk_deposit_bda",
                         "buffer_matmul_coop_splitk_fold_bda",
                         "buffer_matmul_coop_splitk_deposit",
                         "buffer_matmul_coop_splitk_fold", 16, 32, 64, 16, 32,
                         8, 8);
        test_coop_inwg(16, 32, 64, 16, 32);
    } else {
        std::printf("SKIP: 8x8x16 coopmat not listed\n");
    }

    // coop16.comp's INWG build is the 16x16x16 tile: Ada only, and K >= 4
    // K_BLOCK so the cs=2 slices keep >= 2 MMAs each (the route's host guard).
    if (g_dev->supports_coopmat_fp16_tile(16, 16, 16)) {
        test_coop16_inwg(32, 64, 128);
    } else {
        std::printf("SKIP: 16x16x16 coopmat not listed\n");
    }

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    std::fflush(nullptr);
    _exit(failures == 0 ? 0 : 1);
}
