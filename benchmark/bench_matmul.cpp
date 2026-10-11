// Standalone MatMul GEMV microbenchmark (decode-shape fp16, buffer backend).
//
// Drives the real MatMul op through ops::create_from_type, records once via
// the replay cache (two onExecute rounds -> CACHED), then re-submits the kept
// SIMULTANEOUS_USE cmd buffer R times and waits once, so the measurement is
// pure GPU time without per-call sync overhead.
//
// Build (mirrors the llm_chat link line):
//   /usr/local/bin/g++-14 -nostdinc++ \
//     -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
//     -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include \
//     -O2 -std=gnu++17 -DUSE_VMA -DVK_ENABLE_BETA_EXTENSIONS \
//     -I/usr/local/include -I/Users/uqland/github/vkop -I/Users/uqland/github/vkop/model \
//     -I/Users/uqland/github/vkop/model/SYSTEM -I/Users/uqland/github/vkop/model/generated \
//     benchmark/bench_matmul.cpp -o build/bench_matmul -Lbuild libvkop.a -lc++
//
// Env: VKOP_NO_COOP=1 on MoltenVK (16x16 coopmat tiles cannot compile to MSL).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "core/Tensor.hpp"
#include "ops/OperatorFactory.hpp"
#include "include/logger.hpp"

using namespace vkop;

// GCC-14 at -O2 emits calls to __cxa_call_terminate from noexcept cleanup
// paths inside libc++ headers. The system libc++.1.dylib (1900.x) predates
// that ABI symbol (llm_chat never hits it because CMake builds everything at
// -O0). Provide it: the real one just routes to the terminate handler, and no
// noexcept function here ever throws in this harness.
extern "C" void __cxa_call_terminate(void *) { abort(); }

namespace {

struct Shape {
    int K, N;
    const char *name;
};

void fill_ones(const std::shared_ptr<core::Tensor<uint16_t>> &t) {
    t->reserveOnCPU();
    std::fill(t->data().begin(), t->data().end(), 0x3C00); // fp16 1.0
}

} // namespace

int main(int argc, char **argv) {
    Logger::getInstance().setLevel(LOG_ERROR);
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        printf("no vulkan device\n");
        return 1;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
    auto queue = dev->getComputeQueue(0);

    // Qwen3-2B decode shapes: A is one hidden row, B is the projection weight
    // [K, N] (transB == 0). All must hit the ksplit GEMV gate: n%4==0, k%2==0,
    // batch*m<=8, k>=256.
    std::vector<Shape> shapes = {
        {2048, 2048, "q/k/v/o"},
        {2048, 6144, "gate/up"},
        {6144, 2048, "down"},
        {2048, 151936, "lm_head"},
    };
    if (argc > 1) {
        // Override: bench_matmul K N [reps]
        shapes.clear();
        shapes.push_back({std::atoi(argv[1]), std::atoi(argv[2]), "custom"});
    }

    const int reps = (argc > 3) ? std::atoi(argv[3]) : 30;

    // DRAM ceiling probe: vkCmdCopyBuffer of a 256MB region, same submit
    // batch as the GEMV measurements. Read+write so effective bandwidth is
    // bytes_moved/2/s per direction.
    {
        const size_t bytes = 256u << 20;
        auto src = std::make_shared<VulkanBuffer>(
            dev, bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto dst = std::make_shared<VulkanBuffer>(
            dev, bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto cmd = std::make_shared<VulkanCommandBuffer>(cmdpool, 0);
        cmd->set_replayable(true);
        cmd->begin();
        VkBufferCopy region{0, 0, static_cast<VkDeviceSize>(bytes)};
        vkop::vkCmdCopyBuffer(cmd->get(), src->getBuffer(), dst->getBuffer(),
                              1, &region);
        cmd->end();
        cmd->submit(queue);
        dev->wait_all_done();
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            cmd->submit(queue);
        }
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        printf("%-9s %zu MB copy  : %8.3f ms  %7.1f GB/s (read+write)\n",
               "memcpy", bytes >> 20, ms, 2.0 * bytes / ms / 1e6);

        // Per-submit CPU overhead: an empty (replayable) command buffer.
        auto empty = std::make_shared<VulkanCommandBuffer>(cmdpool, 0);
        empty->set_replayable(true);
        empty->begin();
        empty->end();
        empty->submit(queue);
        dev->wait_all_done();
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; ++r) {
            empty->submit(queue);
        }
        dev->wait_all_done();
        t1 = std::chrono::steady_clock::now();
        printf("%-9s empty submit : %8.4f ms\n", "overhead",
               std::chrono::duration<double, std::milli>(t1 - t0).count() /
                   200);
    }

    for (const auto &s : shapes) {
        auto A = std::make_shared<core::Tensor<uint16_t>>(
            std::vector<int>{1, s.K});
        auto B = std::make_shared<core::Tensor<uint16_t>>(
            std::vector<int>{s.K, s.N});
        auto O = std::make_shared<core::Tensor<uint16_t>>(
            std::vector<int>{1, s.N});
        fill_ones(A);
        fill_ones(B);
        // Allocate + upload BEFORE binding: as_storage_buffer only allocates
        // (make_vkbuff), the upload is copyToGPU's staging copy. Binding an
        // unuploaded tensor silently computes on zeros.
        A->as_storage_buffer(dev);
        B->as_storage_buffer(dev);
        A->copyToGPU(cmdpool);
        B->copyToGPU(cmdpool);

        auto op = ops::create_from_type(ops::OpType::MATMUL, /*fp16=*/1,
                                        /*use_tensorcore=*/0,
                                        /*backend_buffer=*/true);
        op->set_name(s.name);
        op->set_runtime_device(dev, cmdpool);
        op->enable_replay(true);
        std::vector<std::shared_ptr<core::ITensor>> in = {A, B}, out = {O};
        op->onExecute(in, out, 0); // round 0: record + store fingerprint
        op->onExecute(in, out, 0); // round 1: match -> CACHED (cmd kept)
        auto cmd = op->get_record();
        if (!cmd) {
            printf("%-9s K=%-6d N=%-6d : no recorded cmd\n", s.name, s.K, s.N);
            continue;
        }

        // warmup submit + sync
        cmd->submit(queue);
        dev->wait_all_done();

        // correctness probe (A=B=1.0 -> every output element must equal K)
        cmd->submit(queue);
        dev->wait_all_done();
        O->reserveOnCPU();
        O->copyToCPU(cmdpool);
        printf("  probe: O[0]=%f O[1]=%f O[N-1]=%f\n",
               core::ITensor::fp16_to_fp32(O->data()[0]),
               core::ITensor::fp16_to_fp32(O->data()[1]),
               core::ITensor::fp16_to_fp32(O->data()[s.N - 1]));
        bool ok = true;
        for (int j = 0; j < s.N; ++j) {
            uint16_t v = O->data()[j];
            float f = vkop::core::ITensor::fp16_to_fp32(v);
            if (f != static_cast<float>(s.K)) {
                ok = false;
                printf("  MISMATCH at %d: %f != %d\n", j, f, s.K);
                break;
            }
        }
        if (!ok) {
            printf("%-9s K=%-6d N=%-6d : WRONG OUTPUT, skip timing\n", s.name,
                   s.K, s.N);
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            cmd->submit(queue);
        }
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        double bytes = static_cast<double>(s.K) * s.N * 2; // fp16 weights
        printf("%-9s K=%-6d N=%-6d : %8.3f ms  %7.1f GB/s\n", s.name, s.K, s.N,
               ms, bytes / ms / 1e6);
    }
    return 0;
}
