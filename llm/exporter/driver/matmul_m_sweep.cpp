// MoE prefill GEMM M-sweep: is MatMul time linear in M?
//
// Grouped (per-expert) GEMM cuts arithmetic -- it runs each expert over only the
// rows routed to it, instead of every expert over every row.  That trade only
// pays if the clock is set by arithmetic.  If the kernel's time is instead
// dominated by a fixed per-dispatch / occupancy cost, cutting the rows changes
// nothing and grouping is a net loss.
//
// Shapes are the three expert projections of Qwen1.5-MoE-A2.7B, in the weight
// layout the real graph uses: the converter rewrites Gemm to MatMul with
// transB=1, so B is stored [N,K] -- exactly as the checkpoint has it.
//
//   gate/up : A[M,2048] x B[1408,2048]^T -> [M,1408]   (fp16)
//   down    : A[M,1408] x B[2048,1408]^T -> [M,2048]   (fp16)
//
// Method is benchmark/bench_matmul.cpp's: record once through the replay cache,
// then submit the same kept SIMULTANEOUS_USE command buffer R times and wait
// once, so the number is replayed GPU execution with no per-call host sync.
//
// VKOP_MATMUL_ROUTE=1 prints the route picked for each shape -- a route flip
// mid-sweep then shows up as a flip instead of being averaged into the slope.
//
// Usage: matmul_m_sweep [reps]          (default 30)
//        BENCH_DEV=<n> picks the physical device (default 0)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "core/Tensor.hpp"
#include "ops/OperatorFactory.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "core/ShaderArchive.hpp"

using namespace vkop;

namespace {

// B[j][k] = wpat(j): column-constant, so a column remap (right number, wrong
// column) is visible rather than hidden behind a symmetric all-ones weight.
inline int wpat(int j) { return (j % 7) - 3; }

struct Row {
    std::string name;
    int m, k, n;
    double ms;
    bool ok;
};

Row run_one(const char *name, int m, int k, int n, int reps,
            std::shared_ptr<VulkanDevice> dev,
            std::shared_ptr<VulkanCommandPool> cp) {
    Row r{name, m, k, n, 0.0, false};
    const auto queue = dev->getComputeQueue(0);

    auto A = std::make_shared<core::Tensor<uint16_t>>(
        std::vector<int>{m, k});
    auto B = std::make_shared<core::Tensor<uint16_t>>(
        std::vector<int>{n, k});
    auto O = std::make_shared<core::Tensor<uint16_t>>(
        std::vector<int>{m, n});

    A->reserveOnCPU();
    std::fill(A->data().begin(), A->data().end(), 0x3C00); // fp16 1.0
    B->reserveOnCPU();
    for (int j = 0; j < n; ++j) {
        const uint16_t v =
            core::ITensor::fp32_to_fp16(static_cast<float>(wpat(j)));
        for (int kk = 0; kk < k; ++kk)
            B->data()[static_cast<size_t>(j) * k + kk] = v;
    }
    // Allocate + upload BEFORE binding: as_storage_buffer only allocates, the
    // upload is copyToGPU's staging copy; binding an unuploaded tensor
    // silently computes on zeros.
    A->as_storage_buffer(dev);
    B->as_storage_buffer(dev);
    O->as_storage_buffer(dev);
    // O is allocated but never uploaded -- the kernel writes every element.
    A->copyToGPU(cp);
    B->copyToGPU(cp);

    std::vector<std::shared_ptr<core::ITensor>> in = {A, B};
    std::vector<std::shared_ptr<core::ITensor>> out = {O};
    auto op = ops::create_from_type(ops::OpType::MATMUL, /*fp16=*/1,
                                    /*use_tensorcore=*/0,
                                    /*backend_buffer=*/true);
    op->set_name(name);
    op->set_runtime_device(dev, cp);
    op->setAttribute({{"transB", "1"}});
    op->enable_replay(true);
    op->onExecute(in, out, 0); // record + store fingerprint
    op->onExecute(in, out, 0); // match -> CACHED, cmd kept
    auto cmd = op->get_record();
    if (!cmd) {
        printf("%-8s m=%-5d no recorded cmd\n", name, m);
        return r;
    }
    if (!op->route_info().empty())
        printf("[route] %-8s m=%-5d k=%-5d n=%-5d %s\n", name, m, k, n,
               op->route_info().c_str());

    cmd->submit(queue);
    dev->wait_all_done();

    // Correctness probe: A is all ones, so every output element must be
    // exactly k * wpat(j). A mismatch here means the timing below is of the
    // wrong arithmetic, so skip rather than report a number.
    O->reserveOnCPU();
    O->copyToCPU(cp);
    bool ok = true;
    for (int j = 0; j < n && ok; ++j) {
        const float got = core::ITensor::fp16_to_fp32(O->data()[j]);
        const float want = static_cast<float>(k) * static_cast<float>(wpat(j));
        if (got != want)
            ok = false;
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int rr = 0; rr < reps; ++rr)
        cmd->submit(queue);
    dev->wait_all_done();
    auto t1 = std::chrono::steady_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    r.ok = ok;
    return r;
}

} // namespace

int main(int argc, char **argv) {
    Logger::getInstance().setLevel(LOG_ERROR);
    // The baked path is this build tree's; a binary shipped to another machine
    // (b3 has no compiler) keeps neither the tree nor the archive location, so
    // let VKOP_DEV_SHADER_ARCHIVE override it. Dev-only: this header is not in
    // shipping binaries at all.
    vkop::dev::seed_shader_archive_from_build_tree();
    if (const char *arch = std::getenv("VKOP_DEV_SHADER_ARCHIVE"))
        core::ShaderArchive::set_process_default(arch);
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        printf("no vulkan device\n");
        return 1;
    }
    const char *dev_env = std::getenv("BENCH_DEV");
    const int dev_idx = dev_env ? std::atoi(dev_env) : 0;
    if (dev_idx < 0 || dev_idx >= static_cast<int>(phydevs.size())) {
        printf("no vulkan device %d\n", dev_idx);
        return 1;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[dev_idx]);
    printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cp = std::make_shared<VulkanCommandPool>(dev);

    const int reps = argc > 1 ? std::atoi(argv[1]) : 30;
    // 1 is the decode row that lands in the GEMV gate, kept only as the "M=0
    // limit"; 8..1024 is the prefill band a grouped kernel would partition.
    const int Ms[] = {1, 8, 32, 64, 128, 256, 512, 1024};

    std::vector<Row> rows;
    for (int m : Ms)
        rows.push_back(run_one("gate_up", m, 2048, 1408, reps, dev, cp));
    for (int m : Ms)
        rows.push_back(run_one("down", m, 1408, 2048, reps, dev, cp));

    printf("\n%-8s %6s %6s %6s %10s %10s %10s %10s\n", "kernel", "M", "K", "N",
           "ms", "GFLOP/s", "GB/s(w)", "vs M=1");
    double base = 0.0;
    std::string fam;
    for (const auto &r : rows) {
        if (r.name != fam) {
            fam = r.name;
            base = r.ms;
        }
        if (!r.ok || r.ms <= 0.0) {
            printf("%-8s %6d %6d %6d %10s\n", r.name.c_str(), r.m, r.k, r.n,
                   "WRONG");
            continue;
        }
        const double gf = 2.0 * r.m * r.k * r.n / (r.ms * 1e6);
        const double gb = double(r.n) * r.k * 2.0 / (r.ms * 1e6);
        printf("%-8s %6d %6d %6d %10.4f %10.1f %10.1f %9.1fx\n", r.name.c_str(),
               r.m, r.k, r.n, r.ms, gf, gb, r.ms / base);
    }

    printf("\nlinearity: time ratio against the M ratio, per doubling\n");
    for (size_t i = 1; i < rows.size(); ++i) {
        const auto &a = rows[i - 1], &b = rows[i];
        if (a.name != b.name || b.m != 2 * a.m || !a.ok || !b.ok)
            continue;
        const double tr = b.ms / a.ms;
        printf("  %-8s M %4d -> %4d : time x%.2f  -> %s\n", a.name.c_str(),
               a.m, b.m, tr,
               tr > 1.6 ? "linear: COMPUTE-BOUND"
                        : (tr > 1.2 ? "sub-linear" : "FLAT: fixed cost dominates"));
    }
    return 0;
}
