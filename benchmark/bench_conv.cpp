// Float Conv2d on the buffer backend (fp32 and fp16 graphs), at the
// Qwen-Image VAE decoder's conv shapes.
//
// Times one recorded conv dispatch per shape and dtype. The host routes these
// shapes to the implicit-GEMM kernels in shaders/buffer/conv2d.comp; set
// VKOP_CONV_NO_IGEMM=1 to time the direct kernels on the same data, or
// VKOP_CONV_IGEMM_M128=1 to pin the 128x128 tile. Correctness is the op
// tests' job (Conv2dTest.Conv2dImplicitGemmBuffer); this only checks that the
// output is finite.
//
// Run:
//   bench_conv [reps]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/ShaderArchive.hpp"
#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using namespace vkop;

namespace {
struct Shape {
    int ic, hw, oc, k;
    const char *name;
};

template <typename T>
void run(const Shape &sh, std::shared_ptr<VulkanDevice> dev,
         std::shared_ptr<VulkanCommandPool> cmdpool, int reps) {
    const bool fp16 = std::is_same_v<T, uint16_t>;
    const int p = sh.k / 2;
    std::vector<float> xv(static_cast<size_t>(sh.ic) * sh.hw * sh.hw);
    for (size_t i = 0; i < xv.size(); i++) {
        xv[i] = static_cast<float>(static_cast<int>(i * 37 % 17) - 8) * 0.0625f;
    }
    std::vector<float> wv(static_cast<size_t>(sh.oc) * sh.ic * sh.k * sh.k);
    for (size_t i = 0; i < wv.size(); i++) {
        wv[i] = static_cast<float>(static_cast<int>(i * 53 % 31) - 15) * 0.002f;
    }
    std::vector<float> bv(sh.oc);
    for (int i = 0; i < sh.oc; i++) {
        bv[i] = 0.125f * static_cast<float>(i % 3);
    }
    auto X = std::make_shared<core::Tensor<T>>(
        std::vector<int>{1, sh.ic, sh.hw, sh.hw});
    auto W = std::make_shared<core::Tensor<T>>(
        std::vector<int>{sh.oc, sh.ic, sh.k, sh.k});
    auto B = std::make_shared<core::Tensor<T>>(std::vector<int>{sh.oc});
    auto Y = std::make_shared<core::Tensor<T>>(
        std::vector<int>{1, sh.oc, sh.hw, sh.hw});
    X->fillFP32ToCPU(xv);
    W->fillFP32ToCPU(wv);
    B->fillFP32ToCPU(bv);
    X->as_storage_buffer(dev);
    W->as_storage_buffer(dev);
    B->as_storage_buffer(dev);
    Y->as_storage_buffer(dev);
    X->copyToGPU(cmdpool);
    W->copyToGPU(cmdpool);
    B->copyToGPU(cmdpool);

    auto op = ops::create_from_type(ops::OpType::CONV2D, fp16 ? 1 : 0,
                                    /*use_tensorcore=*/0,
                                    /*backend_buffer=*/true);
    op->set_name(sh.name);
    op->set_runtime_device(dev, cmdpool);
    op->enable_replay(true);
    op->setAttribute({{"strides", "1"},
                      {"pads", std::to_string(p)},
                      {"dilations", "1"},
                      {"group", "1"},
                      {"kernel_shape", std::to_string(sh.k)}});
    std::vector<std::shared_ptr<core::ITensor>> in = {X, W, B}, out = {Y};
    op->onExecute(in, out, 0);
    op->onExecute(in, out, 0);
    auto cmd = op->get_record();
    if (!cmd) {
        std::printf("%-28s %s: no recorded cmd\n", sh.name,
                    fp16 ? "fp16" : "fp32");
        return;
    }
    cmd->submit(dev->getComputeQueue(0));
    dev->wait_all_done();
    Y->reserveOnCPU();
    Y->copyToCPU(cmdpool);
    int bad = 0;
    for (int i = 0; i < Y->num_elements(); i++) {
        const float v = fp16 ? core::ITensor::fp16_to_fp32((*Y)[i])
                             : static_cast<float>((*Y)[i]);
        bad += std::isfinite(v) ? 0 : 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        cmd->submit(dev->getComputeQueue(0));
    }
    dev->wait_all_done();
    auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    const double flop = 2.0 * sh.oc * sh.hw * sh.hw * sh.ic * sh.k * sh.k;
    std::printf("%-28s %s: %8.2f ms  %5.2f TFLOP/s%s\n", sh.name,
                fp16 ? "fp16" : "fp32", ms, flop / ms / 1e9,
                bad ? "  NON-FINITE OUTPUT" : "");
}
} // namespace

int main(int argc, char **argv) {
    Logger::getInstance().setLevel(LOG_ERROR);
    dev::seed_shader_archive_from_build_tree();
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("no vulkan device\n");
        return 1;
    }
    const char *dev_env = std::getenv("BENCH_DEV");
    auto dev = std::make_shared<VulkanDevice>(
        phydevs[dev_env ? std::atoi(dev_env) : 0]);
    std::printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 5;

    const std::vector<Shape> shapes = {
        {288, 512, 288, 3, "3x3 288->288 @512"},
        {576, 256, 576, 3, "3x3 576->576 @256"},
        {1152, 128, 1152, 3, "3x3 1152->1152 @128"},
        {288, 512, 144, 3, "3x3 288->144 @512"},
        {144, 512, 144, 3, "3x3 144->144 @512"},
        {1152, 32, 1152, 3, "3x3 1152->1152 @32"},
        {288, 512, 144, 1, "1x1 288->144 @512"},
        {1152, 128, 576, 1, "1x1 1152->576 @128"},
    };
    for (const auto &sh : shapes) {
        run<float>(sh, dev, cmdpool, reps);
        run<uint16_t>(sh, dev, cmdpool, reps);
    }
    return 0;
}
