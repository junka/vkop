// Function::preprocess_jpg on a BUFFER-backed graph input.
//
// With backend_buffer (vkbench pins it) Runtime allocates every graph input as
// an SSBO. preprocess_jpg used to call copyToGPUImage unconditionally, which
// dynamic_pointer_casts the SSBO to VulkanImage, gets null, and dereferences it
// -- a segfault in the default `<model> <image>` benchmark path. The buffer
// branch must pack plain NCHW (no RGBA swizzle) and upload it as an SSBO.
//
// The fixture is a solid-color PPM (stb_image reads PNM by content) so the
// resize is a no-op and the expected value is one normalized constant per
// channel -- no reference implementation of stbir in the test.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>

#include "core/Tensor.hpp"
#include "core/function.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using namespace vkop;

namespace {

constexpr int kHeight = 8;
constexpr int kWidth = 8;
constexpr uint8_t kRgb[3] = {10, 20, 30};
const float kMean[3] = {0.485F, 0.456F, 0.406F};
const float kStd[3] = {0.229F, 0.224F, 0.225F};

int failures = 0;

void check(bool ok, const char *what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

const char *write_ppm() {
    const char *path = "/tmp/vkop_preprocess_jpg_buffer.ppm";
    FILE *f = std::fopen(path, "wb");
    if (f == nullptr) {
        return nullptr;
    }
    std::fprintf(f, "P6\n4 4\n255\n");
    for (int i = 0; i < 4 * 4; ++i) {
        std::fwrite(kRgb, 1, 3, f);
    }
    std::fclose(f);
    return path;
}

template <typename T>
void run_case(std::shared_ptr<VulkanDevice> dev,
              const std::shared_ptr<VulkanCommandPool> &pool, const char *path,
              const char *what) {
    auto input = std::make_shared<core::Tensor<T>>(
        std::vector<int>{1, 3, kHeight, kWidth});
    // Exactly what Runtime::LoadModel does for a buffer-backend graph input;
    // this is the storage form that made copyToGPUImage segfault.
    input->as_storage_buffer(dev);
    check(input->is_buffer_backed(), "input is buffer-backed (SSBO)");

    auto dims = core::Function::preprocess_jpg(
        path, pool, input, /*use_letterbox=*/false, core::NormMethod::IMAGENET);
    check(dims.first > 0 && dims.second > 0, "preprocess_jpg returned a size");

    input->copyToCPU(pool);
    double max_err = 0.0;
    for (int c = 0; c < 3; ++c) {
        const float want =
            (static_cast<float>(kRgb[c]) / 255.0F - kMean[c]) / kStd[c];
        const size_t plane = static_cast<size_t>(kHeight) * kWidth;
        for (size_t i = 0; i < plane; ++i) {
            const T raw = (*input)[c * plane + i];
            const float got = std::is_same_v<T, uint16_t>
                                  ? core::ITensor::fp16_to_fp32(raw)
                                  : static_cast<float>(raw);
            max_err = std::max(max_err, double(std::fabs(got - want)));
        }
    }
    const double tol = std::is_same_v<T, uint16_t> ? 2e-3 : 1e-5;
    std::printf("  %s max_err=%.3g tol=%.3g\n", what, max_err, tol);
    check(max_err <= tol, what);
}

} // namespace

int main() {
    Logger::getInstance().setLevel(LOG_ERROR);
    const char *path = write_ppm();
    if (path == nullptr) {
        std::printf("SKIP: cannot write the PPM fixture\n");
        return 0;
    }
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("SKIP: no Vulkan device\n");
        std::remove(path);
        return 0;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    std::printf("GPU: %s\n", dev->getDeviceName().c_str());

    run_case<float>(dev, pool, path, "preprocess_jpg NCHW fp32");
    run_case<uint16_t>(dev, pool, path, "preprocess_jpg NCHW fp16");

    std::remove(path);
    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures ? 1 : 0;
}
