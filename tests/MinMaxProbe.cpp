// Minimal repro probe for the full-DiT decode DEVICE_LOST at /Min_24 + /Max_24
// ([1,1024,4096], scalar in1). Runs BufferBinaryFactory's Min/Max impls
// directly at the failing size, sweeping the suspected variables:
//   - in1 as rank-0 (empty shape) vs rank-1 [1]
//   - fp16 vs fp32 data
//   - total = 4194304 (full) vs 6144 (tiny, known-good control)
// Usage: MinMaxProbe [full|tiny]  (default full)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Min.hpp"
#include "ops/Max.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;

namespace {

int run_case(vkop::ops::OpType op_type, const char *op_name,
             const std::vector<int> &in_shape,
             const std::vector<int> &scalar_shape, bool fp16,
             std::shared_ptr<VulkanDevice> dev,
             std::shared_ptr<VulkanCommandPool> cmdpool) {
    int total = 1;
    for (int d : in_shape) total *= (d > 0 ? d : 1);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);

    std::shared_ptr<ITensor> in0, in1, out;
    std::vector<float> a(total);
    for (int i = 0; i < total; ++i) a[i] = dist(rng);

    if (fp16) {
        auto t = std::make_shared<Tensor<uint16_t>>();
        t->resize(in_shape);
        std::vector<uint16_t> ah(total);
        for (int i = 0; i < total; ++i) ah[i] = ITensor::fp32_to_fp16(a[i]);
        t->fillToCPU(ah);
        in0 = t;
        // Compare against the round-tripped fp16 values, not the pre-rounding
        // fp32 ones: fp16 rounding error (~1e-3 at |x|≈8) exceeds the check
        // threshold and would masquerade as a shader bug.
        for (int i = 0; i < total; ++i) a[i] = ITensor::fp16_to_fp32(ah[i]);
        auto o = std::make_shared<Tensor<uint16_t>>();
        o->resize(in_shape);
        out = o;
        auto s = std::make_shared<Tensor<uint16_t>>();
        s->resize(scalar_shape);
        s->fillToCPU(std::vector<uint16_t>{ITensor::fp32_to_fp16(6.0f)});
        in1 = s;
    } else {
        auto t = std::make_shared<Tensor<float>>();
        t->resize(in_shape);
        t->fillToCPU(a);
        in0 = t;
        auto o = std::make_shared<Tensor<float>>();
        o->resize(in_shape);
        out = o;
        auto s = std::make_shared<Tensor<float>>();
        s->resize(scalar_shape);
        s->fillToCPU(std::vector<float>{6.0f});
        in1 = s;
    }

    std::vector<std::shared_ptr<ITensor>> inputs{in0, in1};
    std::vector<std::shared_ptr<ITensor>> outputs{out};

    // Upload inputs exactly the way the test infra / runtime does.
    if (fp16) {
        vkop::core::as_tensor<uint16_t>(in0)->as_storage_buffer(dev);
        vkop::core::as_tensor<uint16_t>(in0)->copyToGPU(cmdpool);
        vkop::core::as_tensor<uint16_t>(in1)->as_storage_buffer(dev);
        vkop::core::as_tensor<uint16_t>(in1)->copyToGPU(cmdpool);
    } else {
        vkop::core::as_tensor<float>(in0)->as_storage_buffer(dev);
        vkop::core::as_tensor<float>(in0)->copyToGPU(cmdpool);
        vkop::core::as_tensor<float>(in1)->as_storage_buffer(dev);
        vkop::core::as_tensor<float>(in1)->copyToGPU(cmdpool);
    }

    std::unique_ptr<vkop::ops::Operator> op;
    try {
        if (op_type == vkop::ops::OpType::MIN)
            op = std::make_unique<vkop::ops::MinBuffer>(fp16 ? 1 : 0);
        else
            op = std::make_unique<vkop::ops::MaxBuffer>(fp16 ? 1 : 0);
    } catch (const std::exception &e) {
        printf("[%s] construct FAILED: %s\n", op_name, e.what());
        return 1;
    }
    op->set_runtime_device(dev, cmdpool);

    printf("[%s] total=%d in1_rank=%d fp16=%d ... executing\n", op_name, total,
           (int)scalar_shape.size(), fp16 ? 1 : 0);
    fflush(stdout);
    try {
        op->onExecute(inputs, outputs, 0);
        auto cmd = op->get_record();
        std::vector<VkSubmitInfo> info;
        info.push_back(cmd->buildSubmitInfo());
        VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
        cmd->wait();
        dev->wait_all_done();
    } catch (const std::exception &e) {
        printf("[%s] execute FAILED: %s\n", op_name, e.what());
        return 1;
    }

    int bad = 0;
    if (fp16) {
        vkop::core::as_tensor<uint16_t>(out)->copyToCPU(cmdpool);
    } else {
        vkop::core::as_tensor<float>(out)->copyToCPU(cmdpool);
    }
    for (int i = 0; i < total; ++i) {
        float e = (op_type == vkop::ops::OpType::MIN) ? std::min(a[i], 6.0f)
                                                      : std::max(a[i], 6.0f);
        float v;
        if (fp16) {
            v = ITensor::fp16_to_fp32(
                (*vkop::core::as_tensor<uint16_t>(out))[i]);
        } else {
            v = (*vkop::core::as_tensor<float>(out))[i];
        }
        if (std::fabs(v - e) > 1e-3f) {
            if (bad < 5)
                printf("[%s] MISMATCH i=%d got %f want %f\n", op_name, i, v, e);
            ++bad;
        }
    }
    printf("[%s] total=%d in1_rank=%d fp16=%d : %s (%d/%d bad)\n", op_name,
           total, (int)scalar_shape.size(), fp16 ? 1 : 0,
           bad == 0 ? "PASS" : "FAIL", bad, total);
    return bad == 0 ? 0 : 2;
}

} // namespace

int main(int argc, char **argv) {
    bool full = (argc < 2) || std::string(argv[1]) == "full";

    Logger::getInstance().setLevel(LOG_INFO);
    const auto &phydevs =
        vkop::VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);

    std::vector<int> shape = full ? std::vector<int>{1, 1024, 4096}
                                  : std::vector<int>{1, 64, 96};
    int rc = 0;
    for (int fp16 : {1, 0}) {
        for (int rank0 : {1, 0}) {
            std::vector<int> scalar_shape =
                rank0 ? std::vector<int>{} : std::vector<int>{1};
            rc |= run_case(vkop::ops::OpType::MIN, "Min", shape, scalar_shape,
                           fp16 != 0, dev, cmdpool);
            rc |= run_case(vkop::ops::OpType::MAX, "Max", shape, scalar_shape,
                           fp16 != 0, dev, cmdpool);
        }
    }
    printf(rc == 0 ? "ALL PASS\n" : "FAILURES PRESENT\n");
    return rc;
}
