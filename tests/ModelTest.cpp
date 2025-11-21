#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include "core/runtime.hpp"

#include <bits/stdint-uintn.h>
#include <cstdio>
#include <unordered_set>
#include <random>
#include <vector>
#include <cmath>

#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::core::Runtime;

namespace {


template<typename T>
class ModelTest {
public:
    std::vector<T> expectedOutput;

    ModelTest() = default;
    void initTestData(const std::shared_ptr<Tensor<T>>& ta, const std::shared_ptr<Tensor<T>>& tb) {
        expectedOutput.resize(ta->num_elements());
        ta->reserveOnCPU();
        tb->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{-1.0F, 1.0F};
        std::normal_distribution<> inputb_dist{1.0F, 2.0F};
        for (int i = 0; i < ta->num_elements(); i++) {
            auto a = inputa_dist(gen);
            auto b = inputb_dist(gen);
            if (typeid(T) == typeid(uint16_t)) {
                (*ta)[i] = ITensor::fp32_to_fp16(a);
                (*tb)[i] = ITensor::fp32_to_fp16(b);
                expectedOutput[i] = ITensor::fp32_to_fp16(a+b);
            } else {
                (*ta)[i] = a;
                (*tb)[i] = b;
                expectedOutput[i] = a+b;
            }
        }
    }

    void reference_conv2d(const T* input, const std::shared_ptr<Tensor<T>>& weight,
        const std::shared_ptr<Tensor<T>>& bias, std::vector<T>& output, int batch, int ic, int oc,
        int ih, int iw, int pad_h, int pad_w, int kh, int kw, int stride_h, int stride_w,
        int dilation_h, int dilation_w, int group) {
        // 计算输出张量的高度和宽度
        int oh = (ih + 2 * pad_h - (kh - 1) * dilation_h - 1) / stride_h + 1;
        int ow = (iw + 2 * pad_w - (kw - 1) * dilation_w - 1) / stride_w + 1;

        // 确保输出通道数和输入通道数可以被组数整除
        assert(oc % group == 0 && ic % group == 0);

        // 如果输出尺寸无效（例如非正值），清空输出并返回
        if (oh <= 0 || ow <= 0) {
            output.clear();
            return;
        }

        // 初始化输出张量的大小
        output.resize(static_cast<size_t>(batch) * oh * ow * oc); // Ensure proper resizing

        // 每组的输出通道数和输入通道数
        int oc_group = oc / group;
        int ic_group = ic / group;

        // 遍历批量、输出通道、高度和宽度，nchw
        for (int b = 0; b < batch; ++b) {
            for (int oz = 0; oz < oc; ++oz) {
                int g_id = oz / oc_group; // 计算当前通道所属的组

                for (int oy = 0; oy < oh; ++oy) {
                    for (int ox = 0; ox < ow; ++ox) {
                        float sum = 0;
                        
                        // 遍历输入通道和卷积核
                        for (int sz = g_id * ic_group; sz < (g_id + 1) * ic_group; ++sz) {

                            for (int ky = 0; ky < kh; ++ky) {
                                for (int kx = 0; kx < kw; ++kx) {
                                    // 计算输入张量的索引
                                    int ix = ox * stride_w + kx * dilation_w - pad_w;
                                    int iy = oy * stride_h + ky * dilation_h - pad_h;
                                    float x_value = 0.0F;

                                    // 检查索引是否在输入张量范围内
                                    if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                        if (typeid(T) == typeid(float)) {
                                            x_value = input[(((b * ic + sz) * ih + iy) * iw + ix)];
                                        } else if (typeid(T) == typeid(uint16_t)) {
                                            x_value = vkop::core::ITensor::fp16_to_fp32(input[(((b * ic + sz) * ih + iy) * iw + ix)]);
                                        }
                                    }

                                    // 获取卷积核的值
                                    float y_value = 0.F;
                                    if (typeid(T) == typeid(float)) {
                                        y_value = (*weight)[(((oz * ic_group) + (sz % ic_group)) * kh + ky) * kw + kx];
                                    } else {
                                        y_value = vkop::core::ITensor::fp16_to_fp32((*weight)[(((oz * ic_group) + (sz % ic_group)) * kh + ky) * kw + kx]);
                                    }

                                    // 累加卷积结果
                                    sum += x_value * y_value;
                                }
                            }
                        }

                        // 将卷积结果加上偏置并存储到输出张量
                        // 计算输出张量的偏移量
                        auto dest_offset = ((b * oc + oz) * oh + oy) * ow + ox;
                        if (typeid(T) == typeid(float)) {
                            output.at(dest_offset) = sum + (*bias)[oz];
                        } else if (typeid(T) == typeid(uint16_t)) {
                            output.at(dest_offset) = sum + vkop::core::ITensor::fp16_to_fp32((*bias)[oz]);
                        }
                    }
                }
            }
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    std::shared_ptr<VulkanDevice> dev;
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto *pdev : phydevs) {
            auto vdev = std::make_shared<VulkanDevice>(pdev);
            if (vdev->getDeviceName().find("llvmpipe") != std::string::npos) {
                continue;
            }
            dev = vdev;
        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s", e.what());
        return EXIT_FAILURE;
    }

    LOG_INFO("%s",dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);
    std::string binary_file_path = TEST_DATA_PATH"/add_conv_model.bin";
    // This model has two inputs and one output,
    // one add and one conv2d operator

    auto rt = std::make_shared<Runtime>(dev, cmdpool, binary_file_path);
    rt->LoadModel();
#ifdef FP16
    auto t1 = vkop::core::as_tensor<uint16_t>(rt->GetInput("input_x1"));
    auto t2 = vkop::core::as_tensor<uint16_t>(rt->GetInput("input_x2"));
    ModelTest<uint16_t> test;
#else
    auto t1 = vkop::core::as_tensor<float>(rt->GetInput("input_x1"));
    auto t2 = vkop::core::as_tensor<float>(rt->GetInput("input_x2"));

    ModelTest<float> test;
#endif
    test.initTestData(t1, t2);
    rt->Run();
    rt->ReadResult();
#ifdef FP16
    auto bias = vkop::core::as_tensor<uint16_t>(rt->GetInitializer("conv.bias"));
    auto weight = vkop::core::as_tensor<uint16_t>(rt->GetInitializer("conv.weight"));
    auto result = vkop::core::as_tensor<float>(rt->GetOutput("output"));
    std::vector<uint16_t> ref_output_data;
#else
    auto bias = vkop::core::as_tensor<float>(rt->GetInitializer("conv.bias"));
    auto weight = vkop::core::as_tensor<float>(rt->GetInitializer("conv.weight"));
    auto result = vkop::core::as_tensor<float>(rt->GetOutput("output"));
    std::vector<float> ref_output_data;
#endif
    bias->copyToCPU(dev, cmdpool);
    weight->copyToCPU(dev, cmdpool);

    int batch = t1->getShape()[0];
    int ic = t1->getShape()[1];
    int ih = t1->getShape()[2];
    int iw = t1->getShape()[3];
    int oc = result->getShape()[1];
    int pad_h = 1;
    int pad_w = 1;
    int kh = weight->getShape()[2];
    int kw = weight->getShape()[3];
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int group = 1;
    test.reference_conv2d(test.expectedOutput.data(), weight, bias, ref_output_data, batch, ic, oc, ih, iw, pad_h, pad_w, kh, kw, stride_h, stride_w, dilation_h, dilation_w, group);
    for (int i = 0; i < result->num_elements(); ++i) {
        #ifdef FP16
        if (std::fabs(vkop::core::ITensor::fp16_to_fp32(result->at(i)) - vkop::core::ITensor::fp16_to_fp32(ref_output_data[i])) > 1e-5) {
            printf("Failed at %d, %.3f vs %.3f\n", i, vkop::core::ITensor::fp16_to_fp32(result->at(i)), vkop::core::ITensor::fp16_to_fp32((ref_output_data[i])));
            return 1;
        }

        #else
        if (std::fabs(result->at(i) - ref_output_data[i]) > 1e-5) {
            printf("Failed at %d, %.3f vs %.3f\n", i, result->at(i), ref_output_data[i]);
            return 1;
        }
        #endif
    }
    return 0;
}