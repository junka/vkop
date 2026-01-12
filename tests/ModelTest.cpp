#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
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
    std::vector<float> expectedOutput;

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
            } else if (typeid(T) == typeid(float)) {
                (*ta)[i] = a;
                (*tb)[i] = b;
            }
            expectedOutput[i] = a + b;
        }
    }

    void reference_conv2d(const float* input, const std::shared_ptr<Tensor<float>>& weight,
        const std::shared_ptr<Tensor<float>>& bias, std::vector<float>& output, int batch, int ic, int oc,
        int ih, int iw, int pad_h, int pad_w, int kh, int kw, int stride_h, int stride_w,
        int dilation_h, int dilation_w, int group) {
        // 计算输出张量的高度和宽度
        int oh = ((ih + 2 * pad_h - (kh - 1) * dilation_h - 1) / stride_h) + 1;
        int ow = ((iw + 2 * pad_w - (kw - 1) * dilation_w - 1) / stride_w) + 1;

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
                                    int ix = (ox * stride_w) + (kx * dilation_w) - pad_w;
                                    int iy = (oy * stride_h) + (ky * dilation_h) - pad_h;
                                    float x_value = 0.0F;

                                    // 检查索引是否在输入张量范围内
                                    if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                        x_value = input[((((b * ic + sz) * ih + iy) * iw) + ix)];
                                    }

                                    // 获取卷积核的值
                                    float y_value = 0.F;
                                    y_value = (*weight)[((((oz * ic_group) + (sz % ic_group)) * kh + ky) * kw) + kx];

                                    // 累加卷积结果
                                    sum += x_value * y_value;
                                }
                            }
                        }

                        // 将卷积结果加上偏置并存储到输出张量
                        // 计算输出张量的偏移量
                        auto dest_offset = (((b * oc + oz) * oh + oy) * ow) + ox;
                        output.at(dest_offset) = sum + (*bias)[oz];
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
    auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        printf("no valid vulkan device\n");
        return -1;
    }

    LOG_INFO("%s",dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);
    std::string binary_file_path = TEST_DATA_PATH"/add_conv_model.vkopbin";
    // This model has two inputs and one output,
    // one add and one conv2d operator

    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path);
    rt->LoadModel();

    auto t1 = vkop::core::as_tensor<uint16_t>(rt->GetInput("input_x1"));
    auto t2 = vkop::core::as_tensor<uint16_t>(rt->GetInput("input_x2"));

    ModelTest<uint16_t> test;
    test.initTestData(t1, t2);
    t1->copyToGPU(cmdpool);
    t2->copyToGPU(cmdpool);
    rt->Run();
    rt->ReadResult();
    auto bias = vkop::core::as_tensor<float>(rt->GetInitializer("conv.bias"));
    auto weight = vkop::core::as_tensor<int8_t>(rt->GetInitializer("conv.weight"));
    auto scale = vkop::core::as_tensor<float>(rt->GetInitializer("conv.weight_scale"));
    auto result = vkop::core::as_tensor<float>(rt->GetOutput("output"));
    std::vector<float> ref_output_data;
    bias->copyToCPU(cmdpool);
    weight->copyToCPU(cmdpool);
    scale->copyToCPU(cmdpool);
    auto ori_weight = std::make_shared<Tensor<float>>(weight->getShape());
    ori_weight->reserveOnCPU();
    auto weight_shape = weight->getShape();
    for (int i = 0; i < weight_shape[0]; ++i) {
        for (int j = 0; j < weight_shape[1] * weight_shape[2] * weight_shape[3]; ++j) {
            ori_weight->at((i * weight_shape[1] * weight_shape[2] * weight_shape[3]) + j) =
                static_cast<float>(weight->at((i * weight_shape[1] * weight_shape[2] * weight_shape[3]) + j)) * scale->at(i);
        }
    }
#if 0
    printf("input : [%d, %d, %d, %d]\n", t1->get_batch(), t1->get_channel(), t1->get_height(), t1->get_width());
    for (int i = 0; i < t1->get_batch(); ++i) {
        printf("[\n");
        for (int j = 0; j < t1->get_channel(); ++j) {
            printf("[\n");
            for (int k = 0; k < t1->get_height(); ++k) {
                printf("[");
                for (int l = 0; l < t1->get_width(); ++l) {
                    printf("%f, ", test.expectedOutput[(((i * t1->get_channel() + j) * t1->get_height() + k) * t1->get_width()) + l]);
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("Scale is:\n");
    for (int i = 0 ; i < weight_shape[0]; ++i) {
        printf("%f, ", scale->at(i));
    }
    printf("\n");
    for (int i = 0; i < weight_shape[0]; ++i) {
        printf("[");
        for (int j = 0; j < weight_shape[1]; ++j) {
            printf("[");
            for (int k = 0; k < weight_shape[2]; ++k) {
                printf("[");
                for (int l = 0; l < weight_shape[3]; ++l) {
                    printf("%f, ", ori_weight->at((i * weight_shape[1] * weight_shape[2] * weight_shape[3]) + (j * weight_shape[2] * weight_shape[3]) + (k * weight_shape[3]) + l));
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("bias is:\n");
    for (int i = 0; i < bias->getShape()[0]; ++i) {
        printf("%f, ", bias->at(i));
    }
    printf("\n");
#endif
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
    int group = 16;
    test.reference_conv2d(test.expectedOutput.data(), ori_weight, bias, ref_output_data, batch, ic, oc, ih, iw, pad_h, pad_w, kh, kw, stride_h, stride_w, dilation_h, dilation_w, group);
#if 0
    auto output_shape = result->getShape();
    printf("Reference output : [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    for (int i = 0; i < output_shape[0]; ++i) {
        printf("[\n");
        for (int j = 0; j < output_shape[1]; ++j) {
            printf("[\n");
            for (int k = 0; k < output_shape[2]; ++k) {
                printf("[");
                for (int l = 0; l < output_shape[3]; ++l) {
                    printf("%f, ", ref_output_data[(((i * output_shape[1] + j) * output_shape[2] + k) * output_shape[3]) + l]);
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("num_elements: %d\n", result->num_elements());
#endif
    for (int i = 0; i < result->num_elements(); ++i) {
        if (std::isnan(result->at(i)) || std::fabs(result->at(i) - ref_output_data[i]) > 1e-2) {
            printf("Failed at %d, %.5f vs %.5f\n", i, result->at(i), ref_output_data[i]);
            return 1;
        }
    }
    return 0;
}