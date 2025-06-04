
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include <cassert>
#include <chrono>

#include "Tensor.hpp"
#include "VulkanDevice.hpp"
#include "VulkanInstance.hpp"
#include "logger.hpp"
#include "Conv2d.hpp"



using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandPool;
using vkop::ops::Conv2d;
using vkop::core::Tensor;

namespace {

template<typename T>
void generateWeight(std::shared_ptr<Tensor<T>> &weight, int group) {
    auto shape = weight->getTensorShape();
    auto ic = shape[0];
    auto oc = shape[1];
    auto kh = shape[2];
    auto kw = shape[3];
    auto *data_ptr = weight->data();

    for (int i = 0; i < group * (oc/group) * (ic/group) * kw *kh; i++) {
        auto data = ((((i / kw)% 1317) * ((i / kh) % 1317)) % 1317 + i / ic + i / oc + (((oc - i) % 1317) * ic) % 1317 + i * ((oc - i) % 1317)) % 1317;
        auto fdata = static_cast<T>(data % 255) / 255.0F / 1000.0F;
        *(data_ptr++) = fdata;
    }
}


/**
 * @brief 实现一个参考版本的2D卷积操作。
 * 
 * 该函数模拟了2D卷积操作的行为，包括支持填充模式、步幅、膨胀和分组卷积。
 * 它还计算了两种版本的输出：一种是直接应用偏置的输出，另一种是偏置单独应用后的输出。
 * 这对于测试某些硬件实现（例如 bf16 或 fp16）的精度差异非常有用。
 * 
 * @param input 一个扁平化的向量，表示输入张量，形状为 [batch, ic, ih, iw]。
 * @param weight 一个扁平化的向量，表示卷积核权重，形状为 [oc, ic/group, kh, kw]。
 * @param bias 一个向量，表示每个输出通道的偏置，大小为 [oc]。
 * @param output 一个引用，用于存储最终输出张量，形状为 [batch, oc, oh, ow]。
 * @param outputDataSeparateBias 一个引用，用于存储单独应用偏置的输出张量。
 * @param batch 输入的批量大小。
 * @param ic 输入通道数。
 * @param oc 输出通道数。
 * @param ih 输入张量的高度。
 * @param iw 输入张量的宽度。
 * @param pad_h 高度方向的填充大小。
 * @param pad_w 宽度方向的填充大小。
 * @param kh 卷积核的高度。
 * @param kw 卷积核的宽度。
 * @param stride_h 高度方向的步幅。
 * @param stride_w 宽度方向的步幅。
 * @param dilation_h 高度方向的膨胀系数。
 * @param dilation_w 宽度方向的膨胀系数。
 * @param group 分组卷积的组数。`ic` 和 `oc` 必须能被 `group` 整除。
 */
template<typename T>
void reference_conv2d(const T* input, const T* weight,
    const T* bias, std::vector<T>& output, std::vector<T>& outputDataSeparateBias, int batch, int ic, int oc,
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
    output.resize(batch * oh * ow * oc);
    outputDataSeparateBias.resize(batch * oh * ow * oc);

    // 每组的输出通道数和输入通道数
    int oc_group = oc / group;
    int ic_group = ic / group;

    // 遍历批量、输出通道、高度和宽度，nchw
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            int g_id = oz / oc_group; // 计算当前通道所属的组
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    float sum = 0; // 用于存储卷积结果
                    auto dest_offset = ((b * oc + oz) * oh + oy) * ow + ox; // 计算输出张量的偏移量

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
                                    x_value = input[(((b * ic + sz) * ih + iy) * iw + ix)];
                                }

                                // 获取卷积核的值
                                float y_value = weight[(((g_id * oc_group + oz % oc_group) * ic_group + sz % ic_group) * kh + ky) * kw + kx];

                                // 累加卷积结果
                                sum += x_value * y_value;
                            }
                        }
                    }

                    // 将卷积结果加上偏置并存储到输出张量
                    output[dest_offset] = sum + bias[oz];
                    outputDataSeparateBias[dest_offset] = sum;
                }
            }
        }
    }
}

}  // namespace

template<typename T>
class Conv2dTest {
public:
    int b_ = 1;
    int oc_ = 16;
    int ic_ = 16;
    int isw_ = 3;
    int ish_ = 4;
    int kw_ = 4;
    int kh_ = 4;
    int d_ = 1;
    int s_ = 2;
    int p_ = 2;


    std::shared_ptr<Tensor<T>> weight_data_;
    std::shared_ptr<Tensor<T>> bias_data_;

    std::shared_ptr<Tensor<T>> input_data_;
    std::vector<T> output_data_;
    std::vector<T> output_data_separate_bias_;

    Conv2dTest() {
        initTestData();
    }

private:
    void reference_conv2d_depthwise(int batch, int ic, int oc, int ih, int iw, int pad_h, int pad_w, int kh, int kw, int stride_h, int stride_w, int dilation_h, int dilation_w, int group) {
        weight_data_ = std::make_shared<Tensor<T>>(ic, oc, kh, kw);
        generateWeight(weight_data_, group);

        bias_data_ = std::make_unique<Tensor<T>>(oc, 1, 1, 1);
        auto *bias_ptr = bias_data_->data();
        for (int i = 0; i < oc; i++) {
            auto data =  (((i / kw) % 1317) * ((i / kh) % 1317) + i / ic + i / oc + (oc - i) * ic + i * (oc - i)) % 1317;
            auto float_data = static_cast<float>(data % 255) / 255.0F;
            data           = data * data;
            *(bias_ptr++) = float_data;
        }

        input_data_ = std::make_unique<Tensor<T>>(batch, ic, ih, iw);
        auto *input_ptr = input_data_->data();
        for (int i = 0; i < ih * iw * ic * batch; ++i) {
            auto data      = ((i / kw) % 1317) * ((i / kh) % 1317) + ((i / ic)% 1317) * ((i / oc) % 1317) + ((oc - i) % 1317) * ic + (i % 1317) * ((oc - i) % 1317);
            data = data % 1317;
            data           = (data * data) % 1317;
            auto float_data = static_cast<T>(data % 255) / 255.0F;
            *(input_ptr++) = (float_data);
        }
        
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        reference_conv2d(input_data_->data(), weight_data_->data(), bias_data_->data(), output_data_, output_data_separate_bias_, batch, ic, oc, ih, iw, pad_h, pad_w, kh, kw,
                        stride_h, stride_w, dilation_h, dilation_w, group);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        LOG_INFO("reference conv2d depthwise time: %f s", duration.count());
    }

    void initTestData() {
        srand(100);
        // for (int b = 1; b <= 2; b++) {
        //     for (int oc = 4; oc <= 16; oc *= 2) {
        //         for (int ic = oc; ic <= oc; ic++) {
        //             for (int isw = 1; isw <= 8; isw += 2) {
        //                 for (int ish = 1; ish <= 8; ish *= 2) {
        //                     for (int kw = 1; kw <= 4; kw++) {
        //                         for (int kh = 1; kh <= 4; kh++) {
        //                             for (int d = 1; d <= 2; d++) {
        //                                 for (int s = 1; s <= 2; s++) {
        //                                     for (int p = 0; p <= std::min(kw, kh); p++) {
                                                reference_conv2d_depthwise(b_, ic_, oc_, ish_, isw_, p_, p_, kh_, kw_, s_, s_, d_, d_, 1);
        //                                     }
        //                                 }
        //                             }
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
    }
};


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    Conv2dTest<float> tp;
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto *pdev : phydevs) {
            auto dev = std::make_shared<VulkanDevice>(pdev);
            if (dev->getDeviceName().find("llvmpipe")!= std::string::npos) {
                continue;
            }

            VkDevice device = dev->getLogicalDevice();
            VulkanCommandPool cmdpool(device, dev->getComputeQueueFamilyIndex());

            Conv2d cd(tp.ic_, tp.oc_, tp.kh_, tp.kw_, tp.s_, tp.s_, tp.p_, tp.p_, tp.d_, tp.d_, 1, true, vkop::ops::conv2d::PaddingMode::ZEROS);
            cd.set_runtime_device(pdev, dev, &cmdpool);

            auto output = std::make_shared<Tensor<float>>();
            // Ensure shared pointers are retained before cmd.submit
            cd.apply<float>(std::vector<std::shared_ptr<Tensor<float>>> {tp.input_data_, tp.weight_data_, tp.bias_data_}, std::vector<std::shared_ptr<Tensor<float>>> {output});
            auto *out_ptr = output->data();
            for (int i = 0; i < output->num_elements(); i++) {
                if (std::fabs(out_ptr[i] - tp.output_data_[i]) > 0.01) {
                    LOG_ERROR("Test Fail at (%d): %f, %f", i, out_ptr[i], tp.output_data_[i]);
                    return -1;
                }
            }
            LOG_INFO("Test Passed");

        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s\n", e.what());
        return EXIT_FAILURE;
    }

    return 0;
}