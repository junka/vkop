
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"
#include "ops/Conv2d.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Conv2d;

#define USE_CPP_REFER 1

namespace {
#if USE_CPP_REFER

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
void reference_conv2d(const std::shared_ptr<Tensor<T>>& input, const std::shared_ptr<Tensor<T>>& weight,
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
                                    x_value = (*input)[(((b * ic + sz) * ih + iy) * iw + ix)];
                                }

                                // 获取卷积核的值
                                // float y_value = weight[(((g_id * oc_group + oz % oc_group) * ic_group + sz % ic_group) * kh + ky) * kw + kx];
                                float y_value = (*weight)[(((oz * ic_group) + (sz % ic_group)) * kh + ky) * kw + kx];


                                // 累加卷积结果
                                sum += x_value * y_value;
                            }
                        }
                    }

                    // 将卷积结果加上偏置并存储到输出张量
                    // 计算输出张量的偏移量
                    auto dest_offset = ((b * oc + oz) * oh + oy) * ow + ox;
                    output.at(dest_offset) = sum + (*bias)[oz];
                }
            }
        }
    }
}
#endif

}  // namespace


template<typename T>
class Conv2dTest: public TestCase {
public:
    std::vector<int> input_shape_ = {1, 16, 4, 4}; // b, ic, ih, iw
    int kernel_size_ = 2;
    int stride_ = 1;
    int pad_ = 0;
    int group_ = 2;
    int dilation_ = 1;
    int feature_size_ = 4; // oc

    std::unordered_map<std::string, std::string> attributes = {
        {"strides", std::to_string(stride_)},
        {"pads", std::to_string(pad_)},
        {"dilations", std::to_string(dilation_)},
        {"group", std::to_string(group_)},
        {"kernel_shape", std::to_string(kernel_size_)}
    };

    std::shared_ptr<Tensor<T>> weight_data_;
    std::shared_ptr<Tensor<T>> bias_data_;

    std::shared_ptr<Tensor<T>> input_data_;
    std::vector<T> output_data_;

    Conv2dTest(): TestCase("Conv2d") {
        initTestData();
    }

private:
    void initTestData() {
        std::vector<std::vector<int>> shapes;
        shapes.push_back(input_shape_);
        shapes.push_back(std::vector<int>{feature_size_, input_shape_[1]/group_, kernel_size_, kernel_size_});
        shapes.push_back(std::vector<int>{feature_size_});

        std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("conv2d", shapes, attributes);
        std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
        auto torch_output = torch_tensors[0];
        auto torch_input = torch_tensors[1];
        auto torch_weight = torch_tensors[2];
        auto torch_bias = torch_tensors[3];
        std::vector<int> output_shape = std::get<1>(k);

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

#if USE_CPP_REFER

        printf("\n===Input==============\n");
        for (int i = 0; i < input_shape_[0]; i++) {
            printf("[\n");
            for (int j = 0; j < input_shape_[1]; j++) {
                printf("[\n");
                for (int k = 0; k < input_shape_[2]; k++) {
                    printf("[");
                    for (int l = 0; l < input_shape_[3]; l++) {
                        int idx = i * input_shape_[1] * input_shape_[2] * input_shape_[3] +
                                j * input_shape_[2] * input_shape_[3] +
                                k * input_shape_[3] +
                                l;
                        printf("%.4f, ", torch_input[idx]);
                    }
                    printf("],\n");
                }
                printf("],\n");
            }
            printf("]\n");
        }

        printf("\n===Output==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            printf("[\n");
            for (int j = 0; j < output_shape[1]; j++) {
                printf("[\n");
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] + l;
                        printf("%.4f, ", torch_output[idx]);
                    }
                    printf("]\n");
                }
                printf("\n");
            }
            printf("\n");
        }

        printf("========weight ==============\n");
        for (int i = 0; i < feature_size_; i++) {
            printf("[\n");
            for (int j = 0; j < input_shape_[1] / group_; j++) {
                printf("[\n");
                for (int k = 0; k < kernel_size_; k++) {
                    printf("[");
                    for (int l = 0; l < kernel_size_; l++) {
                        int idx = i * input_shape_[1] / group_ * kernel_size_ * kernel_size_ +
                                j * kernel_size_ * kernel_size_ +
                                k * kernel_size_ + l;
                        printf("%.4f, ", torch_weight[idx]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("\n============bias ===========\n");
        for (int i = 0; i < feature_size_; i ++) {
            printf("%.4f, ", torch_bias[i]);
        }
#endif
        input_data_ = std::make_shared<Tensor<float>>(input_shape_);
        input_data_->fillToCPU(torch_input);
        output_data_ = torch_output;
        weight_data_ = std::make_shared<Tensor<T>>(std::vector<int>{feature_size_, input_shape_[1] / group_, kernel_size_, kernel_size_});
        weight_data_->fillToCPU(torch_weight);
        bias_data_ = std::make_shared<Tensor<T>>(std::vector<int>{feature_size_});
        bias_data_->fillToCPU(torch_bias);

#if USE_CPP_REFER

        int batch = input_shape_[0];
        int ic = input_shape_[1];
        int ih = input_shape_[2];
        int iw = input_shape_[3];
        int oc = feature_size_;
        int kh = kernel_size_;
        int kw = kernel_size_;
        int stride_h = stride_;
        int stride_w = stride_;
        int pad_h = pad_;
        int pad_w = pad_;
        int dilation_h = dilation_;
        int dilation_w = dilation_;
        std::vector<float> ref_output_data;
        reference_conv2d(input_data_, weight_data_, bias_data_,
                 ref_output_data, batch, ic, oc,
                 ih, iw, pad_h, pad_w, kh, kw, stride_h, stride_w,
                 dilation_h, dilation_w, group_);
        printf("\n===Reference Output==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            for (int j = 0; j < output_shape[1]; j++) {
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] + l;
                        printf("%.4f, ", ref_output_data[idx]);
                        if (fabs(ref_output_data[idx] - torch_output[idx]) > 0.0001F) {
                            printf("  <-- MISMATCH ");
                        }
                    }
                    printf("]\n");
                }
                printf("\n");
            }
            printf("\n");
        }
#endif
    }
};


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    Conv2dTest<float> ct;

    if (!ct.run_test({ct.input_data_, ct.weight_data_, ct.bias_data_}, ct.output_data_,
        [&ct](std::unique_ptr<vkop::ops::Operator> &op) {
        auto *conv_op = dynamic_cast<Conv2d *>(op.get());
        if (!conv_op) {
            LOG_ERROR("Failed to cast operator to Conv2d");
            return;
        }
        conv_op->setAttribute(ct.attributes);

    })) {
        return -1;
    }


    return 0;
}