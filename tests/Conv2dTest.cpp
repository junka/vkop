
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cassert>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"
#include "ops/Conv2d.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Conv2d;

#define USE_CPP_REFER 0

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
    const std::shared_ptr<Tensor<T>>& bias, std::vector<float>& output, int batch, int ic, int oc,
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
                                        x_value = (*input)[(((b * ic + sz) * ih + iy) * iw + ix)];
                                    } else if (typeid(T) == typeid(uint16_t)) {
                                        x_value = vkop::core::ITensor::fp16_to_fp32((*input)[(((b * ic + sz) * ih + iy) * iw + ix)]);
                                    }
                                }

                                // 获取卷积核的值
                                float y_value = 0.F;
                                if (typeid(T) == typeid(float)) {
                                    y_value = (*weight)[(((oz * ic_group) + (sz % ic_group)) * kh + ky) * kw + kx];
                                } else if (typeid(T) == typeid(uint16_t)) {
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
#endif

}  // namespace

template<typename T>
class Conv2dTest: public TestCase<T> {
public:
    std::vector<int> input_shape_ = {1, 10, 7, 7}; // b, ic, ih, iw
    int kernel_size_ = 2;
    int stride_ = 1;
    int pad_ = 0;
    int group_ = 5;
    int dilation_ = 1;
    int feature_size_ = 5; // oc

    std::unordered_map<std::string, std::string> attributes;

    std::shared_ptr<Tensor<T>> weight_data_;
    std::shared_ptr<Tensor<T>> bias_data_;

    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;

    Conv2dTest(const std::vector<int>& input_shape, int kernel_size, int stride, 
               int pad, int group, int dilation, int feature_size)
        : TestCase<T>("Conv2d"), input_shape_(input_shape), kernel_size_(kernel_size), stride_(stride), 
          pad_(pad), group_(group), dilation_(dilation), feature_size_(feature_size) {
          
        attributes = {
            {"strides", std::to_string(stride_)},
            {"pads", std::to_string(pad_)},
            {"dilations", std::to_string(dilation_)},
            {"group", std::to_string(group_)},
            {"kernel_shape", std::to_string(kernel_size_)}
        };
        initTestData();
    }

private:
    void initTestData() {
        torch::manual_seed(42);
        auto torch_input = torch::randn({input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]}, this->getTorchConf());
        auto torch_weight = torch::randn({feature_size_, input_shape_[1]/group_, kernel_size_, kernel_size_}, this->getTorchConf());
        auto torch_bias = torch::randn({feature_size_}, this->getTorchConf());
        auto torch_output = torch::conv2d(torch_input, torch_weight, torch_bias, torch::IntArrayRef({stride_, stride_}),
            torch::IntArrayRef({pad_, pad_}),
            torch::IntArrayRef({dilation_, dilation_}),
            group_);
        auto output_flat = torch_output.flatten();

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        printf("\n===Input==============\n");
        std::cout << torch_input << std::endl;

        printf("\n===Output==============\n");
        std::cout << torch_output << std::endl;

        printf("========weight ==[%d, %d, %d, %d]============\n", feature_size_, input_shape_[1] / group_, kernel_size_, kernel_size_);
        std::cout << torch_weight << std::endl;

        printf("\n============bias ===========\n");
        std::cout << torch_bias << std::endl;
        bias_data_ = std::make_shared<Tensor<T>>(std::vector<int>{feature_size_});
        this->fillTensorFromTorch(bias_data_, torch_bias);

        input = std::make_shared<Tensor<T>>(input_shape_);
        this->fillTensorFromTorch(input, torch_input);
        output = std::make_shared<Tensor<T>>(output_shape);
        this->fillTensorFromTorch(output, torch_output);

        weight_data_ = std::make_shared<Tensor<T>>(std::vector<int>{feature_size_, input_shape_[1] / group_, kernel_size_, kernel_size_});
        weight_data_->set_transpose();
        this->fillTensorFromTorch(weight_data_, torch_weight);

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
        reference_conv2d(input, weight_data_, bias_data_,
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

TEST(Conv2dTest, Conv2dComprehensiveTest) {

    std::vector<std::tuple<std::vector<int>, int, int, int, int, int, int>> test_cases = {
        {{1, 10, 7, 7}, 2, 1, 0, 5, 1, 5},    // Group convolution
        {{1, 6, 8, 8}, 3, 1, 1, 2, 1, 4},     // With padding
        {{2, 4, 10, 10}, 3, 2, 1, 1, 1, 8},   // Batch size > 1
        {{1, 8, 6, 6}, 2, 1, 0, 4, 2, 4},     // Dilated convolution
        {{1, 16, 5, 5}, 1, 1, 0, 1, 1, 32},   // 1x1 convolution
        {{1, 16, 5, 5}, 1, 1, 0, 16, 1, 32},   // 1x1 group convolution

        {{1, 1, 224, 224}, 3, 1, 1, 1, 1, 32}, // Large input typical in CNNs
        {{4, 3, 32, 32}, 3, 1, 1, 1, 1, 16},   // Larger batch size
        {{1, 32, 8, 8}, 3, 1, 2, 1, 1, 64},    // Larger padding (pad > kernel/2)
        {{1, 12, 15, 15}, 5, 2, 2, 3, 1, 24},  // Larger kernel with stride
        {{1, 16, 7, 7}, 7, 1, 0, 1, 1, 32},    // Kernel size equals input size
        {{1, 4, 5, 5}, 3, 2, 1, 4, 1, 8},      // Stride > 1 with groups
        {{1, 8, 8, 8}, 3, 1, 1, 2, 2, 16},     // Dilation > 2 with groups
        {{1, 1, 3, 3}, 3, 1, 1, 1, 1, 1},      // Minimal case
    };
    for (const auto& test_case : test_cases) {
        auto [input_shape, kernel_size, stride, pad, group, dilation, feature_size] = test_case;
        LOG_INFO("Running test case: input=%d,%d,%d,%d, kernel=%d, stride=%d, pad=%d, group=%d, dilation=%d, feature_size=%d",
               input_shape[0], input_shape[1], input_shape[2], input_shape[3],
               kernel_size, stride, pad, group, dilation, feature_size);
        
        Conv2dTest<uint16_t> ct16(input_shape, kernel_size, stride, pad, group, dilation, feature_size);
        EXPECT_TRUE(ct16.run_test({ct16.input, ct16.weight_data_, ct16.bias_data_}, {ct16.output},
            [&ct16](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *conv_op = dynamic_cast<Conv2d *>(op.get());
            if (!conv_op) {
                LOG_ERROR("Failed to cast operator to Conv2d");
                return;
            }
            conv_op->setAttribute(ct16.attributes);
        }));

        Conv2dTest<float> ct(input_shape, kernel_size, stride, pad, group, dilation, feature_size);
        EXPECT_TRUE(ct.run_test({ct.input, ct.weight_data_, ct.bias_data_}, {ct.output},
            [&ct](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *conv_op = dynamic_cast<Conv2d *>(op.get());
            if (!conv_op) {
                LOG_ERROR("Failed to cast operator to Conv2d");
                return;
            }
            conv_op->setAttribute(ct.attributes);
        }));
    }
}