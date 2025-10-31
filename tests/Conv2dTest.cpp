
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

namespace {

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
    const T* bias, std::vector<T>& output, int batch, int ic, int oc,
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
                                // float y_value = weight[(((g_id * oc_group + oz % oc_group) * ic_group + sz % ic_group) * kh + ky) * kw + kx];
                                float y_value = weight[(((oz * ic_group) + (sz % ic_group)) * kh + ky) * kw + kx];


                                // 累加卷积结果
                                sum += x_value * y_value;
                            }
                        }
                    }

                    // 将卷积结果加上偏置并存储到输出张量
                    output.at(dest_offset) = sum + bias[oz];
                }
            }
        }
    }
}

}  // namespace


template<typename T>
class Conv2dTest: public TestCase {
public:
    std::vector<int> input_shape_ = {1, 16, 8, 8}; // b, ic, ih, iw
    int kernel_size_ = 4;
    int stride_ = 2;
    int pad_ = 0;
    int group_ = 1;
    int dilation_ = 1;
    int feature_size_ = 16; // oc

    std::unordered_map<std::string, std::string> attributes = {
        {"kernel_shape", std::to_string(kernel_size_)},
        {"strides", std::to_string(stride_)},
        {"pads", std::to_string(pad_)},
        {"dilations", std::to_string(dilation_)},
        {"group", "1"}
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

        const std::unordered_map<std::string, std::string> param = {{"kernel_shape", std::to_string(kernel_size_)},
                                                                    {"stride", std::to_string(stride_)},
                                                                    {"padding", std::to_string(pad_)},
                                                                    {"dilations", std::to_string(dilation_)},
                                                                    {"groups", "1"}};
        std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("conv2d", shapes, param);
        std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
        auto torch_output = torch_tensors[0];
        auto torch_input = torch_tensors[1];
        std::vector<int> output_shape = std::get<1>(k);

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

#if 1
        printf("\n===Input==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            printf("[\n");
            for (int j = 0; j < output_shape[1]; j++) {
                printf("[\n");
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] +
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
            for (int j = 0; j < output_shape[1]; j++) {
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] +
                                l;
                        printf("%.4f, ", torch_output[idx]);
                    }
                    printf("]\n");
                }
                printf("\n");
            }
            printf("\n");
        }
#endif
        input_data_ = std::make_shared<Tensor<float>>(input_shape_);
        for (int i = 0; i < input_data_->num_elements(); i++) {
            input_data_->at(i) = torch_input[i];
        }
        output_data_ = torch_output;
#if 0
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};

        // Initialize weight data
        // M, C/group, kh, kw
        weight_data_ = std::make_shared<Tensor<T>>(std::vector<int>{feature_size_, input_shape_[1] / group_, kernel_size_, kernel_size_});
        for (int i = 0; i < weight_data_->num_elements(); i++) {
            weight_data_->at(i) = static_cast<T>(input_dist(gen));
        }

        // Initialize bias data
        input_dist = std::normal_distribution<>(0.0F, 2.0F);
        bias_data_ = std::make_shared<Tensor<T>>(std::vector<int>{feature_size_});
        for (int i = 0; i < bias_data_->num_elements(); ++i) {
            bias_data_->at(i) = static_cast<T>(input_dist(gen));
        }

        // Initialize input data
        input_dist = std::normal_distribution<>(0.0F, 255.0F);
        input_data_ = std::make_shared<Tensor<T>>(input_shape_);
        for (int i = 0; i < input_data_->num_elements(); ++i) {
            input_data_->at(i) = static_cast<T>(input_dist(gen));
        }

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

        reference_conv2d(input_data_->data(), weight_data_->data(), bias_data_->data(),
                 output_data_, batch, ic, oc,
                 ih, iw, pad_h, pad_w, kh, kw, stride_h, stride_w,
                 dilation_h, dilation_w, group_);
        
        printf("%f\n", output_data_[0]);
#endif
    }
};


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    Conv2dTest<float> ct;

    std::vector<std::vector<int>> shapes;
    shapes.push_back(ct.input_shape_);
    shapes.push_back(std::vector<int>{ct.feature_size_, ct.input_shape_[1]/ct.group_, ct.kernel_size_, ct.kernel_size_});
    shapes.push_back(std::vector<int>{ct.feature_size_});
    std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("conv2d", shapes, ct.attributes);
    std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
    auto torch_output = torch_tensors[0];
    auto torch_input = torch_tensors[1];
    std::vector<int> output_shape = std::get<1>(k);
    printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    // for (int i = 0; i < static_cast<int>(torch_input.size()); i++) {
    //     if (i % 8 == 0) {
    //         printf("\n");
    //     }
    //     printf("%0.4f\t", torch_input[i]);
    // }

    if (!ct.run_test({ct.input_data_, ct.weight_data_, ct.bias_data_}, ct.output_data_,
        [&ct](std::unique_ptr<vkop::ops::Operator> &op) {
        auto *conv_op = dynamic_cast<Conv2d *>(op.get());
        if (!conv_op) {
            LOG_ERROR("Failed to cast operator to Softmax");
            return;
        }
        conv_op->setAttribute(ct.attributes);

    })) {
        return -1;
    }


    return 0;
}