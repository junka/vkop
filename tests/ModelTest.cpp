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
        int oh = ((ih + 2 * pad_h - (kh - 1) * dilation_h - 1) / stride_h) + 1;
        int ow = ((iw + 2 * pad_w - (kw - 1) * dilation_w - 1) / stride_w) + 1;

        assert(oc % group == 0 && ic % group == 0);

        if (oh <= 0 || ow <= 0) {
            output.clear();
            return;
        }

        output.resize(static_cast<size_t>(batch) * oh * ow * oc);

        int oc_group = oc / group;
        int ic_group = ic / group;

        for (int b = 0; b < batch; ++b) {
            for (int oz = 0; oz < oc; ++oz) {
                int g_id = oz / oc_group;

                for (int oy = 0; oy < oh; ++oy) {
                    for (int ox = 0; ox < ow; ++ox) {
                        float sum = 0;
                        for (int sz = g_id * ic_group; sz < (g_id + 1) * ic_group; ++sz) {

                            for (int ky = 0; ky < kh; ++ky) {
                                for (int kx = 0; kx < kw; ++kx) {
                                    int ix = (ox * stride_w) + (kx * dilation_w) - pad_w;
                                    int iy = (oy * stride_h) + (ky * dilation_h) - pad_h;
                                    float x_value = 0.0F;

                                    if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                        x_value = input[((((b * ic + sz) * ih + iy) * iw) + ix)];
                                    }

                                    float y_value = 0.F;
                                    y_value = (*weight)[((((oz * ic_group) + (sz % ic_group)) * kh + ky) * kw) + kx];

                                    sum += x_value * y_value;
                                }
                            }
                        }

                        auto dest_offset = (((b * oc + oz) * oh + oy) * ow) + ox;
                        output.at(dest_offset) = sum + (*bias)[oz];
                    }
                }
            }
        }
    }
};
}

// The whole model check, once per storage type. DataT is the container the
// graph inputs and the graph output actually use: the fp16 image backend stores
// halfs in uint16_t words, the buffer backend stores the model's own float32.
template <typename DataT>
int run_model_case(const std::shared_ptr<Runtime> &rt,
                   const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
                   const std::shared_ptr<Tensor<DataT>> &t1,
                   const std::shared_ptr<Tensor<DataT>> &t2) {
    ModelTest<DataT> test;
    test.initTestData(t1, t2);
    t1->copyToGPU(cmdpool);
    t2->copyToGPU(cmdpool);
    rt->Run();
    rt->ReadResult();
    auto bias = vkop::core::as_tensor<float>(rt->GetInitializer("conv.bias"));
    auto weight = vkop::core::as_tensor<int8_t>(rt->GetInitializer("conv.weight"));
    auto scale = vkop::core::as_tensor<float>(rt->GetInitializer("conv.weight_scale"));
    auto result = vkop::core::as_tensor<DataT>(rt->GetOutput("output"));
    if (!bias || !weight || !scale || !result) {
        printf("missing initializer or output tensor\n");
        return 1;
    }
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
    int batch = t1->getShape()[0];
    int ic = t1->getShape()[1];
    int ih = t1->getShape()[2];
    int iw = t1->getShape()[3];
    int oc = result->getShape()[1];
    int kh = weight->getShape()[2];
    int kw = weight->getShape()[3];
    test.reference_conv2d(test.expectedOutput.data(), ori_weight, bias,
                          ref_output_data, batch, ic, oc, ih, iw, 1, 1, kh, kw,
                          1, 1, 1, 1, 16);
    for (int i = 0; i < result->num_elements(); ++i) {
        // fp16 storage holds a raw half; float32 storage is already a float.
        const float got = (typeid(DataT) == typeid(float))
                              ? static_cast<float>((*result)[i])
                              : ITensor::fp16_to_fp32(result->at(i));
        if (std::isnan(got) || std::fabs(got - ref_output_data[i]) > 1e-2) {
            printf("Failed at %d, %.5f vs %.5f\n", i, got, ref_output_data[i]);
            return 1;
        }
    }
    return 0;
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
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

    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path, 0);
    // No set_backend_buffer call: LoadModel picks from the device (images when
    // present, buffers otherwise), and the dtype checks below accept whichever
    // containers that choice produces.
    rt->LoadModel();

    // The image backend hands every graph input back as an fp16 container and
    // its int8-conv chain writes an fp16 output. The buffer backend builds the
    // inputs from the dtype the file declares -- float32 for this model -- and
    // writes an output matching precision_ (also float32 here). Read whichever
    // container the runtime actually made: a null as_tensor used to reach
    // initTestData and crash rather than fail the test.
    if (auto f1 = vkop::core::as_tensor<float>(rt->GetInput("input_x1"))) {
        auto f2 = vkop::core::as_tensor<float>(rt->GetInput("input_x2"));
        if (!f2) {
            printf("float32 input_x1 but input_x2 is not float32\n");
            return 1;
        }
        return run_model_case(rt, cmdpool, f1, f2);
    }

    auto t1 = vkop::core::as_tensor<uint16_t>(rt->GetInput("input_x1"));
    auto t2 = vkop::core::as_tensor<uint16_t>(rt->GetInput("input_x2"));
    if (!t1 || !t2) {
        printf("graph inputs are neither float32 nor fp16\n");
        return 1;
    }

    return run_model_case(rt, cmdpool, t1, t2);
}