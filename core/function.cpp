// Copyright 2025 @junka
#include "function.hpp"
#include "core/runtime.hpp"

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize2.h"
}

namespace vkop {
namespace core {
namespace {
const float kImagenetMean[] = {0.485F, 0.456F, 0.406F};
const float kImagenetStdvar[] = {0.229F, 0.224F, 0.225F};
const float kClipMean[] = {0.48145466F, 0.4578275F, 0.40821073F};
const float kClipStdvar[] = {0.26862954F, 0.26130258F, 0.27577711F};
} // namespace

/*
 * imagenet = true:
 * 像素值缩放到 [0,1] + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
 * 0.225]) ResNet, ResNeXt, DenseNet, EfficientNet, MobileNet, Vit, Swin
 * Transformer, ConvNeXt, RegNet, Inception
 * imagenet = false:
 * 像素值归一化：img = img / 255.0, 不需要mean std 标准化
 */
void Function::preprocess_jpg(const char *input_file,
                              const std::shared_ptr<VulkanCommandPool> &cmdpool,
                              const std::shared_ptr<core::ITensor> &input,
                              NormMethod method) {
    int image_h;
    int image_w;
    int channels;
    auto *raw = stbi_load(input_file, &image_w, &image_h, &channels, 3);

    int resize_h = input->getShape()[2];
    int resize_w = input->getShape()[3];
    auto *resized = static_cast<uint8_t *>(malloc(resize_h * resize_w * 3));
    stbir_resize_uint8_linear(raw, image_w, image_h, 0, resized, resize_w,
                              resize_h, 0, STBIR_RGB);

    stbi_image_free(raw);

    auto normalize = [&method](float val, int c) -> float {
        switch (method) {
        case NormMethod::IMAGENET:
            return (val / 255.0F - kImagenetMean[c]) / kImagenetStdvar[c];
        case NormMethod::INCEPTION:
            return (val / 127.5F) - 1.0F;
        case NormMethod::CLIP:
            return (val / 255.0F - kClipMean[c]) / kClipStdvar[c];
        default:
            return val / 255.0F;
        }
    };

    // 1, 3, h, w, RGBA copy directly
    if (input->dtype() == typeid(float)) {
        std::vector<float> normalized_data(resize_h * resize_w * 4);
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < resize_h * resize_w; i++) {
                normalized_data[(i * 4) + c] =
                    normalize(static_cast<float>(resized[(i * 3) + c]), c);
            }
        }
        auto t = vkop::core::as_tensor<float>(input);
        t->copyToGPUImage(cmdpool, normalized_data.data(), true);
    } else if (input->dtype() == typeid(uint16_t)) {
        std::vector<uint16_t> normalized_data(resize_h * resize_w * 4);
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < resize_h * resize_w; i++) {
                normalized_data[(i * 4) + c] =
                    vkop::core::ITensor::fp32_to_fp16(
                        normalize(static_cast<float>(resized[(i * 3) + c]), c));
            }
        }
        auto t = vkop::core::as_tensor<uint16_t>(input);
        t->copyToGPUImage(cmdpool, normalized_data.data(), true);
    }
    free(resized);
}

std::vector<std::pair<int, float>>
Function::get_top_k_predictions(const std::vector<float> &probs, int k) {
    std::vector<float> softmax_probs = probs;

    float max_val =
        *std::max_element(softmax_probs.begin(), softmax_probs.end());
    float sum = 0.0F;
    for (auto &val : softmax_probs) {
        val = std::exp(val - max_val);
        sum += val;
    }

    for (auto &val : softmax_probs) {
        val /= sum;
    }

    std::vector<std::pair<int, float>> indexed_probs;
    indexed_probs.reserve(softmax_probs.size());
    for (size_t i = 0; i < softmax_probs.size(); ++i) {
        indexed_probs.emplace_back(i, softmax_probs[i]);
    }

    std::sort(
        indexed_probs.begin(), indexed_probs.end(),
        [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
            return a.second > b.second;
        });

    if (indexed_probs.size() > static_cast<size_t>(k)) {
        indexed_probs.resize(k);
    }

    return indexed_probs;
}

} // namespace core
} // namespace vkop
