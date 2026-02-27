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
constexpr uint8_t kLetterboxPadColor = 114;
} // namespace

/*
 * imagenet = true:
 * normlize to [0,1] + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
 * 0.225]) ResNet, ResNeXt, DenseNet, EfficientNet, MobileNet, Vit, Swin
 * Transformer, ConvNeXt, RegNet, Inception
 * imagenet = false:
 * normlize: img = img / 255.0
 */
std::pair<int, int>
Function::preprocess_jpg(const char *input_file,
                         const std::shared_ptr<VulkanCommandPool> &cmdpool,
                         const std::shared_ptr<core::ITensor> &input,
                         bool letterbox, NormMethod method) {
    int image_h;
    int image_w;
    int channels;
    auto *raw = stbi_load(input_file, &image_w, &image_h, &channels, 3);
    if (!raw) {
        std::cerr << "Failed to load image: " << input_file << std::endl;
        return {};
    }
    int resize_h = input->getShape()[2];
    int resize_w = input->getShape()[3];

    uint8_t *processed_image = nullptr;

    if (letterbox) {
        // Letterbox resize for YOLO models - maintains aspect ratio
        float scale = std::min(static_cast<float>(resize_w) / image_w,
                               static_cast<float>(resize_h) / image_h);

        int new_w = static_cast<int>(image_w * scale);
        int new_h = static_cast<int>(image_h * scale);

        // Allocate memory for scaled image
        auto *scaled = static_cast<uint8_t *>(malloc(new_h * new_w * 3));
        if (!scaled) {
            std::cerr << "Failed to allocate memory for scaled image"
                      << std::endl;
            return {};
        }
        // Resize with maintained aspect ratio
        stbir_resize_uint8_linear(raw, image_w, image_h, 0, scaled, new_w,
                                  new_h, 0, STBIR_RGB);

        // Create letterbox image with padding
        processed_image =
            static_cast<uint8_t *>(calloc(resize_h * resize_w * 3, 1));
        if (!processed_image) {
            std::cerr << "Failed to allocate memory for processed image"
                      << std::endl;
            return {};
        }
        // Fill with padding color
        memset(processed_image, kLetterboxPadColor, resize_h * resize_w * 3);

        // Copy scaled image to center of letterbox
        int pad_x = (resize_w - new_w) / 2;
        int pad_y = (resize_h - new_h) / 2;
        std::cout << "Padding letterbox (" << pad_y << ", " << pad_x << ")"
                  << std::endl;

        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                for (int c = 0; c < 3; c++) {
                    int src_idx = ((y * new_w + x) * 3) + c;
                    int dst_idx =
                        (((pad_y + y) * resize_w + (pad_x + x)) * 3) + c;
                    processed_image[dst_idx] = scaled[src_idx];
                }
            }
        }

        free(scaled);
    } else {
        processed_image =
            static_cast<uint8_t *>(malloc(resize_h * resize_w * 3));
        if (!processed_image) {
            std::cerr << "Failed to allocate memory for processed image"
                      << std::endl;
            return {};
        }
        stbir_resize_uint8_linear(raw, image_w, image_h, 0, processed_image,
                                  resize_w, resize_h, 0, STBIR_RGB);
    }

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
                normalized_data[(i * 4) + c] = normalize(
                    static_cast<float>(processed_image[(i * 3) + c]), c);
            }
        }
        auto t = vkop::core::as_tensor<float>(input);
        t->copyToGPUImage(cmdpool, normalized_data.data(), true);
    } else if (input->dtype() == typeid(uint16_t)) {
        std::vector<uint16_t> normalized_data(resize_h * resize_w * 4);
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < resize_h * resize_w; i++) {
                normalized_data[(i * 4) + c] =
                    vkop::core::ITensor::fp32_to_fp16(normalize(
                        static_cast<float>(processed_image[(i * 3) + c]), c));
            }
        }
        auto t = vkop::core::as_tensor<uint16_t>(input);
        t->copyToGPUImage(cmdpool, normalized_data.data(), true);
    }
    free(processed_image);
    return {image_h, image_w};
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
