// Copyright 2025 @junka
#include "function.hpp"
#include "core/runtime.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize2.h"

namespace vkop {
namespace core {
namespace {
const float kMean[] = {0.485F, 0.456F, 0.406F};
const float kStdvar[] = {0.229F, 0.224F, 0.225F};
} // namespace

Function::Function() = default;

void Function::preprocess_jpg(const char *input_file,
                              const std::shared_ptr<VulkanCommandPool> &cmdpool,
                              const std::shared_ptr<core::ITensor> &input) {
    int image_h;
    int image_w;
    int channels;
    auto *raw = stbi_load(input_file, &image_w, &image_h, &channels, 3);

    auto t = vkop::core::as_tensor<float>(input);
    int resize_h = t->getShape()[2];
    int resize_w = t->getShape()[3];
    auto *resized = static_cast<uint8_t *>(malloc(resize_h * resize_w * 3));
    stbir_resize_uint8_linear(raw, image_w, image_h, 0, resized, resize_w,
                              resize_h, 0, STBIR_RGB);

    stbi_image_free(raw);
    std::vector<float> normalized_data(resize_h * resize_w * 4);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < resize_h * resize_w; i++) {
            normalized_data[(i * 4) + c] =
                ((static_cast<float>(resized[(i * 3) + c]) / 255.0F) -
                 kMean[c]) /
                kStdvar[c];
        }
    }
    free(resized);

    // 1, 3, h, w, RGBA copy directly
    t->copyToGPUImage(cmdpool, normalized_data.data(), true);
    normalized_data.clear();
    normalized_data.shrink_to_fit();
}

} // namespace core
} // namespace vkop
