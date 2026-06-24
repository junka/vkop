// Copyright 2025 @junka
#include "function.hpp"
#include "core/runtime.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
        stbir_resize(raw, image_w, image_h, 0, processed_image, resize_w,
                     resize_h, 0, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_TRIANGLE);
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

namespace {
struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<char> data; // little-endian
    bool little_endian = true;
    char dtype_kind = 'f'; // 'f' float
    int dtype_size = 4;
};

inline char read_u1(std::istream &is) {
    char c;
    is.read(&c, 1);
    return c;
}

inline std::string read_magic_string(std::istream &is) {
    char magic[6];
    is.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not a npy file (bad magic)");
    }
    return std::string(magic, 6);
}

inline NpyArray parse_npy(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open npy: " + path);
    }
    read_magic_string(f);
    // version: major, minor
    int major = static_cast<unsigned char>(read_u1(f));
    static_cast<void>(read_u1(f)); // minor version, ignored

    // header len
    size_t header_len = 0;
    if (major == 1) {
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char *>(&hlen), 2);
        header_len = hlen;
    } else if (major == 2) {
        uint32_t hlen = 0;
        f.read(reinterpret_cast<char *>(&hlen), 4);
        header_len = hlen;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // parse dict: {'descr': '<f4', 'fortran_order': False, 'shape':
    // (1,3,224,224), }
    NpyArray arr;
    // descr：locate 'descr' key，strings as alue
    // header shape: {'descr': '<f4', 'fortran_order': False, 'shape':
    // (1,3,224,224), }
    {
        auto key = header.find("'descr'");
        if (key == std::string::npos) {
            throw std::runtime_error("npy header missing 'descr'");
        }
        auto vstart = header.find('\'', key + 7); // skip 'descr'
        auto vend = header.find('\'', vstart + 1);
        if (vstart == std::string::npos || vend == std::string::npos) {
            throw std::runtime_error("npy header bad descr value");
        }
        std::string descr = header.substr(vstart + 1, vend - vstart - 1);
        if (!descr.empty()) {
            arr.little_endian = (descr[0] == '<' || descr[0] == '|');
            arr.dtype_kind = descr[descr.size() - 2];
            arr.dtype_size = descr[descr.size() - 1] - '0';
        }
    }
    // shape
    {
        auto p1 = header.find('(');
        auto p2 = header.find(')', p1);
        if (p1 != std::string::npos && p2 != std::string::npos) {
            std::string s = header.substr(p1 + 1, p2 - p1 - 1);
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                size_t b = tok.find_first_not_of(" ");
                if (b == std::string::npos)
                    continue;
                tok = tok.substr(b);
                if (tok.empty())
                    continue;
                arr.shape.push_back(std::stoll(tok));
            }
        }
    }

    std::vector<char> body((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    arr.data = std::move(body);
    return arr;
}
} // namespace

bool Function::preprocess_npy(const std::string &npy_path,
                              const std::shared_ptr<VulkanCommandPool> &cmdpool,
                              const std::shared_ptr<core::ITensor> &input) {
    NpyArray arr;
    try {
        arr = parse_npy(npy_path);
    } catch (const std::exception &e) {
        std::cerr << "preprocess_npy: " << e.what() << std::endl;
        return false;
    }
    if (arr.dtype_kind != 'f' || (arr.dtype_size != 4 && arr.dtype_size != 2)) {
        std::cerr << "preprocess_npy: only float32/float16 supported, got kind="
                  << arr.dtype_kind << " size=" << arr.dtype_size << std::endl;
        return false;
    }
    if (arr.shape.size() != 4) {
        std::cerr << "preprocess_npy: expect NCHW 4D, got " << arr.shape.size()
                  << "D shape=[";
        for (size_t i = 0; i < arr.shape.size(); ++i)
            std::cerr << arr.shape[i] << (i + 1 < arr.shape.size() ? "," : "");
        std::cerr << "]" << std::endl;
        return false;
    }
    const int64_t N = arr.shape[0];
    const int64_t C = arr.shape[1];
    const int64_t H = arr.shape[2];
    const int64_t W = arr.shape[3];

    auto in_shape = input->getShape();
    if (in_shape.size() != 4 || in_shape[0] != N || in_shape[1] != C ||
        in_shape[2] != H || in_shape[3] != W) {
        std::cerr << "preprocess_npy: shape mismatch npy=[" << N << "," << C
                  << "," << H << "," << W << "] input=[";
        for (size_t i = 0; i < in_shape.size(); ++i) {
            std::cerr << in_shape[i] << (i + 1 < in_shape.size() ? "," : "");
        }
        std::cerr << "]" << std::endl;
        return false;
    }

    std::vector<float> fp32_data(static_cast<size_t>(N) * C * H * W);
    const char *raw = arr.data.data();
    if (arr.dtype_size == 4) {
        std::memcpy(fp32_data.data(), raw, fp32_data.size() * sizeof(float));
    } else {
        // float16 -> float32
        for (size_t i = 0; i < fp32_data.size(); ++i) {
            uint16_t h;
            std::memcpy(&h, raw + i * 2, 2);
            fp32_data[i] = ITensor::fp16_to_fp32(h);
        }
    }

    std::cout << "[preprocess_npy] " << npy_path << " [" << N << "," << C << ","
              << H << "," << W << "] first 8 (NCHW c0): ";
    for (int i = 0; i < 8 && i < static_cast<int>(fp32_data.size()); ++i) {
        std::cout << fp32_data[i] << " ";
    }
    std::cout << std::endl;
    const int chan4 = (C + 3) / 4;
    if (input->dtype() == typeid(float)) {
        std::vector<float> rgba(static_cast<size_t>(N) * H * W * chan4 * 4,
                                0.0f);
        for (int n = 0; n < N; ++n) {
            for (int c4 = 0; c4 < chan4; ++c4) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        for (int k = 0; k < 4; ++k) {
                            int c = c4 * 4 + k;
                            if (c < C) {
                                size_t idx =
                                    (((size_t)n * C + c) * H + h) * W + w;
                                size_t layer_stride = (size_t)N * H * (W * 4);
                                size_t row_pitch = (size_t)W * 4;
                                size_t dst = (size_t)c4 * layer_stride +
                                             ((size_t)n * H + h) * row_pitch +
                                             (size_t)w * 4 + k;
                                rgba[dst] = fp32_data[idx];
                            }
                        }
                    }
                }
            }
        }
        auto t = as_tensor<float>(input);
        t->copyToGPUImage(cmdpool, rgba.data(), true);
    } else if (input->dtype() == typeid(uint16_t)) {
        std::vector<uint16_t> rgba(static_cast<size_t>(N) * H * W * chan4 * 4,
                                   0);
        for (int n = 0; n < N; ++n) {
            for (int c4 = 0; c4 < chan4; ++c4) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        for (int k = 0; k < 4; ++k) {
                            int c = c4 * 4 + k;
                            if (c < C) {
                                size_t idx =
                                    (((size_t)n * C + c) * H + h) * W + w;
                                size_t layer_stride = (size_t)N * H * (W * 4);
                                size_t row_pitch = (size_t)W * 4;
                                size_t dst = (size_t)c4 * layer_stride +
                                             ((size_t)n * H + h) * row_pitch +
                                             (size_t)w * 4 + k;
                                rgba[dst] =
                                    ITensor::fp32_to_fp16(fp32_data[idx]);
                            }
                        }
                    }
                }
            }
        }
        auto t = as_tensor<uint16_t>(input);
        t->copyToGPUImage(cmdpool, rgba.data(), true);
    } else {
        std::cerr << "preprocess_npy: unsupported input dtype" << std::endl;
        return false;
    }
    return true;
}

} // namespace core
} // namespace vkop
