#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize2.h"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <cmath>
#include <iomanip>

#include <unistd.h>
#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
// using vkop::core::Tensor;
using vkop::core::Runtime;

namespace {
const float mean[] = {0.485f, 0.456f, 0.406f};
const float stdvar[] = {0.229f, 0.224f, 0.225f};
std::vector<std::string> load_labels(const std::string& label_path) {
    std::vector<std::string> labels;
    std::ifstream file(label_path);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Could not open label file: " << label_path << std::endl;
        return labels;
    }
    
    while (std::getline(file, line)) {
        // Remove carriage return if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        labels.push_back(line);
    }
    
    file.close();
    return labels;
}

// Function to get top-K predictions
std::vector<std::pair<int, float>> get_top_k_predictions(const std::vector<float>& probs, int k) {
    std::vector<float> softmax_probs = probs;

    float max_val = *std::max_element(softmax_probs.begin(), softmax_probs.end());
    printf("max val %f\n", max_val);
    // Compute exp(x - max) for numerical stability
    float sum = 0.0f;
    for (auto& val : softmax_probs) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    // Normalize to get probabilities
    for (auto& val : softmax_probs) {
        val /= sum;
    }

    std::vector<std::pair<int, float>> indexed_probs;
    indexed_probs.reserve(softmax_probs.size());
    for (size_t i = 0; i < softmax_probs.size(); ++i) {
        indexed_probs.emplace_back(i, softmax_probs[i]);
    }

    std::sort(indexed_probs.begin(),
              indexed_probs.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

    if (indexed_probs.size() > static_cast<size_t>(k)) {
        indexed_probs.resize(k);
    }
    
    for (const auto& pair : indexed_probs) {
        printf("%d: %.4f\n", pair.first, pair.second);
    }

    return indexed_probs;
}
template<typename T>
std::vector<T> resize_rgb_unorm(const uint8_t *raw_image, int image_h, int image_w, int in_h, int in_w) {
    float x_ratio = static_cast<float>(image_w - 1) / (in_w - 1);
    float y_ratio = static_cast<float>(image_h - 1) / (in_h - 1);

    std::vector<T> resized_image(in_h * in_w * 3);

    for (int dy = 0; dy < in_h; ++dy) {
        for (int dx = 0; dx < in_w; ++dx) {
            float src_x = dx * x_ratio;
            float src_y = dy * y_ratio;
            int x1 = static_cast<int>(src_x);
            int y1 = static_cast<int>(src_y);
            int x2 = std::min(x1 + 1, image_w - 1);
            int y2 = std::min(y1 + 1, image_h - 1);

            float dx_ratio = src_x - x1;
            float dy_ratio = src_y - y1;

            // 对每个颜色通道分别进行双线性插值
            for (int c = 0; c < 3; c++) {
                auto interpolate = [](const uint8_t* image_data, int width, int x1, int y1, int x2, int y2, float dx, float dy, int channel) {
                    uint8_t p11 = image_data[(y1 * width + x1) * 3 + channel];
                    uint8_t p12 = image_data[(y1 * width + x2) * 3 + channel];
                    uint8_t p21 = image_data[(y2 * width + x1) * 3 + channel];
                    uint8_t p22 = image_data[(y2 * width + x2) * 3 + channel];
                    return static_cast<uint8_t>(
                        (p11 * (1 - dx) * (1 - dy)) +
                        (p12 * dx * (1 - dy)) +
                        (p21 * (1 - dx) * dy) +
                        (p22 * dx * dy)
                    );
                };

                int dst_idx = (dy * in_w + dx) * 3 + c;
                float scale = 1.0f;
                // if (std::is_same_v<T, uint16_t>) {
                //     scale = 257.0f;
                // } else if (std::is_same_v<T, float>) {
                //     scale = 1.0f/255.0f;
                // }
                resized_image[dst_idx] = interpolate(raw_image, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio, c) * scale;
            }
        }
    }
    return resized_image;
}
}

int main(int argc, char *argv[]) {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        printf("Please set env VK_ICD_FILENAMES for a valid GPU\n");
        return -1;
    }
    printf("using %s\n",dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    if (argc < 3) {
        std::cerr << "download model from https://media.githubusercontent.com/media/onnx/models/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx?download=true" << std::endl;
        std::cerr << "download class tag from https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <binary_file_path> <image> [labels.txt]" << std::endl;
        return 1;
    }
    std::string binary_file_path = argv[1];
    std::string image_file_path = argv[2];
    std::string labels_file_path = (argc > 3) ? argv[3] : "imagenet_classes.txt";

    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path);
    rt->LoadModel();

    int image_h;
    int image_w;
    int channels;
    auto *raw = stbi_load(image_file_path.c_str(), &image_w, &image_h, &channels, 3);
    // printf("%d, %d, %d\n", image_h, image_w, channels);
    // size_t data_size = image_w * image_h * channels;
    // for (int i = 0; i < image_w; i++) {
    //     printf("%d, ", raw[(image_h -1) * image_w * 3 + i * 3 + 2]);
    // }

    auto input = rt->GetInput("data");
    auto t = vkop::core::as_tensor<float>(input);
    int resize_h = t->getShape()[2];
    int resize_w = t->getShape()[3];
    uint8_t *resized = static_cast<uint8_t *>(malloc(resize_h * resize_w * 3));
    stbir_resize_uint8_linear(raw, image_w, image_h, 0, resized, resize_w, resize_h, 0, STBIR_RGB);

    // auto resized = resize_rgb_unorm<float>(raw, image_h, image_w, resize_h, resize_w);
    stbi_image_free(raw);
    // for (int i = 0; i < resize_w; i++) {
    //     printf("%d, ", resized[(resize_h -1) * resize_w * 3 + i * 3 + 2]);
    // }
    // printf("\n");
    std::vector<float> normalized_data(resize_h * resize_w * 4);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < resize_h * resize_w; i++) {
            normalized_data[i * 4  + c] = ((static_cast<float>(resized[i * 3 + c])/255.0F) - mean[c]) / stdvar[c];
        }
    }
    // for (int h = 0; h < resize_h; h++) {
    //     for (int w = 0; w < resize_w; w++) {
    //         for (int c = 0; c < 3; c++) {
    //             int rgb_idx = (h * resize_w * 3) + (w * 3) + c;         // RGB格式索引
    //             int chw_idx = (c * resize_h * resize_w) + (h * resize_w) + w;  // CHW格式索引
    //             normalized_data[chw_idx] = ((static_cast<float>(resized[rgb_idx])/255.0F) - mean[c]) / stdvar[c];
    //         }
    //     }
    // }
    free(resized);
    // 打印blue分量最后一行
    // for (int i = 0; i < resize_w; i++) {
    //     printf("%.4f, ", normalized_data[(resize_h -1) * resize_w * 4 + i * 4 + 2]);
    // }
    // printf("\n");


    // 1, 3, h, w, RGBA copy directly
    t->fillToGPUImage(cmdpool, normalized_data.data());
    // t->copyToGPU(cmdpool, normalized_data.data());
    normalized_data.clear();
    normalized_data.shrink_to_fit();

    double tot_lat = 0.0F;
    int count = 10;
    for (int i = 0; i < count; i ++) {
        auto lat = rt->Run();
        tot_lat += lat;
        std::cout << "inference time:" << lat << " ms" << std::endl;
    }
    std::cout << "avg time:" << tot_lat/count << " ms" << std::endl;
    rt->ReadResult();
    auto cls = vkop::core::as_tensor<float>(rt->GetOutput("resnetv24_dense0_fwd"));
    // for (auto& val : cls->data()) {
    //     std::cout<< val << " ";
    // }
    auto res = get_top_k_predictions(cls->data(), 10);
    std::cout << "\nTop-10 Predictions:\n";
    std::cout << std::fixed << std::setprecision(4);
    
    auto labels = load_labels(labels_file_path);

    for (int i = 0; i < 10 && i < static_cast<int>(res.size()); ++i) {
        int index = res[i].first;
        float probability = res[i].second;
        
        std::string label = "Unknown";
        if (index < static_cast<int>(labels.size())) {
            label = labels[index];
        }
        
        std::cout << (i + 1) << ": " << label << " (" << probability << ")\n";
    }

    return EXIT_SUCCESS;
}
