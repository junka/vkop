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
    // Create index-value pairs
    std::vector<std::pair<int, float>> indexed_probs;
    indexed_probs.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed_probs.emplace_back(i, probs[i]);
    }
    
    // Sort by probability in descending order
    std::partial_sort(indexed_probs.begin(), 
                      indexed_probs.begin() + std::min(k, static_cast<int>(indexed_probs.size())), 
                      indexed_probs.end(),
                      [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                          return a.second > b.second;
                      });
    
    // Return top K results
    if (indexed_probs.size() > static_cast<size_t>(k)) {
        indexed_probs.resize(k);
    }
    
    return indexed_probs;
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
    stbir_resize_uint8_linear(raw, image_w, image_h, 0, resized, resize_h, resize_w, 0, STBIR_RGB);
    stbi_image_free(raw);
    // for (int i = 0; i < resize_w; i++) {
    //     printf("%d, ", resized[(resize_h -1) * resize_w * 3 + i * 3 + 2]);
    // }
    // printf("\n");
    std::vector<float> normalized_data(resize_h * resize_w * 4);
    printf("tensor %d, %ld\n", t->size(), normalized_data.size()* 4);
    // for (int c = 0; c < 3; c++) {
    //     for (int i = 0; i < resize_h * resize_w; i++) {
    //         normalized_data[i * 4  + c] = ((static_cast<float>(resized[i * 3 + c])/255.0F) - mean[c]) / stdvar[c];
    //     }
    // }
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < resize_h; h++) {
            for (int w = 0; w < resize_w; w++) {
                int rgb_idx = h * resize_w * 3 + w * 3 + c;         // RGB格式索引
                int chw_idx = c * resize_h * resize_w + h * resize_w + w;  // CHW格式索引
                normalized_data[chw_idx] = ((static_cast<float>(resized[rgb_idx])/255.0F) - mean[c]) / stdvar[c];
            }
        }
    }
    free(resized);
    // 打印blue分量最后一行
    // for (int i = 0; i < resize_w; i++) {
    //     printf("%.4f, ", normalized_data[(resize_h -1) * resize_w * 4 + i * 4 + 2]);
    // }
    // printf("\n");


    // 1, 3, h, w, RGBA copy directly
    // t->fillToGPUImage(cmdpool, normalized_data.data());
    t->copyToGPU(cmdpool, normalized_data.data());
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
