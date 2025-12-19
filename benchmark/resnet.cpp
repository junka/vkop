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
#include <string>
#include <iomanip>

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
    float sum = 0.0f;
    for (auto& val : softmax_probs) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    for (auto& val : softmax_probs) {
        val /= sum;
    }

    std::vector<std::pair<int, float>> indexed_probs;
    indexed_probs.reserve(softmax_probs.size());
    for (size_t i = 0; i < softmax_probs.size(); ++i) {
        if (softmax_probs[i] > 0.1F) {
            indexed_probs.emplace_back(i, softmax_probs[i]);
        }
    }

    std::sort(indexed_probs.begin(),
              indexed_probs.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

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
        std::cerr << "download resnet model from https://github.com/onnx/models/tree/main/validated/vision/classification/resnet/model" << std::endl;
        std::cerr << "convert onnx to vkopbin using onnx2vkop.py from directory model" << std::endl;
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

    auto input = rt->GetInput(); // data
    auto t = vkop::core::as_tensor<float>(input);
    int resize_h = t->getShape()[2];
    int resize_w = t->getShape()[3];
    uint8_t *resized = static_cast<uint8_t *>(malloc(resize_h * resize_w * 3));
    stbir_resize_uint8_linear(raw, image_w, image_h, 0, resized, resize_w, resize_h, 0, STBIR_RGB);

    stbi_image_free(raw);
    std::vector<float> normalized_data(resize_h * resize_w * 4);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < resize_h * resize_w; i++) {
            normalized_data[(i * 4)  + c] = ((static_cast<float>(resized[i * 3 + c])/255.0F) - mean[c]) / stdvar[c];
        }
    }
    free(resized);

    // 1, 3, h, w, RGBA copy directly
    t->copyToGPUImage(cmdpool, normalized_data.data(), true);
    normalized_data.clear();
    normalized_data.shrink_to_fit();

    double tot_lat = 0.0F;
    int count = 1000;
    for (int i = 0; i < count; i ++) {
        auto lat = rt->Run();
        tot_lat += lat;
        std::cout << "inference time:" << lat << " ms" << std::endl;
    }
    std::cout << "avg time:" << tot_lat / count << " ms" << std::endl;
    rt->ReadResult();
    auto cls = vkop::core::as_tensor<float>(rt->GetOutput());

    auto res = get_top_k_predictions(cls->data(), 5);
    std::cout << "\nPredictions:\n";
    std::cout << std::fixed << std::setprecision(4);
    
    auto labels = load_labels(labels_file_path);

    for (int i = 0; i < 5 && i < static_cast<int>(res.size()); ++i) {
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
