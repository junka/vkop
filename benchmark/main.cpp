#include "ops/Ops.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "core/function.hpp"

#include <memory>
#include <cmath>
#include <string>
#include <iomanip>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Runtime;

namespace {
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

    vkop::core::Function::preprocess_jpg(image_file_path.c_str(), cmdpool, rt->GetInput(), true);

    auto cls = vkop::core::as_tensor<float>(rt->GetOutput());
    auto shape = cls->getShape();
    auto indexs = std::make_shared<vkop::core::Tensor<int>>(shape, true);
    auto values = std::make_shared<vkop::core::Tensor<float>>(shape, true);

    if (binary_file_path.find("inception") != std::string::npos) {
        rt->RegisterPostProcess(vkop::ops::OpType::TOPK, {{"k", "10"}}, {cls}, {values, indexs});
    } else {
        auto sf = std::make_shared<vkop::core::Tensor<float>>(shape, true);
        rt->RegisterPostProcess(vkop::ops::OpType::SOFTMAX, {{"axis", "-1"}}, {cls}, {sf});
        rt->RegisterPostProcess(vkop::ops::OpType::TOPK, {{"k", "10"}}, {sf}, {values, indexs});
    }
    double tot_lat = 0.0F;
    int count = 10;
    printf("run inference %d times...\n", count);
    for (int i = 0; i < count; i ++) {
        auto lat = rt->Run();
        tot_lat += lat;
        std::cout << "inference time:" << lat << " ms" << std::endl;
    }
    std::cout << "avg time:" << tot_lat / count << " ms" << std::endl;
    rt->ReadResult();

    std::cout << "\nPredictions:\n";
    std::cout << std::fixed << std::setprecision(4);
    
    auto labels = load_labels(labels_file_path);

    for (int i = 0; i < 10; ++i) {
        int index = (*indexs)[i];
        float value = (*values)[i];

        std::string label = "Unknown";
        if (index < static_cast<int>(labels.size())) {
            label = labels[index];
        }
        
        std::cout << (i + 1) << ": " << label << " (" << value << ")\n";
    }

    return EXIT_SUCCESS;
}
