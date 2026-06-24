#include "ops/Ops.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCudaLaunch.hpp"

#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "core/function.hpp"
#include "core/cpu_postprocess.hpp"

#include <cstdint>
#include <memory>
#include <cmath>
#include <string>
#include <iomanip>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Runtime;
#define USE_GPU_POSTPROCESS

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
        std::cerr << "down models using model/download_models.py for benchmark" << std::endl;
        std::cerr << "convert onnx to vkopbin using model/onnx2vkop" << std::endl;
        std::cerr << "download class tag from https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <binary_file_path> <image> [labels.txt]" << std::endl;
        return 1;
    }
    std::string binary_file_path = argv[1];
    std::string image_file_path = argv[2];
    std::string labels_file_path = (argc > 3) ? argv[3] : "imagenet_classes.txt";

    int precision = 0;
    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path, precision);
    rt->LoadModel();
    /* example for debug one node */
    // rt->TraceNode("node_Conv_291");

    vkop::core::NormMethod method = vkop::core::NormMethod::IMAGENET;
    if (binary_file_path.find("inception") != std::string::npos) {
        method = vkop::core::NormMethod::INCEPTION;
    }

    bool input_loaded = false;
    if (image_file_path.size() >= 4 &&
        image_file_path.compare(image_file_path.size() - 4, 4, ".npy") == 0) {
        input_loaded = vkop::core::Function::preprocess_npy(
            image_file_path, cmdpool, rt->GetInput());
        if (!input_loaded) {
            std::cerr << "load npy input failed: " << image_file_path
                      << std::endl;
            return 1;
        }
        std::cout << "[main] using npy input: " << image_file_path << std::endl;
    } else {
        vkop::core::Function::preprocess_jpg(image_file_path.c_str(), cmdpool,
                                             rt->GetInput(), false, method);
    }
#ifdef USE_GPU_POSTPROCESS
    vkop::core::Function::preprocess_jpg(image_file_path.c_str(), cmdpool, rt->GetInput(), false, method);
    std::vector<int> shape;

    std::shared_ptr<vkop::core::Tensor<int>> indexs;
    std::shared_ptr<vkop::core::Tensor<float>> values_float;
    std::shared_ptr<vkop::core::Tensor<uint16_t>> values_half;
    auto register_pipeline = [&shape, &rt, &indexs, &values_float, &values_half](auto tensor_type) {
        using T = decltype(tensor_type);
        auto cls = vkop::core::as_tensor<T>(rt->GetOutput());
        shape = cls->getShape();
        auto sf = std::make_shared<vkop::core::Tensor<T>>(shape, true);
        rt->RegisterPostProcess(vkop::ops::OpType::SOFTMAX, {{"axis", "-1"}}, {cls}, {sf});

        indexs = std::make_shared<vkop::core::Tensor<int>>(shape, true);
        auto values = std::make_shared<vkop::core::Tensor<T>>(shape, true);
        rt->RegisterPostProcess(vkop::ops::OpType::TOPK, {{"k", "10"}}, {sf}, {values, indexs});

        if constexpr (std::is_same_v<T, float>) {
            values_float = values;
        } else {
            values_half = values;
        }
    };

    if (precision == 1) {
        register_pipeline(uint16_t{});
    } else {
        register_pipeline(float{});
    }
#endif

    double tot_lat = 0.0F;
    int count = 100;
    printf("run inference %d times...\n", count);
    for (int i = 0; i < count; i ++) {
        auto lat = rt->Run();
        tot_lat += lat;
        std::cout << "inference time:" << lat << " ms" << std::endl;
    }
    std::cout << "avg time:" << tot_lat / count << " ms" << std::endl;

    rt->ReadResult();
#ifndef USE_GPU_POSTPROCESS
    auto out = rt->GetOutput();
    auto out_shape = out->getShape();
    std::vector<float> logits;
    if (out->dtype() == typeid(float)) {
        auto t = vkop::core::as_tensor<float>(out);
        logits.resize(t->num_elements());
        for (int i = 0; i < t->num_elements(); ++i) {
            logits[i] = (*t)[i];
        }
    } else if (out->dtype() == typeid(uint16_t)) {
        auto t = vkop::core::as_tensor<uint16_t>(out);
        logits.resize(t->num_elements());
        for (int i = 0; i < t->num_elements(); ++i) {
            logits[i] = vkop::core::ITensor::fp16_to_fp32((*t)[i]);
        }
    } else {
        std::cerr << "unsupported output dtype" << std::endl;
        return 1;
    }

    // CPU softmax(axis=-1) + top10
    auto probs = vkop::core::cpu::softmax(logits, out_shape, -1);
    auto topk_result = vkop::core::cpu::topk(probs, out_shape, 10, true, true);
    const auto &top_vals = topk_result.first;
    const auto &top_idx = topk_result.second;
#else
    const auto &top_idx = *indexs;
    auto &top_vals = *values_float;
    if (precision == 1) {
        auto top_vals_half = values_half;
        for (int i = 0; i < top_vals_half->num_elements(); ++i) {
            top_vals[i] = vkop::core::ITensor::fp16_to_fp32((*top_vals_half)[i]);
        }
    }
#endif
    std::cout << "\nPredictions:\n";
    std::cout << std::fixed << std::setprecision(3);

    auto labels = load_labels(labels_file_path);

    for (int i = 0; i < 10; ++i) {
        int index = top_idx[i];
        float value = top_vals[i];

        std::string label = "Unknown";
        if (index < static_cast<int>(labels.size())) {
            label = labels[index];
        }

        std::cout << (i + 1) << ": " << label << " (" << value << ")\n";
    }

    return EXIT_SUCCESS;
}
