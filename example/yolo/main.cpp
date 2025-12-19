#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize2.h"

#include <cstdint>
#include <memory>
#include <cmath>

#include <unistd.h>
#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::core::Runtime;



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
    printf("sizeof Tensor<float>: %zu\n", sizeof(Tensor<float>));
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <binary_file_path> <image.yuv>" << std::endl;
        return 1;
    }

    std::string binary_file_path = argv[1];
    std::string image_file_path = argv[2];
    std::vector<uint8_t> frame;

    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path);
    rt->LoadModel();

    int image_h;
    int image_w;
    int channels;
    auto *raw = stbi_load(image_file_path.c_str(), &image_w, &image_h, &channels, 3);

    printf("model Loaded done\n");
    auto input = rt->GetInput(); // 1, 3, 640, 640
    auto t = vkop::core::as_tensor<float>(input);

    int resize_h = t->getShape()[2];
    int resize_w = t->getShape()[3];
    auto *resized = static_cast<uint8_t *>(malloc(resize_h * resize_w * 3));
    stbir_resize_uint8_linear(raw, image_w, image_h, 0, resized, resize_w, resize_h, 0, STBIR_RGB);
    stbi_image_free(raw);
    std::vector<float> normalized_data(resize_h * resize_w * 4);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < resize_h * resize_w; i++) {
            normalized_data[(i * 4)  + c] = ((static_cast<float>(resized[i * 3 + c])/255.0F));
        }
    }
    t->copyToGPUImage(cmdpool, normalized_data.data(), true);
    normalized_data.clear();
    normalized_data.shrink_to_fit();

    for (int i = 0; i < 1000; i ++) {
        rt->Run();
    }
    rt->ReadResult();
    auto out = vkop::core::as_tensor<float>(rt->GetOutput("output0"));


    return EXIT_SUCCESS;
}
