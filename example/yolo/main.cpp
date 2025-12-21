#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "core/function.hpp"

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

    printf("model Loaded done\n");

    vkop::core::Function::preprocess_jpg(image_file_path.c_str(), cmdpool, rt->GetInput());


    for (int i = 0; i < 1000; i ++) {
        rt->Run();
    }
    rt->ReadResult();
    auto out = vkop::core::as_tensor<float>(rt->GetOutput());


    return EXIT_SUCCESS;
}
