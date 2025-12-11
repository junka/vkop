#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <cmath>

#include <sys/types.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::core::Runtime;

template<typename T>
std::vector<T> resize_YUV(std::vector<uint8_t> raw_image, int image_h, int image_w, std::shared_ptr<Tensor<T>> &t) {
    int in_h = t->getShape()[2];
    int in_w = t->getShape()[3];

    float x_ratio = static_cast<float>(image_w - 1) / (in_w - 1);
    float y_ratio = static_cast<float>(image_h - 1) / (in_h - 1);

    const uint8_t* y_src = raw_image.data();
    const uint8_t* u_src = raw_image.data() + image_w * image_h;
    const uint8_t* v_src = raw_image.data() + 2 * image_w * image_h;

    int u_offset = in_w * in_h;
    int v_offset = 2 * in_w * in_h;

    std::vector<T> resized_image(in_w * in_h * 3);

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

            // 对 Y, U, V 分量分别进行双线性插值
            auto interpolate = [](const uint8_t* plane, int w, int x1, int y1, int x2, int y2, float dx, float dy) {
                uint8_t p11 = plane[y1 * w + x1];
                uint8_t p12 = plane[y1 * w + x2];
                uint8_t p21 = plane[y2 * w + x1];
                uint8_t p22 = plane[y2 * w + x2];
                return static_cast<uint8_t>(
                    p11 * (1 - dx) * (1 - dy) +
                    p12 * dx * (1 - dy) +
                    p21 * (1 - dx) * dy +
                    p22 * dx * dy
                );
            };

            int dst_idx = dy * in_w + dx;
            resized_image[dst_idx] = interpolate(y_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
            resized_image[u_offset+dst_idx] = interpolate(u_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
            resized_image[v_offset+dst_idx] = interpolate(v_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
        }
    }
    return resized_image;
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
        std::cerr << "Usage: " << argv[0] << " <binary_file_path> <image.yuv>" << std::endl;
        return 1;
    }
    std::string binary_file_path = argv[1];
    std::string image_file_path = argv[2];
    std::vector<uint8_t> frame;

    std::ifstream infile(image_file_path, std::ios::in | std::ios::binary);
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    frame.resize(file_size);
    infile.read(reinterpret_cast<char*>(frame.data()), file_size);
    infile.close();

    int image_h = 1080;
    int image_w = 1920;

    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path);
    rt->LoadModel();

    auto input = rt->GetInput("data");
    auto t = vkop::core::as_tensor<float>(input);
    auto data = resize_YUV(frame, image_h, image_w, t);
    frame.clear();
    frame.shrink_to_fit();
    t->copyToGPU(cmdpool, data.data());
    data.clear();
    data.shrink_to_fit();

    for (int i = 0; i < 1000; i ++) {
        rt->Run();
    }
    rt->ReadResult();
    auto cls = vkop::core::as_tensor<float>(rt->GetOutput("resnetv24_dense0_fwd"));
    
    return EXIT_SUCCESS;
}
