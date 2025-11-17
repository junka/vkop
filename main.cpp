#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include "core/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <unordered_set>

#include <sys/types.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::core::Runtime;

enum class Category : int {
    UNKNOWN = 0,
    HUMAN_FACE = 1,
    LICENSE_PLATE = 2,
    OTHERS = 999
};

using MaskInfo = struct MaskInfo {
    float x1;
    float y1;
    float x2;
    float y2;

    float score;
    Category category;
};

static float score_threshold[2] = {0.35F, 0.5F};

std::vector<MaskInfo> postProcessNMS(
    const float* hm_data, const float* hm_nms_data,
    const float* reg_data, const float* dim_data,
    const float* cls_data, int H, int W, int image_h, int image_w,
    int tensor_h, int tensor_w
) {
    auto sigmoid = [](float x) -> float {
        return 1.0f / (1.0f + std::exp(-x));
    };
    const int num_points = H * W;

    std::vector<int> indices(num_points);
    for (int i = 0; i < num_points; i++) {
        indices[i] = i;
    }
    std::partial_sort(indices.begin(), indices.begin() + 64, indices.end(),
        [hm_nms_data](const int& i1, const int& i2) {
            return hm_nms_data[i1] > hm_nms_data[i2];
        });

    std::vector<MaskInfo> detections;
    for (int i = 0; i < 64; i++) {
        int idx = indices.at(i);

        if (std::fabs(hm_data[idx] - hm_nms_data[idx]) > 1e-6F) {
            continue;
        }

        float score = sigmoid(hm_nms_data[idx]);
        int category = 0;
        const float cls_value1 = cls_data[idx];
        const float cls_value2 = cls_data[num_points + idx];
        if (cls_value1 > cls_value2) {
            category = 0;
        } else {
            category = 1;
        }

        if (score < score_threshold[category]) {
            continue;
        }

        int y = idx / W;
        int x = idx % W;
        float xo = sigmoid(reg_data[idx]) + x;
        float yo = sigmoid(reg_data[num_points + idx]) + y;
        // std::cout << "point at index " << idx << " has (x, y): (" << xo << ", " << yo << ")" << std::endl;
        xo *= (tensor_w / W);
        yo *= (tensor_h / H);

        float wo = std::exp(dim_data[idx]) * tensor_w / W;
        float ho = std::exp(dim_data[num_points + idx]) * tensor_h / H;

        // std::cout << "index " << idx << " score " << score <<" Raw bbox (center x, center y, width, height): (" << xo << ", " << yo << ", " << wo << ", " << ho << ")" << std::endl;
        xo = xo * W / tensor_w;
        wo = wo * W / tensor_w;
        yo = yo * H / tensor_h;
        ho = ho * H / tensor_h;

        float scale_x = static_cast<float>(tensor_w) / W;
        float scale_y = static_cast<float>(tensor_h) / H;

        float x1 = (xo - wo * 0.5F) * scale_x * image_w / tensor_w;
        float y1 = (yo - ho * 0.5F) * scale_y * image_h / tensor_h;
        float x2 = (xo + wo * 0.5F) * scale_x * image_w / tensor_w;
        float y2 = (yo + ho * 0.5F) * scale_y * image_h / tensor_h;
        x1 = std::max(0, static_cast<int>(x1));
        y1 = std::max(0, static_cast<int>(y1));
        x2 = std::min(static_cast<int>(x2), image_w - 1);
        y2 = std::min(static_cast<int>(y2), image_h - 1);

        std::cout << "Detected plate at: (" << x1 << ", " << y1 << ") to (" << x2 << ", " << y2 << ") with score: " << score
                  << " and category: " << category << std::endl;

        // detections.push_back({x1, y1, x2, y2, score, Category::category});
    }
    return detections;
}

void resize_YUV(std::vector<uint8_t> raw_image, int image_h, int image_w, std::shared_ptr<Tensor<float>> &t) {
    int in_h = t->getTensorShape()[2];
    int in_w = t->getTensorShape()[3];

    float* data_ptr = t->data();

    float x_ratio = float(image_w - 1) / (in_w - 1);
    float y_ratio = float(image_h - 1) / (in_h - 1);

    const uint8_t* y_src = raw_image.data();
    const uint8_t* u_src = raw_image.data() + image_w * image_h;
    const uint8_t* v_src = raw_image.data() + 2 * image_w * image_h;

    float* y_dst = data_ptr;
    float* u_dst = data_ptr + in_w * in_h;
    float* v_dst = data_ptr + 2 * in_w * in_h;

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
            y_dst[dst_idx] = interpolate(y_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
            u_dst[dst_idx] = interpolate(u_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
            v_dst[dst_idx] = interpolate(v_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) / 255.0F;
        }
    }
}

int main(int argc, char *argv[]) {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    std::shared_ptr<VulkanDevice> dev;
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto *pdev : phydevs) {
            auto vdev = std::make_shared<VulkanDevice>(pdev);
            if (vdev->getDeviceName().find("llvmpipe") != std::string::npos) {
                continue;
            }
            dev = vdev;
        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s", e.what());
        return EXIT_FAILURE;
    }
    printf("using %s\n",dev->getDeviceName().c_str());
    auto *device = dev->getLogicalDevice();
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <binary_file_path> <image.yuv>" << std::endl;
        return 1;
    }
    try {
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

        auto rt = std::make_shared<Runtime>(dev, cmdpool, binary_file_path);
        rt->LoadModel();

        auto t = vkop::core::as_tensor<float>(rt->GetInput("input.1"));
        resize_YUV(frame, image_h, image_w, t);

        for (int i = 0; i < 1; i ++) {
            rt->Run();
        }
        rt->ReadResult();
        auto hm = vkop::core::as_tensor<float>(rt->GetOutput("hm"));
        auto reg = vkop::core::as_tensor<float>(rt->GetOutput("reg"));
        auto dim = vkop::core::as_tensor<float>(rt->GetOutput("dim"));
        auto cls = vkop::core::as_tensor<float>(rt->GetOutput("cls"));
        auto hm_nms = vkop::core::as_tensor<float>(rt->GetOutput("hm_nms"));
        assert(hm != nullptr);

        postProcessNMS(hm->data(), hm_nms->data(), reg->data(), dim->data(), cls->data(), hm_nms->getTensorShape()[2], hm_nms->getTensorShape()[3], image_h, image_w,
            t->getTensorShape()[2], t->getTensorShape()[3]);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return EXIT_SUCCESS;
}
