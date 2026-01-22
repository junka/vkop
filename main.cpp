#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <cmath>

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

namespace {
float score_threshold[2] = {0.35F, 0.5F};

std::vector<MaskInfo> postProcessNMS(
    const std::shared_ptr<class vkop::core::Tensor<float> >& hm_data, const std::shared_ptr<class vkop::core::Tensor<float> >& hm_nms_data,
    const std::shared_ptr<class vkop::core::Tensor<float> >& reg_data, const std::shared_ptr<class vkop::core::Tensor<float> >& dim_data,
    const std::shared_ptr<class vkop::core::Tensor<float> >& cls_data, int H, int W, int image_h, int image_w,
    int tensor_h, int tensor_w
) {
    auto sigmoid = [](float x) -> float {
        return 1.0F / (1.0F + std::exp(-x));
    };
    const int num_points = H * W;

    std::vector<int> indices(num_points);
    for (int i = 0; i < num_points; i++) {
        indices[i] = i;
    }
    std::partial_sort(indices.begin(), indices.begin() + 64, indices.end(),
        [hm_nms_data](const int& i1, const int& i2) {
            return (*hm_nms_data)[i1] > (*hm_nms_data)[i2];
        });

    std::vector<MaskInfo> detections;
    for (int i = 0; i < 64; i++) {
        int idx = indices.at(i);

        if (std::fabs((*hm_data)[idx] - (*hm_nms_data)[idx]) > 1e-6F) {
            continue;
        }

        float score = sigmoid((*hm_nms_data)[idx]);
        int category = 0;
        const float cls_value1 = (*cls_data)[idx];
        const float cls_value2 = (*cls_data)[num_points + idx];
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
        float xo = sigmoid((*reg_data)[idx]) + x;
        float yo = sigmoid((*reg_data)[num_points + idx]) + y;

        xo *= (static_cast<float>(tensor_w) / W);
        yo *= (static_cast<float>(tensor_h) / H);

        float wo = std::exp((*dim_data)[idx]) * tensor_w / W;
        float ho = std::exp((*dim_data)[num_points + idx]) * tensor_h / H;

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
        int xi1 = std::max(0, static_cast<int>(x1));
        int yi1 = std::max(0, static_cast<int>(y1));
        int xi2 = std::min(static_cast<int>(x2), image_w - 1);
        int yi2 = std::min(static_cast<int>(y2), image_h - 1);

        std::cout << "Detected plate at: (" << xi1 << ", " << yi1 << ") to (" << xi2 << ", " << yi2 << ") with score: " << score
                  << " and category: " << category << std::endl;

        // detections.push_back({x1, y1, x2, y2, score, Category::category});
    }
    return detections;
}


template<typename T>
std::vector<T> resize_yuv444(const std::vector<uint8_t> &raw_image, int image_h, int image_w, std::shared_ptr<Tensor<T>> &t) {
    int in_h = t->getShape()[2];
    int in_w = t->getShape()[3];

    float x_ratio = static_cast<float>(image_w - 1) / (in_w - 1);
    float y_ratio = static_cast<float>(image_h - 1) / (in_h - 1);

    const uint8_t* y_src = raw_image.data();
    const uint8_t* u_src = raw_image.data() + (image_w * image_h);
    const uint8_t* v_src = raw_image.data() + (2 * image_w * image_h);

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

            auto interpolate = [](const uint8_t* plane, int w, int x1, int y1, int x2, int y2, float dx, float dy) {
                uint8_t p11 = plane[(y1 * w) + x1];
                uint8_t p12 = plane[(y1 * w) + x2];
                uint8_t p21 = plane[(y2 * w) + x1];
                uint8_t p22 = plane[(y2 * w) + x2];
                return static_cast<uint8_t>(
                    (p11 * (1 - dx) * (1 - dy)) +
                    (p12 * dx * (1 - dy)) +
                    (p21 * (1 - dx) * dy) +
                    (p22 * dx * dy)
                );
            };

            int dst_idx = (dy * in_w) + dx;
            float scale = 1.0F/255.0F;
            if (sizeof(T) == 2) {
                resized_image[dst_idx] = vkop::core::ITensor::fp32_to_fp16(interpolate(y_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) * scale);
                resized_image[u_offset+dst_idx] = vkop::core::ITensor::fp32_to_fp16(interpolate(u_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) * scale);
                resized_image[v_offset+dst_idx] = vkop::core::ITensor::fp32_to_fp16(interpolate(v_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) * scale);
            } else if (sizeof(T) == 4) {
                resized_image[dst_idx] = interpolate(y_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) * scale;
                resized_image[u_offset+dst_idx] = interpolate(u_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) * scale;
                resized_image[v_offset+dst_idx] = interpolate(v_src, image_w, x1, y1, x2, y2, dx_ratio, dy_ratio) * scale;
            }
        }
    }
    return resized_image;
}
template<typename T>
void processTensorInput(const std::shared_ptr<vkop::core::ITensor>& input, 
                    std::vector<uint8_t>& frame, 
                    int image_h, int image_w,
                    int& tensor_h, int& tensor_w,
                    const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool) {
    auto t = vkop::core::as_tensor<T>(input);
    tensor_h = t->getShape()[2];
    tensor_w = t->getShape()[3];
    auto data = resize_yuv444(frame, image_h, image_w, t);
    frame.clear();
    frame.shrink_to_fit();
    t->copyToGPU(cmdpool, data.data());
    data.clear();
}
}


int main(int argc, char *argv[]) {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    std::shared_ptr<VulkanDevice> dev;
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        printf("Found %ld physical devices\n", phydevs.size());
        for (auto *pdev : phydevs) {
            auto vdev = std::make_shared<VulkanDevice>(pdev);
            if (vdev->getDeviceName().find("llvmpipe") != std::string::npos) {
                vdev.reset();
                continue;
            }
            dev = vdev;
        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s", e.what());
        return EXIT_FAILURE;
    }
    printf("using %s\n",dev->getDeviceName().c_str());
    printf("sizeof Tensor<float>: %zu\n", sizeof(Tensor<float>));
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

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

        auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path);
        rt->LoadModel();

        auto input = rt->GetInput("input.1");
        int tensor_h = 0;
        int tensor_w = 0;
        if (!input) {
            printf("Input tensor not found\n");
            return -1;
        }

        if (input->dtype() == typeid(uint8_t)) {
            processTensorInput<uint8_t>(input, frame, image_h, image_w, tensor_h, tensor_w, cmdpool);
        } else if (input->dtype() == typeid(uint16_t)) {
            processTensorInput<uint16_t>(input, frame, image_h, image_w, tensor_h, tensor_w, cmdpool);
        } else if (input->dtype() == typeid(float)) {
            processTensorInput<float>(input, frame, image_h, image_w, tensor_h, tensor_w, cmdpool);
        } else {
            printf("Unsupported input tensor data type\n");
            return -1;
        }
        double tot_lat = 0.0F;
        int count = 10;
        for (int i = 0; i < count; i ++) {
            auto lat = rt->Run();
            tot_lat += lat;
            std::cout << "inference time:" << lat << " ms" << std::endl;
        }
        std::cout << "avg time:" << tot_lat/count << " ms" << std::endl;

        rt->ReadResult();
        auto hm = vkop::core::as_tensor<float>(rt->GetOutput("hm"));
        auto reg = vkop::core::as_tensor<float>(rt->GetOutput("reg"));
        auto dim = vkop::core::as_tensor<float>(rt->GetOutput("dim"));
        auto cls = vkop::core::as_tensor<float>(rt->GetOutput("cls"));
        auto hm_nms = vkop::core::as_tensor<float>(rt->GetOutput("hm_nms"));
        assert(hm != nullptr);

        postProcessNMS(hm, hm_nms, reg, dim, cls, hm_nms->getShape()[2], hm_nms->getShape()[3],
            image_h, image_w, tensor_h, tensor_w);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return EXIT_SUCCESS;
}
