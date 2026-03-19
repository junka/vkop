#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "core/function.hpp"

#include <cmath>
#include <cstdint>
#include <memory>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::core::Runtime;

namespace{
#if 1
template<typename T>
void postProcess(const std::shared_ptr<vkop::core::Tensor<T> >& out, int img_height, int img_width)
{
    auto shape = out->getShape();
    printf("output shape: [%d, %d, %d]\n", shape[0], shape[1], shape[2]);

    const float confidence_threshold = 0.5F;
    const float nms_threshold = 0.4F;

    // Assuming shape[1] = 84 (4 bbox coords + 80 class probs for COCO)
    // - First 4 channels: bounding box coordinates (x, y, w, h)
    // - Next 80 channels: class probabilities
    const int num_classes = 80;  // COCO dataset classes
    const int num_detections = shape[2]; // 8400
    // const int rows = shape[1]; // 84
    
    float gain = std::min(640.0F / img_height, 640.0F / img_width);
    printf("image %d, %d, gain %f\n", img_height, img_width, gain);

    int new_width = static_cast<int>(img_width * gain);
    int new_height = static_cast<int>(img_height * gain);

    int pad_w = 640 - new_width;
    int pad_h = 640 - new_height;
    int pad_top = pad_h / 2;
    int pad_left = pad_w / 2;

    printf("Original image: %d x %d\n", img_width, img_height);
    printf("Scale: %f, New size: %d x %d\n", gain, new_width, new_height);
    printf("Padding: top=%d, left=%d\n", pad_top, pad_left);

    struct Detection {
        float bbox[4];      // x1, y1, x2, y2
        float confidence;   // objectness score
        int class_id;       // predicted class
        float class_prob;   // probability of the predicted class
    };

    std::vector<Detection> detections;

    // Process each detection - 修正数据访问方式
    for (int det_idx = 0; det_idx < num_detections; det_idx++) {
        float cx;
        float cy;
        float w;
        float h;
        if constexpr (std::is_same_v<T, float>) {
            cx = (*out)[(0 * num_detections) + det_idx];  // x coordinate
            cy = (*out)[(1 * num_detections) + det_idx];  // y coordinate
            w = (*out)[(2 * num_detections) + det_idx];  // width
            h = (*out)[(3 * num_detections) + det_idx];  // height
        } else if constexpr (std::is_same_v<T, uint16_t>) { 
            cx = vkop::core::ITensor::fp16_to_fp32(out->data()[(0 * num_detections) + det_idx]);
            cy = vkop::core::ITensor::fp16_to_fp32(out->data()[(1 * num_detections) + det_idx]);
            w = vkop::core::ITensor::fp16_to_fp32(out->data()[(2 * num_detections) + det_idx]);
            h = vkop::core::ITensor::fp16_to_fp32(out->data()[(3 * num_detections) + det_idx]);

        }
        cx -= pad_left;
        cy -= pad_top;

        // 转换为像素坐标并应用缩放因子
        float center_x = cx / gain;
        float center_y = cy / gain;
        float width = w / gain;
        float height = h / gain;

        // 转换为中心格式到角点格式
        float x1 = center_x - (width / 2.0F);
        float y1 = center_y - (height / 2.0F);
        float x2 = center_x + (width / 2.0F);
        float y2 = center_y + (height / 2.0F);

        x1 = std::max(0.0F, std::min(static_cast<float>(img_width), x1));
        y1 = std::max(0.0F, std::min(static_cast<float>(img_height), y1));
        x2 = std::max(0.0F, std::min(static_cast<float>(img_width), x2));
        y2 = std::max(0.0F, std::min(static_cast<float>(img_height), y2));

        // Find the class with highest probability
        float max_class_prob = 0.0F;
        int max_class_id = 0;

        for (int cls_idx = 0; cls_idx < num_classes; cls_idx++) {
            float class_prob;
            if constexpr (std::is_same_v<T, float>) {
                class_prob = (*out)[((4 + cls_idx) * num_detections) + det_idx];
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                class_prob = vkop::core::ITensor::fp16_to_fp32(out->data()[((4 + cls_idx) * num_detections) + det_idx]);
            }
            if (class_prob > max_class_prob) {
                max_class_prob = class_prob;
                max_class_id = cls_idx;
            }
        }

        // Filter out low-confidence detections
        if (max_class_prob > confidence_threshold) {
            Detection det;
            det.bbox[0] = x1;
            det.bbox[1] = y1;
            det.bbox[2] = x2;
            det.bbox[3] = y2;
            det.confidence = max_class_prob;
            det.class_id = max_class_id;
            det.class_prob = max_class_prob;

            detections.push_back(det);
        }
    }

    printf("Found %zu potential detections above threshold\n", detections.size());

    for (const auto& det : detections) {
        printf("Detection - BBox: [%.2f, %.2f, %.2f, %.2f], Confidence: %.2f, Class ID: %d\n",
               det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.confidence, det.class_id);
    }

    // Perform Non-Maximum Suppression (NMS)
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            // Calculate IoU between boxes i and j
            float x1_i = detections[i].bbox[0];
            float y1_i = detections[i].bbox[1];
            float x2_i = detections[i].bbox[2];
            float y2_i = detections[i].bbox[3];

            float x1_j = detections[j].bbox[0];
            float y1_j = detections[j].bbox[1];
            float x2_j = detections[j].bbox[2];
            float y2_j = detections[j].bbox[3];

            // Calculate intersection area
            float x1_inter = std::max(x1_i, x1_j);
            float y1_inter = std::max(y1_i, y1_j);
            float x2_inter = std::min(x2_i, x2_j);
            float y2_inter = std::min(y2_i, y2_j);

            float inter_area = std::max(0.0F, x2_inter - x1_inter) * std::max(0.0F, y2_inter - y1_inter);
            
            float area_i = (x2_i - x1_i) * (y2_i - y1_i);
            float area_j = (x2_j - x1_j) * (y2_j - y1_j);
            float union_area = area_i + area_j - inter_area;
            
            float iou = (union_area > 0) ? inter_area / union_area : 0.0F;
            
            if (iou > nms_threshold) {
                // Suppress the detection with lower confidence
                if (detections[i].confidence < detections[j].confidence) {
                    suppressed[i] = true;
                    break;
                }
                suppressed[j] = true;
            }
        }
    }
    
    printf("After NMS: %zu final detections\n", 
           std::count_if(suppressed.begin(), suppressed.end(), [](bool s) { return !s; }));

    // Print final detections after NMS
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            const auto& det = detections[i];
            printf("Final Detection - BBox: [%.2f, %.2f, %.2f, %.2f], Confidence: %.2f, Class ID: %d\n",
                   det.bbox[0], det.bbox[1], 
                   det.bbox[2], det.bbox[3],
                   det.confidence, det.class_id);
        }
    }
}
#endif
} // namespace
int main(int argc, char *argv[]) {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
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
    std::string tracenodename = argc > 3 ? argv[3] : "";

    int precision = 0;
    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path, precision);
    rt->LoadModel();
    if (tracenodename.size() > 0) {
        rt->TraceNode(tracenodename);
    }
    printf("model Loaded done\n");

    auto [image_h, image_w] = vkop::core::Function::preprocess_jpg(image_file_path.c_str(), cmdpool, rt->GetInput(), true);

    rt->Run();
    rt->ReadResult();
    if (precision == 0) {
        auto out = vkop::core::as_tensor<float>(rt->GetOutput());
        if (tracenodename.size() > 0) {
            out->print_tensor();
        }
        postProcess(out, image_h, image_w);
    } else if (precision == 1) {
        auto out = vkop::core::as_tensor<uint16_t>(rt->GetOutput());
        if (tracenodename.size() > 0) {
            out->print_tensor();
        }
        postProcess(out, image_h, image_w);
    }
    

    return EXIT_SUCCESS;
}
