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
void postProcess(const std::shared_ptr<class vkop::core::Tensor<float> >& out)
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

    struct Detection {
        float bbox[4];      // x_center, y_center, width, height (normalized)
        float confidence;   // objectness score
        int class_id;       // predicted class
        float class_prob;   // probability of the predicted class
    };
    
    std::vector<Detection> detections;
    
    // Process each detection
    for (int det_idx = 0; det_idx < num_detections; det_idx++) {
        // Extract bounding box center coordinates, width, height
        float x_center = out->data()[det_idx];
        float y_center = out->data()[det_idx + (num_detections * 1)];
        float width = out->data()[det_idx + (num_detections * 2)];
        float height = out->data()[det_idx + (num_detections * 3)];

        // Find the class with highest probability
        float max_class_prob = 0.0F;
        int max_class_id = 0;

        for (int cls_idx = 0; cls_idx < num_classes; cls_idx++) {
            float class_prob = out->data()[det_idx + (num_detections * (4 + cls_idx))];
            if (class_prob > max_class_prob) {
                max_class_prob = class_prob;
                max_class_id = cls_idx;
            }
        }
        float total_confidence = max_class_prob;
        
        // Filter out low-confidence detections
        if (total_confidence > confidence_threshold) {
            Detection det;
            det.bbox[0] = x_center;
            det.bbox[1] = y_center;
            det.bbox[2] = width;
            det.bbox[3] = height;
            det.confidence = total_confidence;
            det.class_id = max_class_id;
            det.class_prob = max_class_prob;
            
            detections.push_back(det);
        }
    }
    
    printf("Found %zu potential detections above threshold\n", detections.size());
    
    // Convert normalized bbox to pixel coordinates and convert from center format to corner format
    const int img_width = 640;  // Assuming input image size, adjust as needed
    const int img_height = 640;
    
    for (auto& det : detections) {
        // Convert from center format to corner format
        float x_center = det.bbox[0] * img_width;
        float y_center = det.bbox[1] * img_height;
        float width = det.bbox[2] * img_width;
        float height = det.bbox[3] * img_height;
        
        float x1 = x_center - (width / 2.0F);
        float y1 = y_center - (height / 2.0F);
        float x2 = x_center + (width / 2.0F);
        float y2 = y_center + (height / 2.0F);
        
        // Clamp to image boundaries
        x1 = std::fmax(0.0F, std::fmin(static_cast<float>(img_width), x1));
        y1 = std::fmax(0.0F, std::fmin(static_cast<float>(img_height), y1));
        x2 = std::fmax(0.0F, std::fmin(static_cast<float>(img_width), x2));
        y2 = std::fmax(0.0F, std::fmin(static_cast<float>(img_height), y2));
        
        printf("Detection - BBox: [%.2f, %.2f, %.2f, %.2f], Confidence: %.2f, Class ID: %d\n",
               x1, y1, x2, y2, det.confidence, det.class_id);
    }
    
    // Perform Non-Maximum Suppression (NMS) to remove duplicate overlapping detections
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            // Calculate IoU (Intersection over Union) between boxes i and j
            float x1_i = (detections[i].bbox[0] * img_width) - ((detections[i].bbox[2] * img_width) / 2.0F);
            float y1_i = (detections[i].bbox[1] * img_height) - ((detections[i].bbox[3] * img_height) / 2.0F);
            float x2_i = (detections[i].bbox[0] * img_width) + ((detections[i].bbox[2] * img_width) / 2.0F);
            float y2_i = (detections[i].bbox[1] * img_height) + ((detections[i].bbox[3] * img_height) / 2.0F);
            
            float x1_j = (detections[j].bbox[0] * img_width) - ((detections[j].bbox[2] * img_width) / 2.0F);
            float y1_j = (detections[j].bbox[1] * img_height) - ((detections[j].bbox[3] * img_height) / 2.0F);
            float x2_j = (detections[j].bbox[0] * img_width) + ((detections[j].bbox[2] * img_width) / 2.0F);
            float y2_j = (detections[j].bbox[1] * img_height) + ((detections[j].bbox[3] * img_height) / 2.0F);

            // Calculate intersection area
            float x1_inter = std::fmax(x1_i, x1_j);
            float y1_inter = std::fmax(y1_i, y1_j);
            float x2_inter = std::fmin(x2_i, x2_j);
            float y2_inter = std::fmin(y2_i, y2_j);

            float inter_area = std::fmax(0.0F, x2_inter - x1_inter) * std::fmax(0.0F, y2_inter - y1_inter);
            float area_i = (x2_i - x1_i) * (y2_i - y1_i);
            float area_j = (x2_j - x1_j) * (y2_j - y1_j);
            float union_area = area_i + area_j - inter_area;
            
            float iou = inter_area / union_area;
            
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
            float x_center = detections[i].bbox[0] * img_width;
            float y_center = detections[i].bbox[1] * img_height;
            float width = detections[i].bbox[2] * img_width;
            float height = detections[i].bbox[3] * img_height;

            float x1 = x_center - (width / 2.0F);
            float y1 = y_center - (height / 2.0F);
            float x2 = x_center + (width / 2.0F);
            float y2 = y_center + (height / 2.0F);

            printf("Final Detection - BBox: [%.2f, %.2f, %.2f, %.2f], Confidence: %.2f, Class ID: %d\n",
                   x1, y1, x2, y2, detections[i].confidence, detections[i].class_id);
        }
    }
}
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
    std::vector<uint8_t> frame;

    auto rt = std::make_shared<Runtime>(cmdpool, binary_file_path);
    rt->LoadModel();

    printf("model Loaded done\n");

    vkop::core::Function::preprocess_jpg(image_file_path.c_str(), cmdpool, rt->GetInput());

    int count = 1;
    for (int i = 0; i < count; i ++) {
        rt->Run();
    }
    rt->ReadResult();
    auto out = vkop::core::as_tensor<float>(rt->GetOutput());
    out->print_tensor();
    postProcess(out);

    return EXIT_SUCCESS;
}
