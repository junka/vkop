// coco_labels.hpp — COCO 80-class names, YOLOv8 canonical order (indices 0..79).
// Correctness matters for detection label display — do not reorder.
#pragma once
#include <array>

namespace yolo_coco {

// COCO 2017 class names in standard 0..79 order.
// Index matches class_id in YOLOv8 output.
static constexpr const char* LABELS[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

inline const char* label(int class_id) {
    if (class_id < 0 || class_id >= 80) return "?";
    return LABELS[class_id];
}

} // namespace yolo_coco
