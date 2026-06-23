

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include "ops/Nms.hpp"


using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Nms;

namespace {
template <typename T>
class NmsTest: public TestCase<T> {
public:
    std::vector<int> box_shape_;
    std::vector<int> score_shape_;
    std::shared_ptr<Tensor<T>> boxes;
    std::shared_ptr<Tensor<T>> scores;
    std::shared_ptr<Tensor<int>> indices;
    float iou_threshold_;
    float score_threshold_;
    int max_per_class_;

    std::unordered_map<std::string, std::string> attributes;
    NmsTest(std::vector<int> &box_shape, std::vector<int>& score_shape, int out_per_class, float iou_threshold, float score_threshold): TestCase<T>("Nms"), box_shape_(box_shape), score_shape_(score_shape), iou_threshold_(iou_threshold), score_threshold_(score_threshold), max_per_class_(out_per_class) {
        attributes = {
            {"max_output_boxes_per_class", std::to_string(out_per_class)},
            {"iou_threshold", std::to_string(iou_threshold)},
            {"score_threshold", std::to_string(score_threshold)},
        };
        init_testdata();
    }

    // NMS output is an unordered set of [batch, class, box] rows, padded to a
    // fixed max size. Each (batch,class) workgroup claims its output slot via
    // atomicAdd, so the row order is non-deterministic. Compare as sets: read
    // the count the shader wrote, truncate both sides to [count], sort, then
    // compare element-wise.
    bool verify_output(const std::unique_ptr<vkop::ops::Operator> &op,
                       int idx,
                       const std::shared_ptr<vkop::core::ITensor> &output,
                       const std::shared_ptr<vkop::core::ITensor> &expect) override {
        auto *nms_op = dynamic_cast<vkop::ops::Nms *>(op.get());
        if (!nms_op) {
            LOG_ERROR("Failed to cast operator to Nms for count readback");
            return false;
        }
        auto count_buf = nms_op->get_count_buffer();
        uint32_t count = 0;
        {
            vkop::VulkanCommandBuffer cmd(this->cmdpool_);
            auto stpool = this->cmdpool_->getStagingBufferPool();
            auto b = stpool->allocate(sizeof(uint32_t));
            if (b) {
                cmd.begin();
                count_buf->copyBufferToStageBuffer(cmd.get(), b->buffer,
                                                    b->offset,
                                                    sizeof(uint32_t));
                cmd.end();
                cmd.submit(this->dev_->getComputeQueue());
                cmd.wait();
                std::memcpy(&count, b->ptr, sizeof(uint32_t));
                stpool->reset();
            }
        }
        auto out_tensor = vkop::core::as_tensor<int>(output);
        auto exp_tensor = vkop::core::as_tensor<int>(expect);
        if (static_cast<int>(count) != exp_tensor->num_elements() / 3) {
            LOG_ERROR("NMS count mismatch: gpu=%u expect=%d",
                      count, exp_tensor->num_elements() / 3);
            return false;
        }
        auto to_rows = [&](const std::shared_ptr<vkop::core::Tensor<int>> &t, int n) {
            std::vector<std::array<int,3>> rows;
            rows.reserve(n);
            for (int r = 0; r < n; ++r) {
                rows.push_back({(*t)[r*3+0], (*t)[r*3+1], (*t)[r*3+2]});
            }
            std::sort(rows.begin(), rows.end());
            return rows;
        };
        auto out_rows = to_rows(out_tensor, static_cast<int>(count));
        auto exp_rows = to_rows(exp_tensor, static_cast<int>(count));
        for (int r = 0; r < static_cast<int>(count); ++r) {
            for (int k = 0; k < 3; ++k) {
                if (out_rows[r][k] != exp_rows[r][k]) {
                    LOG_ERROR("NMS row %d col %d: %d vs %d",
                              r, k, out_rows[r][k], exp_rows[r][k]);
                    return false;
                }
            }
        }
        return true;
    }
private:
    void init_testdata() {
        std::vector<int64_t> boxshape(box_shape_.begin(), box_shape_.end());
        std::vector<int64_t> scoreshape(score_shape_.begin(), score_shape_.end());

        auto torch_scores = torch::rand(scoreshape, this->getTorchConf());

        auto torch_boxes_raw = torch::rand({boxshape[0], boxshape[1], 2}, this->getTorchConf()); // 生成中心点
        auto torch_boxes_size = torch::rand({boxshape[0], boxshape[1], 2}, this->getTorchConf()) * 1.3F; // 生成尺寸

        // 构造有效的边界框 [x1, y1, x2, y2] 其中 x1 < x2, y1 < y2
        auto x_center = torch_boxes_raw.select(-1, 0);
        auto y_center = torch_boxes_raw.select(-1, 1);
        auto width = torch_boxes_size.select(-1, 0);
        auto height = torch_boxes_size.select(-1, 1);

        auto x1 = torch::clamp(x_center - (width/2), 0.0F, 1.0F);
        auto y1 = torch::clamp(y_center - (height/2), 0.0F, 1.0F);
        auto x2 = torch::clamp(x_center + (width/2), 0.0F, 1.0F);
        auto y2 = torch::clamp(y_center + (height/2), 0.0F, 1.0F);

        auto torch_boxes = torch::stack({x1, y1, x2, y2}, -1);

        boxes = std::make_shared<Tensor<T>>(box_shape_);
        this->fillTensorFromTorch(boxes, torch_boxes);
        scores = std::make_shared<Tensor<T>>(score_shape_);
        this->fillTensorFromTorch(scores, torch_scores);

        indices = compute_expected_nms(torch_boxes, torch_scores);
    }

    std::shared_ptr<Tensor<int>> compute_expected_nms(const torch::Tensor& boxes_tensor, 
                                                     const torch::Tensor& scores_tensor) {
        
        int64_t num_batches = scores_tensor.size(0);
        int64_t num_classes = scores_tensor.size(1);
        int64_t spatial_dimension = scores_tensor.size(2);

        std::vector<std::vector<int>> keep_boxes;

        // 对每个批次和每个类别分别进行NMS
        for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            auto batch_boxes = boxes_tensor[batch_idx]; // [spatial_dimension, 4]
            
            for (int64_t class_idx = 0; class_idx < num_classes; ++class_idx) {
                auto class_scores = scores_tensor[batch_idx][class_idx]; // [spatial_dimension]

                // 按分数降序排列
                auto sorted_result = torch::sort(class_scores, /*dim=*/0, /*descending=*/true);
                auto sorted_scores = std::get<0>(sorted_result);
                auto sorted_indices = std::get<1>(sorted_result);

                // 过滤低于阈值的分数
                auto valid_mask = sorted_scores >= score_threshold_;
                auto valid_indices = sorted_indices.masked_select(valid_mask);
                auto valid_scores = sorted_scores.masked_select(valid_mask);

                if (valid_indices.numel() == 0) {
                    continue;
                }

                auto selected_boxes = batch_boxes.index({valid_indices});
                std::vector<bool> suppressed(valid_indices.numel(), false);

                // ONNX NMS: max_output_boxes_per_class caps the kept boxes per
                // (batch, class). 0 means unlimited. The GPU shader applies the
                // same cap, so the reference must too or the counts diverge.
                int kept_this_class = 0;
                for (int64_t i = 0; i < valid_indices.numel(); ++i) {
                    if (suppressed[i]) continue;

                    if (max_per_class_ > 0 && kept_this_class >= max_per_class_) {
                        break;
                    }

                    // 添加当前框到保留列表
                    std::vector<int> result_entry = {
                        static_cast<int>(batch_idx),
                        static_cast<int>(class_idx),
                        static_cast<int>(valid_indices[i].item<int64_t>())
                    };
                    keep_boxes.push_back(result_entry);
                    kept_this_class++;

                    // 计算当前框与后续所有框的IoU
                    auto box_i = selected_boxes[i];
                    auto x1_i = box_i[0].item<float>();
                    auto y1_i = box_i[1].item<float>();
                    auto x2_i = box_i[2].item<float>();
                    auto y2_i = box_i[3].item<float>();
                    auto area_i = (x2_i - x1_i) * (y2_i - y1_i);

                    for (int64_t j = i + 1; j < valid_indices.numel(); ++j) {
                        if (suppressed[j]) continue;

                        auto box_j = selected_boxes[j];
                        auto x1_j = box_j[0].item<float>();
                        auto y1_j = box_j[1].item<float>();
                        auto x2_j = box_j[2].item<float>();
                        auto y2_j = box_j[3].item<float>();

                        // 计算交集
                        auto xx1 = std::max(x1_i, x1_j);
                        auto yy1 = std::max(y1_i, y1_j);
                        auto xx2 = std::min(x2_i, x2_j);
                        auto yy2 = std::min(y2_i, y2_j);

                        auto w = std::max(0.0F, xx2 - xx1);
                        auto h = std::max(0.0F, yy2 - yy1);
                        auto inter = w * h;

                        // 计算IoU
                        auto area_j = (x2_j - x1_j) * (y2_j - y1_j);
                        auto union_area = area_i + area_j - inter;
                        float iou = 0.0F;
                        if (union_area > 1e-8F) {
                            iou = inter / union_area;
                        }

                        if (iou > iou_threshold_) {
                            suppressed[j] = true;
                        }
                    }
                }
            }
        }

        // 创建期望输出张量
        if (keep_boxes.empty()) {
            auto result_tensor = std::make_shared<Tensor<int>>(std::vector<int>{0, 3});
            return result_tensor;
        }

        auto result_tensor = std::make_shared<Tensor<int>>(std::vector<int>{
            static_cast<int>(keep_boxes.size()), 3
        });

        std::vector<int> result_data;
        for (const auto& box : keep_boxes) {
            result_data.insert(result_data.end(), box.begin(), box.end());
        }
        result_tensor->fillToCPU(result_data);

        std::cout << "boxes_tensor:" << std::endl; 
        std::cout << boxes_tensor << std::endl;
        std::cout << "scores_tensor:" << std::endl; 
        std::cout << scores_tensor << std::endl;
        result_tensor->print_tensor();

        return result_tensor;
    }
};
}

TEST(NmsTest, NmsComprehensiveTest) {
    const std::vector<std::tuple<std::vector<int>, std::vector<int>, int, float, float>> test_cases = {
        {{1, 10, 4}, {1, 2, 10}, 0, 0.5, 0.5},
        {{1, 200, 4}, {1, 1, 200}, 50, 0.7F, 0.4F},
        {{1, 50, 4}, {1, 7, 50}, 0, 0.3F, 0.2F},
    };

    for (const auto &test_case : test_cases) {
        auto [box_shape, scores_shape, max_per_class, iou_threshold, score_threshold] = test_case; 
        LOG_INFO("NMS for max_per_class: %d, iou_threshold: %f, score_threshold: %f", max_per_class, iou_threshold, score_threshold);
        LOG_INFO("Testing FP32");
        NmsTest<float> nmstest(box_shape, scores_shape, max_per_class, iou_threshold, score_threshold);
        EXPECT_TRUE(nmstest.run_test({nmstest.boxes, nmstest.scores}, {nmstest.indices}, [&nmstest](std::unique_ptr<vkop::ops::Operator> &op){
            auto *topk_op = dynamic_cast<Nms *>(op.get());
            if (!topk_op) {
                LOG_ERROR("Failed to cast operator to Topk");
                return;
            }
            topk_op->setAttribute(nmstest.attributes);
        }));

        LOG_INFO("Testing FP16");
        NmsTest<uint16_t> nmstest1(box_shape, scores_shape, max_per_class, iou_threshold, score_threshold);
        EXPECT_TRUE(nmstest1.run_test({nmstest1.boxes, nmstest1.scores}, {nmstest1.indices}, [&nmstest1](std::unique_ptr<vkop::ops::Operator> &op){
            auto *topk_op = dynamic_cast<Nms *>(op.get());
            if (!topk_op) {
                LOG_ERROR("Failed to cast operator to Topk");
                return;
            }
            topk_op->setAttribute(nmstest1.attributes);
        }));
    }
}
