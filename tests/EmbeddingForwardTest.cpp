// Copyright 2025 @junka
// Embedding forward test: validates the Vulkan buffer EmbeddingForward op
// against a port of cuEmbed's CPU reference (EmbeddingForwardCpu). fp32,
// int32 indices, covering Sum/Mean/Concat x fixed/CSR x weighted/unweighted.

#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/EmbeddingForward.hpp"

#include <gtest/gtest.h>

using vkop::core::Tensor;
namespace ops = vkop::ops;

namespace {

// cuEmbed CombineMode codes (mirror ops/EmbeddingForward.hpp).
constexpr int kSum = 0;
constexpr int kMean = 1;
constexpr int kConcat = 2;

struct EmbedCase {
    int batch;
    int embed_width;
    int hotness;
    int num_categories;
    int combine_mode;
    bool csr;
    bool weighted;
};

// Port of cuEmbed's EmbeddingForwardCpu (fp32, int32). Produces the
// expected output for comparison.
std::vector<float> cpu_reference(const std::vector<float> &embedding,
                                 int embed_width,
                                 const std::vector<int> &indices,
                                 const std::vector<int> *offsets,
                                 const std::vector<float> *weights,
                                 int batch_size, int num_hots,
                                 int combine_mode) {
    int out_size;
    if (combine_mode == kConcat) {
        out_size = batch_size * num_hots * embed_width;
    } else {
        out_size = batch_size * embed_width;
    }
    std::vector<float> ret(out_size, 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        for (int k = 0; k < embed_width; ++k) {
            float sum = 0.0f;
            int hotness =
                (offsets == nullptr) ? num_hots : ((*offsets)[i + 1] - (*offsets)[i]);
            int index_start =
                (offsets == nullptr) ? i * num_hots : (*offsets)[i];
            int write_idx = i * embed_width + k;
            for (int j = 0; j < hotness; ++j) {
                int64_t read_idx = static_cast<int64_t>(indices[index_start + j]) *
                                   embed_width + k;
                if (combine_mode == kConcat) {
                    write_idx = index_start * embed_width + j * embed_width + k;
                    ret[write_idx] = embedding[read_idx];
                } else {
                    float weight = (weights == nullptr)
                                       ? 1.0f
                                       : (*weights)[index_start + j];
                    sum += embedding[read_idx] * weight;
                }
            }
            if (combine_mode == kSum) {
                ret[write_idx] = sum;
            } else if (combine_mode == kMean) {
                ret[write_idx] = (hotness > 0) ? sum / float(hotness) : 0.0f;
            }
        }
    }
    return ret;
}

// Drive the Vulkan buffer op for one case and compare to the CPU reference.
bool run_case(const EmbedCase &c) {
    std::mt19937 rng(42 + c.batch * 7 + c.embed_width);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> idx_dist(0, c.num_categories - 1);

    // Generate inputs.
    std::vector<float> embedding(c.num_categories * c.embed_width);
    for (auto &v : embedding) v = dist(rng);

    std::vector<int> indices;
    std::vector<int> offsets;
    std::vector<float> weights;
    if (c.csr) {
        offsets.push_back(0);
        for (int b = 0; b < c.batch; ++b) {
            int hot = c.hotness; // CSR variable hotness; use fixed for test
            for (int j = 0; j < hot; ++j) indices.push_back(idx_dist(rng));
            offsets.push_back(static_cast<int>(indices.size()));
        }
    } else {
        for (int i = 0; i < c.batch * c.hotness; ++i)
            indices.push_back(idx_dist(rng));
    }
    if (c.weighted) {
        int n_idx = static_cast<int>(indices.size());
        weights.resize(n_idx);
        for (auto &w : weights) w = dist(rng);
    }

    // CPU reference.
    int num_hots = c.csr ? 0 : c.hotness;
    auto expected = cpu_reference(embedding, c.embed_width, indices,
                                  c.csr ? &offsets : nullptr,
                                  c.weighted ? &weights : nullptr,
                                  c.batch, num_hots, c.combine_mode);

    // Build Vulkan tensors.
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();

    auto embed_t = std::make_shared<Tensor<float>>(
        std::vector<int>{c.num_categories, c.embed_width});
    embed_t->fillToCPU(embedding);
    auto idx_t = std::make_shared<Tensor<int>>(
        std::vector<int>{static_cast<int>(indices.size())});
    idx_t->fillToCPU(indices);

    std::shared_ptr<Tensor<int>> off_t;
    if (c.csr) {
        off_t = std::make_shared<Tensor<int>>(
            std::vector<int>{static_cast<int>(offsets.size())});
        off_t->fillToCPU(offsets);
    }
    std::shared_ptr<Tensor<float>> wt_t;
    if (c.weighted) {
        wt_t = std::make_shared<Tensor<float>>(
            std::vector<int>{static_cast<int>(weights.size())});
        wt_t->fillToCPU(weights);
    }
    int out_elems = static_cast<int>(expected.size());
    auto out_t = std::make_shared<Tensor<float>>(std::vector<int>{out_elems});

    // Create + configure op.
    auto op = ops::create_from_type(ops::OpType::EMBEDDING_FORWARD,
                                    /*fp16=*/0, /*tensorcore=*/0,
                                    /*backend_buffer=*/true);
    if (!op) return false;
    op->set_runtime_device(dev, cmdpool);
    std::unordered_map<std::string, std::string> attrs = {
        {"embed_width", std::to_string(c.embed_width)},
        {"combine_mode", std::to_string(c.combine_mode)},
        {"is_csr", std::to_string(c.csr ? 1 : 0)},
        {"is_weighted", std::to_string(c.weighted ? 1 : 0)},
    };
    if (!c.csr) attrs["num_hots"] = std::to_string(c.hotness);
    op->setAttribute(attrs);

    // Bind SSBOs and upload.
    embed_t->as_storage_buffer(dev); embed_t->copyToGPU(cmdpool);
    idx_t->as_storage_buffer(dev);   idx_t->copyToGPU(cmdpool);
    if (off_t) { off_t->as_storage_buffer(dev); off_t->copyToGPU(cmdpool); }
    if (wt_t)  { wt_t->as_storage_buffer(dev);  wt_t->copyToGPU(cmdpool); }
    out_t->as_storage_buffer(dev); out_t->toGPU();

    // Execute. Inputs layout: [embed, idx, off?, wt?].
    std::vector<std::shared_ptr<vkop::core::ITensor>> ins = {embed_t, idx_t};
    if (off_t) ins.push_back(off_t);
    if (wt_t) ins.push_back(wt_t);
    op->onExecute(ins, {out_t}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait(); dev->wait_all_done();
    out_t->copyToCPU(cmdpool);

    // Compare.
    const float tol = 1e-4f;
    for (int i = 0; i < out_elems; ++i) {
        float got = (*out_t)[i];
        float exp = expected[i];
        if (std::abs(got - exp) > tol) {
            LOG_ERROR("case (bs=%d ew=%d hot=%d mode=%d csr=%d wt=%d) mismatch %d: %f vs %f",
                      c.batch, c.embed_width, c.hotness, c.combine_mode,
                      c.csr, c.weighted, i, got, exp);
            return false;
        }
    }
    return true;
}

} // namespace

TEST(EmbeddingForwardTest, Sum) {
    EXPECT_TRUE(run_case({3, 2, 4, 100, kSum, false, false}));
    EXPECT_TRUE(run_case({1023, 32, 26, 10000, kSum, false, false}));
    EXPECT_TRUE(run_case({3, 512, 63, 10000, kSum, false, false}));
    EXPECT_TRUE(run_case({1023, 514, 63, 10000, kSum, false, false}));
}

TEST(EmbeddingForwardTest, SumCsr) {
    EXPECT_TRUE(run_case({3, 2, 4, 100, kSum, true, false}));
    EXPECT_TRUE(run_case({1023, 32, 26, 10000, kSum, true, false}));
    EXPECT_TRUE(run_case({3, 512, 63, 10000, kSum, true, false}));
}

TEST(EmbeddingForwardTest, SumWeighted) {
    EXPECT_TRUE(run_case({3, 2, 4, 100, kSum, false, true}));
    EXPECT_TRUE(run_case({1023, 32, 26, 10000, kSum, false, true}));
    EXPECT_TRUE(run_case({1023, 36, 26, 10000, kSum, false, true}));
}

TEST(EmbeddingForwardTest, SumCsrWeighted) {
    EXPECT_TRUE(run_case({3, 2, 4, 100, kSum, true, true}));
    EXPECT_TRUE(run_case({1023, 32, 26, 10000, kSum, true, true}));
}

TEST(EmbeddingForwardTest, Mean) {
    EXPECT_TRUE(run_case({3, 2, 4, 100, kMean, false, false}));
    EXPECT_TRUE(run_case({1023, 32, 26, 10000, kMean, false, false}));
    EXPECT_TRUE(run_case({3, 512, 63, 10000, kMean, false, false}));
}

TEST(EmbeddingForwardTest, MeanCsr) {
    EXPECT_TRUE(run_case({3, 2, 4, 100, kMean, true, false}));
    EXPECT_TRUE(run_case({1023, 32, 26, 10000, kMean, true, false}));
}

TEST(EmbeddingForwardTest, Concat) {
    EXPECT_TRUE(run_case({3, 2, 4, 100, kConcat, false, false}));
    EXPECT_TRUE(run_case({1023, 32, 26, 10000, kConcat, false, false}));
    EXPECT_TRUE(run_case({3, 512, 63, 10000, kConcat, false, false}));
    EXPECT_TRUE(run_case({1023, 514, 63, 10000, kConcat, false, false}));
}

TEST(EmbeddingForwardTest, OddEmbedWidth) {
    EXPECT_TRUE(run_case({3, 36, 26, 10000, kSum, false, false}));
    EXPECT_TRUE(run_case({1023, 514, 63, 10000, kMean, false, false}));
    EXPECT_TRUE(run_case({3, 36, 26, 10000, kConcat, false, false}));
}
