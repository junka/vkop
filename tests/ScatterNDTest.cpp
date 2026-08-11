// Copyright 2026 @junka
#include "setup.hpp"
#include "core/Tensor.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using vkop::core::Tensor;
using vkop::tests::TestCase;
namespace ops = vkop::ops;

// ScatterND is a GPU buffer shader op. It mirrors the real LLM usage
// (llm.vkopbin): data float [1,1,64], indices int64 [1,1,1,3] = {0,0,k},
// updates float [1,1,1], output float [1,1,64]. The shader does two passes
// in one command buffer: copy data->out, then scatter updates.

namespace {

static void run_scatternd(const std::shared_ptr<vkop::core::ITensor> &data,
                          const std::shared_ptr<vkop::core::ITensor> &indices,
                          const std::shared_ptr<vkop::core::ITensor> &updates,
                          const std::shared_ptr<vkop::core::ITensor> &output) {
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    auto op = ops::create_from_type(ops::OpType::SCATTER_ND, 0, 0, true);
    op->set_runtime_device(dev, cmdpool);

    for (auto &t : {data, output}) {
        if (t->dtype() == typeid(float)) {
            vkop::core::as_tensor<float>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<float>(t)->copyToGPU(cmdpool);
        }
    }
    if (indices->dtype() == typeid(int64_t)) {
        vkop::core::as_tensor<int64_t>(indices)->as_storage_buffer(dev);
        vkop::core::as_tensor<int64_t>(indices)->copyToGPU(cmdpool);
    } else if (indices->dtype() == typeid(int)) {
        vkop::core::as_tensor<int>(indices)->as_storage_buffer(dev);
        vkop::core::as_tensor<int>(indices)->copyToGPU(cmdpool);
    }
    if (updates->dtype() == typeid(float)) {
        vkop::core::as_tensor<float>(updates)->as_storage_buffer(dev);
        vkop::core::as_tensor<float>(updates)->copyToGPU(cmdpool);
    }

    op->onExecute({data, indices, updates}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    vkop::core::as_tensor<float>(output)->copyToCPU(cmdpool);
}

TEST(ScatterNDTest, Basic) {
    // data [1,1,64] = 0..63 ; scatter updates at indices {0,0,k}
    // into the output copy.
    int D = 64;
    auto data = torch::arange(0, D).to(torch::kFloat32).view({1, 1, D});

    // k values to scatter
    std::vector<int> ks = {0, 2, 4, 6, 8, 10};
    auto updates = torch::tensor({100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f})
                       .view({6, 1, 1});
    // indices [6,1,1,3] each {0,0,k}
    auto indices = torch::zeros({6, 1, 1, 3}, torch::kInt64);
    for (size_t i = 0; i < ks.size(); ++i) {
        indices[i][0][0][0] = 0;
        indices[i][0][0][1] = 0;
        indices[i][0][0][2] = ks[i];
    }

    // expected output
    auto ref = data.clone();
    for (size_t i = 0; i < ks.size(); ++i) {
        ref[0][0][ks[i]] = updates[i][0][0];
    }

    auto tdata = std::make_shared<Tensor<float>>(std::vector<int>{1, 1, D});
    auto tindices = std::make_shared<Tensor<int64_t>>(std::vector<int>{6, 1, 1, 3});
    auto tupd = std::make_shared<Tensor<float>>(std::vector<int>{6, 1, 1});
    auto tout = std::make_shared<Tensor<float>>(std::vector<int>{1, 1, D});

    auto flat = data.cpu().contiguous().flatten();
    tdata->fillToCPU(std::vector<float>(flat.data_ptr<float>(),
                                        flat.data_ptr<float>() + D));
    auto ind_flat = indices.cpu().contiguous().flatten();
    const int64_t *ip = ind_flat.data_ptr<int64_t>();
    tdata->fillToCPU(std::vector<float>(flat.data_ptr<float>(),
                                        flat.data_ptr<float>() + D));
    tindices->fillToCPU(std::vector<int64_t>(ip, ip + ind_flat.numel()));
    auto up_flat = updates.cpu().contiguous().flatten();
    tupd->fillToCPU(std::vector<float>(up_flat.data_ptr<float>(),
                                       up_flat.data_ptr<float>() + 6));
    // output initially uninitialized; ScatterND pass 0 copies data into it.
    auto out_flat = ref.cpu().contiguous().flatten();
    tout->fillToCPU(std::vector<float>(out_flat.data_ptr<float>(),
                                       out_flat.data_ptr<float>() + D));

    run_scatternd(tdata, tindices, tupd, tout);

    for (int i = 0; i < D; ++i) {
        EXPECT_FLOAT_EQ((*tout)[i], ref.flatten()[i].item<float>())
            << "mismatch at " << i;
    }
}

} // namespace
