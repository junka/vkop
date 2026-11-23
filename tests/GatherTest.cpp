#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Gather.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Gather;

namespace {

template<typename T>
class GatherTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    std::vector<int> indices_shape_;
    std::vector<int> indices_data_;
    int axis;

    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<int64_t>> indices;
    std::shared_ptr<Tensor<T>> output;
    std::unordered_map<std::string, std::string> attributes;

    GatherTest(std::vector<int> &input_shape, int axis, std::vector<int>& indices_shape, std::vector<int>& indices_data): TestCase<T>("Gather"),input_shape_(input_shape), indices_shape_(indices_shape), indices_data_(indices_data), axis(axis) {
        attributes = {
            {"axis", std::to_string(axis)},
        };
        initTestdata();
    }
    torch::Tensor onnx_gather(const torch::Tensor& data, const torch::Tensor& indices, int axis) {
        if (axis < 0) {
            axis += data.dim();
        }

        std::vector<torch::indexing::TensorIndex> index_list;
        index_list.reserve(data.dim());

        for (int64_t i = 0; i < axis; ++i) {
            index_list.emplace_back(torch::indexing::Slice()); 
        }

        index_list.emplace_back(indices);

        for (int64_t i = axis + 1; i < data.dim(); ++i) {
            index_list.emplace_back(torch::indexing::Slice());
        }

        return data.index(index_list);
    }

private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<T>>(input_shape_);

        indices = std::make_shared<Tensor<int64_t>>(indices_shape_);
        std::vector<int64_t> indices_data_i64(indices_data_.begin(),
                                              indices_data_.end());
        indices->fillToCPU(indices_data_i64);
        indices->print_tensor();

        std::vector<int64_t> inshape(input_shape_.begin(), input_shape_.end());
        std::vector<int64_t> indshape(indices_shape_.begin(), indices_shape_.end());
        auto torch_input = torch::randn(inshape, this->getTorchConf());
        auto torch_indices = torch::tensor(indices_data_, torch::TensorOptions().dtype(torch::kInt64)).reshape(indshape);

        std::cout << "axis " << axis << std::endl;
        std::cout << "torch_input: " << torch_input << std::endl;
        std::cout << "torch_indices: " << torch_indices << std::endl;
        auto torch_output = onnx_gather(torch_input, torch_indices, axis);
        std::cout << "torch_output: " << torch_output << std::endl;
        this->fillTensorFromTorch(input, torch_input);

        std::vector<int> output_shape;
        for (auto s : torch_output.sizes()) {
            output_shape.push_back(static_cast<int>(s));
        }
        output = std::make_shared<Tensor<T>>(output_shape);
        this->fillTensorFromTorch(output, torch_output);
        // output->print_tensor();
    }
};
}

TEST(GatherTest, GatherComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>, int, std::vector<int>, std::vector<int>>> test_cases = {
        {{3, 4}, 0, {3}, {1, 0, 2}},
        {{3, 4}, 1, {3, 2}, {1, 0, 2, 1, 0, 3}},  // Shape: [3, 4], axis: 1, indices shape: [3, 2], indices: [[1, 0], [2, 1], [0, 3]]
        {{2, 4}, 0, {1, 2, 1}, {0, 1}},         // Shape: [2, 4], axis: 0, indices shape: [1, 2, 1], indices: [[[0], [1]]]
        {{2, 3, 4}, 2, {2, 3}, {0, 1, 1, 0, 1, 2}}  // Shape: [2, 3, 4], axis: 2, indices shape: [2, 3, 2], indices: complex
    };
    for (const auto& test_case : test_cases) {
        auto [input_shape, axis, indices_shape, indices_data] = test_case;

        LOG_INFO("Testing Gather fp32");
        GatherTest<float> gather_test(input_shape, axis, indices_shape, indices_data);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
            gather_test.input,
            gather_test.indices
        };
        EXPECT_TRUE(gather_test.run_test(inputs, {gather_test.output}, [&gather_test](std::unique_ptr<vkop::ops::Operator> &op){
            auto *gather_op = dynamic_cast<Gather *>(op.get());
            if (!gather_op) {
                LOG_ERROR("Failed to cast operator to Gather");
                return;
            }
            gather_op->setAttribute(gather_test.attributes);
        }));


        LOG_INFO("Testing Gather fp16");
        GatherTest<uint16_t> gather_test1(input_shape, axis, indices_shape, indices_data);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs1 = {
            gather_test1.input,
            gather_test1.indices
        };
        EXPECT_TRUE(gather_test1.run_test(inputs1, {gather_test1.output}, [&gather_test1](std::unique_ptr<vkop::ops::Operator> &op){
            auto *gather_op = dynamic_cast<Gather *>(op.get());
            if (!gather_op) {
                LOG_ERROR("Failed to cast operator to Gather");
                return;
            }
            gather_op->setAttribute(gather_test1.attributes);
        }));
    }
}

// Pure math behind the Gather zero-copy fast path: a gather is one contiguous
// run of the data's bytes iff every dim before `axis` is 1 (no outer stride
// between blocks) and the indices are consecutive ascending. A wrong answer
// here is silently-wrong data (a bad offset binds the wrong bytes), so pin the
// interesting cases.
TEST(GatherTest, ContiguousGatherRangeMath) {
    using vkop::ops::contiguous_gather_range;
    VkDeviceSize bo = 0;
    VkDeviceSize bl = 0;
    // axis 0, consecutive {2,3} on {8,4,4}: inner 16 -> offset 2*16 elements.
    EXPECT_TRUE(contiguous_gather_range({8, 4, 4}, {2, 3}, 0, sizeof(float), bo,
                                        bl));
    EXPECT_EQ(bo, VkDeviceSize(2 * 16) * sizeof(float));
    EXPECT_EQ(bl, VkDeviceSize(2 * 16) * sizeof(float));
    // A single index is trivially contiguous.
    EXPECT_TRUE(
        contiguous_gather_range({8, 4, 4}, {5}, 0, sizeof(float), bo, bl));
    EXPECT_EQ(bo, VkDeviceSize(5 * 16) * sizeof(float));
    // Non-consecutive / descending -> a real gather.
    EXPECT_FALSE(
        contiguous_gather_range({8, 4, 4}, {0, 2}, 0, sizeof(float), bo, bl));
    EXPECT_FALSE(
        contiguous_gather_range({8, 4, 4}, {3, 2}, 0, sizeof(float), bo, bl));
    // An outer dim > 1 strides between the gathered blocks -> not one run.
    EXPECT_FALSE(
        contiguous_gather_range({2, 4, 4}, {1}, 1, sizeof(float), bo, bl));
    // prefix all 1 lets an inner axis qualify: [1,4,4] axis 1 idx {1,2}.
    EXPECT_TRUE(
        contiguous_gather_range({1, 4, 4}, {1, 2}, 1, sizeof(float), bo, bl));
    EXPECT_EQ(bo, VkDeviceSize(1 * 4) * sizeof(float));
    EXPECT_EQ(bl, VkDeviceSize(2 * 4) * sizeof(float));
    // Out-of-range block and empty indices.
    EXPECT_FALSE(
        contiguous_gather_range({8, 4, 4}, {7, 8}, 0, sizeof(float), bo, bl));
    EXPECT_FALSE(
        contiguous_gather_range({8, 4, 4}, {}, 0, sizeof(float), bo, bl));
}

// Zero-copy contiguous Gather end to end. When a gather selects a contiguous
// run of the data, Gather binds the data's VkBuffer at the block's byte offset
// (VkDescriptorBufferInfo.offset, transparent to the shader) instead of
// dispatching gather.comp. Two assertions: the numbers still match torch (this
// exercises the offset forwarding in VulkanBuffer::getDescriptorInfo /
// copyBufferToStageBuffer — read at the parent's start instead of +offset and
// the output would be the data's head), and the alias actually fired, since a
// regression that silently disables the fast path would still pass a
// correctness-only check.
TEST(GatherTest, GatherContiguousBufferAlias) {
    vkop::tests::ScopedBufferBackend buffer;
    std::vector<int> input_shape{8, 4, 8};
    std::vector<int> indices_shape{2};
    std::vector<int> indices_data{2, 3};

    LOG_INFO("Testing Gather contiguous alias fp32");
    GatherTest<float> fp32(input_shape, 0, indices_shape, indices_data);
    fp32.indices->set_host_authoritative();
    auto before = vkop::core::ITensor::shape_view_stats().buffer_aliases;
    EXPECT_TRUE(fp32.run_test({fp32.input, fp32.indices}, {fp32.output}));
    EXPECT_EQ(vkop::core::ITensor::shape_view_stats().buffer_aliases,
              before + 1);

    LOG_INFO("Testing Gather contiguous alias fp16");
    GatherTest<uint16_t> fp16(input_shape, 0, indices_shape, indices_data);
    fp16.indices->set_host_authoritative();
    before = vkop::core::ITensor::shape_view_stats().buffer_aliases;
    EXPECT_TRUE(fp16.run_test({fp16.input, fp16.indices}, {fp16.output}));
    EXPECT_EQ(vkop::core::ITensor::shape_view_stats().buffer_aliases,
              before + 1);
}

// A contiguous run whose length is not a multiple of four (the ViT's 42-entry
// position-table rows). The alias window used to be exactly the run, but a
// tensor's buffer is sized to its element count rounded up to four, so the
// first READER that bound the output (Tensor::make_vkbuff via
// as_storage_buffer) found the view too small, dropped it and allocated a
// fresh, zeroed buffer: every consumer read zeros while a readback of the
// view itself still looked right. Bind the output the way a reader does and
// check it is still the view.
TEST(GatherTest, GatherAliasSurvivesReaderBind) {
    vkop::tests::ScopedBufferBackend buffer;
    auto dev = vkop::tests::TestEnv::get_device();
    auto src = std::make_shared<vkop::VulkanBuffer>(
        dev, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // What Gather now asks for: the padded window (44 halves = 88 bytes).
    vkop::core::Tensor<uint16_t> padded(std::vector<int>{42});
    padded.alias_storage_buffer_range(src, 16, 88, nullptr);
    EXPECT_TRUE(padded.as_storage_buffer(dev)->is_view_range());
    // What it used to ask for: exactly the run. A reader's bind replaces it.
    vkop::core::Tensor<uint16_t> exact(std::vector<int>{42});
    exact.alias_storage_buffer_range(src, 16, 84, nullptr);
    EXPECT_FALSE(exact.as_storage_buffer(dev)->is_view_range());
}

// The complement: non-consecutive indices must NOT alias — they fall through to
// the real gather dispatch and still produce correct numbers. Guards against a
// fast path that fires too eagerly and binds the wrong block.
TEST(GatherTest, GatherNonContiguousNoAlias) {
    vkop::tests::ScopedBufferBackend buffer;
    std::vector<int> input_shape{8, 4, 8};
    std::vector<int> indices_shape{2};
    std::vector<int> indices_data{0, 2};

    GatherTest<float> fp32(input_shape, 0, indices_shape, indices_data);
    fp32.indices->set_host_authoritative();
    auto before = vkop::core::ITensor::shape_view_stats().buffer_aliases;
    EXPECT_TRUE(fp32.run_test({fp32.input, fp32.indices}, {fp32.output}));
    EXPECT_EQ(vkop::core::ITensor::shape_view_stats().buffer_aliases, before);
}
// ---------------------------------------------------------------------------
// Out-of-range negative index (the MiniMax-H3 crash)
// ---------------------------------------------------------------------------
//
// ONNX Gather wraps a negative index ONCE (`idx += axis_size`) and then
// requires 0 <= idx < axis_size. The host int64 path (cpuComputeInt64, taken
// whenever VKOP_HOST_SHAPE is on — the default) used to skip the range check
// and index the data tensor with a still-negative number. With an empty-ish
// table and index -1 that read element -1, i.e. heap in front of the buffer;
// it segfaulted at /transformer_blocks.0/ff/net.0/Gather in a converted
// MiniMax-H3 transformer and VAE decoder, while onnxruntime ran the identical
// ONNX fine.
//
// Defined result: a skipped element. Both the host path and gather.comp /
// gather_int64.comp now do exactly that, and the output tensor starts zeroed,
// so an out-of-range index yields 0. Skipping (rather than clamping, wrapping
// twice, or throwing) is the right answer because:
//   * it is what the GPU kernels already did for the indices-side miss, so the
//     same graph cannot change behaviour when VKOP_HOST_SHAPE is toggled;
//   * clamping would invent a value ONNX never asked for, and wrapping twice
//     would silently turn -1 on a 0-length axis into a different read;
//   * it is a defined, reproducible value in an output whose other elements
//     are still correct — the alternative is a crash or a stale word.
namespace {

// data [4] int64, indices [3] int64 = {-1 (wraps to 3, valid), 9 (>= axis),
// -9 (wraps to -5, still negative)} at axis 0.
const std::vector<int64_t> kGatherOobData = {10, 20, 30, 40};
const std::vector<int64_t> kGatherOobIdx = {-1, 9, -9};
const std::vector<int64_t> kGatherOobExpect = {40, 0, 0};

bool run_gather_int64_oob_case() {
    auto &dev = []() -> std::shared_ptr<vkop::VulkanDevice> & {
        static std::shared_ptr<vkop::VulkanDevice> d = [] {
            if (!vkop::tests::TestEnv::is_initialized())
                vkop::tests::TestEnv::initialize();
            return vkop::tests::TestEnv::get_device();
        }();
        return d;
    }();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();

    auto data = std::make_shared<Tensor<int64_t>>(std::vector<int>{4});
    data->fillToCPU(kGatherOobData);
    // Keep the CPU copy alive: the host path reads data_ directly.
    auto keep_data = data->data();
    data->as_storage_buffer(dev);
    data->copyToGPU(cmdpool, keep_data.data());

    auto indices = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    indices->fillToCPU(kGatherOobIdx);
    auto keep_idx = indices->data();
    indices->as_storage_buffer(dev);
    indices->copyToGPU(cmdpool, keep_idx.data());

    auto output = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    output->as_storage_buffer(dev);
    output->toGPU();

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::GATHER, 0, 0,
                                          /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type(GATHER) returned null");
        return false;
    }
    op->set_runtime_device(dev, cmdpool);
    op->setAttribute({{"axis", "0"}});

    // The whole point: this must not crash on the out-of-range element.
    op->onExecute({data, indices}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    output->copyToCPU(cmdpool);
    const auto &got = output->data();
    if (got.size() != kGatherOobExpect.size()) {
        LOG_ERROR("Gather int64 OOB: output has %zu elems, want %zu",
                  got.size(), kGatherOobExpect.size());
        return false;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != kGatherOobExpect[i]) {
            LOG_ERROR("Gather int64 OOB: elem %zu = %lld, want %lld", i,
                      static_cast<long long>(got[i]),
                      static_cast<long long>(kGatherOobExpect[i]));
            return false;
        }
    }
    return true;
}

} // namespace

// Default path: cpuComputeInt64 (VKOP_HOST_SHAPE on). This is the one that
// crashed.
TEST(GatherTest, Int64NegativeIndexOutOfRangeHostPath) {
    EXPECT_TRUE(run_gather_int64_oob_case());
}

// The same case through the GPU kernel (gather_int64.comp), so the two paths
// are pinned to the same answer. host_shape_enabled() latches its env read in
// a function-local static, so this only switches paths when it runs before any
// other Gather execute in the process — which is the case under ctest (each
// TEST is its own process). If it does run second, it simply re-checks the
// host path against the same expectation, which must also hold.
TEST(GatherTest, Int64NegativeIndexOutOfRangeGpuShader) {
    setenv("VKOP_HOST_SHAPE", "0", 1);
    EXPECT_TRUE(run_gather_int64_oob_case());
}
