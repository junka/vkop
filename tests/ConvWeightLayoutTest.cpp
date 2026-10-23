// Copyright 2026 @junka
// Runtime half of the Conv weight-image layout.
//
// Both fixtures come from the same graph (model/conv_weight_layout.onnx): a
// rank-4 Conv whose weight is [16,3,3,3] and a rank-5 Conv whose weight is
// [16,3,2,3,3]. The image conv shader addresses a weight through two push
// constants, transpose and pack; the converter writes them on the node as
// weight_transpose / weight_pack and this loader reads them verbatim.
//
// The rank-5 case is the regression this pins. The loader used to decide both
// flags itself -- a name -> op_type map applied set_transpose() to every
// Conv-consumed weight, while set_pack() was gated on rank == 4. A 5-D weight
// therefore came back transpose=true, and Tensor::getGPUShape() reads dims_[0]
// as the batch only at rank 4 (:1 otherwise), so transpose swapped a synthetic
// 1 into the channel slot and promoted dims_[2] into the batch -- a geometry
// the shader cannot have addressed. Neither flag may be set on a rank-5 weight.
//
// Conv3d is unlowered on both backends, so what is testable here is the layout
// decision, not the convolution. The second fixture is the same graph converted
// with the pass ablated: with no attributes on the wire the loader falls back
// to the rank-gated geometry rule, which is what pre-existing .vkopbin files
// (and VKOP_NO_FUSIONS runs) still go through.

#include <memory>
#include <string>

#include "core/runtime.hpp"
#include "setup.hpp"

#include <gtest/gtest.h>

namespace {

using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::tests::TestEnv;

std::shared_ptr<Runtime> Load(const std::string &file) {
    auto rt = std::make_shared<Runtime>(TestEnv::get_command_pool(),
                                        CONV_WEIGHT_LAYOUT_DATA "/" + file, 0);
    rt->LoadModel();
    return rt;
}

void ExpectLayout(const std::shared_ptr<Runtime> &rt, const char *name,
                  bool transpose, bool pack) {
    auto t = rt->GetInitializer(name);
    ASSERT_TRUE(t != nullptr) << "no initializer named " << name;
    EXPECT_EQ(t->get_transpose(), transpose)
        << name << " (rank " << int(t->num_dims()) << ") transpose";
    EXPECT_EQ(t->get_pack(), pack)
        << name << " (rank " << int(t->num_dims()) << ") pack";
}

}  // namespace

// The converter's attributes, read verbatim. W_r4 is the ordinary case: the
// shader wants the transposed batch/chan addressing with the channel group
// folded into the width. W_r5 must get neither.
TEST(ConvWeightLayoutTest, Rank4TransposedRank5Plain) {
    auto rt = Load("conv_weight_layout.vkopbin");
    ExpectLayout(rt, "W_r4", /*transpose=*/true, /*pack=*/true);
    ExpectLayout(rt, "W_r5", /*transpose=*/false, /*pack=*/false);
}

// Same model, converted with ConvWeightLayoutPass ablated: no attributes reach
// the file, so this is the rank-gated geometry fallback. It has to agree with
// the attribute path on both ranks -- in particular it must NOT hand a 5-D
// weight transpose=true, which is exactly what the old loader did.
TEST(ConvWeightLayoutTest, GeometryFallbackMatchesAttributes) {
    auto rt = Load("conv_weight_layout_fallback.vkopbin");
    ExpectLayout(rt, "W_r4", /*transpose=*/true, /*pack=*/true);
    ExpectLayout(rt, "W_r5", /*transpose=*/false, /*pack=*/false);
}
