// Copyright 2025 @junka
#ifndef OPS_REDUCE_TYPES_HPP_
#define OPS_REDUCE_TYPES_HPP_

namespace vkop {
namespace ops {
namespace reduce {

enum class ReduceType {
    L1 = 0,
    L2,
    LOGSUM,
    LOGSUMEXP,
    MAX,
    MEAN,
    MIN,
    PROD,
    SUM,
    SUMSQUARE,
};

struct GpuReduceParam {
    ivec4 shape;
    int reduce_op;
    int axes_mask;
};

} // namespace reduce
} // namespace ops
} // namespace vkop

#endif // OPS_REDUCE_TYPES_HPP_
