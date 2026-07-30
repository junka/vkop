// Copyright 2025 @junka
#ifndef OPS_SLICE_CALC_HPP_
#define OPS_SLICE_CALC_HPP_

#include <cstdio>
#include <stdexcept>
#include <vector>

// ONNX slice output-shape + starts/ends/steps computation, shared by the
// image Slice op (SliceImage) and the buffer BufferSlice op. Extracted to a
// free header so neither depends on the other's class definition.

namespace vkop {
namespace ops {
namespace slice_calc {

template <typename T>
std::vector<std::vector<int>>
calculate_output_shape(const std::vector<int> &input_shape,
                       const std::vector<T> &starts, const std::vector<T> &ends,
                       const std::vector<T> &axes,
                       const std::vector<T> &steps) {
    assert(input_shape.size() >= 3);
    const int dims = static_cast<int>(input_shape.size());
    std::vector<std::vector<int>> ret;

    std::vector<T> norm_axes = axes;
    if (norm_axes.empty()) {
        for (int i = 0; i < dims; ++i) {
            norm_axes.push_back(static_cast<T>(i));
        }
    }

    for (auto &ax : norm_axes) {
        if (ax < 0)
            ax += dims;
        if (ax < 0 || ax >= dims) {
            throw std::out_of_range("axis out of range");
        }
    }
    if (starts.size() != norm_axes.size() || ends.size() != norm_axes.size()) {
        printf("%zd %zd %zd\n", norm_axes.size(), starts.size(), ends.size());
        throw std::invalid_argument("starts/ends length must match axes");
    }

    std::vector<int> full_starts(dims);
    std::vector<int> full_ends(dims);
    std::vector<int> full_steps(dims, 1);

    for (int i = 0; i < dims; ++i) {
        full_starts[i] = 0;
        full_ends[i] = input_shape[i];
    }

    for (size_t i = 0; i < norm_axes.size(); ++i) {
        auto axis = norm_axes[i];
        auto dim_size = input_shape[axis];

        int start = static_cast<int>(starts[i]);
        int end = static_cast<int>(ends[i]);
        int step = (steps.size() > i) ? static_cast<int>(steps[i]) : 1;

        if (step == 0)
            step = 1;

        if (start < 0)
            start += dim_size;
        if (end < 0)
            end += dim_size;

        start = std::max(0, std::min(start, dim_size));
        end = std::max(0, std::min(end, dim_size));

        full_starts[axis] = start;
        full_ends[axis] = end;
        full_steps[axis] = step;
    }

    std::vector<int> output_shape(dims);
    for (int i = 0; i < dims; ++i) {
        auto start = full_starts[i];
        auto end = full_ends[i];
        auto step = full_steps[i];

        if (step > 0) {
            if (start >= end) {
                output_shape[i] = 0;
            } else {
                output_shape[i] = (end - start + step - 1) / step;
            }
        } else {
            if (start <= end) {
                output_shape[i] = 0;
            } else {
                output_shape[i] = (start - end - step - 1) / (-step);
            }
        }
        output_shape[i] = std::max(0, output_shape[i]);
    }
    ret.emplace_back(output_shape);
    ret.emplace_back(full_starts);
    ret.emplace_back(full_ends);
    ret.emplace_back(full_steps);

    return ret;
}

} // namespace slice_calc
} // namespace ops
} // namespace vkop

#endif // OPS_SLICE_CALC_HPP_
