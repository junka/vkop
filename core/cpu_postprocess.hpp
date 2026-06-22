// Copyright 2025 @junka
#ifndef CORE_CPU_POSTPROCESS_HPP_
#define CORE_CPU_POSTPROCESS_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace vkop {
namespace core {
namespace cpu {

/**
 * @brief CPU implementation of numerically stable Softmax.
 *
 * Computes softmax along the given axis using the standard
 * max-subtraction trick for numerical stability:
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
 *
 * @param input  Input data in row-major (NCHW) layout
 * @param shape  Dimensions of the tensor (up to 4D)
 * @param axis   Axis along which to compute softmax (supports negative indices)
 * @return       Softmax output, same shape as input
 */
inline std::vector<float> softmax(const std::vector<float> &input,
                                   const std::vector<int> &shape, int axis) {
    if (shape.empty()) {
        return input;
    }

    // Handle negative axis
    if (axis < 0) {
        axis += static_cast<int>(shape.size());
    }

    // Compute total elements and axis size
    int total = 1;
    for (auto d : shape) {
        total *= d;
    }

    int axis_size = shape[axis];

    // Compute outer and inner dimensions
    int outer = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= shape[i];
    }
    int inner = 1;
    for (int i = axis + 1; i < static_cast<int>(shape.size()); ++i) {
        inner *= shape[i];
    }

    std::vector<float> output(total);

    // For each position outside the softmax axis
    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < inner; ++i) {
            // Find max along axis for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int a = 0; a < axis_size; ++a) {
                int idx = (o * axis_size + a) * inner + i;
                if (input[idx] > max_val) {
                    max_val = input[idx];
                }
            }

            // Compute exp(x - max) and sum
            float sum_exp = 0.0f;
            for (int a = 0; a < axis_size; ++a) {
                int idx = (o * axis_size + a) * inner + i;
                output[idx] = std::exp(input[idx] - max_val);
                sum_exp += output[idx];
            }

            // Normalize
            for (int a = 0; a < axis_size; ++a) {
                int idx = (o * axis_size + a) * inner + i;
                output[idx] /= sum_exp;
            }
        }
    }

    return output;
}

/**
 * @brief CPU implementation of TopK.
 *
 * Returns the k largest (or smallest) values and their indices
 * along the last dimension of the input.
 *
 * @param input  Input data in row-major layout
 * @param shape  Dimensions of the tensor (up to 4D)
 * @param k      Number of top elements to select
 * @param largest If true, return largest; if false, return smallest
 * @param sorted  If true, results are sorted in descending (largest) or
 *                ascending (smallest) order
 * @return        Pair of (values, indices), each of shape with last dim = k
 */
inline std::pair<std::vector<float>, std::vector<int>>
topk(const std::vector<float> &input, const std::vector<int> &shape, int k,
     bool largest = true, bool sorted = true) {
    if (shape.empty()) {
        return {{}, {}};
    }

    int total = 1;
    for (auto d : shape) {
        total *= d;
    }

    int last_dim = shape.back();
    int rows = total / last_dim;

    std::vector<float> values(rows * k);
    std::vector<int> indices(rows * k);

    for (int r = 0; r < rows; ++r) {
        // Build index array for this row
        std::vector<int> row_indices(last_dim);
        std::iota(row_indices.begin(), row_indices.end(), 0);

        // Partial sort: select top-k
        if (largest) {
            std::partial_sort(
                row_indices.begin(), row_indices.begin() + k,
                row_indices.end(),
                [&input, r, last_dim](int a, int b) {
                    return input[r * last_dim + a] > input[r * last_dim + b];
                });
        } else {
            std::partial_sort(
                row_indices.begin(), row_indices.begin() + k,
                row_indices.end(),
                [&input, r, last_dim](int a, int b) {
                    return input[r * last_dim + a] < input[r * last_dim + b];
                });
        }

        // Extract top-k values and indices
        for (int i = 0; i < k; ++i) {
            indices[r * k + i] = row_indices[i];
            values[r * k + i] = input[r * last_dim + row_indices[i]];
        }

        // Sort within top-k if requested
        if (sorted) {
            std::vector<int> sort_idx(k);
            std::iota(sort_idx.begin(), sort_idx.end(), 0);
            if (largest) {
                std::sort(sort_idx.begin(), sort_idx.end(),
                          [&values, r, k](int a, int b) {
                              return values[r * k + a] > values[r * k + b];
                          });
            } else {
                std::sort(sort_idx.begin(), sort_idx.end(),
                          [&values, r, k](int a, int b) {
                              return values[r * k + a] < values[r * k + b];
                          });
            }
            // Apply permutation
            std::vector<float> tmp_val(k);
            std::vector<int> tmp_idx(k);
            for (int i = 0; i < k; ++i) {
                tmp_val[i] = values[r * k + sort_idx[i]];
                tmp_idx[i] = indices[r * k + sort_idx[i]];
            }
            for (int i = 0; i < k; ++i) {
                values[r * k + i] = tmp_val[i];
                indices[r * k + i] = tmp_idx[i];
            }
        }
    }

    return {values, indices};
}

} // namespace cpu
} // namespace core
} // namespace vkop

#endif // CORE_CPU_POSTPROCESS_HPP_
