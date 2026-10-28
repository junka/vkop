/* Copyright (c) 2026 junka. All Rights Reserved. */
// The only test TU that includes the real <torch/torch.h>. Every name declared
// in torch_subset.hpp is defined here by forwarding to it, so the other test
// files get the torch surface without the parse cost.
//
// Deliberately NOT included before that: torch_subset.hpp (through setup.hpp).
// The relay declarations in it are exact matches for what torch.h declares, and
// reaching them first would make torch.h's own declarations ambiguous.

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torch {

// ---------------------------------------------------------------------------
// Construction: nothing to declare. torch.h's variable_factories.h, which
// torch_subset.hpp reaches through torch/types.h, already provides every
// factory the suite uses.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Comparisons and reductions
// ---------------------------------------------------------------------------

bool allclose(const Tensor& a, const Tensor& b, double rtol, double atol,
              bool equal_nan) {
    return at::allclose(a, b, rtol, atol, equal_nan);
}
Tensor isclose(const Tensor& a, const Tensor& b, double rtol, double atol,
               bool equal_nan) {
    return at::isclose(a, b, rtol, atol, equal_nan);
}
Tensor amax(const Tensor& t, at::IntArrayRef dim, bool keepdim) {
    return at::amax(t, dim, keepdim);
}
Tensor amin(const Tensor& t, at::IntArrayRef dim, bool keepdim) {
    return at::amin(t, dim, keepdim);
}
Tensor argmax(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    return at::argmax(t, dim, keepdim);
}
Tensor logsumexp(const Tensor& t, at::IntArrayRef dim, bool keepdim) {
    return at::logsumexp(t, dim, keepdim);
}
Tensor sum(const Tensor& t, at::IntArrayRef dim, bool keepdim) {
    return at::sum(t, dim, keepdim);
}
Tensor prod(const Tensor& t, int64_t dim, bool keepdim) {
    return at::prod(t, dim, keepdim);
}
Tensor mean(const Tensor& t) { return at::mean(t); }
Tensor mean(const Tensor& t, at::IntArrayRef dim, bool keepdim) {
    return at::mean(t, dim, keepdim);
}
std::tuple<Tensor, Tensor> sort(const Tensor& t, int64_t dim, bool descending) {
    return at::sort(t, dim, descending);
}
std::tuple<Tensor, Tensor> topk(const Tensor& t, int64_t k, int64_t dim,
                                bool largest, bool sorted) {
    return at::topk(t, k, dim, largest, sorted);
}
Tensor stack(at::TensorList ts, int64_t dim) {
    return at::stack(ts, dim);
}
Tensor cat(at::TensorList ts, int64_t dim) {
    return at::cat(ts, dim);
}
std::vector<Tensor> split_with_sizes(const Tensor& t, at::IntArrayRef sizes,
                                     int64_t dim) {
    return at::split_with_sizes(t, sizes, dim);
}
Tensor matmul(const Tensor& a, const Tensor& b) { return at::matmul(a, b); }
Tensor where(const Tensor& cond, const Tensor& a, const Tensor& b) {
    return at::where(cond, a, b);
}
Tensor round(const Tensor& t, int64_t decimals) {
    return at::round(t, decimals);
}
Tensor nan_to_num(const Tensor& t) { return at::nan_to_num(t); }

// ---------------------------------------------------------------------------
// Elementwise, unary and shape ops
// ---------------------------------------------------------------------------

Tensor abs(const Tensor& t) { return at::abs(t); }
Tensor atan(const Tensor& t) { return at::atan(t); }
Tensor clone(const Tensor& t) { return at::clone(t); }
Tensor erf(const Tensor& t) { return at::erf(t); }
Tensor floor(const Tensor& t) { return at::floor(t); }
Tensor log(const Tensor& t) { return at::log(t); }
Tensor relu(const Tensor& t) { return at::relu(t); }
Tensor rsqrt(const Tensor& t) { return at::rsqrt(t); }
Tensor sigmoid(const Tensor& t) { return at::sigmoid(t); }
Tensor sqrt(const Tensor& t) { return at::sqrt(t); }
Tensor square(const Tensor& t) { return at::square(t); }
Tensor tanh(const Tensor& t) { return at::tanh(t); }
Tensor softplus(const Tensor& t, double beta, double threshold) {
    return at::softplus(t, beta, threshold);
}
Tensor pow(const Tensor& t, const at::Scalar& exponent) {
    return at::pow(t, exponent);
}
Tensor pow(const Tensor& base, const Tensor& exponent) {
    return at::pow(base, exponent);
}
Tensor pow(double base, const Tensor& exponent) {
    return at::pow(at::Scalar(base), exponent);
}
Tensor clamp(const Tensor& t, std::optional<at::Scalar> min,
             std::optional<at::Scalar> max) {
    return at::clamp(t, min, max);
}
Tensor softmax(const Tensor& t, int64_t dim,
               std::optional<at::ScalarType> dtype) {
    return at::softmax(t, dim, dtype);
}
Tensor log_softmax(const Tensor& t, int64_t dim,
                   std::optional<at::ScalarType> dtype) {
    return at::log_softmax(t, dim, dtype);
}
Tensor reshape(const Tensor& t, at::IntArrayRef shape) {
    return at::reshape(t, shape);
}
Tensor permute(const Tensor& t, at::IntArrayRef dims) {
    return at::permute(t, dims);
}
Tensor transpose(const Tensor& t, int64_t dim0, int64_t dim1) {
    return at::transpose(t, dim0, dim1);
}
Tensor squeeze(const Tensor& t) { return at::squeeze(t); }
Tensor unsqueeze(const Tensor& t, int64_t dim) { return at::unsqueeze(t, dim); }
Tensor expand(const Tensor& t, at::IntArrayRef size) {
    return t.expand(size);
}

// Tensor's own indexing operator: no at:: free function does this, so forward
// to the member. TensorIndex comes from ATen/TensorIndexing.h.
Tensor tensor_index(const Tensor& t, at::ArrayRef<at::indexing::TensorIndex> indices) {
    return t.index(indices);
}

// ---------------------------------------------------------------------------
// Op-head references, in torch::tests (see torch_subset.hpp)
// ---------------------------------------------------------------------------

namespace tests {

// Forwarded to at::, not torch::nn::functional: this definition sits inside
// namespace torch, where an unqualified torch::silu would find itself.
Tensor silu(const Tensor& t) { return at::silu(t); }
Tensor prelu(const Tensor& input, const Tensor& weight) {
    return at::prelu(input, weight);
}
Tensor layer_norm(const Tensor& input, at::IntArrayRef normalized_shape,
                  const Tensor& weight, const Tensor& bias, double eps) {
    return at::layer_norm(input, normalized_shape, weight, bias, eps);
}
Tensor layer_norm(const Tensor& input, at::IntArrayRef normalized_shape,
                  const Tensor& weight, const Tensor& bias, double eps,
                  bool cudnn_enable) {
    return at::layer_norm(input, normalized_shape, weight, bias, eps,
                             cudnn_enable);
}
// ATen's batch_norm takes optional<> biases and a trailing cudnn_enabled the
// test-facing signature hides.
Tensor batch_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  const Tensor& running_mean, const Tensor& running_var,
                  bool training, double momentum, double eps,
                  bool cudnn_enable) {
    return at::batch_norm(input, weight, bias, running_mean, running_var,
                          training, momentum, eps, cudnn_enable);
}
// The tests drive torch's nn::functional::interpolate through a mode constant
// (ResizeTest's kBilinear/kBicubic/kTrilinear/kLinear/kNearest). Reproduce
// the mapping here, minus the InterpolateFuncOptions builder.
Tensor interpolate(const Tensor& input, at::IntArrayRef output_size,
                   int64_t mode, bool align_corners) {
    const int64_t rank = input.dim();
    const int64_t spatial = static_cast<int64_t>(output_size.size());
    if (mode == 0) {  // kBilinear
        return at::upsample_bilinear2d(input, output_size, align_corners);
    }
    if (mode == 1) {  // kBicubic
        return at::upsample_bicubic2d(input, output_size, align_corners);
    }
    if (mode == 2) {  // kBilinear, trilinear when the input is 5-D
        if (rank == 5) {
            return at::upsample_trilinear3d(input, output_size, align_corners);
        }
        return at::upsample_bilinear2d(input, output_size, align_corners);
    }
    if (mode == 3) {  // kNearest
        if (spatial == 3) {
            return at::upsample_nearest3d(input, output_size,
                                          std::nullopt, std::nullopt,
                                          std::nullopt);
        }
        if (rank == 3) {
            return at::upsample_nearest1d(input, output_size, std::nullopt);
        }
        return at::upsample_nearest2d(input, output_size, std::nullopt,
                                      std::nullopt);
    }
    return at::upsample_linear1d(input, output_size, align_corners,
                                 std::nullopt);
}

}  // namespace tests

}  // namespace torch
