/* Copyright (c) 2026 junka. All Rights Reserved. */
#ifndef VKOP_TESTS_PYTORCH_SUBSET_HPP_
#define VKOP_TESTS_PYTORCH_SUBSET_HPP_

// The torch surface the test suite uses, available without <torch/torch.h>.
//
// Every TU that parses torch.h pays ~5.7s and ~1.0 GB of RSS before it reads a
// line of the test, and every test file was paying it through setup.hpp. The
// suite only uses torch for CPU reference data: build a tensor, slice/reshape
// it, run one op, compare. That is a closed set of ~80 names, so this header
// declares exactly that set -- same signatures as upstream -- and
// torch_subset.cpp (the only test TU that includes the real header) defines
// them by forwarding to it.
//
// Disjoint overloads rather than inline definitions: if a test also includes
// <torch/torch.h> it sees both the real declarations and these, and the real
// one wins for every call whose arguments convert to it. Only the subset's own
// argument sets (notably IntArrayRef, where torch templates on initializer
// lists) land on these, and those are the ones torch_subset.cpp defines. This
// is what lets tests be converted one file at a time and still link.
//
// Adding a torch call a test needs: declare it here and forward it in
// torch_subset.cpp, keeping the upstream signature.

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

// The real types only: ATen's lean headers, which pull in no autograd and no
// nn. Together with the vkop headers they cost ~3.2s / 0.6 GB against ~9.5s /
// 1.6 GB for <torch/torch.h>. Two of them are load-bearing:
//   - TensorIndexing.h supplies the indexing machinery. It has no include guard
//     and defines the `t[{...}]` operators, so it must come BEFORE TensorBody.h
//     or those operators never see their argument type.
//   - torch/types.h transitively includes autograd's variable_factories.h,
//     which already declares every torch:: factory below with the upstream
//     defaults. Never re-declare one of those here.
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <torch/types.h>

namespace torch {

// `t[{i, Slice(...)}]` / `t.index({...})` parse against the real index type.
namespace indexing {

using SmallVectorSizeType = c10::SmallVectorSizeType<at::Tensor>;
using Slice = at::indexing::Slice;
using TensorIndex = at::indexing::TensorIndex;

}  // namespace indexing


// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
// randn/rand/zeros/ones/ones_like/full/arange/empty/from_blob/tensor and the
// manual_seed overloads are NOT declared here: torch/types.h's
// variable_factories.h already declares them with the upstream signatures and
// defaults, so a second declaration would collide on the default arguments.
// Artists who use them need no relay at all.

// ---------------------------------------------------------------------------
// Comparisons and reductions
// ---------------------------------------------------------------------------

bool allclose(const Tensor& a, const Tensor& b, double rtol = 1e-05,
              double atol = 1e-08, bool equal_nan = false);
Tensor isclose(const Tensor& a, const Tensor& b, double rtol = 1e-05,
               double atol = 1e-08, bool equal_nan = false);
Tensor amax(const Tensor& t, at::IntArrayRef dim, bool keepdim = false);
Tensor amin(const Tensor& t, at::IntArrayRef dim, bool keepdim = false);
Tensor argmax(const Tensor& t, std::optional<int64_t> dim = {},
              bool keepdim = false);
Tensor logsumexp(const Tensor& t, at::IntArrayRef dim, bool keepdim = false);
Tensor sum(const Tensor& t, at::IntArrayRef dim, bool keepdim = false);
Tensor prod(const Tensor& t, int64_t dim, bool keepdim = false);
Tensor mean(const Tensor& t);
Tensor mean(const Tensor& t, at::IntArrayRef dim, bool keepdim = false);
std::tuple<Tensor, Tensor> sort(const Tensor& t, int64_t dim,
                                bool descending = false);
std::tuple<Tensor, Tensor> topk(const Tensor& t, int64_t k, int64_t dim,
                                bool largest = true, bool sorted = true);
Tensor stack(at::TensorList ts, int64_t dim = 0);
Tensor cat(at::TensorList ts, int64_t dim = 0);
std::vector<Tensor> split_with_sizes(const Tensor& t, at::IntArrayRef sizes,
                                     int64_t dim = 0);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor where(const Tensor& cond, const Tensor& a, const Tensor& b);
Tensor round(const Tensor& t, int64_t decimals = 0);
Tensor nan_to_num(const Tensor& t);

// ---------------------------------------------------------------------------
// Elementwise, unary and shape ops (reference math)
// ---------------------------------------------------------------------------

Tensor abs(const Tensor& t);
Tensor atan(const Tensor& t);
Tensor clone(const Tensor& t);
Tensor erf(const Tensor& t);
Tensor floor(const Tensor& t);
Tensor log(const Tensor& t);
Tensor relu(const Tensor& t);
Tensor rsqrt(const Tensor& t);
Tensor sigmoid(const Tensor& t);
Tensor sqrt(const Tensor& t);
Tensor square(const Tensor& t);
Tensor tanh(const Tensor& t);
Tensor softplus(const Tensor& t, double beta = 1.0, double threshold = 20.0);
Tensor pow(const Tensor& t, const at::Scalar& exponent);
Tensor pow(const Tensor& base, const Tensor& exponent);
Tensor pow(double base, const Tensor& exponent);
Tensor clamp(const Tensor& t, std::optional<at::Scalar> min,
             std::optional<at::Scalar> max);
Tensor softmax(const Tensor& t, int64_t dim,
               std::optional<at::ScalarType> dtype = {});
Tensor log_softmax(const Tensor& t, int64_t dim,
                   std::optional<at::ScalarType> dtype = {});
Tensor reshape(const Tensor& t, at::IntArrayRef shape);
Tensor permute(const Tensor& t, at::IntArrayRef dims);
Tensor transpose(const Tensor& t, int64_t dim0, int64_t dim1);
Tensor squeeze(const Tensor& t);
Tensor unsqueeze(const Tensor& t, int64_t dim);
Tensor expand(const Tensor& t, at::IntArrayRef size);

// ---------------------------------------------------------------------------
// Op-head references
// ---------------------------------------------------------------------------
// These are torch:: names too, and <torch/torch.h> declares them itself in this
// namespace (torch::prelu, torch::max_pool2d, ...), with argument sets that
// overlap ours. A test file that includes one before the other would pick its
// own relay inside torch.h's inline definitions -- infinite recursion or a
// wrong overload inside the header. So the relays live in torch::tests instead:
// tests call torch::tests::silu etc., which no torch header touches.

namespace tests {

Tensor silu(const Tensor& t);
Tensor interpolate(const Tensor& input, at::IntArrayRef output_size,
                   int64_t mode, bool align_corners);
Tensor prelu(const Tensor& input, const Tensor& weight);
Tensor layer_norm(const Tensor& input, at::IntArrayRef normalized_shape,
                  const Tensor& weight, const Tensor& bias, double eps = 1e-05);
Tensor layer_norm(const Tensor& input, at::IntArrayRef normalized_shape,
                  const Tensor& weight, const Tensor& bias, double eps,
                  bool cudnn_enable);
Tensor batch_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  const Tensor& running_mean, const Tensor& running_var,
                  bool training, double momentum, double eps, bool cudnn_enable);
Tensor max_pool2d(const Tensor& input, at::IntArrayRef kernel,
                  at::IntArrayRef stride, at::IntArrayRef padding,
                  int64_t dilation, bool ceil_mode);
Tensor avg_pool2d(const Tensor& input, at::IntArrayRef kernel,
                  at::IntArrayRef stride, at::IntArrayRef padding,
                  bool ceil_mode, bool count_include_pad,
                  std::optional<int64_t> divisor_override = {});

}  // namespace tests

// The one Tensor member the tests use that has no free-function equivalent:
// `t.index({...})` / `t[{...}]`, spelled torch::tensor_index(t, {...}) here.
Tensor tensor_index(const Tensor& t, at::ArrayRef<at::indexing::TensorIndex> indices);

}  // namespace torch

#endif  // VKOP_TESTS_PYTORCH_SUBSET_HPP_
