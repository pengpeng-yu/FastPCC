#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.h"

namespace space_filling_curves {

namespace {

__device__ __constant__ uint8_t kMortonToHilbertTable[96] = {
  48, 33, 27, 34, 47, 78, 28, 77,
  66, 29, 51, 52, 65, 30, 72, 63,
  76, 95, 75, 24, 53, 54, 82, 81,
  18,  3, 17, 80, 61,  4, 62, 15,
   0, 59, 71, 60, 49, 50, 86, 85,
  84, 83,  5, 90, 79, 56,  6, 89,
  32, 23,  1, 94, 11, 12,  2, 93,
  42, 41, 13, 14, 35, 88, 36, 31,
  92, 37, 87, 38, 91, 74,  8, 73,
  46, 45,  9, 10,  7, 20, 64, 19,
  70, 25, 39, 16, 69, 26, 44, 43,
  22, 55, 21, 68, 57, 40, 58, 67,
};

}

template <int XIndex, int YIndex, int ZIndex>
__global__ void hilbert3d_encode_lut_kernel(
  const int32_t* __restrict__ coords,
  int64_t* __restrict__ out,
  int64_t N,
  int32_t bits,
  int64_t stride0,
  int64_t stride1)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const int64_t row_offset = idx * stride0;
  uint32_t x = static_cast<uint32_t>(coords[row_offset + static_cast<int64_t>(XIndex) * stride1]);
  uint32_t y = static_cast<uint32_t>(coords[row_offset + static_cast<int64_t>(YIndex) * stride1]);
  uint32_t z = static_cast<uint32_t>(coords[row_offset + static_cast<int64_t>(ZIndex) * stride1]);

  uint32_t transform = 0;
  uint64_t key = 0;

  #pragma unroll
  for (int32_t b = bits - 1; b >= 0; --b) {
    uint32_t input =
      ((x >> b) & 1u) |
      (((y >> b) & 1u) << 1) |
      (((z >> b) & 1u) << 2);

    uint32_t t = kMortonToHilbertTable[transform | input];
    key = (key << 3) | static_cast<uint64_t>(t & 7u);
    transform = t & ~7u;
  }

  out[idx] = static_cast<int64_t>(key);
}

at::Tensor hilbert3d_encode_lut(
  const at::Tensor& coords,
  int64_t bits,
  const std::string& axis_order) {
  TORCH_CHECK(coords.is_cuda());
  TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 3);
  TORCH_CHECK(coords.scalar_type() == at::kInt);
  TORCH_CHECK(bits >= 1 && bits <= 21);

  const int64_t N = coords.size(0);
  auto out = at::empty({N}, coords.options().dtype(at::kLong));
  if (N == 0) return out;

  constexpr int threads = 256;
  const int blocks = static_cast<int>((N + threads - 1) / threads);
  const int32_t bits_i32 = static_cast<int32_t>(bits);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(coords.device().index());
  const int32_t* coords_ptr = coords.data_ptr<int32_t>();
  int64_t* out_ptr = out.data_ptr<int64_t>();
  const int64_t stride0 = coords.stride(0);
  const int64_t stride1 = coords.stride(1);

  dispatch_axis_order_3d(axis_order, [&](auto indices) {
    using AxisOrder = decltype(indices);
    hilbert3d_encode_lut_kernel<
      AxisOrder::XIndex,
      AxisOrder::YIndex,
      AxisOrder::ZIndex><<<blocks, threads, 0, stream>>>(
        coords_ptr,
        out_ptr,
        N,
        bits_i32,
        stride0,
        stride1);
  });

  return out;
}

}  // namespace space_filling_curves
