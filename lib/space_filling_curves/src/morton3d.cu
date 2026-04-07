#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.h"

namespace space_filling_curves {

__device__ __forceinline__ uint64_t split_by_3_u32(uint32_t a) {
  uint64_t x = static_cast<uint64_t>(a) & 0x1fffffu;
  x = (x | (x << 32)) & 0x1f00000000ffffULL;
  x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
  x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
  x = (x | (x << 2)) & 0x1249249249249249ULL;
  return x;
}

template <int XIndex, int YIndex, int ZIndex>
__global__ void morton3d_encode_magicbits_kernel(
  const int32_t* __restrict__ coords,
  int64_t* __restrict__ out,
  int64_t N,
  int64_t stride0,
  int64_t stride1) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const int64_t row_offset = idx * stride0;
  uint32_t x = static_cast<uint32_t>(coords[row_offset + static_cast<int64_t>(XIndex) * stride1]);
  uint32_t y = static_cast<uint32_t>(coords[row_offset + static_cast<int64_t>(YIndex) * stride1]);
  uint32_t z = static_cast<uint32_t>(coords[row_offset + static_cast<int64_t>(ZIndex) * stride1]);

  uint64_t xx = split_by_3_u32(x);
  uint64_t yy = split_by_3_u32(y) << 1;
  uint64_t zz = split_by_3_u32(z) << 2;
  out[idx] = static_cast<int64_t>(xx | yy | zz);
}

at::Tensor morton3d_encode_magicbits(
  const at::Tensor& coords,
  const std::string& axis_order) {
  TORCH_CHECK(coords.is_cuda());
  TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 3);
  TORCH_CHECK(coords.scalar_type() == at::kInt);

  const int64_t N = coords.size(0);
  auto out = at::empty({N}, coords.options().dtype(at::kLong));
  if (N == 0) return out;

  constexpr int threads = 256;
  const int blocks = static_cast<int>((N + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(coords.device().index());
  const int32_t* coords_ptr = coords.data_ptr<int32_t>();
  int64_t* out_ptr = out.data_ptr<int64_t>();
  const int64_t stride0 = coords.stride(0);
  const int64_t stride1 = coords.stride(1);

  dispatch_axis_order_3d(axis_order, [&](auto indices) {
    using AxisOrder = decltype(indices);
    morton3d_encode_magicbits_kernel<
      AxisOrder::XIndex,
      AxisOrder::YIndex,
      AxisOrder::ZIndex><<<blocks, threads, 0, stream>>>(
        coords_ptr,
        out_ptr,
        N,
        stride0,
        stride1);
  });

  return out;
}

}  // namespace space_filling_curves
