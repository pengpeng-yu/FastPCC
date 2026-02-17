#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace int_sparse_conv {

__inline__ __device__ int32_t scalar_prelu(
  const int32_t input,
  const int32_t slope  // Q6.25
) {
  int64_t val = (int64_t)input;

  if (val < 0) {
    val = val * slope;
    int64_t half = int64_t(1) << 24;
    if (val >= 0) val = (val + half) >> 25;
    else val = -((-val + half) >> 25);
  }

  val = ::min(::max(val, int64_t(INT32_MIN)), int64_t(INT32_MAX));
  return int32_t(val);
}

__global__ void prelu_kernel(
  const int32_t* __restrict__ input,
  const int32_t* __restrict__ slope,
  int32_t* __restrict__ out,
  uint64_t N, uint64_t Ch)
{
  const int32_t slope_val = slope[0];
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
  uint64_t total = N * Ch;

  for (; idx < total; idx += stride) {
    out[idx] = scalar_prelu(input[idx], slope_val);
  }
}

at::Tensor prelu(
  const at::Tensor &input,
  const at::Tensor &slope)
{
  at::Device device = input.device();
  TORCH_CHECK(device.is_cuda());
  TORCH_CHECK(slope.device() == device);
  TORCH_CHECK(input.is_contiguous() && slope.is_contiguous());
  TORCH_CHECK(input.dim() == 2);

  int64_t N = input.size(0);
  int64_t Ch = input.size(1);
  TORCH_CHECK(N > 0 && Ch > 0);
  TORCH_CHECK(1 == slope.numel());

  at::Tensor out = at::empty_like(input);

  int64_t threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  int64_t total = N * Ch;
  int64_t blocks = std::min(int64_t(4096), (total + threads - 1) / threads);
  prelu_kernel<<<blocks, threads, 0, stream>>>(
    input.data_ptr<int32_t>(),
    slope.data_ptr<int32_t>(),
    out.data_ptr<int32_t>(),
    N, Ch);
  return out;
}

}
