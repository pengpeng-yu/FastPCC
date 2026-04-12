#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace int_sparse_conv {

template <typename T>
__inline__ __device__ T scalar_requant(
  const int32_t input,
  const uint32_t requant_mul,
  const int64_t zero_point,
  const int right_shift
) {
  int64_t prod = (int64_t)input * (int64_t)requant_mul + zero_point;

  int64_t out64 = prod;
  if (right_shift > 0) {
    int64_t half = int64_t(1) << (right_shift - 1);
    if (prod >= 0) out64 = (prod + half) >> right_shift;
    else out64 = -((-prod + half) >> right_shift);
  }

  constexpr int64_t low = (int64_t)(std::numeric_limits<T>::min());
  constexpr int64_t high = (int64_t)(std::numeric_limits<T>::max());
  T out = (T)(::min(::max(out64, low), high));
  return out;
}

template <typename T>
__global__ void requant_kernel(
  const int32_t* __restrict__ input,
  const uint32_t* __restrict__ requant_mul,
  const int64_t* zero_point,
  const int right_shift,
  T* __restrict__ out,
  uint64_t N, uint64_t Ch)
{
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
  uint64_t total = N * Ch;

  for (; idx < total; idx += stride) {
    uint64_t ch = idx % Ch;
    out[idx] = scalar_requant<T>(input[idx], requant_mul[ch], zero_point[0], right_shift);
  }
}

template <typename T>
at::Tensor requant(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int right_shift,
  at::ScalarType out_dtype)
{
  at::Device device = input.device();
  TORCH_CHECK(device.is_cuda());
  TORCH_CHECK(requant_mul.device() == device && zero_point.device() == device);
  TORCH_CHECK(input.is_contiguous() && requant_mul.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && requant_mul.dim() == 1);

  int64_t N = input.size(0);
  int64_t Ch = input.size(1);
  TORCH_CHECK(N > 0 && Ch > 0 && right_shift >= 0);
  TORCH_CHECK(Ch == requant_mul.size(0) && 1 == zero_point.numel());

  auto options = at::TensorOptions().dtype(out_dtype).device(device);
  at::Tensor out = at::empty({N, Ch}, options);

  int64_t threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  int64_t total = N * Ch;
  const auto* prop = at::cuda::getDeviceProperties(device.index());
  int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, prop->maxGridSize[0]);
  requant_kernel<T><<<blocks, threads, 0, stream>>>(
    input.data_ptr<int32_t>(),
    requant_mul.data_ptr<uint32_t>(),
    zero_point.data_ptr<int64_t>(),
    right_shift,
    out.data_ptr<T>(),
    N, Ch);
  return out;
}

at::Tensor requant_to_int8(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int right_shift)
{
  return requant<int8_t>(input, requant_mul, zero_point, right_shift, at::kChar);
}

at::Tensor requant_to_int16(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int right_shift)
{
  return requant<int16_t>(input, requant_mul, zero_point, right_shift, at::kShort);
}

at::Tensor requant_to_int32(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int right_shift)
{
  return requant<int32_t>(input, requant_mul, zero_point, right_shift, at::kInt);
}

}
