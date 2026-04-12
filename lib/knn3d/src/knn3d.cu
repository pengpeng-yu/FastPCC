/*
 * Adapted from PyTorch3D KNN for a minimal CUDA path:
 * float32 + L2 + D=3 + single point set pair.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <iostream>
#include <tuple>

#include "utils/dispatch.cuh"
#include "utils/mink.cuh"

namespace knn3d {

namespace {

constexpr int64_t D_FIXED = 3;
constexpr int TILE_P2 = 512;
constexpr int V2_MAX_K = 32;

__global__ void KNearestNeighborKernelV0(
    const float* __restrict__ points1,
    const float* __restrict__ points2,
    float* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const int64_t P1,
    const int64_t P2,
    const int64_t K) {
  __shared__ float points2_tile[TILE_P2 * D_FIXED];
  const int64_t chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  for (int64_t chunk = blockIdx.x; chunk < chunks_per_cloud; chunk += gridDim.x) {
    const int64_t start_point = blockDim.x * chunk;
    const int64_t p1 = start_point + threadIdx.x;
    const bool valid_p1 = p1 < P1;
    float cur_point[D_FIXED];
    if (valid_p1) {
      for (int d = 0; d < D_FIXED; ++d) {
        cur_point[d] = points1[p1 * D_FIXED + d];
      }
    }
    const int64_t safe_p1 = valid_p1 ? p1 : 0;
    const int64_t offset = safe_p1 * K;
    MinK<float, int64_t> mink(dists + offset, idxs + offset, K);
    for (int64_t tile_start = 0; tile_start < P2; tile_start += TILE_P2) {
      const int tile_size = min(static_cast<int>(P2 - tile_start), TILE_P2);
      for (int load_idx = threadIdx.x; load_idx < tile_size; load_idx += blockDim.x) {
        const int64_t src_offset = (tile_start + load_idx) * D_FIXED;
        const int dst_offset = load_idx * D_FIXED;
        points2_tile[dst_offset + 0] = points2[src_offset + 0];
        points2_tile[dst_offset + 1] = points2[src_offset + 1];
        points2_tile[dst_offset + 2] = points2[src_offset + 2];
      }
      __syncthreads();

      if (valid_p1) {
        for (int local_p2 = 0; local_p2 < tile_size; ++local_p2) {
          const int tile_offset = local_p2 * D_FIXED;
          const float dx = cur_point[0] - points2_tile[tile_offset + 0];
          const float dy = cur_point[1] - points2_tile[tile_offset + 1];
          const float dz = cur_point[2] - points2_tile[tile_offset + 2];
          const float dist = dx * dx + dy * dy + dz * dz;
          mink.add(dist, tile_start + local_p2);
        }
      }
      __syncthreads();
    }
  }
}

template <int64_t K>
__global__ void KNearestNeighborKernelV2(
    const float* __restrict__ points1,
    const float* __restrict__ points2,
    float* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const int64_t P1,
    const int64_t P2) {
  __shared__ float points2_tile[TILE_P2 * D_FIXED];
  const int64_t chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  for (int64_t chunk = blockIdx.x; chunk < chunks_per_cloud; chunk += gridDim.x) {
    const int64_t start_point = blockDim.x * chunk;
    const int64_t p1 = start_point + threadIdx.x;
    const bool valid_p1 = p1 < P1;
    float cur_point[D_FIXED];
    float min_dists[K];
    int64_t min_idxs[K];
    if (valid_p1) {
      for (int d = 0; d < D_FIXED; ++d) {
        cur_point[d] = points1[p1 * D_FIXED + d];
      }
    }
    const int64_t length2 = P2;
    MinK<float, int64_t> mink(min_dists, min_idxs, K);
    for (int64_t tile_start = 0; tile_start < length2; tile_start += TILE_P2) {
      const int tile_size = min(static_cast<int>(length2 - tile_start), TILE_P2);
      for (int load_idx = threadIdx.x; load_idx < tile_size; load_idx += blockDim.x) {
        const int64_t src_offset = (tile_start + load_idx) * D_FIXED;
        const int dst_offset = load_idx * D_FIXED;
        points2_tile[dst_offset + 0] = points2[src_offset + 0];
        points2_tile[dst_offset + 1] = points2[src_offset + 1];
        points2_tile[dst_offset + 2] = points2[src_offset + 2];
      }
      __syncthreads();

      if (valid_p1) {
        for (int local_p2 = 0; local_p2 < tile_size; ++local_p2) {
          const int tile_offset = local_p2 * D_FIXED;
          const float dx = cur_point[0] - points2_tile[tile_offset + 0];
          const float dy = cur_point[1] - points2_tile[tile_offset + 1];
          const float dz = cur_point[2] - points2_tile[tile_offset + 2];
          const float dist = dx * dx + dy * dy + dz * dz;
          mink.add(dist, tile_start + local_p2);
        }
      }
      __syncthreads();
    }

    if (valid_p1) {
      const int64_t offset = p1 * K;
      for (int k = 0; k < mink.size(); ++k) {
        dists[offset + k] = min_dists[k];
        idxs[offset + k] = min_idxs[k];
      }
    }
  }
}

template <int64_t K>
struct KNearestNeighborKernelV2Functor {
  static void run(
      size_t blocks,
      size_t threads,
      const float* __restrict__ points1,
      const float* __restrict__ points2,
      float* __restrict__ dists,
      int64_t* __restrict__ idxs,
      const int64_t P1,
      const int64_t P2) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    KNearestNeighborKernelV2<K><<<blocks, threads, 0, stream>>>(
        points1, points2, dists, idxs, P1, P2);
  }
};

bool KnnCheckVersion(int version, const int64_t K) {
  if (version == 0) {
    return true;
  }
  if (version == 2) {
    return K <= V2_MAX_K;
  }
  return false;
}

int ChooseVersion(const int64_t K) {
  return KnnCheckVersion(2, K) ? 2 : 0;
}

} // namespace

std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdx(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const int K,
    int version) {
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2};
  at::CheckedFrom c = "KNearestNeighborIdx";
  at::checkAllSameGPU(c, {p1_t, p2_t});
  at::checkAllSameType(c, {p1_t, p2_t});

  TORCH_CHECK(p1.is_cuda(), "p1 must be a CUDA tensor");
  TORCH_CHECK(p2.is_cuda(), "p2 must be a CUDA tensor");
  TORCH_CHECK(p1.is_contiguous(), "p1 must be contiguous");
  TORCH_CHECK(p2.is_contiguous(), "p2 must be contiguous");
  TORCH_CHECK(p1.scalar_type() == at::kFloat, "p1 must be float32");
  TORCH_CHECK(p2.scalar_type() == at::kFloat, "p2 must be float32");
  TORCH_CHECK(p1.dim() == 2 && p1.size(1) == D_FIXED, "p1 must have shape (N, 3)");
  TORCH_CHECK(p2.dim() == 2 && p2.size(1) == D_FIXED, "p2 must have shape (M, 3)");
  TORCH_CHECK(K > 0, "K must be positive");
  TORCH_CHECK(p2.size(0) >= K, "p2.size(0) must be >= K");

  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto P1 = p1.size(0);
  const auto P2 = p2.size(0);
  auto idxs = at::empty({P1, K}, p1.options().dtype(at::kLong));
  auto dists = at::empty({P1, K}, p1.options());

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(idxs, dists);
  }

  if (version < 0) {
    version = ChooseVersion(K);
  } else if (!KnnCheckVersion(version, K)) {
    const int new_version = ChooseVersion(K);
    std::cout << "WARNING: Requested KNN version " << version
              << " is not compatible with K = " << K
              << ". Falling back to version = " << new_version << std::endl;
    version = new_version;
  }
  AT_ASSERTM(KnnCheckVersion(version, K), "Invalid version");

  const size_t threads = 256;
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t chunks = (P1 + threads - 1) / threads;
  const int64_t blocks = std::min<int64_t>(chunks, static_cast<int64_t>(sm_count) * 4);

  if (version == 0) {
    KNearestNeighborKernelV0<<<blocks, threads, 0, stream>>>(
        p1.data_ptr<float>(),
        p2.data_ptr<float>(),
        dists.data_ptr<float>(),
        idxs.data_ptr<int64_t>(),
        P1,
        P2,
        K);
  } else if (version == 2) {
    DispatchKernel1D<
        KNearestNeighborKernelV2Functor,
        1,
        V2_MAX_K>(
        K,
        blocks,
        threads,
        p1.data_ptr<float>(),
        p2.data_ptr<float>(),
        dists.data_ptr<float>(),
        idxs.data_ptr<int64_t>(),
        P1,
        P2);
  } else {
    KNearestNeighborKernelV0<<<blocks, threads, 0, stream>>>(
        p1.data_ptr<float>(),
        p2.data_ptr<float>(),
        dists.data_ptr<float>(),
        idxs.data_ptr<int64_t>(),
        P1,
        P2,
        K);
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(idxs, dists);
}

} // namespace knn3d
