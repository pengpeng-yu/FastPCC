/*
 * Adapted from PyTorch3D KNN for a minimal CUDA path:
 * float32 + L2 + D=3 + single point set pair.
 */

#pragma once

#include <torch/extension.h>
#include <tuple>

namespace knn3d {

std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdx(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const int K,
    const int version);

} // namespace knn3d
