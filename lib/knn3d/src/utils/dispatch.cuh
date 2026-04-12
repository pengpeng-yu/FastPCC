/*
 * Adapted from PyTorch3D 1D dispatch helper.
 */

#pragma once

#include <torch/extension.h>

template <template <int64_t> class Kernel, int64_t minN, int64_t maxN, int64_t curN, typename... Args>
struct DispatchKernelHelper1D {
  static void run(const int64_t N, Args... args) {
    if (N == curN) {
      Kernel<curN>::run(args...);
    } else {
      DispatchKernelHelper1D<Kernel, minN, maxN, curN + 1, Args...>::run(N, args...);
    }
  }
};

template <template <int64_t> class Kernel, int64_t minN, int64_t maxN, typename... Args>
struct DispatchKernelHelper1D<Kernel, minN, maxN, maxN, Args...> {
  static void run(const int64_t N, Args... args) {
    if (N == maxN) {
      Kernel<maxN>::run(args...);
    } else {
      TORCH_CHECK(false, "Invalid dispatch value");
    }
  }
};

template <template <int64_t> class Kernel, int64_t minN, int64_t maxN, typename... Args>
void DispatchKernel1D(const int64_t N, Args... args) {
  TORCH_CHECK(N >= minN && N <= maxN, "Dispatch value out of range");
  DispatchKernelHelper1D<Kernel, minN, maxN, minN, Args...>::run(N, args...);
}
