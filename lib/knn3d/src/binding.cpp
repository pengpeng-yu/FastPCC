#include <torch/extension.h>

#include "knn3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn3d", &knn3d::KNearestNeighborIdx);
}
