#include <torch/extension.h>

#include "common.h"

namespace space_filling_curves {

at::Tensor hilbert3d_encode_lut(
  const at::Tensor& coords,
  int64_t bits,
  const std::string& axis_order = "xyz");
at::Tensor morton3d_encode_magicbits(
  const at::Tensor& coords,
  const std::string& axis_order = "xyz");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "hilbert3d_encode_lut",
    &space_filling_curves::hilbert3d_encode_lut,
    pybind11::arg("coords"),
    pybind11::arg("bits"),
    pybind11::arg("axis_order") = "xyz");
  m.def(
    "morton3d_encode_magicbits",
    &space_filling_curves::morton3d_encode_magicbits,
    pybind11::arg("coords"),
    pybind11::arg("axis_order") = "xyz");
}
