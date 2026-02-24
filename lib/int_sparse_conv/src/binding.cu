#include <torch/extension.h>

#include "hashmap/hashmap_cuda.cuh"

namespace int_sparse_conv {

void cutlass_gather_gemm_scatter_int8(
  const at::Tensor &A,
  const at::Tensor &B,
  const at::Tensor &C,
  const at::Tensor &D,
  const at::Tensor &gather_idx,
  const at::Tensor &scatter_idx);

void cutlass_gemm_int8(
  const at::Tensor &A,
  const at::Tensor &B,
  const at::Tensor &C,
  const at::Tensor &D);

at::Tensor softmax_int32(const at::Tensor &input);

at::Tensor requant_to_int8(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor requant_to_int16(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor requant_to_int32(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor bias_prelu_requant_to_int8(
  const at::Tensor &input,
  const at::Tensor &bias,
  const at::Tensor &slope,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor bias_prelu_requant_to_int16(
  const at::Tensor &input,
  const at::Tensor &bias,
  const at::Tensor &slope,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor bias_prelu_requant_to_int32(
  const at::Tensor &input,
  const at::Tensor &bias,
  const at::Tensor &slope,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor bias_requant_to_int8(
  const at::Tensor &input,
  const at::Tensor &bias,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor bias_requant_to_int16(
  const at::Tensor &input,
  const at::Tensor &bias,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor bias_requant_to_int32(
  const at::Tensor &input,
  const at::Tensor &bias,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor prelu_requant_to_int8(
  const at::Tensor &input,
  const at::Tensor &slope,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor prelu_requant_to_int16(
  const at::Tensor &input,
  const at::Tensor &slope,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor prelu_requant_to_int32(
  const at::Tensor &input,
  const at::Tensor &slope,
  const at::Tensor &requant_mul,
  const at::Tensor &zero_point,
  const int shift);

at::Tensor prelu(
  const at::Tensor &input,
  const at::Tensor &slope);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_gather_gemm_scatter_int8", &int_sparse_conv::cutlass_gather_gemm_scatter_int8);
  m.def("cutlass_gemm_int8", &int_sparse_conv::cutlass_gemm_int8);
  m.def("softmax_int32", &int_sparse_conv::softmax_int32);
  m.def("requant_to_int8", &int_sparse_conv::requant_to_int8);
  m.def("requant_to_int16", &int_sparse_conv::requant_to_int16);
  m.def("requant_to_int32", &int_sparse_conv::requant_to_int32);
  m.def("bias_prelu_requant_to_int8", &int_sparse_conv::bias_prelu_requant_to_int8);
  m.def("bias_prelu_requant_to_int16", &int_sparse_conv::bias_prelu_requant_to_int16);
  m.def("bias_prelu_requant_to_int32", &int_sparse_conv::bias_prelu_requant_to_int32);
  m.def("bias_requant_to_int8", &int_sparse_conv::bias_requant_to_int8);
  m.def("bias_requant_to_int16", &int_sparse_conv::bias_requant_to_int16);
  m.def("bias_requant_to_int32", &int_sparse_conv::bias_requant_to_int32);
  m.def("prelu_requant_to_int8", &int_sparse_conv::prelu_requant_to_int8);
  m.def("prelu_requant_to_int16", &int_sparse_conv::prelu_requant_to_int16);
  m.def("prelu_requant_to_int32", &int_sparse_conv::prelu_requant_to_int32);
  m.def("prelu", &int_sparse_conv::prelu);
  py::class_<int_sparse_conv::hashtable>(m, "GPUHashTable")
    .def(py::init<const int>())
    .def(py::init<torch::Tensor, torch::Tensor>())
    .def("insert_vals", &int_sparse_conv::hashtable::insert_vals)
    .def("lookup_vals", &int_sparse_conv::hashtable::lookup_vals)
    .def("insert_coords", &int_sparse_conv::hashtable::insert_coords)
    .def("lookup_coords", &int_sparse_conv::hashtable::lookup_coords);
  py::class_<int_sparse_conv::hashtable32>(m, "GPUHashTable32")
    .def(py::init<const int>())
    .def(py::init<torch::Tensor, torch::Tensor>())
    .def("insert_vals", &int_sparse_conv::hashtable32::insert_vals)
    .def("lookup_vals", &int_sparse_conv::hashtable32::lookup_vals)
    .def("insert_coords", &int_sparse_conv::hashtable32::insert_coords)
    .def("lookup_coords", &int_sparse_conv::hashtable32::lookup_coords);
}
