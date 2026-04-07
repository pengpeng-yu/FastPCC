#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"

namespace int_sparse_conv {

template <typename SmArch, int AlignInputElements, int AlignOutputElements, bool UseTensorOp>
void cutlass_gather_gemm_scatter_int8_impl(
    const at::Tensor &A,           // (M x K) int8
    const at::Tensor &B,           // (N x K) int8
    const at::Tensor &C,           // (D_rows x N) int32 or (N,) int32 or (0,) int32
    const at::Tensor &D,           // (D_rows x N) int32
    const at::Tensor &gather_idx,  // (L,) int32
    const at::Tensor &scatter_idx  // (L,) int32
) {
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementInputC = int32_t;
  using ElementOutput = ElementInputC;
  using ElementAccumulator = ElementOutput;
  using ElementComputeEpilogue = ElementOutput;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutInputC = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = std::conditional_t<
    UseTensorOp, cutlass::arch::OpClassTensorOp, cutlass::arch::OpClassSimt
  >;
  using ThreadblockShape = std::conditional_t<
    UseTensorOp, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<64, 64, 8>
  >;
  using WarpShape = std::conditional_t<
    UseTensorOp, cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<32, 32, 8>
  >;
  using InstructionShape = std::conditional_t<
    UseTensorOp,
    std::conditional_t<
      std::is_same_v<SmArch, cutlass::arch::Sm75>,
      cutlass::gemm::GemmShape<8, 8, 16>,
      cutlass::gemm::GemmShape<16, 8, 32>
    >,
    cutlass::gemm::GemmShape<1, 1, 1>
  >;
  static constexpr int NumStages = UseTensorOp ? (std::is_same_v<SmArch, cutlass::arch::Sm75> ? 2 : 2) : 2;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    AlignOutputElements,
    ElementAccumulator,
    ElementComputeEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementInputC,
    LayoutInputC,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    AlignInputElements,   // alignmentA
    AlignInputElements,   // alignmentB
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    true,  // GatherA = true
    false, // GatherB = false
    true   // ScatterD = true
  >;

  at::Device device = A.device();
  TORCH_CHECK(device.is_cuda());
  TORCH_CHECK(B.device() == device && C.device() == device && D.device() == device
              && gather_idx.device() == device && scatter_idx.device() == device);
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() && D.is_contiguous()
              && gather_idx.is_contiguous() && scatter_idx.is_contiguous());
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2
              && gather_idx.dim() == 1 && scatter_idx.dim() == 1);

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(0);
  int64_t D_rows = D.size(0);
  int64_t L = gather_idx.numel();
  TORCH_CHECK(B.size(1) == K);
  TORCH_CHECK((C.dim() == 1 && C.numel() == N)
              || (C.dim() == 2 && C.size(0) == D_rows && C.size(1) == N)
              || C.numel() == 0);
  TORCH_CHECK(D.size(1) == N);
  TORCH_CHECK(scatter_idx.size(0) == L);
  TORCH_CHECK(L <= D_rows);

  cutlass::gemm::GemmCoord problem_size_real(L, N, K);

  int split_k_slices = 1;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size_real,
    split_k_slices,
    {1, C.numel() == 0 ? 0 : 1},
    A.data_ptr<ElementInputA>(),
    B.data_ptr<ElementInputB>(),
    C.data_ptr<ElementInputC>(),
    D.data_ptr<ElementOutput>(),
    M * K, N * K, C.numel(), D_rows * N,
    K, K, C.numel() == D_rows * N ? N : 0, N,
    gather_idx.data_ptr<int32_t>(),
    nullptr,
    scatter_idx.data_ptr<int32_t>()
  };

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlass::cutlassGetStatusString(status));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void *workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          A.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  status = gemm_op(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlass::cutlassGetStatusString(status));
}

namespace {

template <typename Element>
__inline__ bool is_aligned_address(const at::Tensor &tensor, int alignment_elements) {
  std::uintptr_t address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr<Element>());
  size_t alignment_bytes = alignment_elements * sizeof(Element);
  return (address % alignment_bytes) == 0;
}

}

template <typename SmArch>
void cutlass_gather_gemm_scatter_int8_dispatch_align(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &D,
    const at::Tensor &gather_idx,
    const at::Tensor &scatter_idx) {
  int64_t K = A.size(1);
  int64_t N = B.size(0);
  bool input_align16 = (K % 16 == 0) &&
       is_aligned_address<int8_t>(A, 16) &&
       is_aligned_address<int8_t>(B, 16);
  bool input_align8 = (K % 8 == 0) &&
       is_aligned_address<int8_t>(A, 8) &&
       is_aligned_address<int8_t>(B, 8);
  bool input_align4 = (K % 4 == 0) &&
       is_aligned_address<int8_t>(A, 4) &&
       is_aligned_address<int8_t>(B, 4);

  bool output_align4 = (N % 4 == 0) &&
       is_aligned_address<int32_t>(D, 4) &&
       ((C.numel() == 0) || is_aligned_address<int32_t>(C, 4));
  bool output_align2 = (N % 2 == 0) &&
       is_aligned_address<int32_t>(D, 2) &&
       ((C.numel() == 0) || is_aligned_address<int32_t>(C, 2));

  if (input_align16 && output_align4) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 16, 4, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align16 && output_align2) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 16, 2, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align16) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 16, 1, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align8 && output_align4) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 8, 4, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align8 && output_align2) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 8, 2, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align8) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 8, 1, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align4 && output_align4) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 4, 4, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align4 && output_align2) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 4, 2, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else if (input_align4) {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 4, 1, true>(A, B, C, D, gather_idx, scatter_idx);
  }
  else {
    cutlass_gather_gemm_scatter_int8_impl<SmArch, 1, 1, false>(A, B, C, D, gather_idx, scatter_idx);
  }
}

void cutlass_gather_gemm_scatter_int8(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &D,
    const at::Tensor &gather_idx,
    const at::Tensor &scatter_idx) {
  const auto prop = at::cuda::getDeviceProperties(A.device().index());
  int sm_arch = prop->major * 10 + prop->minor;

#define CASE_SM(SM) \
  else if (sm_arch == (SM)) cutlass_gather_gemm_scatter_int8_dispatch_align<cutlass::arch::Sm##SM>(A, B, C, D, gather_idx, scatter_idx)

  if (0) {}
#if defined(TARGET_SM120) || defined(TARGET_SM120A) || defined(TARGET_SM120F)
  CASE_SM(120);
#endif
#if defined(TARGET_SM103) || defined(TARGET_SM103A) || defined(TARGET_SM103F)
  CASE_SM(103);
#endif
#if defined(TARGET_SM101) || defined(TARGET_SM101A) || defined(TARGET_SM101F)
  CASE_SM(101);
#endif
#if defined(TARGET_SM100) || defined(TARGET_SM100A) || defined(TARGET_SM100F)
  CASE_SM(100);
#endif
#if defined(TARGET_SM90) || defined(TARGET_SM90A)
  CASE_SM(90);
#endif
#ifdef TARGET_SM89
  CASE_SM(89);
#endif
#ifdef TARGET_SM86
  CASE_SM(86);
#endif
#ifdef TARGET_SM80
  CASE_SM(80);
#endif
#ifdef TARGET_SM75
  CASE_SM(75);
#endif
#ifdef TARGET_SM72
  else if (sm_arch == 72) cutlass_gather_gemm_scatter_int8_impl<cutlass::arch::Sm72, 1, 1, false>(A, B, C, D, gather_idx, scatter_idx);
#endif
#ifdef TARGET_SM70
  else if (sm_arch == 70) cutlass_gather_gemm_scatter_int8_impl<cutlass::arch::Sm70, 1, 1, false>(A, B, C, D, gather_idx, scatter_idx);
#endif
  else {
    TORCH_CHECK(false, "No available arch");
  }

#undef CASE_SM
}

}
