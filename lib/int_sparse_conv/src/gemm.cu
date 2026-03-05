#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"

namespace int_sparse_conv {

template <typename SmArch, int AlignInputElements, int AlignOutputElements, bool UseTensorOp>
void cutlass_gemm_int8_impl(
    const at::Tensor &A,           // (M x K) int8
    const at::Tensor &B,           // (N x K) int8
    const at::Tensor &C,           // (N,)    int32 or (M x N) int32 or (0,) int32
    const at::Tensor &D            // (M x N) int32
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
    UseTensorOp, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<128, 128, 64>
  >;
  using WarpShape = std::conditional_t<
    UseTensorOp, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<64, 64, 64>
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
  static constexpr int NumStages = UseTensorOp ? (std::is_same_v<SmArch, cutlass::arch::Sm75> ? 2 : 4) : 2;

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
    cutlass::arch::OpMultiplyAddSaturate
  >;

  at::Device device = A.device();
  TORCH_CHECK(device.is_cuda());
  TORCH_CHECK(B.device() == device && C.device() == device && D.device() == device);
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() && D.is_contiguous());
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2);

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(0);
  TORCH_CHECK(B.size(1) == K);
  TORCH_CHECK((C.dim() == 1 && C.numel() == N)
            || (C.dim() == 2 && C.size(0) == M && C.size(1) == N)
            || C.numel() == 0);
  TORCH_CHECK(D.size(0) == M && D.size(1) == N);

  cutlass::gemm::GemmCoord problem_size_real(M, N, K);

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
    M * K, N * K, C.numel(), M * N,
    K, K, C.numel() == M * N ? N : 0, N,
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
  status = gemm_op.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlass::cutlassGetStatusString(status));

  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlass::cutlassGetStatusString(status));
}

template <typename SmArch>
void cutlass_gemm_int8_dispatch_align(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &D) {
  int64_t K = A.size(1);
  int64_t N = B.size(0);

  if ((K % 16 == 0) && (N % 4 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 16, 4, true>(A, B, C, D);
  }
  else if ((K % 16 == 0) && (N % 2 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 16, 2, true>(A, B, C, D);
  }
  else if ((K % 16 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 16, 1, true>(A, B, C, D);
  }
  else if ((K % 8 == 0) && (N % 4 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 8, 4, true>(A, B, C, D);
  }
  else if ((K % 8 == 0) && (N % 2 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 8, 2, true>(A, B, C, D);
  }
  else if ((K % 8 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 8, 1, true>(A, B, C, D);
  }
  else if ((K % 4 == 0) && (N % 4 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 4, 4, true>(A, B, C, D);
  }
  else if ((K % 4 == 0) && (N % 2 == 0)) {
    cutlass_gemm_int8_impl<SmArch, 4, 2, true>(A, B, C, D);
  }
  else if (K % 4 == 0) {
    cutlass_gemm_int8_impl<SmArch, 4, 1, true>(A, B, C, D);
  }
  else {
    cutlass_gemm_int8_impl<SmArch, 1, 1, false>(A, B, C, D);
  }
}

void cutlass_gemm_int8(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &D) {
  const auto prop = at::cuda::getDeviceProperties(A.device().index());
  int sm_arch = prop->major * 10 + prop->minor;

#define CASE_SM(SM) \
  else if (sm_arch == (SM)) cutlass_gemm_int8_dispatch_align<cutlass::arch::Sm##SM>(A, B, C, D)

  if (0) {}
#ifdef TARGET_SM120
  CASE_SM(120);
#endif
#ifdef TARGET_SM103
  CASE_SM(103);
#endif
#ifdef TARGET_SM101
  CASE_SM(101);
#endif
#ifdef TARGET_SM100
  CASE_SM(100);
#endif
#ifdef TARGET_SM90
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
  else if (sm_arch == 72) cutlass_gemm_int8_impl<cutlass::arch::Sm72, 1, 1, false>(A, B, C, D);
#endif
#ifdef TARGET_SM70
  else if (sm_arch == 70) cutlass_gemm_int8_impl<cutlass::arch::Sm70, 1, 1, false>(A, B, C, D);
#endif
  else {
    TORCH_CHECK(false, "No available arch");
  }

#undef CASE_SM
}

}
