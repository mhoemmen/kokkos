// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {

constexpr int M = 8;
constexpr int N = 16;
constexpr int K = 24;

// Tile shapes from the CUDA Tile C++ GEMM example (blog / programming guide).
constexpr int TILE_M = 4;
constexpr int TILE_N = 4;
constexpr int TILE_K = 8;

constexpr int TILE_GRID_M = M / TILE_M;  // 2
constexpr int TILE_GRID_N = N / TILE_N;  // 4
constexpr int NUM_TILE_BLOCKS = TILE_GRID_M * TILE_GRID_N;

using matrix_type =
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>;

// Host allocates Kokkos::Views and the functor holds them by value. View's
// default/copy/move construction, destruction, extent(), stride(), and
// data() are __tile__-callable, so make_tensor_span(view) can rebuild the
// tensor_span directly from the captured View inside the tile kernel body.
struct CuTileGemmFunctor {
  matrix_type A;
  matrix_type B;
  matrix_type C;

  __tile__ void operator()(int linear) const {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    // Linearize a 2D tile grid into RangePolicy (begin == 0 => bid.x == linear).
    int const x_block = linear / TILE_GRID_N;
    int const y_block = linear % TILE_GRID_N;

    auto a_span = Kokkos::make_tensor_span(A);
    auto b_span = Kokkos::make_tensor_span(B);
    auto c_span = Kokkos::make_tensor_span(C);

    auto a_view = ct::partition_view{a_span, ct::shape{4_ic, 8_ic}};
    auto b_view = ct::partition_view{b_span, ct::shape{8_ic, 4_ic}};
    auto c_view = ct::partition_view{c_span, ct::shape{4_ic, 4_ic}};

    using f32x4x4 = ct::tile<float, ct::shape<4, 4>>;
    auto acc_tile = ct::full<f32x4x4>(0);

    int const length_k = A.extent(1);
    int const k_tiles  = 1 + (length_k - 1) / TILE_K;
    for (int idx = 0; idx < k_tiles; ++idx) {
      auto a_tile = a_view.load_masked(x_block, idx);
      auto b_tile = b_view.load_masked(idx, y_block);
      acc_tile    = ct::mma(a_tile, b_tile, acc_tile);
    }
    c_view.store_masked(acc_tile, x_block, y_block);
  }
};

// Kokkos::View is standard-layout but never std::is_trivially_copyable under
// nvcc (see Kokkos_CuTile_KernelLaunch.hpp); CuTile's parallel_for driver
// only requires standard layout for its raw cudaMemcpyAsync-based transfer.
static_assert(std::is_standard_layout_v<CuTileGemmFunctor>);

void host_reference_gemm(
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> const& A,
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> const& B,
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>& C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int status = EXIT_SUCCESS;
  {
    matrix_type A("A", M, K);
    matrix_type B("B", K, N);
    matrix_type C("C", M, N);

    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);
    auto C_h = Kokkos::create_mirror_view(C);

    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        A_h(i, k) = static_cast<float>(i + k + 1);
      }
    }
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        B_h(k, j) = static_cast<float>((k + 1) * 0.1f + j * 0.01f);
      }
    }
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C_h(i, j) = 0.0f;
      }
    }

    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);
    Kokkos::deep_copy(C, C_h);

    Kokkos::CuTile exec;
    Kokkos::parallel_for(
        "example::cutile_gemm",
        Kokkos::RangePolicy<Kokkos::CuTile>(exec, 0, NUM_TILE_BLOCKS),
        CuTileGemmFunctor{A, B, C});
    exec.fence();

    Kokkos::deep_copy(C_h, C);

    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> C_ref(
        "C_ref", M, N);
    host_reference_gemm(A_h, B_h, C_ref);

    float max_err = 0.0f;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        max_err = std::fmax(max_err, std::fabs(C_h(i, j) - C_ref(i, j)));
      }
    }

    std::printf("cutile_gemm: M=%d N=%d K=%d max_err=%e\n", M, N, K, max_err);
    // float mma accumulation vs host reference; allow modest ulp slack
    constexpr float tol = 1.0e-3f;
    if (max_err > tol) {
      std::printf("FAIL: max_err exceeds tolerance %e\n", tol);
      status = EXIT_FAILURE;
    } else {
      std::printf("PASS\n");
    }
  }
  Kokkos::finalize();
  return status;
}
