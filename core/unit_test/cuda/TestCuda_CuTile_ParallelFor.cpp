// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <TestCuda_Category.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace Test {

struct CuTileFillFunctor {
  int* ptr;

  // cuTile: __tile_global__ may only call __tile__ functions (not plain
  // __device__ / KOKKOS_FUNCTION alone).
  __tile__ void operator()(int i) const { ptr[i] = i; }
};

static_assert(std::is_trivially_copyable_v<CuTileFillFunctor>);

TEST(cuda, cutile_parallel_for_range) {
  constexpr int N = 64;

  int* d_out = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_out, sizeof(int) * N));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemset(d_out, 0, sizeof(int) * N));

  Kokkos::Cuda exec;
  Kokkos::parallel_for("Test::cuda::cutile_parallel_for_range",
                       Kokkos::TileRangePolicy<Kokkos::Cuda>(exec, 0, N),
                       CuTileFillFunctor{d_out});
  exec.fence();

  std::array<int, N> h_out{};
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(h_out.data(), d_out, sizeof(int) * N, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(h_out[i], i) << "mismatch at index " << i;
  }

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_out));
}

}  // namespace Test
