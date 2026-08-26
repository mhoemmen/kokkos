// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestCuda_Category.hpp>

#include <cuda_tile.h>

#include <array>
#include <cstddef>

#include <Cuda/Kokkos_Cuda_KernelLaunchTile.hpp>
#include <Cuda/Kokkos_Cuda_TileView.hpp>

namespace Test {

__tile_global__ void tile_vector_add(float* __restrict__ a,
                                     float* __restrict__ b,
                                     float* __restrict__ out, size_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  a   = ct::assume_aligned(a, 16_ic);
  b   = ct::assume_aligned(b, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  constexpr auto tile_size = 8_ic;
  auto shape               = ct::shape{tile_size};
  auto extent              = ct::extents{n};

  auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
  auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
  auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

  const size_t n_tile       = n / tile_size;
  const size_t n_tile_block = n_tile / ct::num_blocks().x;
  const size_t tile_start   = ct::bid().x * n_tile_block;
  const size_t tile_end     = tile_start + n_tile_block;
  for (size_t tile_index : ct::irange(tile_start, tile_end)) {
    auto a_plus_b =
        a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
    o_view.store_masked(a_plus_b, tile_index);
  }
}

TEST(cuda, tile_vector_add) {
  constexpr std::size_t N          = 256;
  constexpr std::size_t num_blocks = 2;

  std::array<float, N> h_a;
  std::array<float, N> h_b;
  for (std::size_t i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(2 * i);
  }

  float* d_a   = nullptr;
  float* d_b   = nullptr;
  float* d_out = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_a, sizeof(float) * N));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_b, sizeof(float) * N));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_out, sizeof(float) * N));

  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(d_a, h_a.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(d_b, h_b.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

  // A tile kernel must be launched with a block dimension of one; the
  // compiler decides how many threads actually execute the kernel.
  tile_vector_add<<<num_blocks, 1>>>(d_a, d_b, d_out, N);
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetLastError());
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());

  std::array<float, N> h_out;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpy(h_out.data(), d_out, sizeof(float) * N,
                                        cudaMemcpyDeviceToHost));

  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out[i] != h_a[i] + h_b[i]) errors++;
  }
  errors = 0;

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_a));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_b));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_out));
}

// This struct implements the minimal compatibility requirements for Driver
// handed to the implementation tile launch function
struct TileVectorAddDummyDriver {
  float* a;
  float* b;
  float* out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    constexpr auto tile_size = 8_ic;
    auto shape               = ct::shape{tile_size};
    auto extent              = ct::extents{n};

    auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
    auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
    auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

    const size_t n_tile       = n / tile_size;
    const size_t n_tile_block = n_tile / ct::num_blocks().x;
    const size_t tile_start   = ct::bid().x * n_tile_block;
    const size_t tile_end     = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b =
          a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
      o_view.store_masked(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_kernel_invoker() {
  constexpr std::size_t N = 256;

  Kokkos::View<float*, Kokkos::Cuda> a("A", N);
  Kokkos::View<float*, Kokkos::Cuda> b("A", N);
  Kokkos::View<float*, Kokkos::Cuda> out("A", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = 2 * i;
      });

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileVectorAddDummyDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  TileVectorAddDummyDriver driver{a.data(), b.data(), out.data(), N};

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(out);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out(i) != float(3 * i)) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_kernel_invoker) { cuda_tile_kernel_invoker(); }

// This struct shows how Kokkos::TileView bridges Kokkos::View data with
// cuTile's partition_view inside a tile kernel driver.  Each TileView is
// constructed on the host from a Kokkos::View, then embedded by value in
// the Driver struct that gets memcpy'd to device scratch memory.
struct TileViewVectorAddDriver {
  using ViewType   = Kokkos::View<float*, Kokkos::Cuda>;
  using ShapeType  = cuda::tiles::shape<8>;
  using TileViewType = Kokkos::TileView<ViewType, ShapeType>;

  TileViewType a;
  TileViewType b;
  TileViewType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    TileViewType::partition_view_type a_view = a;
    TileViewType::partition_view_type b_view = b;
    TileViewType::partition_view_type o_view = out;

    constexpr std::size_t tile_size = 8;
    const size_t n_tile             = n / tile_size;
    const size_t n_tile_block       = n_tile / ct::num_blocks().x;
    const size_t tile_start         = ct::bid().x * n_tile_block;
    const size_t tile_end           = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b =
          a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
      o_view.store_masked(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_view_kernel_invoker() {
  constexpr std::size_t N = 256;

  Kokkos::View<float*, Kokkos::Cuda> a("A", N);
  Kokkos::View<float*, Kokkos::Cuda> b("B", N);
  Kokkos::View<float*, Kokkos::Cuda> out("Out", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = 2 * i;
      });

  using ShapeType = TileViewVectorAddDriver::ShapeType;
  TileViewVectorAddDriver driver{
      Kokkos::TileView<decltype(a), ShapeType>(a),
      Kokkos::TileView<decltype(b), ShapeType>(b),
      Kokkos::TileView<decltype(out), ShapeType>(out), N};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewVectorAddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(out);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out(i) != float(3 * i)) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_vector_add) { cuda_tile_view_kernel_invoker(); }

// Exercises Kokkos::TileView with a noncontiguous, rank-1 Kokkos::View:
// a and b are columns of rank-2 backing views, picked out with a subview,
// so consecutive elements are actually two floats apart in memory.
struct TileViewNoncontiguousRank1AddDriver {
  using BackingViewType =
      Kokkos::View<float* [2], Kokkos::LayoutRight, Kokkos::Cuda>;
  using ViewType =
      decltype(Kokkos::subview(std::declval<BackingViewType const&>(),
                                Kokkos::ALL, 0));
  using ShapeType    = cuda::tiles::shape<8>;
  using TileViewType = Kokkos::TileView<ViewType, ShapeType>;

  TileViewType a;
  TileViewType b;
  TileViewType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    TileViewType::partition_view_type a_view = a;
    TileViewType::partition_view_type b_view = b;
    TileViewType::partition_view_type o_view = out;

    constexpr std::size_t tile_size = 8;
    const size_t n_tile             = n / tile_size;
    const size_t n_tile_block       = n_tile / ct::num_blocks().x;
    const size_t tile_start         = ct::bid().x * n_tile_block;
    const size_t tile_end           = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b =
          a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
      o_view.store_masked(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_view_noncontiguous_rank1_add() {
  constexpr std::size_t N = 256;

  TileViewNoncontiguousRank1AddDriver::BackingViewType a_full("a_full", N);
  TileViewNoncontiguousRank1AddDriver::BackingViewType b_full("b_full", N);
  TileViewNoncontiguousRank1AddDriver::BackingViewType out_full("out_full",
                                                                 N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a_full(i, 0)   = i;
        a_full(i, 1)   = -1;
        b_full(i, 0)   = 2 * i;
        b_full(i, 1)   = -1;
        out_full(i, 0) = -1;
        out_full(i, 1) = -1;
      });

  auto a   = Kokkos::subview(a_full, Kokkos::ALL, 0);
  auto b   = Kokkos::subview(b_full, Kokkos::ALL, 0);
  auto out = Kokkos::subview(out_full, Kokkos::ALL, 0);

  using ShapeType = TileViewNoncontiguousRank1AddDriver::ShapeType;
  TileViewNoncontiguousRank1AddDriver driver{
      Kokkos::TileView<decltype(a), ShapeType>(a),
      Kokkos::TileView<decltype(b), ShapeType>(b),
      Kokkos::TileView<decltype(out), ShapeType>(out), N};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewNoncontiguousRank1AddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out_full = Kokkos::create_mirror_view_and_copy(out_full);
  int errors       = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out_full(i, 0) != float(3 * i)) errors++;
    // The untouched neighbor column must be unaffected; if TileView were
    // treating the strided view as contiguous, this would be clobbered.
    if (h_out_full(i, 1) != -1.0f) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_noncontiguous_rank1_add) {
  cuda_tile_view_noncontiguous_rank1_add();
}

// Exercises Kokkos::TileView with a noncontiguous, rank-2 Kokkos::View:
// a and b are 2-D "slabs" of rank-3 backing views, picked out with a
// subview, so consecutive elements along the second dimension are two
// floats apart in memory.
struct TileViewNoncontiguousRank2AddDriver {
  using BackingViewType =
      Kokkos::View<float* [4][2], Kokkos::LayoutRight, Kokkos::Cuda>;
  using ViewType =
      decltype(Kokkos::subview(std::declval<BackingViewType const&>(),
                                Kokkos::ALL, Kokkos::ALL, 0));
  using ShapeType    = cuda::tiles::shape<8, 4>;
  using TileViewType = Kokkos::TileView<ViewType, ShapeType>;

  TileViewType a;
  TileViewType b;
  TileViewType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    TileViewType::partition_view_type a_view = a;
    TileViewType::partition_view_type b_view = b;
    TileViewType::partition_view_type o_view = out;

    // The second dimension has static extent 4, which fits in a single
    // tile of shape <8, 4>; only the first dimension needs to be tiled.
    constexpr std::size_t tile_size = 8;
    const size_t n_tile             = n / tile_size;
    const size_t n_tile_block       = n_tile / ct::num_blocks().x;
    const size_t tile_start         = ct::bid().x * n_tile_block;
    const size_t tile_end           = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b = a_view.load_masked(tile_index, 0) +
                       b_view.load_masked(tile_index, 0);
      o_view.store_masked(a_plus_b, tile_index, 0);
    }
  }
};

void cuda_tile_view_noncontiguous_rank2_add() {
  constexpr std::size_t N = 256;

  TileViewNoncontiguousRank2AddDriver::BackingViewType a_full("a_full", N);
  TileViewNoncontiguousRank2AddDriver::BackingViewType b_full("b_full", N);
  TileViewNoncontiguousRank2AddDriver::BackingViewType out_full("out_full",
                                                                 N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < 4; ++j) {
          a_full(i, j, 0)   = i * 4 + j;
          a_full(i, j, 1)   = -1;
          b_full(i, j, 0)   = 2 * (i * 4 + j);
          b_full(i, j, 1)   = -1;
          out_full(i, j, 0) = -1;
          out_full(i, j, 1) = -1;
        }
      });

  auto a   = Kokkos::subview(a_full, Kokkos::ALL, Kokkos::ALL, 0);
  auto b   = Kokkos::subview(b_full, Kokkos::ALL, Kokkos::ALL, 0);
  auto out = Kokkos::subview(out_full, Kokkos::ALL, Kokkos::ALL, 0);

  using ShapeType = TileViewNoncontiguousRank2AddDriver::ShapeType;
  TileViewNoncontiguousRank2AddDriver driver{
      Kokkos::TileView<decltype(a), ShapeType>(a),
      Kokkos::TileView<decltype(b), ShapeType>(b),
      Kokkos::TileView<decltype(out), ShapeType>(out), N};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewNoncontiguousRank2AddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out_full = Kokkos::create_mirror_view_and_copy(out_full);
  int errors       = 0;
  for (std::size_t i = 0; i < N; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (h_out_full(i, j, 0) != float(3 * (i * 4 + j))) errors++;
      // The untouched neighbor slab must be unaffected; if TileView were
      // treating the strided view as contiguous, this would be clobbered.
      if (h_out_full(i, j, 1) != -1.0f) errors++;
    }
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_noncontiguous_rank2_add) {
  cuda_tile_view_noncontiguous_rank2_add();
}

}  // namespace Test
