// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <TestCuda_Category.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Test {

TEST(cuda, cutile_make_tensor_span_layout_right) {
  using view_type =
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>;
  view_type v("v", 3, 5);

  auto span = Kokkos::make_tensor_span(v);
  using span_type = decltype(span);

  // The View overload always builds the tensor_span from View's actual
  // strides (see Kokkos_CuTile_TensorSpan.hpp), so LayoutLeft/LayoutRight
  // views yield a layout_strided tensor_span, not cuTile's compact
  // layout_left/layout_right.
  static_assert(span_type::rank() == 2);
  ASSERT_EQ(span.extent(0), 3u);
  ASSERT_EQ(span.extent(1), 5u);
  ASSERT_EQ(span.data_handle(), v.data());
  // Row-major: stride(0) = extent(1), stride(1) = 1
  ASSERT_EQ(span.mapping().stride(0), 5u);
  ASSERT_EQ(span.mapping().stride(1), 1u);
}

TEST(cuda, cutile_make_tensor_span_layout_left) {
  using view_type =
      Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
  view_type v("v", 4, 7);

  auto span = Kokkos::make_tensor_span(v);
  using span_type = decltype(span);

  static_assert(span_type::rank() == 2);
  ASSERT_EQ(span.extent(0), 4u);
  ASSERT_EQ(span.extent(1), 7u);
  ASSERT_EQ(span.data_handle(), v.data());
  // Column-major: stride(0) = 1, stride(1) = extent(0)
  ASSERT_EQ(span.mapping().stride(0), 1u);
  ASSERT_EQ(span.mapping().stride(1), 4u);
}

TEST(cuda, cutile_make_tensor_span_layout_stride) {
  // Non-contiguous strides: extent (2,3), strides (6,2).
  Kokkos::LayoutStride layout(2, 6, 3, 2);
  using view_type =
      Kokkos::View<float**, Kokkos::LayoutStride, Kokkos::CudaSpace>;
  view_type v("v", layout);

  auto span = Kokkos::make_tensor_span(v);
  using span_type = decltype(span);

  static_assert(span_type::rank() == 2);
  ASSERT_EQ(span.extent(0), 2u);
  ASSERT_EQ(span.extent(1), 3u);
  ASSERT_EQ(span.mapping().stride(0), 6u);
  ASSERT_EQ(span.mapping().stride(1), 2u);
  ASSERT_EQ(span.data_handle(), v.data());
}

TEST(cuda, cutile_make_tensor_span_cuda_uvm_space) {
  using view_type =
      Kokkos::View<float*, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
  view_type v("v", 16);

  auto span = Kokkos::make_tensor_span(v);
  using span_type = decltype(span);

  static_assert(span_type::rank() == 1);
  ASSERT_EQ(span.extent(0), 16u);
  ASSERT_EQ(span.data_handle(), v.data());
}

// Holds Kokkos::View members directly (not decomposed into pointer +
// mapping) and builds tensor_span from them inside the tile kernel body,
// using the tile-callable View overload of Kokkos::make_tensor_span.
//
// Kokkos::View is never std::is_trivially_copyable under nvcc (its copy/
// move constructors are user-provided to work around an nvcc overload
// ambiguity bug -- see KOKKOS_IMPL_VIEW_HOOKS_NVCC_WORKAROUND), but its
// layout is still safe for the raw cudaMemcpyAsync host->device transfer
// the CuTile launch mechanism uses (which never invokes a copy/move
// constructor or destructor on the transferred bytes), so
// CuTileParallelLaunch / ParallelFor<..., CuTile> only require the driver
// to be standard layout. Views are unmanaged since accessing them from a
// tile kernel never runs their destructor to release a reference count.
struct CuTileTensorSpanFromViewVectorAdd {
  using view_type =
      Kokkos::View<float*, Kokkos::LayoutLeft, Kokkos::CudaSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  view_type a;
  view_type b;
  view_type out;

  __tile__ void operator()(int) const {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    auto a_span   = Kokkos::make_tensor_span(a);
    auto b_span   = Kokkos::make_tensor_span(b);
    auto out_span = Kokkos::make_tensor_span(out);

    auto a_view   = ct::partition_view{a_span, ct::shape{8_ic}};
    auto b_view   = ct::partition_view{b_span, ct::shape{8_ic}};
    auto out_view = ct::partition_view{out_span, ct::shape{8_ic}};

    int bx = ct::bid().x;
    out_view.store_masked(a_view.load_masked(bx) + b_view.load_masked(bx),
                          bx);
  }
};

static_assert(std::is_standard_layout_v<CuTileTensorSpanFromViewVectorAdd>);

TEST(cuda, cutile_make_tensor_span_from_view_parallel_for_vector_add) {
  constexpr int N = 8;

  using managed_type =
      Kokkos::View<float*, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
  using unmanaged_type = CuTileTensorSpanFromViewVectorAdd::view_type;

  managed_type a_m("a", N);
  managed_type b_m("b", N);
  managed_type out_m("out", N);

  auto a_h   = Kokkos::create_mirror_view(a_m);
  auto b_h   = Kokkos::create_mirror_view(b_m);
  auto out_h = Kokkos::create_mirror_view(out_m);
  for (int i = 0; i < N; ++i) {
    a_h(i) = static_cast<float>(i);
    b_h(i) = static_cast<float>(2 * i);
  }
  Kokkos::deep_copy(a_m, a_h);
  Kokkos::deep_copy(b_m, b_h);
  Kokkos::deep_copy(out_m, out_h);

  constexpr int nblocks = 1;
  Kokkos::CuTile exec;
  Kokkos::parallel_for(
      "Test::cuda::cutile_make_tensor_span_from_view_vector_add",
      Kokkos::RangePolicy<Kokkos::CuTile>(exec, 0, nblocks),
      CuTileTensorSpanFromViewVectorAdd{unmanaged_type(a_m),
                                        unmanaged_type(b_m),
                                        unmanaged_type(out_m)});
  exec.fence();

  Kokkos::deep_copy(out_h, out_m);
  for (int i = 0; i < N; ++i) {
    ASSERT_FLOAT_EQ(out_h(i), a_h(i) + b_h(i)) << "mismatch at index " << i;
  }
}

// Rank-2 coverage for all three tensor_span-eligible layouts, each built
// directly from a Kokkos::View held by the functor (LayoutLeft and
// LayoutRight tested directly; LayoutStride is exercised through a
// non-contiguous unmanaged view over the same LayoutLeft view's storage).
struct CuTileTensorSpanFromViewLayoutProbe {
  using left_type =
      Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using right_type =
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using stride_type =
      Kokkos::View<float**, Kokkos::LayoutStride, Kokkos::CudaSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using out_type =
      Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  left_type left;
  right_type right;
  stride_type strided;
  out_type out;

  __tile__ void operator()(int) const {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    auto left_span    = Kokkos::make_tensor_span(left);
    auto right_span   = Kokkos::make_tensor_span(right);
    auto strided_span = Kokkos::make_tensor_span(strided);
    auto out_span     = Kokkos::make_tensor_span(out);

    auto left_view    = ct::partition_view{left_span, ct::shape{1_ic, 1_ic}};
    auto right_view   = ct::partition_view{right_span, ct::shape{1_ic, 1_ic}};
    auto strided_view = ct::partition_view{strided_span, ct::shape{1_ic, 1_ic}};
    auto out_view     = ct::partition_view{out_span, ct::shape{1_ic, 1_ic}};

    out_view.store_masked(left_view.load_masked(1, 0), 0, 0);
    out_view.store_masked(right_view.load_masked(1, 0), 1, 0);
    out_view.store_masked(strided_view.load_masked(1, 0), 2, 0);
  }
};

static_assert(std::is_standard_layout_v<CuTileTensorSpanFromViewLayoutProbe>);

TEST(cuda, cutile_make_tensor_span_from_view_layout_left_right_stride) {
  constexpr int N0 = 2;
  constexpr int N1 = 2;
  constexpr int span_size = N0 * N1;

  // Physical contents: [10, 20, 30, 40]
  // LayoutLeft  (1,0) -> offset 1 -> 20
  // LayoutRight (1,0) -> offset 2 -> 30
  // LayoutStride mirrors LayoutLeft's strides -> (1,0) -> offset 1 -> 20
  Kokkos::View<float*, Kokkos::CudaSpace> buffer("buffer", span_size);
  {
    auto h = Kokkos::create_mirror_view(buffer);
    h(0) = 10.0f;
    h(1) = 20.0f;
    h(2) = 30.0f;
    h(3) = 40.0f;
    Kokkos::deep_copy(buffer, h);
  }

  using probe_type = CuTileTensorSpanFromViewLayoutProbe;
  Kokkos::LayoutStride stride_layout(N0, 1, N1, N0);

  probe_type::left_type left(buffer.data(), N0, N1);
  probe_type::right_type right(buffer.data(), N0, N1);
  probe_type::stride_type strided(buffer.data(), stride_layout);

  // 3x1 so we can store 1x1 tiles at (0,0), (1,0), and (2,0).
  Kokkos::View<float*, Kokkos::CudaSpace> out_m("out", 3);
  probe_type::out_type out(out_m.data(), 3, 1);

  Kokkos::CuTile exec;
  Kokkos::parallel_for(
      "Test::cuda::cutile_make_tensor_span_from_view_layout_probe",
      Kokkos::RangePolicy<Kokkos::CuTile>(exec, 0, 1),
      probe_type{left, right, strided, out});
  exec.fence();

  auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out_m);
  ASSERT_FLOAT_EQ(out_h(0), 20.0f);  // LayoutLeft (1,0)
  ASSERT_FLOAT_EQ(out_h(1), 30.0f);  // LayoutRight (1,0)
  ASSERT_FLOAT_EQ(out_h(2), 20.0f);  // LayoutStride (1,0), left-equivalent strides
}

}  // namespace Test
