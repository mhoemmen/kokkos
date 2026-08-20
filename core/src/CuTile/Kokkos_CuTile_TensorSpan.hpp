// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUTILE_TENSOR_SPAN_HPP
#define KOKKOS_CUTILE_TENSOR_SPAN_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE)

#include <Kokkos_Abort.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_View.hpp>
#include <Cuda/Kokkos_CudaSpace.hpp>

#include <cuda_tile.h>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace Kokkos {
namespace Impl {

// cuTile extents use uint32_t by CTAD. View extents/strides are size_t.
// Host path aborts on overflow; device/tile path narrows (sizes must fit).
__tile__ __host__ __device__ inline std::uint32_t to_cutile_index(
    std::size_t value) {
#ifndef __CUDA_ARCH__
  constexpr std::size_t max_u32 = static_cast<std::size_t>(UINT32_MAX);
  if (value > max_u32) {
    Kokkos::abort(
        "Kokkos::make_tensor_span: extent/stride exceeds uint32_t range");
  }
#endif
  return static_cast<std::uint32_t>(value);
}

template <class ViewType>
constexpr bool is_cutile_tensor_span_memory_space_v =
    std::is_same_v<typename ViewType::memory_space, CudaSpace> ||
    std::is_same_v<typename ViewType::memory_space, CudaUVMSpace>;

template <class ViewType>
constexpr bool is_cutile_tensor_span_layout_v =
    std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
    std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
    std::is_same_v<typename ViewType::array_layout, LayoutStride>;

template <class T, class Extents, class Strides>
__tile__ __host__ __device__ auto make_tensor_span_strided(T* ptr,
                                                          Extents const& ex,
                                                          Strides const& st) {
  cuda::tiles::layout_strided_mapping mapping{ex, st};
  return cuda::tiles::tensor_span{ptr, mapping};
}

template <class ViewType, std::size_t... Is>
__tile__ __host__ __device__ auto make_cutile_extents(
    ViewType const& view, std::index_sequence<Is...>) {
  return cuda::tiles::extents{to_cutile_index(view.extent(Is))...};
}

template <class ViewType, std::size_t... Is>
__tile__ __host__ __device__ auto make_cutile_strides(
    ViewType const& view, std::index_sequence<Is...>) {
  return cuda::tiles::extents{to_cutile_index(view.stride(Is))...};
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// Tile-callable overloads (pointer + layout + extents).
//
// View::data / extent / stride are only __host__ __device__, so they cannot be
// called from __tile__ code. Tile kernels capture View::data() and extents on
// the host into a trivially copyable functor, then call these overloads.
// ---------------------------------------------------------------------------

template <class T, class... SizeTypes>
__tile__ __host__ __device__ auto make_tensor_span(T* ptr, LayoutLeft,
                                                   SizeTypes... sizes) {
  auto extents = cuda::tiles::extents{
      Impl::to_cutile_index(static_cast<std::size_t>(sizes))...};
  return cuda::tiles::tensor_span{ptr, extents, cuda::tiles::layout_left{}};
}

template <class T, class... SizeTypes>
__tile__ __host__ __device__ auto make_tensor_span(T* ptr, LayoutRight,
                                                   SizeTypes... sizes) {
  auto extents = cuda::tiles::extents{
      Impl::to_cutile_index(static_cast<std::size_t>(sizes))...};
  return cuda::tiles::tensor_span{ptr, extents, cuda::tiles::layout_right{}};
}

template <class T, class Extents, class Strides>
__tile__ __host__ __device__ auto make_tensor_span(T* ptr, LayoutStride,
                                                   Extents const& extents,
                                                   Strides const& strides) {
  return Impl::make_tensor_span_strided(ptr, extents, strides);
}

// ---------------------------------------------------------------------------
// View overload, directly callable from __tile__ kernels.
//
// The View functions this depends on (default/copy/move construction,
// destruction, extent(), stride(), data()) are marked __tile__ so a View
// held by value in a trivially copyable functor can be used to build a
// tensor_span from inside the tile kernel body, without pre-extracting
// pointer + extents on the host.
// ---------------------------------------------------------------------------

template <class ViewType>
__tile__ __host__ __device__ auto make_tensor_span(ViewType const& view) {
  static_assert(is_view_v<ViewType>,
                "Kokkos::make_tensor_span requires a Kokkos::View");
  static_assert(Impl::is_cutile_tensor_span_layout_v<ViewType>,
                "Kokkos::make_tensor_span requires LayoutLeft, LayoutRight, "
                "or LayoutStride");
  static_assert(Impl::is_cutile_tensor_span_memory_space_v<ViewType>,
                "Kokkos::make_tensor_span requires CudaSpace or CudaUVMSpace");
  static_assert(ViewType::rank() >= 1 &&
                    ViewType::rank() <= ARRAY_LAYOUT_MAX_RANK,
                "Kokkos::make_tensor_span rank must be in [1, 8]");

  using layout_type          = typename ViewType::array_layout;
  constexpr std::size_t rank = ViewType::rank();
  auto extents =
      Impl::make_cutile_extents(view, std::make_index_sequence<rank>{});
  auto* ptr = view.data();

  if constexpr (std::is_same_v<layout_type, LayoutLeft>) {
    return cuda::tiles::tensor_span{ptr, extents, cuda::tiles::layout_left{}};
  } else if constexpr (std::is_same_v<layout_type, LayoutRight>) {
    return cuda::tiles::tensor_span{ptr, extents, cuda::tiles::layout_right{}};
  } else {
    static_assert(std::is_same_v<layout_type, LayoutStride>);
    auto strides =
        Impl::make_cutile_strides(view, std::make_index_sequence<rank>{});
    return Impl::make_tensor_span_strided(ptr, extents, strides);
  }
}

}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA && KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUTILE_TENSOR_SPAN_HPP
