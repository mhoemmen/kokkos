// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUDA_TILEVIEW_HPP
#define KOKKOS_CUDA_TILEVIEW_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_CUDA_TILE

#include <cuda_tile.h>

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_TILEVIEW
#endif

#include <Kokkos_Layout.hpp>
#include <Kokkos_View.hpp>

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_TILEVIEW
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_TILEVIEW
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace Kokkos {
namespace Impl {

namespace ct = cuda::tiles;

// Builds a cuda::tiles::extents object matching a View's rank, with every
// dimension dynamic, filled in from the View's runtime extents.
template <class ViewType, size_t... Ranks>
KOKKOS_INLINE_FUNCTION auto cuda_tile_extents_from_view(
    ViewType const& view, std::index_sequence<Ranks...>) {
  return ct::extents<uint32_t, ((void)Ranks, ct::dynamic_extent)...>(
      static_cast<uint32_t>(view.extent(Ranks))...);
}

// Builds a cuda::tiles::extents object (used as a strides descriptor)
// matching a View's rank, with every dimension dynamic, filled in from the
// View's runtime strides.
template <class ViewType, size_t... Ranks>
KOKKOS_INLINE_FUNCTION auto cuda_tile_strides_from_view(
    ViewType const& view, std::index_sequence<Ranks...>) {
  return ct::extents<uint32_t, ((void)Ranks, ct::dynamic_extent)...>(
      static_cast<uint32_t>(view.stride(Ranks))...);
}

}  // namespace Impl

/// \brief Bridges a Kokkos::View, constructed on the host, with cuTile's
///        tensor_span/partition_view so its data can be used from CUDA
///        Tile kernels.
///
/// A TileView is constructed on the host from a Kokkos::View and a
/// compile-time tile shape.  It stores a cuda::tiles::tensor_span (a raw
/// device pointer plus the View's runtime extents and strides) and a
/// cuda::tiles::shape tag describing how to partition the tensor into
/// register tiles.  The tensor_span's layout is always
/// cuda::tiles::layout_strided, built from the View's runtime strides,
/// regardless of the View's array_layout (Kokkos::LayoutLeft,
/// Kokkos::LayoutRight, and Kokkos::LayoutStride do not correspond to
/// cuda::tiles::layout_left/layout_right).  Both are trivially copyable,
/// so a TileView may be
/// embedded by value in a Driver struct handed to
/// Kokkos::Impl::CudaParallelLaunchTileKernelInvoker (see
/// Kokkos_Cuda_KernelLaunchTile.hpp), which forbids passing struct
/// arguments directly to a __tile_global__ kernel.
///
/// Inside the Driver's operator(), TileView::partition() builds the actual
/// cuda::tiles::partition_view used to load/store register tiles.
template <class ViewType, class TileShape>
class TileView {
  static_assert(Kokkos::is_view_v<ViewType>,
                "Kokkos::TileView requires a Kokkos::View");
  static_assert(TileShape::rank() == ViewType::rank(),
                "Kokkos::TileView: TileShape rank must match View rank");

 public:
  using view_type       = ViewType;
  using element_type    = typename ViewType::value_type;
  using value_type      = typename ViewType::non_const_value_type;
  using tile_shape_type = TileShape;
  using extents_type    = decltype(Impl::cuda_tile_extents_from_view(
      std::declval<ViewType const&>(),
      std::make_index_sequence<ViewType::rank()>{}));
  using strides_type = decltype(Impl::cuda_tile_strides_from_view(
      std::declval<ViewType const&>(),
      std::make_index_sequence<ViewType::rank()>{}));
  using layout_type  = Impl::ct::layout_strided<strides_type>;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using span_type =
      Impl::ct::tensor_span<element_type, extents_type, layout_type>;
  using partition_view_type =
      Impl::ct::partition_view<span_type, TileShape>;
  using data_handle_type = typename span_type::data_handle_type;

  TileView() = default;

  // Construct from a Kokkos::View on the host.
  TileView(ViewType const& view, tile_shape_type shape = {})
      : span_(view.data(),
              mapping_type(
                  Impl::cuda_tile_extents_from_view(
                      view, std::make_index_sequence<ViewType::rank()>{}),
                  Impl::cuda_tile_strides_from_view(
                      view, std::make_index_sequence<ViewType::rank()>{}))),
        shape_(shape) {}

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  span_type span() const noexcept { return span_; }

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  tile_shape_type shape() const noexcept { return shape_; }

  // Builds the cuda::tiles::partition_view used inside a tile kernel body
  // to load/store register tiles.
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  operator partition_view_type() const noexcept {
    return partition_view_type(span_, shape_);
  }

 private:
  span_type span_;
  tile_shape_type shape_;
};

}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUDA_TILEVIEW_HPP
