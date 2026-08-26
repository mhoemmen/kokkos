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

// Builds an instance of TargetExtents (a cuda::tiles::extents
// specialization, possibly with some static dimensions) matching a View's
// rank, from the View's runtime extents.  This works regardless of which
// of TargetExtents's dimensions are static, because
// cuda::tiles::extents's constructor accepts either exactly
// rank_dynamic() values (one per dynamic dimension) or exactly rank()
// values (one per dimension, ignoring the ones at static positions); we
// always supply rank() values here.
template <class TargetExtents, class ViewType, size_t... Ranks>
KOKKOS_INLINE_FUNCTION TargetExtents cuda_tile_extents_from_view(
    ViewType const& view, std::index_sequence<Ranks...>) {
  return TargetExtents(
      static_cast<typename TargetExtents::index_type>(view.extent(Ranks))...);
}

// Same as cuda_tile_extents_from_view, but pulls values from the View's
// runtime strides instead of its extents.
template <class TargetExtents, class ViewType, size_t... Ranks>
KOKKOS_INLINE_FUNCTION TargetExtents cuda_tile_strides_from_view(
    ViewType const& view, std::index_sequence<Ranks...>) {
  return TargetExtents(
      static_cast<typename TargetExtents::index_type>(view.stride(Ranks))...);
}

// A cuda::tiles::extents specialization with the given rank, all of whose
// dimensions are dynamic.
template <size_t Rank, class Ranks = std::make_index_sequence<Rank>>
struct AllDynamicCudaTileExtents;

template <size_t Rank, size_t... Ranks>
struct AllDynamicCudaTileExtents<Rank, std::index_sequence<Ranks...>> {
  using type = ct::extents<uint32_t, ((void)Ranks, ct::dynamic_extent)...>;
};

}  // namespace Impl

/// \brief Bridges a Kokkos::View, constructed on the host, with cuTile's
///        tensor_span/partition_view so its data can be used from CUDA
///        Tile kernels.
///
/// TileView is templated on \c TileType, a cuda::tiles::tile
/// specialization (e.g. cuda::tiles::tile<float, cuda::tiles::shape<8>>)
/// describing the element type and shape of the register tiles it will be
/// partitioned into, and on \c Extents, a cuda::tiles::extents
/// specialization describing the index space of the whole tensor that
/// gets partitioned into those tiles.
///
/// A TileView is constructed on the host from a Kokkos::View and a
/// cuda::tiles tile shape tag (e.g. cuda::tiles::shape<8>{}); template
/// argument deduction picks TileType and Extents automatically, so
/// callers need not spell out TileView<TileType, Extents> explicitly.
/// It stores
/// a cuda::tiles::tensor_span (a raw device pointer plus the View's
/// runtime extents and strides), which is trivially copyable, so a
/// TileView may be embedded by value in a Driver struct handed to
/// Kokkos::Impl::CudaParallelLaunchTileKernelInvoker (see
/// Kokkos_Cuda_KernelLaunchTile.hpp), which forbids passing struct
/// arguments directly to a __tile_global__ kernel.
///
/// The tensor_span's layout is always cuda::tiles::layout_strided,
/// built from the View's runtime strides, regardless of the View's
/// array_layout (Kokkos::Layout{Left,Right} do not correspond exactly
/// to cuda::tiles::layout_{left,right}).
///
/// Inside a tile kernel body, a TileView converts implicitly to the
/// cuda::tiles::partition_view used to load/store register tiles.
template <class TileType, class Extents>
class TileView {
  static_assert(Impl::ct::tile_shape<typename TileType::shape_type>,
                "Kokkos::TileView: TileType::shape_type must be a "
                "cuda::tiles::tile_shape");
  static_assert(TileType::shape_type::rank() == Extents::rank(),
                "Kokkos::TileView: TileType's shape rank must match "
                "Extents rank");

 public:
  using tile_type       = TileType;
  using element_type    = typename TileType::element_type;
  using value_type      = std::remove_cv_t<element_type>;
  using tile_shape_type = typename TileType::shape_type;
  using extents_type    = Extents;
  using strides_type    = typename Impl::AllDynamicCudaTileExtents<
      extents_type::rank()>::type;
  using layout_type  = Impl::ct::layout_strided<strides_type>;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using span_type =
      Impl::ct::tensor_span<element_type, extents_type, layout_type>;
  using partition_view_type =
      Impl::ct::partition_view<span_type, tile_shape_type>;

  // Construct from a Kokkos::View and a tile shape tag on the host.  The
  // shape argument's type must match tile_shape_type; its value is
  // unused (cuda::tiles shapes are stateless tags), but the parameter is
  // required so that TileView's template arguments can be deduced via
  // CTAD -- see the deduction guide below.
  template <class ViewType>
  TileView(ViewType const& view, tile_shape_type) noexcept
      : span_(view.data(),
              mapping_type(
                  Impl::cuda_tile_extents_from_view<extents_type>(
                      view, std::make_index_sequence<ViewType::rank()>{}),
                  Impl::cuda_tile_strides_from_view<strides_type>(
                      view, std::make_index_sequence<ViewType::rank()>{}))) {
    static_assert(Kokkos::is_view_v<ViewType>,
                  "Kokkos::TileView requires a Kokkos::View");
    static_assert(static_cast<size_t>(ViewType::rank()) ==
                      extents_type::rank(),
                  "Kokkos::TileView: View rank must match Extents rank");
    static_assert(
        std::is_same_v<typename ViewType::non_const_value_type, value_type>,
        "Kokkos::TileView: View's value_type must match TileType's "
        "element_type");
  }

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  span_type span() const noexcept { return span_; }

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  tile_shape_type shape() const noexcept { return tile_shape_type{}; }

  // Builds the cuda::tiles::partition_view used inside a tile kernel body
  // to load/store register tiles.
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  operator partition_view_type() const noexcept {
    return partition_view_type(span_, tile_shape_type{});
  }

 private:
  span_type span_;
};

// Deduces TileView's template arguments from a Kokkos::View and a
// cuda::tiles tile shape tag, so that callers can write, e.g.,
// Kokkos::TileView(view, cuda::tiles::shape<8>{}) without spelling out
// TileView<TileType, Extents> explicitly.  This can't be synthesized from
// the constructor alone, since neither TileType nor Extents appears
// directly among the constructor's parameter types.
template <class ViewType, class TileShape>
TileView(ViewType const&, TileShape) -> TileView<
    Impl::ct::tile<typename ViewType::non_const_value_type, TileShape>,
    typename Impl::AllDynamicCudaTileExtents<ViewType::rank()>::type>;

}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUDA_TILEVIEW_HPP
