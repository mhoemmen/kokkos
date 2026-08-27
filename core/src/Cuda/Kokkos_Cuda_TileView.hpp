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

/// \brief Applies a tile partitioning to a Kokkos::View
///
/// \tparam TileType cuda::tiles::tile specialization describing the
///   element type and shape of the tiles into which the View will be
///   partitioned
///
/// \tparam Extents cuda::tiles::extents specialization describing the
///    "tile index space type," that is, the type of the index space
///    over which tile loads and stores iterate
///
/// Idiomatic construction happens on host from a Kokkos::View (the
/// array to partition) and a cuda::tiles::shape (the shape of each
/// tile in the partition).  The resulting object can be cudaMemcpy'd
/// to device for use in a Tile kernel.  There, it behaves like a
/// cuda::tiles::partition_view.
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

  // The remaining public type aliases mirror
  // cuda::tiles::partition_view's, so that TileView can be used
  // anywhere a partition_view would be, both for type introspection and
  // for the member functions below.
  using index_type      = typename partition_view_type::index_type;
  using rank_type       = typename partition_view_type::rank_type;
  using view_shape_type = typename partition_view_type::view_shape_type;
  using view_tile_type  = typename partition_view_type::view_tile_type;

  // Construct from a Kokkos::View and a tile shape on the host.
  template <class ViewType>
    requires(Kokkos::is_view_v<ViewType>)
  TileView(ViewType const& view, tile_shape_type) noexcept
      : span_(view.data(),
              mapping_type(
                  Impl::cuda_tile_extents_from_view<extents_type>(
                      view, std::make_index_sequence<ViewType::rank()>{}),
                  Impl::cuda_tile_strides_from_view<strides_type>(
                      view, std::make_index_sequence<ViewType::rank()>{}))) {
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

  /// \brief Convert this object into a partition_view.
  ///
  /// Use this only if you need to call an interface that expects a
  /// partition_view.  Otherwise, just call load and store directly on
  /// this object.
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  operator partition_view_type() const noexcept {
    return partition_view_type(span_, tile_shape_type{});
  }

  // The remaining member functions mirror cuda::tiles::partition_view's
  // load/store interface (same names, template parameters, and function
  // parameters), each forwarding to the partition_view built by the
  // conversion operator above.  This lets a TileView be used directly in
  // a tile kernel body wherever a partition_view would be.
  //
  // TileView doesn't yet implement masked loads or stores.  For now,
  // we would like to explore an interface that only exposes in-bounds
  // loads and stores.  On the other hand, masking is useful for
  // reasons other than simplifying boundary handling.
  //
  // TileView doesn't yet implement atomic loads or stores.

  template <class... Idx>
    requires (sizeof...(Idx) == extents_type::rank())
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION view_tile_type
  load(Idx... idx) const noexcept {
    return static_cast<partition_view_type>(*this).load(idx...);
  }

  template <Impl::ct::tile_like Value, class... Idx>
    requires (
      sizeof...(Idx) == extents_type::rank()
      // && is_convertible_v<Value, view_tile_type>
    )
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION void
  store(Value tile_value, Idx... idx) const noexcept {
    static_cast<partition_view_type>(*this).store(tile_value, idx...);
  }

 private:
  span_type span_;
};

// TileView needs a deduction guide because the constructor's
// parameters don't have the same types as its template parameters.
template <class ViewType, class TileShape>
TileView(ViewType const&, TileShape) -> TileView<
    Impl::ct::tile<typename ViewType::non_const_value_type, TileShape>,
    typename Impl::AllDynamicCudaTileExtents<ViewType::rank()>::type>;

}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUDA_TILEVIEW_HPP
