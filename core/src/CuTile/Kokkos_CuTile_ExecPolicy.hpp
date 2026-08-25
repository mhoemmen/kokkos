// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_CUTILE_EXEC_POLICY_HPP
#define KOKKOS_CUTILE_EXEC_POLICY_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE)

#include <Kokkos_ExecPolicy.hpp>

namespace Kokkos {

/// \class TileRangePolicy
/// \brief Range policy whose iterations are dispatched one per cuTile
///        (__tile_global__) tile block, rather than one per SIMT thread.
///
/// Structurally mirrors Kokkos::RangePolicy (same backing
/// Impl::ImplRangePolicy) rather than deriving from it, because
/// Impl::ParallelFor dispatch matches on the exact policy template-id: a
/// class deriving from RangePolicy<Traits...> would still resolve to
/// RangePolicy's own ParallelFor specialization, not one written for
/// TileRangePolicy.
template <typename... Properties>
class TileRangePolicy
    : public Impl::ImplRangePolicy<
          typename Impl::PolicyTraits<Properties...>::execution_type,
          Properties...> {
 public:
  using execution_type =
      typename Impl::PolicyTraits<Properties...>::execution_type;
  using base_t = Impl::ImplRangePolicy<execution_type, Properties...>;
  using base_t::base_t;

  // Shadow ImplRangePolicy::execution_policy (which names
  // Kokkos::RangePolicy<Properties...>, not this class): the
  // Kokkos::ExecutionPolicy concept requires T::execution_policy to be (or be
  // a base of) T itself, and RangePolicy is not a base of TileRangePolicy.
  using execution_policy = TileRangePolicy<Properties...>;
};

}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA && KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUTILE_EXEC_POLICY_HPP
