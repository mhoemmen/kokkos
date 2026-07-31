// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUTILE_PARALLEL_RANGE_HPP
#define KOKKOS_CUTILE_PARALLEL_RANGE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE)

#include <Kokkos_Parallel.hpp>
#include <CuTile/Kokkos_CuTile.hpp>
#include <CuTile/Kokkos_CuTile_KernelLaunch.hpp>
#include <Kokkos_StandardLayoutTuple.hpp>

#include <cuda_tile.h>

#include <type_traits>

namespace Kokkos {
namespace Impl {

// Must be a namespace-scope type: __tile_global__ templates cannot use
// private nested class types as template arguments.
template <class FunctorType, class Member, class WorkTag>
struct CuTileRangeParallelForDriver {
  FunctorType m_functor;
  Member m_begin;
  Member m_end;

  template <class TagType>
  __tile__ std::enable_if_t<std::is_void_v<TagType>> exec_range(
      Member i) const {
    m_functor(i);
  }

  template <class TagType>
  __tile__ std::enable_if_t<!std::is_void_v<TagType>> exec_range(
      Member i) const {
    m_functor(TagType(), i);
  }

  __tile__ void operator()() const {
    // One iteration index per tile block. Use cuTile bid(), not SIMT blockIdx.
    Member const iwork = m_begin + static_cast<Member>(cuda::tiles::bid().x);
    if (iwork < m_end) {
      this->template exec_range<WorkTag>(iwork);
    }
  }
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::CuTile> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 private:
  using Member  = typename Policy::member_type;
  using WorkTag = typename Policy::work_tag;
  using Driver  = CuTileRangeParallelForDriver<FunctorType, Member, WorkTag>;

  static_assert(std::is_trivially_copyable_v<Driver>,
                "CuTile RangePolicy parallel_for requires a trivially "
                "copyable functor");

  const FunctorType m_functor;
  const Policy m_policy;

 public:
  using functor_type = FunctorType;

  ParallelFor() = delete;

  Policy const& get_policy() const { return m_policy; }

  inline void execute() const {
    Member const nwork = m_policy.end() - m_policy.begin();
    if (nwork <= 0) return;

    // Pack parameters into a standard-layout / trivially copyable driver
    // (cuTile kernel args may only be pointers or arithmetic types).
    auto packed = Kokkos::make_standard_layout_tuple(
        m_functor, m_policy.begin(), m_policy.end());
    Driver driver{Kokkos::get<0>(packed), Kokkos::get<1>(packed),
                  Kokkos::get<2>(packed)};
    dim3 grid(static_cast<unsigned>(nwork), 1, 1);

    CuTileParallelLaunch<Driver>(driver, grid,
                                 m_policy.space().cuda_stream());
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA && KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUTILE_PARALLEL_RANGE_HPP
