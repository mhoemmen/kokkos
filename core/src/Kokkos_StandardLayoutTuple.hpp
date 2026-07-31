// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// Adapted from CUTLASS cute::tuple (BSD-3-Clause, NVIDIA Corporation & Affiliates).
// Original: include/cute/container/tuple.hpp in https://github.com/NVIDIA/cutlass
//
// Kokkos::standard_layout_tuple is like std::tuple, with differences:
//
// 1. It works on both host and device.
// 2. Its template arguments must be semiregular types (no references).
// 3. It is always a standard-layout type if all of its template arguments are
//    standard-layout types.
// 4. It is always an empty type if all of its template arguments are empty types.
// 5. When all template arguments are trivially copyable, the tuple itself is
//    trivially copyable (safe to cudaMemcpyAsync to device).
//
// The empty-structure optimization (ESO) implementation is adapted from CuTe
// rather than classic EBO inheritance, so standard-layout / trivial-copyability
// are preserved across host-device boundaries.

#ifndef KOKKOS_STANDARD_LAYOUT_TUPLE_HPP
#define KOKKOS_STANDARD_LAYOUT_TUPLE_HPP

#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace Kokkos {
namespace Impl {
namespace standard_layout_tuple_detail {

template <std::size_t I, class... T>
struct type_at;

template <class T0, class... Ts>
struct type_at<0, T0, Ts...> {
  using type = T0;
};

template <std::size_t I, class T0, class... Ts>
struct type_at<I, T0, Ts...> {
  using type = typename type_at<I - 1, Ts...>::type;
};

template <std::size_t I, class... T>
using type_at_t = typename type_at<I, T...>::type;

// ESO stands for "empty structure optimization."
// Empty types in the template argument list are not stored and are not
// constructed; get() constructs and returns an instance on demand.

template <bool IsFirstEmpty, bool IsRestEmpty, class... T>
struct ESO;

template <class First, class... Rest>
inline constexpr bool is_first_empty_v = std::is_empty_v<First>;

template <class First, class... Rest>
inline constexpr bool is_rest_empty_v = (std::is_empty_v<Rest> && ...);

template <class... T>
using ESO_t = ESO<is_first_empty_v<T...>, is_rest_empty_v<T...>, T...>;

// Empty First and Empty Rest...
template <class First, class... Rest>
struct ESO<true, true, First, Rest...> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO() {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO(First const&, Rest const&...) {}
};

// NonEmpty First and Empty Rest...
template <class First, class... Rest>
struct ESO<false, true, First, Rest...> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO() : first_{} {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO(First const& first, Rest const&...)
      : first_{first} {}

  First first_;
};

// Empty First and NonEmpty Rest...
template <class First, class... Rest>
struct ESO<true, false, First, Rest...> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO() : rest_{} {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO(First const&, Rest const&... rest)
      : rest_{rest...} {}

  ESO_t<Rest...> rest_;
};

// NonEmpty First and NonEmpty Rest...
template <class First, class... Rest>
struct ESO<false, false, First, Rest...> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO() : first_{}, rest_{} {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr ESO(First const& first,
                                            Rest const&... rest)
      : first_{first}, rest_{rest...} {}

  First first_;
  ESO_t<Rest...> rest_;
};

template <class R, std::size_t N, class S>
KOKKOS_FORCEINLINE_FUNCTION constexpr R getr(S&& s) noexcept {
  if constexpr (N == 0) {
    return static_cast<S&&>(s).first_;
  } else {
    return getr<R, N - 1>(static_cast<S&&>(s).rest_);
  }
}

template <std::size_t N, bool F, bool R, class... T>
KOKKOS_FORCEINLINE_FUNCTION constexpr std::conditional_t<
    std::is_empty_v<type_at_t<N, T...>>, type_at_t<N, T...>,
    type_at_t<N, T...> const&>
getv_cr(ESO<F, R, T...> const& s) noexcept {
  if constexpr (std::is_empty_v<type_at_t<N, T...>>) {
    return {};
  } else {
    return getr<type_at_t<N, T...> const&, N>(s);
  }
}

template <std::size_t N, bool F, bool R, class... T>
KOKKOS_FORCEINLINE_FUNCTION constexpr std::conditional_t<
    std::is_empty_v<type_at_t<N, T...>>, type_at_t<N, T...>, type_at_t<N, T...>&>
getv_r(ESO<F, R, T...>& s) noexcept {
  if constexpr (std::is_empty_v<type_at_t<N, T...>>) {
    return {};
  } else {
    return getr<type_at_t<N, T...>&, N>(s);
  }
}

template <std::size_t N, bool F, bool R, class... T>
KOKKOS_FORCEINLINE_FUNCTION constexpr std::conditional_t<
    std::is_empty_v<type_at_t<N, T...>>, type_at_t<N, T...>,
    type_at_t<N, T...>&&>
getv_rr(ESO<F, R, T...>&& s) noexcept {
  if constexpr (std::is_empty_v<type_at_t<N, T...>>) {
    return {};
  } else {
    return getr<type_at_t<N, T...>&&, N>(static_cast<ESO<F, R, T...>&&>(s));
  }
}

}  // namespace standard_layout_tuple_detail
}  // namespace Impl

template <class... T>
struct standard_layout_tuple
    : Impl::standard_layout_tuple_detail::ESO_t<T...> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr standard_layout_tuple() {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr standard_layout_tuple(T const&... t)
      : Impl::standard_layout_tuple_detail::ESO_t<T...>(t...) {}
};

template <>
struct standard_layout_tuple<> {};

template <class... T>
KOKKOS_FORCEINLINE_FUNCTION constexpr standard_layout_tuple<T...>
make_standard_layout_tuple(T const&... t) {
  return {t...};
}

template <std::size_t I, class... T>
KOKKOS_FORCEINLINE_FUNCTION constexpr decltype(auto) get(
    standard_layout_tuple<T...> const& t) noexcept {
  static_assert(I < sizeof...(T), "Index out of range");
  return Impl::standard_layout_tuple_detail::getv_cr<I>(t);
}

template <std::size_t I, class... T>
KOKKOS_FORCEINLINE_FUNCTION constexpr decltype(auto) get(
    standard_layout_tuple<T...>& t) noexcept {
  static_assert(I < sizeof...(T), "Index out of range");
  return Impl::standard_layout_tuple_detail::getv_r<I>(t);
}

template <std::size_t I, class... T>
KOKKOS_FORCEINLINE_FUNCTION constexpr decltype(auto) get(
    standard_layout_tuple<T...>&& t) noexcept {
  static_assert(I < sizeof...(T), "Index out of range");
  return Impl::standard_layout_tuple_detail::getv_rr<I>(
      static_cast<standard_layout_tuple<T...>&&>(t));
}

}  // namespace Kokkos

namespace std {

template <class... T>
struct tuple_size<Kokkos::standard_layout_tuple<T...>>
    : std::integral_constant<std::size_t, sizeof...(T)> {};

template <std::size_t I, class... T>
struct tuple_element<I, Kokkos::standard_layout_tuple<T...>> {
  using type =
      typename Kokkos::Impl::standard_layout_tuple_detail::type_at_t<I, T...>;
};

}  // namespace std

#endif  // KOKKOS_STANDARD_LAYOUT_TUPLE_HPP
