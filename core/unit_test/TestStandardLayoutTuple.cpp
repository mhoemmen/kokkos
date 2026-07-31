// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_StandardLayoutTuple.hpp>

#include <cstring>
#include <tuple>
#include <type_traits>

namespace {

template <typename... Ts>
KOKKOS_FUNCTION constexpr void maybe_unused(Ts&&...) {}

KOKKOS_FUNCTION constexpr bool test_empty_tuple() {
  Kokkos::standard_layout_tuple<> t{};
  maybe_unused(t);
  static_assert(std::tuple_size_v<Kokkos::standard_layout_tuple<>> == 0);
  static_assert(std::is_standard_layout_v<Kokkos::standard_layout_tuple<>>);
  static_assert(std::is_trivially_copyable_v<Kokkos::standard_layout_tuple<>>);
  return true;
}
static_assert(test_empty_tuple());

KOKKOS_FUNCTION constexpr bool test_single_element() {
  Kokkos::standard_layout_tuple<int> t{42};
  if (Kokkos::get<0>(t) != 42) return false;
  Kokkos::get<0>(t) = 7;
  if (Kokkos::get<0>(t) != 7) return false;
  static_assert(std::tuple_size_v<Kokkos::standard_layout_tuple<int>> == 1);
  static_assert(std::is_same_v<
                std::tuple_element_t<0, Kokkos::standard_layout_tuple<int>>,
                int>);
  static_assert(std::is_standard_layout_v<Kokkos::standard_layout_tuple<int>>);
  static_assert(
      std::is_trivially_copyable_v<Kokkos::standard_layout_tuple<int>>);
  return true;
}
static_assert(test_single_element());

KOKKOS_FUNCTION constexpr bool test_multi_element() {
  auto t = Kokkos::make_standard_layout_tuple(1, 2.5f, true);
  if (Kokkos::get<0>(t) != 1) return false;
  if (Kokkos::get<1>(t) != 2.5f) return false;
  if (Kokkos::get<2>(t) != true) return false;

  static_assert(
      std::tuple_size_v<Kokkos::standard_layout_tuple<int, float, bool>> == 3);
  static_assert(std::is_standard_layout_v<
                Kokkos::standard_layout_tuple<int, float, bool>>);
  static_assert(std::is_trivially_copyable_v<
                Kokkos::standard_layout_tuple<int, float, bool>>);
  static_assert(std::is_trivially_copyable_v<
                Kokkos::standard_layout_tuple<int*, double, std::size_t>>);
  return true;
}
static_assert(test_multi_element());

// Contrast with std::tuple, which is not required to be (and typically is not)
// standard-layout or trivially copyable.
static_assert(!std::is_trivially_copyable_v<std::tuple<int>>);

KOKKOS_FUNCTION constexpr bool test_pointer_and_arithmetic() {
  int x = 11;
  auto t =
      Kokkos::make_standard_layout_tuple(static_cast<int*>(&x), std::size_t{3});
  if (Kokkos::get<0>(t) != &x) return false;
  if (Kokkos::get<1>(t) != 3) return false;
  if (*Kokkos::get<0>(t) != 11) return false;
  return true;
}
static_assert(test_pointer_and_arithmetic());

// Host memcpy round-trip (runtime, exercised by the dedicated CUDA gtest).
[[maybe_unused]] bool test_host_memcpy_roundtrip() {
  using Tuple = Kokkos::standard_layout_tuple<int, double, float*>;
  float f     = 1.25f;
  Tuple src{17, 3.5, &f};
  Tuple dst{};
  std::memcpy(&dst, &src, sizeof(Tuple));
  return Kokkos::get<0>(dst) == 17 && Kokkos::get<1>(dst) == 3.5 &&
         Kokkos::get<2>(dst) == &f && *Kokkos::get<2>(dst) == 1.25f;
}

}  // namespace
