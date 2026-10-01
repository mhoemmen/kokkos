// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_UNITTEST_MDSPAN_HPP
#define KOKKOS_UNITTEST_MDSPAN_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <gtest/gtest.h>

namespace {
void test_mdspan_minimal_functional() {
  int N = 100;
  Kokkos::View<int*, TEST_EXECSPACE> a("A", N);
  Kokkos::parallel_for(
      "FillSequence", Kokkos::RangePolicy<TEST_EXECSPACE>(0, N),
      KOKKOS_LAMBDA(int i) { a(i) = i; });

  Kokkos::mdspan<int, Kokkos::dextents<int, 1>> a_mds(a.data(), N);
  int errors;
  Kokkos::parallel_reduce(
      "CheckMinimalMDSpan", Kokkos::RangePolicy<TEST_EXECSPACE>(0, N),
      KOKKOS_LAMBDA(int i, int& err) {
        Kokkos::mdspan<int, Kokkos::dextents<int, 1>> b_mds(a.data(), N);
        if (a_mds(i) != i) err++;
        if (b_mds(i) != i) err++;
      },
      errors);
  ASSERT_EQ(errors, 0);
}
}  // namespace

namespace {

TEST(TEST_CATEGORY, mdspan_minimal_functional) {
#if defined(KOKKOS_ENABLE_OPENACC) && (KOKKOS_COMPILER_NVHPC > 240500) && \
    (KOKKOS_COMPILER_NVHPC <= 260500)
  // FIXME_OPENACC: Test is known to fail for 24.5 < NVHPC version <= 26.5.
  GTEST_SKIP() << "skipping since the OpenACC backend fails to compile this "
                  "test if 24.5 < NVHPC version <= 26.5";
#else
  test_mdspan_minimal_functional();
#endif
}

}  // namespace

#endif
