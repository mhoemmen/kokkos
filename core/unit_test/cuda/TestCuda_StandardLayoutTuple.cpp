// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <Kokkos_StandardLayoutTuple.hpp>
#include <TestCuda_Category.hpp>

#include <cstring>
#include <type_traits>

namespace Test {

TEST(cuda, standard_layout_tuple_host_memcpy) {
  using Tuple = Kokkos::standard_layout_tuple<int, double, float*>;
  static_assert(std::is_standard_layout_v<Tuple>);
  static_assert(std::is_trivially_copyable_v<Tuple>);

  float f = 1.25f;
  Tuple src{17, 3.5, &f};
  Tuple dst{};
  std::memcpy(&dst, &src, sizeof(Tuple));

  ASSERT_EQ(Kokkos::get<0>(dst), 17);
  ASSERT_EQ(Kokkos::get<1>(dst), 3.5);
  ASSERT_EQ(Kokkos::get<2>(dst), &f);
  ASSERT_FLOAT_EQ(*Kokkos::get<2>(dst), 1.25f);
}

TEST(cuda, standard_layout_tuple_device_memcpy) {
  using Tuple = Kokkos::standard_layout_tuple<int, float>;
  static_assert(std::is_standard_layout_v<Tuple>);
  static_assert(std::is_trivially_copyable_v<Tuple>);

  Tuple h_in{9, 1.5f};
  Tuple* d_ptr = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_ptr, sizeof(Tuple)));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(d_ptr, &h_in, sizeof(Tuple), cudaMemcpyHostToDevice));

  Tuple h_out{};
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(&h_out, d_ptr, sizeof(Tuple), cudaMemcpyDeviceToHost));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_ptr));

  ASSERT_EQ(Kokkos::get<0>(h_out), 9);
  ASSERT_FLOAT_EQ(Kokkos::get<1>(h_out), 1.5f);
}

}  // namespace Test
