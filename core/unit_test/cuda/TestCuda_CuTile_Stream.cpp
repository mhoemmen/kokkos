// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <TestCuda_Category.hpp>

namespace Test {

TEST(cuda, cutile_holds_stream) {
  cudaStream_t stream;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));

  {
    Kokkos::CuTile exec(stream);
    ASSERT_EQ(exec.cuda_stream(), stream);
    exec.fence();
  }

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
}

TEST(cuda, cutile_default_stream) {
  Kokkos::CuTile exec;
  ASSERT_NE(exec.cuda_stream(), static_cast<cudaStream_t>(nullptr));
  exec.fence();
}

TEST(cuda, cutile_managed_streams_are_distinct) {
  cudaStream_t s0;
  cudaStream_t s1;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&s0));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&s1));

  Kokkos::CuTile exec0(s0, Kokkos::Impl::ManageStream::yes);
  Kokkos::CuTile exec1(s1, Kokkos::Impl::ManageStream::yes);

  ASSERT_EQ(exec0.cuda_stream(), s0);
  ASSERT_EQ(exec1.cuda_stream(), s1);
  ASSERT_NE(exec0.cuda_stream(), exec1.cuda_stream());
  // Streams are destroyed when exec0/exec1 go out of scope.
}

}  // namespace Test
