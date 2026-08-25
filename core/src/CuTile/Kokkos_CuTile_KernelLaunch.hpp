// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUTILE_KERNEL_LAUNCH_HPP
#define KOKKOS_CUTILE_KERNEL_LAUNCH_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE)

#include <Cuda/Kokkos_Cuda_Error.hpp>

#include <cuda_tile.h>

#include <type_traits>

namespace Kokkos {
namespace Impl {

// cuTile currently only allows __tile_global__ parameters that are pointers or
// arithmetic types. Pack the driver on the host, cudaMemcpyAsync it to device,
// and launch with a device pointer argument only.

// Kokkos::View is never std::is_trivially_copyable when built with nvcc: its
// copy/move constructors are user-provided to work around an nvcc overload
// ambiguity bug (KOKKOS_IMPL_VIEW_HOOKS_NVCC_WORKAROUND in Kokkos_View.hpp),
// not because a View's members are actually unsafe to relocate by raw byte
// copy. A driver holding Views (or plain aggregates) is standard-layout,
// which is the property this raw cudaMemcpyAsync-based transfer actually
// depends on; it never invokes Driver's copy/move constructor or destructor
// on the transferred bytes.
template <class Driver>
__tile_global__ void cutile_parallel_for_kernel(Driver const* driver) {
  driver->operator()();
}

template <class Driver>
struct CuTileParallelLaunch {
  static_assert(std::is_standard_layout_v<Driver>,
                "CuTile parallel_for driver must be standard layout");

  CuTileParallelLaunch(Driver const& driver, dim3 const& grid,
                       cudaStream_t stream) {
    Driver* d_driver = nullptr;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_driver, sizeof(Driver)));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpyAsync(
        d_driver, &driver, sizeof(Driver), cudaMemcpyHostToDevice, stream));

    // Tile kernels must launch with block dimension 1.
    cutile_parallel_for_kernel<Driver><<<grid, dim3(1, 1, 1), 0, stream>>>(
        d_driver);
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetLastError());

    // Free after the stream completes this launch. A later refinement can use
    // stream-ordered allocation to avoid the synchronize.
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_driver));
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA && KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUTILE_KERNEL_LAUNCH_HPP
