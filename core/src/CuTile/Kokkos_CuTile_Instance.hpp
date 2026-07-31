// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUTILE_INSTANCE_HPP
#define KOKKOS_CUTILE_INSTANCE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE)

#include <Cuda/Kokkos_Cuda_Error.hpp>
#include <impl/Kokkos_HostSharedPtr.hpp>
#include <impl/Kokkos_Tools.hpp>

#include <cuda_runtime_api.h>

#include <iosfwd>
#include <string>

namespace Kokkos {
class CuTile;
namespace Impl {

class CuTileInternal {
 public:
  int m_cudaDev         = -1;
  cudaStream_t m_stream = nullptr;
  uint32_t m_instance_id =
      Kokkos::Tools::Experimental::Impl::idForInstance<Kokkos::CuTile>(
          reinterpret_cast<uintptr_t>(this));

  static HostSharedPtr<CuTileInternal> default_instance;

  explicit CuTileInternal(cudaStream_t stream);
  ~CuTileInternal() = default;

  CuTileInternal(const CuTileInternal&)            = delete;
  CuTileInternal& operator=(const CuTileInternal&) = delete;

  void fence(const std::string& name) const;
  void fence() const;
  uint32_t impl_get_instance_id() const { return m_instance_id; }
  void print_configuration(std::ostream& os) const;
  int verify_is_initialized(const char* const label) const;
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA && KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUTILE_INSTANCE_HPP
