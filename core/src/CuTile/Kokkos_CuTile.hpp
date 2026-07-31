// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_CUTILE_HPP
#define KOKKOS_CUTILE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE)

#include <Kokkos_Core_fwd.hpp>

#include <iosfwd>

#include <Cuda/Kokkos_Cuda.hpp>  // ManageStream, CudaSpace
#include <Cuda/Kokkos_CudaSpace.hpp>
#include <impl/Kokkos_HostSharedPtr.hpp>
#include <impl/Kokkos_InitializationSettings.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Layout.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {
class CuTileInternal;
}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

/// \class CuTile
/// \brief Kokkos execution space that launches NVIDIA cuTile (__tile_global__)
///        kernels on a CUDA stream.
class CuTile {
 public:
  using execution_space = CuTile;
  using memory_space    = CudaSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using index_type      = memory_space::index_type;
  using size_type       = memory_space::size_type;
  using array_layout    = LayoutLeft;
  using scratch_memory_space = ScratchMemorySpace<CuTile>;

  static void impl_static_fence(const std::string& name);

  void fence(const std::string& name =
                 "Kokkos::CuTile::fence(): Unnamed Instance Fence") const;

  int concurrency() const;

  void print_configuration(std::ostream& os, bool verbose = false) const;

  KOKKOS_DEFAULTED_FUNCTION CuTile(const CuTile&) = default;
  KOKKOS_FUNCTION CuTile(CuTile&& other) noexcept
      : CuTile(static_cast<const CuTile&>(other)) {}
  KOKKOS_DEFAULTED_FUNCTION CuTile& operator=(const CuTile&) = default;
  KOKKOS_FUNCTION CuTile& operator=(CuTile&& other) noexcept {
    return *this = static_cast<const CuTile&>(other);
  }
  CuTile();

  KOKKOS_FUNCTION ~CuTile() {
    KOKKOS_IF_ON_HOST(
        (Impl::check_execution_space_destructor_precondition(name());))
  }

  explicit CuTile(cudaStream_t stream)
      : CuTile(stream, Impl::ManageStream::no) {}

  CuTile(cudaStream_t stream, Impl::ManageStream manage_stream);

  static void impl_finalize();
  static void impl_initialize(InitializationSettings const&);

  cudaStream_t cuda_stream() const;
  int cuda_device() const;

  static const char* name();

  inline Impl::CuTileInternal* impl_internal_space_instance() const {
    return m_space_instance.get();
  }
  uint32_t impl_instance_id() const noexcept;

 private:
  friend bool operator==(CuTile const& lhs, CuTile const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(CuTile const& lhs, CuTile const& rhs) {
    return !(lhs == rhs);
  }
  Kokkos::Impl::HostSharedPtr<Impl::CuTileInternal> m_space_instance;
};

namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<CuTile> {
  // Reuse Cuda tool device type for this initial integration.
  static constexpr DeviceType id = DeviceType::Cuda;
  static int device_id(const CuTile& exec) { return exec.cuda_device(); }
};
}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace,
                         Kokkos::CuTile::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
};

}  // namespace Impl
}  // namespace Kokkos

#endif /* #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE) */
#endif /* #ifndef KOKKOS_CUTILE_HPP */
