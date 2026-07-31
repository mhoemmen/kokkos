// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#ifndef KOKKOS_CUTILE_INSTANCE_CPP_
#define KOKKOS_CUTILE_INSTANCE_CPP_

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_TILE)

#include <Kokkos_Core.hpp>
#include <CuTile/Kokkos_CuTile.hpp>
#include <CuTile/Kokkos_CuTile_Instance.hpp>

#include <impl/Kokkos_DeviceManagement.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>
#include <impl/Kokkos_CheckUsage.hpp>

#include <iostream>

namespace Kokkos {
namespace Impl {

HostSharedPtr<CuTileInternal> CuTileInternal::default_instance;

CuTileInternal::CuTileInternal(cudaStream_t stream) : m_stream(stream) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDevice(&m_cudaDev));
}

void CuTileInternal::fence(const std::string& name) const {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Kokkos::CuTile>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{m_instance_id},
      [stream = m_stream] {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
      });
}

void CuTileInternal::fence() const {
  fence("Kokkos::CuTileInternal::fence(): Unnamed Instance Fence");
}

void CuTileInternal::print_configuration(std::ostream& os) const {
  os << "  CuTile device: " << m_cudaDev << '\n';
  os << "  CuTile stream: " << static_cast<void*>(m_stream) << '\n';
}

int CuTileInternal::verify_is_initialized(const char* const label) const {
  if (m_cudaDev < 0) {
    Kokkos::abort((std::string("Kokkos::CuTile::") + label +
                   " : ERROR device not initialized\n")
                      .c_str());
  }
  return 0 <= m_cudaDev;
}

}  // namespace Impl

void CuTile::impl_initialize(InitializationSettings const& settings) {
  const std::vector<int>& visible_devices = Impl::get_visible_devices();
  const int cuda_device_id =
      Impl::get_gpu(settings).value_or(visible_devices[0]);

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(cuda_device_id));

  cudaStream_t stream;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));

  Impl::CuTileInternal::default_instance = Impl::HostSharedPtr(
      new Impl::CuTileInternal(stream), [](Impl::CuTileInternal* ptr) {
        cudaStream_t s = ptr->m_stream;
        delete ptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(s));
      });
}

void CuTile::impl_finalize() {
  Impl::CuTileInternal::default_instance = nullptr;
}

CuTile::CuTile()
    : m_space_instance(
          (Impl::check_execution_space_constructor_precondition(name()),
           Impl::CuTileInternal::default_instance)) {}

CuTile::CuTile(cudaStream_t stream, Impl::ManageStream manage_stream)
    : m_space_instance(
          (Impl::check_execution_space_constructor_precondition(name()),
           static_cast<bool>(manage_stream)
               ? Impl::HostSharedPtr(
                     new Impl::CuTileInternal(stream),
                     [](Impl::CuTileInternal* ptr) {
                       cudaStream_t s = ptr->m_stream;
                       delete ptr;
                       KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(s));
                     })
               : Impl::HostSharedPtr(new Impl::CuTileInternal(stream)))) {}

void CuTile::print_configuration(std::ostream& os, bool /*verbose*/) const {
  os << "Device Execution Space:\n";
  os << "  KOKKOS_ENABLE_CUDA_TILE: yes\n";
  os << "\nCuTile Runtime Configuration:\n";
  m_space_instance->print_configuration(os);
}

void CuTile::impl_static_fence(const std::string& name) {
  Kokkos::Impl::cuda_device_synchronize(name);
}

void CuTile::fence(const std::string& name) const {
  m_space_instance->fence(name);
}

const char* CuTile::name() { return "CuTile"; }

uint32_t CuTile::impl_instance_id() const noexcept {
  return m_space_instance->impl_get_instance_id();
}

cudaStream_t CuTile::cuda_stream() const { return m_space_instance->m_stream; }

int CuTile::cuda_device() const { return m_space_instance->m_cudaDev; }

int CuTile::concurrency() const {
  cudaDeviceProp prop;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, cuda_device()));
  return static_cast<int>(prop.maxThreadsPerMultiProcessor *
                          prop.multiProcessorCount);
}

namespace Impl {

int g_cutile_space_factory_initialized =
    initialize_space_factory<CuTile>("151_CuTile");

}  // namespace Impl

}  // namespace Kokkos

#else

void KOKKOS_CORE_SRC_CUTILE_IMPL_PREVENT_LINK_ERROR() {}

#endif  // KOKKOS_ENABLE_CUDA && KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUTILE_INSTANCE_CPP_
