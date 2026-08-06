kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

kokkos_add_executable(example_cutile_gemm SOURCES cutile_gemm.cpp)

add_test(NAME Kokkos_Example_CuTileGemm COMMAND Kokkos_example_cutile_gemm)
