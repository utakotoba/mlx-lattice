# MARK: - target

add_library(mlx_lattice_backend_metal STATIC)

target_sources(
  mlx_lattice_backend_metal
  PRIVATE
    conv/runtime.cpp
    coords/runtime.cpp
    pool/runtime.cpp
    runtime_utils.cpp
    tensor_ops/capabilities.cpp
    tensor_ops/conv/weight_grad/runtime.cpp)

target_include_directories(
  mlx_lattice_backend_metal
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_compile_definitions(
  mlx_lattice_backend_metal
  PUBLIC MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL})

target_link_libraries(
  mlx_lattice_backend_metal
  PUBLIC mlx)

# MARK: - shaders

if(MLX_LATTICE_HAS_METAL)
  mlx_build_metallib(
    TARGET mlx_lattice_metallib
    TITLE mlx_lattice
    SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/conv/kernels.metal
      ${CMAKE_CURRENT_LIST_DIR}/coords/hash.metal
      ${CMAKE_CURRENT_LIST_DIR}/coords/kernels.metal
      ${CMAKE_CURRENT_LIST_DIR}/coords/quantization.metal
      ${CMAKE_CURRENT_LIST_DIR}/coords/set.metal
      ${CMAKE_CURRENT_LIST_DIR}/pool/autodiff.metal
      ${CMAKE_CURRENT_LIST_DIR}/pool/forward.metal
      ${CMAKE_CURRENT_LIST_DIR}/tensor_ops/conv/weight_grad/contract.metal
      ${CMAKE_CURRENT_LIST_DIR}/tensor_ops/conv/weight_grad/reduce.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_lattice_backend_metal mlx_lattice_metallib)
endif()
