# MARK: - target

add_library(mlx_lattice_backend_metal STATIC)

target_sources(
  mlx_lattice_backend_metal
  PRIVATE
    coords/runtime.cpp
    exec/runtime.cpp)

target_include_directories(
  mlx_lattice_backend_metal
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_link_libraries(
  mlx_lattice_backend_metal
  PUBLIC mlx)

# MARK: - shaders

if(MLX_LATTICE_HAS_METAL)
  mlx_build_metallib(
    TARGET mlx_lattice_metallib
    TITLE mlx_lattice
    SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/coords/kernels.metal
      ${CMAKE_CURRENT_LIST_DIR}/exec/kernels.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_lattice_backend_metal mlx_lattice_metallib)
endif()
