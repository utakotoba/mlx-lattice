# MARK: - shared platform infrastructure

add_library(mlx_lattice_platform INTERFACE)
target_include_directories(
  mlx_lattice_platform
  INTERFACE ${PROJECT_SOURCE_DIR}/native)
target_link_libraries(mlx_lattice_platform INTERFACE mlx)

add_library(mlx_lattice_platform_metal STATIC)
target_sources(
  mlx_lattice_platform_metal
  PRIVATE
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/runtime_utils.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/capabilities.cpp>)
target_include_directories(
  mlx_lattice_platform_metal
  PUBLIC ${PROJECT_SOURCE_DIR}/native)
target_compile_definitions(
  mlx_lattice_platform_metal
  PUBLIC MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL})
target_link_libraries(
  mlx_lattice_platform_metal
  PUBLIC mlx_lattice_platform)
