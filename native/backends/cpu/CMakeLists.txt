# MARK: - target

add_library(mlx_lattice_backend_cpu STATIC)

target_sources(
  mlx_lattice_backend_cpu
  PRIVATE coords/algorithms.cpp)

target_include_directories(
  mlx_lattice_backend_cpu
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_link_libraries(
  mlx_lattice_backend_cpu
  PUBLIC mlx)
