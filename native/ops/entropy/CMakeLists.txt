# MARK: - target

add_library(mlx_lattice_ops_entropy STATIC)

target_sources(
  mlx_lattice_ops_entropy
  PRIVATE api.cpp range.cpp)

target_include_directories(
  mlx_lattice_ops_entropy
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_link_libraries(
  mlx_lattice_ops_entropy
  PUBLIC mlx)
