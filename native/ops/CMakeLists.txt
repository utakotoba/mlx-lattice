# MARK: - targets

add_subdirectory(coords)

add_library(mlx_lattice_ops INTERFACE)

target_link_libraries(
  mlx_lattice_ops
  INTERFACE mlx_lattice_ops_coords)
