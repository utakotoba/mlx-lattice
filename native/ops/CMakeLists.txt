# MARK: - targets

add_subdirectory(coords)
add_subdirectory(exec)

add_library(mlx_lattice_ops INTERFACE)

target_link_libraries(
  mlx_lattice_ops
  INTERFACE
    mlx_lattice_ops_coords
    mlx_lattice_ops_exec)
