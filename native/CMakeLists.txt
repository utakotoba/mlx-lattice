# MARK: - config

add_library(mlx_lattice_config INTERFACE)

target_compile_definitions(
  mlx_lattice_config
  INTERFACE MLX_LATTICE_VERSION="${PROJECT_VERSION}")

# MARK: - targets

add_subdirectory(backends)
add_subdirectory(ops)
add_subdirectory(lattice)
add_subdirectory(bindings)
