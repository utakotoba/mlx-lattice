# MARK: - target

add_library(mlx_lattice_ops_exec STATIC)

target_sources(
  mlx_lattice_ops_exec
  PRIVATE
    api.cpp
    dispatch.cpp
    validation.cpp)

target_include_directories(
  mlx_lattice_ops_exec
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_link_libraries(
  mlx_lattice_ops_exec
  PUBLIC
    mlx
    mlx_lattice_backends)
