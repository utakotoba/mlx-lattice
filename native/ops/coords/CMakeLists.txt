# MARK: - target

add_library(mlx_lattice_ops_coords STATIC)

target_sources(
  mlx_lattice_ops_coords
  PRIVATE
    api.cpp
    lookup.cpp
    neighbors.cpp
    offsets.cpp
    quantization.cpp
    relations.cpp
    set.cpp
    streams.cpp
    validation.cpp)

target_include_directories(
  mlx_lattice_ops_coords
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_link_libraries(
  mlx_lattice_ops_coords
  PUBLIC
    mlx
    mlx_lattice_backends)
