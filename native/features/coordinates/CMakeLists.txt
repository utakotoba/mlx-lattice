add_library(mlx_lattice_feature_coordinates STATIC)
target_sources(
  mlx_lattice_feature_coordinates
  PRIVATE
    api.cpp
    lookup.cpp
    neighbors.cpp
    occupancy.cpp
    offsets.cpp
    ordering.cpp
    quantization.cpp
    relations.cpp
    set.cpp
    streams.cpp
    validation.cpp
    cpu/neighbor_algorithms.cpp
    cpu/occupancy_algorithms.cpp
    cpu/quantization_algorithms.cpp
    cpu/relation_algorithms.cpp
    cpu/set_algorithms.cpp
    $<$<NOT:$<BOOL:${MLX_LATTICE_HAS_METAL}>>:metal/stub.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/neighbors_runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/occupancy_runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/ordering_runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/quantization_runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/relations_runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/set_runtime.cpp>)
target_include_directories(
  mlx_lattice_feature_coordinates
  PUBLIC ${PROJECT_SOURCE_DIR}/native)
target_link_libraries(
  mlx_lattice_feature_coordinates
  PUBLIC mlx_lattice_platform mlx_lattice_platform_metal)
