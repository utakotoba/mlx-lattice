add_library(mlx_lattice_feature_pooling STATIC)
target_sources(
  mlx_lattice_feature_pooling
  PRIVATE
    api.cpp
    primitive.cpp
    streams.cpp
    validation.cpp
    cpu/algorithms.cpp
    $<$<NOT:$<BOOL:${MLX_LATTICE_HAS_METAL}>>:metal/stub.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/runtime.cpp>)
target_include_directories(
  mlx_lattice_feature_pooling
  PUBLIC ${PROJECT_SOURCE_DIR}/native)
target_link_libraries(
  mlx_lattice_feature_pooling
  PUBLIC
    mlx_lattice_feature_coordinates
    mlx_lattice_platform
    mlx_lattice_platform_metal)
