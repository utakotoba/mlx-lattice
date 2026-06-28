add_library(mlx_lattice_feature_convolution STATIC)
target_sources(
  mlx_lattice_feature_convolution
  PRIVATE
    api.cpp
    primitive.cpp
    quantized.cpp
    streams.cpp
    cpu/algorithms.cpp
    cpu/quantized.cpp
    $<$<NOT:$<BOOL:${MLX_LATTICE_HAS_METAL}>>:metal/stub.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/quantized_runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/tensor_ops/input_grad/runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/tensor_ops/sorted_igemm/runtime.cpp>
    $<$<BOOL:${MLX_LATTICE_HAS_METAL}>:metal/tensor_ops/weight_grad/runtime.cpp>)
target_include_directories(
  mlx_lattice_feature_convolution
  PUBLIC ${PROJECT_SOURCE_DIR}/native)
target_link_libraries(
  mlx_lattice_feature_convolution
  PUBLIC
    mlx_lattice_feature_coordinates
    mlx_lattice_platform
    mlx_lattice_platform_metal)
