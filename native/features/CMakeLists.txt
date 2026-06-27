add_subdirectory(coordinates)
add_subdirectory(convolution)
add_subdirectory(pooling)
add_subdirectory(entropy)

add_library(mlx_lattice_features INTERFACE)
target_link_libraries(
  mlx_lattice_features
  INTERFACE
    mlx_lattice_feature_coordinates
    mlx_lattice_feature_convolution
    mlx_lattice_feature_pooling
    mlx_lattice_feature_entropy)

if(MLX_LATTICE_HAS_METAL)
  mlx_build_metallib(
    TARGET mlx_lattice_metallib
    TITLE mlx_lattice
    SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/forward.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/input_grad.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/sorted_igemm.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/specialized_forward.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/specialized_input_grad.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/specialized_weight_grad.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/weight_grad.metal
      ${CMAKE_CURRENT_LIST_DIR}/coordinates/metal/hash.metal
      ${CMAKE_CURRENT_LIST_DIR}/coordinates/metal/relation_kernels.metal
      ${CMAKE_CURRENT_LIST_DIR}/coordinates/metal/neighbor_kernels.metal
      ${CMAKE_CURRENT_LIST_DIR}/coordinates/metal/occupancy.metal
      ${CMAKE_CURRENT_LIST_DIR}/coordinates/metal/ordering.metal
      ${CMAKE_CURRENT_LIST_DIR}/coordinates/metal/quantization.metal
      ${CMAKE_CURRENT_LIST_DIR}/coordinates/metal/set.metal
      ${CMAKE_CURRENT_LIST_DIR}/pooling/metal/autodiff.metal
      ${CMAKE_CURRENT_LIST_DIR}/pooling/metal/forward.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/tensor_ops/input_grad/contract.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/tensor_ops/weight_grad/contract.metal
      ${CMAKE_CURRENT_LIST_DIR}/convolution/metal/tensor_ops/weight_grad/reduce.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(mlx_lattice_feature_convolution mlx_lattice_metallib)
endif()
