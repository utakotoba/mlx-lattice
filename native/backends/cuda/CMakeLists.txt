# MARK: - target

add_library(mlx_lattice_backend_cuda STATIC)

target_sources(
  mlx_lattice_backend_cuda
  PRIVATE $<$<BOOL:${MLX_LATTICE_HAS_CUDA}>:conv/runtime.cpp>
          $<$<BOOL:${MLX_LATTICE_HAS_CUDA}>:conv/kernels.cu>
          $<$<BOOL:${MLX_LATTICE_HAS_CUDA}>:coords/runtime.cpp>
          $<$<BOOL:${MLX_LATTICE_HAS_CUDA}>:coords/kernels.cu>
          $<$<BOOL:${MLX_LATTICE_HAS_CUDA}>:pool/kernels.cu>
          $<$<BOOL:${MLX_LATTICE_HAS_CUDA}>:pool/runtime.cpp>
          $<$<BOOL:${MLX_LATTICE_HAS_CUDA}>:runtime_utils.cpp>
          $<$<NOT:$<BOOL:${MLX_LATTICE_HAS_CUDA}>>:stub.cpp>)

target_include_directories(
  mlx_lattice_backend_cuda
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_compile_definitions(
  mlx_lattice_backend_cuda
  PUBLIC MLX_LATTICE_HAS_CUDA=${MLX_LATTICE_HAS_CUDA})

target_link_libraries(
  mlx_lattice_backend_cuda
  PUBLIC mlx)

if(MLX_LATTICE_HAS_CUDA)
  set_target_properties(
    mlx_lattice_backend_cuda
    PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_SEPARABLE_COMPILATION ON)

  target_link_libraries(
    mlx_lattice_backend_cuda
    PRIVATE CUDA::cuda_driver CUDA::cudart)
endif()
