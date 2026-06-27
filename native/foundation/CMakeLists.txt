add_library(mlx_lattice_core STATIC)
target_sources(
  mlx_lattice_core
  PRIVATE runtime.cpp)
target_include_directories(
  mlx_lattice_core
  PUBLIC ${PROJECT_SOURCE_DIR}/native)
target_link_libraries(
  mlx_lattice_core
  PUBLIC
    mlx
    mlx_lattice_config
    mlx_lattice_features)
