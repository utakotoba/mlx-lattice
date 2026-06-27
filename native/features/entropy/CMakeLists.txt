add_library(mlx_lattice_feature_entropy STATIC)
target_sources(
  mlx_lattice_feature_entropy
  PRIVATE api.cpp range.cpp)
target_include_directories(
  mlx_lattice_feature_entropy
  PUBLIC ${PROJECT_SOURCE_DIR}/native)
target_link_libraries(
  mlx_lattice_feature_entropy
  PUBLIC mlx_lattice_platform)
