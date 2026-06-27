# MARK: - config

add_library(mlx_lattice_config INTERFACE)

target_compile_definitions(
  mlx_lattice_config
  INTERFACE MLX_LATTICE_VERSION="${PROJECT_VERSION}")

# MARK: - capabilities

option(MLX_LATTICE_BUILD_METAL "Build mlx-lattice Metal backend" ON)

set(MLX_LATTICE_HAS_METAL 0)
if(APPLE)
  cmake_host_system_information(
    RESULT MLX_LATTICE_MACOS_VERSION QUERY OS_RELEASE)
  string(REGEX MATCH "^[0-9]+" MLX_LATTICE_MACOS_MAJOR
               "${MLX_LATTICE_MACOS_VERSION}")
  execute_process(
    COMMAND xcrun -sdk macosx -find metal
    RESULT_VARIABLE MLX_LATTICE_METAL_FIND_RESULT
    OUTPUT_QUIET
    ERROR_QUIET)
  if(MLX_BUILD_METAL
     AND MLX_LATTICE_BUILD_METAL
     AND MLX_LATTICE_MACOS_MAJOR GREATER_EQUAL 25
     AND MLX_LATTICE_METAL_FIND_RESULT EQUAL 0)
    set(MLX_LATTICE_HAS_METAL 1)
  endif()
endif()

target_compile_definitions(
  mlx_lattice_config
  INTERFACE
    MLX_LATTICE_HAS_CPU=1
    MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL})

# MARK: - targets

add_subdirectory(platform)
add_subdirectory(features)
add_subdirectory(foundation)
add_subdirectory(bindings)
