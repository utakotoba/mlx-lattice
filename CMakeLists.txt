cmake_minimum_required(VERSION 3.27)

# MARK: - project

if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME mlx_lattice)
endif()
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.1.0)
endif()

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# MARK: - platform

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MLX_LATTICE_BUILD_METAL "Build the mlx-lattice Metal library" ON)
option(MLX_LATTICE_BUILD_CUDA "Build the mlx-lattice CUDA kernels" ON)
option(
  MLX_LATTICE_REQUIRE_CUDA
  "Fail configuration when CUDA kernels cannot be built"
  OFF)
set(MLX_LATTICE_HAS_METAL 0)
set(MLX_LATTICE_HAS_CUDA 0)

# MARK: - paths

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

# MARK: - dependencies

if(DEFINED ENV{VIRTUAL_ENV}
   AND (NOT DEFINED Python_EXECUTABLE OR NOT EXISTS "${Python_EXECUTABLE}"))
  set(Python_EXECUTABLE
      "$ENV{VIRTUAL_ENV}/bin/python"
      CACHE FILEPATH "Python executable from active virtual environment" FORCE)
endif()
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT DEFINED nanobind_DIR OR NOT EXISTS "${nanobind_DIR}")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_DIR
    COMMAND_ERROR_IS_FATAL ANY)
  set(nanobind_DIR
      "${nanobind_DIR}"
      CACHE PATH "nanobind CMake package directory" FORCE)
endif()
find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_DIR}" NO_DEFAULT_PATH)

if(NOT DEFINED MLX_DIR OR NOT EXISTS "${MLX_DIR}")
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import importlib.util, pathlib, sys; spec = importlib.util.find_spec('mlx'); roots = list(spec.submodule_search_locations or []) if spec else []; candidates = [pathlib.Path(root) / 'share' / 'cmake' / 'MLX' for root in roots]; matches = [path for path in candidates if (path / 'MLXConfig.cmake').exists()]; print(matches[0] if matches else ''); sys.exit(0 if matches else 1)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_DIR
    COMMAND_ERROR_IS_FATAL ANY)
  set(MLX_DIR "${MLX_DIR}" CACHE PATH "MLX CMake package directory" FORCE)
endif()
find_package(MLX CONFIG REQUIRED PATHS "${MLX_DIR}" NO_DEFAULT_PATH)

# MARK: - native

set(MLX_LATTICE_CPU_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cpu/coords.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cpu/conv3d_backward.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cpu/conv3d_forward.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cpu/conv3d_pool.cpp)

set(MLX_LATTICE_METAL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/conv3d_backward.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/conv3d_forward.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/conv3d_pool.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/coords.cpp)

set(MLX_LATTICE_CUDA_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cuda/conv3d.cu
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cuda/coords.cu)

set(MLX_LATTICE_OP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/conv3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/conv3d/dispatch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/conv3d/primitives.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/conv3d/validation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/coords.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/coords/dispatch.cpp)

set(MLX_LATTICE_RUNTIME_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/native/lattice/runtime.cpp)

add_library(mlx_lattice_native STATIC)
target_sources(
  mlx_lattice_native
  PRIVATE
    ${MLX_LATTICE_CPU_SOURCES}
    ${MLX_LATTICE_OP_SOURCES}
    ${MLX_LATTICE_RUNTIME_SOURCES})
target_include_directories(
  mlx_lattice_native
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/native)
target_link_libraries(mlx_lattice_native PUBLIC mlx)
target_compile_definitions(
  mlx_lattice_native
  PUBLIC
    MLX_LATTICE_VERSION="${PROJECT_VERSION}"
    MLX_LATTICE_HAS_CPU=1
    MLX_LATTICE_HAS_ROCM=0)

# MARK: - metal

if(APPLE)
  execute_process(
    COMMAND xcrun -sdk macosx -find metal
    RESULT_VARIABLE MLX_LATTICE_METAL_FIND_RESULT
    OUTPUT_QUIET
    ERROR_QUIET)

  if(MLX_BUILD_METAL AND MLX_LATTICE_BUILD_METAL
     AND MLX_LATTICE_METAL_FIND_RESULT EQUAL 0)
    set(MLX_LATTICE_HAS_METAL 1)
    target_sources(
      mlx_lattice_native
      PRIVATE ${MLX_LATTICE_METAL_SOURCES})
    mlx_build_metallib(
      TARGET mlx_lattice_metallib
      TITLE mlx_lattice
      SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/conv3d.metal
        ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/coords.metal
      INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
      OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    add_dependencies(mlx_lattice_native mlx_lattice_metallib)
  endif()
endif()

target_compile_definitions(
  mlx_lattice_native
  PUBLIC MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL})

# MARK: - cuda

if(MLX_LATTICE_BUILD_CUDA AND NOT APPLE)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(MLX_LATTICE_HAS_CUDA 1)
      target_sources(
        mlx_lattice_native
        PRIVATE ${MLX_LATTICE_CUDA_SOURCES})
      target_link_libraries(
        mlx_lattice_native
        PUBLIC CUDA::cudart CUDA::cuda_driver)
      if(DEFINED MLX_LATTICE_CUDA_INCLUDE_DIRS)
        target_include_directories(
          mlx_lattice_native
          SYSTEM
          PRIVATE ${MLX_LATTICE_CUDA_INCLUDE_DIRS})
      endif()
      if(DEFINED MLX_LATTICE_CUDA_LIBRARY_DIRS)
        target_link_directories(
          mlx_lattice_native
          PRIVATE ${MLX_LATTICE_CUDA_LIBRARY_DIRS})
      endif()
      target_compile_options(
        mlx_lattice_native
        PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      set_target_properties(
        mlx_lattice_native
        PROPERTIES
          CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED ON
          CUDA_SEPARABLE_COMPILATION ON)
    endif()
  endif()
endif()

if(MLX_LATTICE_REQUIRE_CUDA AND NOT MLX_LATTICE_HAS_CUDA)
  message(
    FATAL_ERROR
      "MLX_LATTICE_REQUIRE_CUDA is ON, but CUDA compiler/toolkit was not found."
  )
endif()

target_compile_definitions(
  mlx_lattice_native
  PUBLIC MLX_LATTICE_HAS_CUDA=${MLX_LATTICE_HAS_CUDA})

# MARK: - extension

nanobind_add_module(
  _ext
  NB_STATIC
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/native/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_lattice_native)

# MARK: - install

install(TARGETS _ext LIBRARY DESTINATION mlx_lattice)
install(
  FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mlx_lattice.metallib
  DESTINATION mlx_lattice
  OPTIONAL)
