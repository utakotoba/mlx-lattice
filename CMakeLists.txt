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

if(NOT APPLE)
  message(FATAL_ERROR "mlx-lattice currently supports macOS only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MLX_LATTICE_BUILD_METAL "Build the mlx-lattice Metal library" ON)
set(MLX_LATTICE_HAS_METAL 0)

# MARK: - paths

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

# MARK: - dependencies

if(DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE
      "$ENV{VIRTUAL_ENV}/bin/python"
      CACHE FILEPATH "Python executable from active virtual environment")
endif()
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT DEFINED nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_DIR
    COMMAND_ERROR_IS_FATAL ANY)
endif()
find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_DIR}" NO_DEFAULT_PATH)

if(NOT DEFINED MLX_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_DIR
    COMMAND_ERROR_IS_FATAL ANY)
endif()
find_package(MLX CONFIG REQUIRED PATHS "${MLX_DIR}" NO_DEFAULT_PATH)

# MARK: - native

add_library(mlx_lattice_native STATIC)
target_sources(
  mlx_lattice_native
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cpu/coords.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/cpu/conv3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/conv3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/backends/metal/coords.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/conv3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/ops/coords.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/lattice/runtime.cpp)
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
    MLX_LATTICE_HAS_CUDA=0
    MLX_LATTICE_HAS_ROCM=0)

# MARK: - metal

execute_process(
  COMMAND xcrun -sdk macosx -find metal
  RESULT_VARIABLE MLX_LATTICE_METAL_FIND_RESULT
  OUTPUT_QUIET
  ERROR_QUIET)

if(MLX_BUILD_METAL AND MLX_LATTICE_BUILD_METAL
   AND MLX_LATTICE_METAL_FIND_RESULT EQUAL 0)
  set(MLX_LATTICE_HAS_METAL 1)
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

target_compile_definitions(
  mlx_lattice_native
  PUBLIC MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL})

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
