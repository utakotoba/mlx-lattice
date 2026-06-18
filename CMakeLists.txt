cmake_minimum_required(VERSION 3.27)

if(NOT SKBUILD)
  message(
    FATAL_ERROR
      "This project is built with scikit-build-core.")
endif()

# MARK: - project

project(
  mlx_lattice
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# MARK: - language

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MARK: - paths

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

# MARK: - python

if(DEFINED ENV{VIRTUAL_ENV}
   AND DEFINED Python_EXECUTABLE
   AND NOT EXISTS "${Python_EXECUTABLE}")
  set(Python_EXECUTABLE
      "$ENV{VIRTUAL_ENV}/bin/python"
      CACHE FILEPATH "Active virtual environment Python executable" FORCE)
endif()

find_package(
  Python 3.12
  REQUIRED
  COMPONENTS Interpreter Development.Module)

# MARK: - nanobind

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
find_package(nanobind CONFIG REQUIRED)

# MARK: - mlx

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

add_subdirectory(native)
