# MARK: - target

nanobind_add_module(
  _ext
  NB_STATIC
  NB_DOMAIN mlx
  coords.cpp
  entropy.cpp
  convolution.cpp
  pooling.cpp
  runtime.cpp
  module.cpp)

target_link_libraries(_ext PRIVATE mlx_lattice_core)

# MARK: - typing

set(MLX_LATTICE_STUB_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")
set(MLX_LATTICE_EXT_STUB "${MLX_LATTICE_STUB_DIR}/_ext.pyi")
set(MLX_LATTICE_SOURCE_EXT_STUB "${PROJECT_SOURCE_DIR}/mlx_lattice/_ext.pyi")

add_custom_command(
  OUTPUT ${MLX_LATTICE_EXT_STUB}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MLX_LATTICE_STUB_DIR}
  COMMAND
    ${Python_EXECUTABLE} -m nanobind.stubgen --quiet --module _ext
    --import $<TARGET_FILE_DIR:_ext> --output-file ${MLX_LATTICE_EXT_STUB}
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different ${MLX_LATTICE_EXT_STUB}
    ${MLX_LATTICE_SOURCE_EXT_STUB}
  DEPENDS _ext
  VERBATIM)

add_custom_target(_ext_stub ALL DEPENDS ${MLX_LATTICE_EXT_STUB})
add_custom_target(
  sync_ext_stub
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different ${MLX_LATTICE_EXT_STUB}
    ${MLX_LATTICE_SOURCE_EXT_STUB}
  DEPENDS ${MLX_LATTICE_EXT_STUB}
  VERBATIM)

# MARK: - install

install(TARGETS _ext LIBRARY DESTINATION mlx_lattice)
install(FILES ${MLX_LATTICE_EXT_STUB} DESTINATION mlx_lattice)
install(
  FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mlx_lattice.metallib
  DESTINATION mlx_lattice
  OPTIONAL)
