#pragma once

#include <nanobind/nanobind.h>

namespace mlx_lattice::bindings {

namespace nb = nanobind;

void register_runtime(nb::module_& module);
void register_coords(nb::module_& module);
void register_exec(nb::module_& module);

} // namespace mlx_lattice::bindings
