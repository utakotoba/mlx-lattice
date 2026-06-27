#pragma once

#include <nanobind/nanobind.h>

namespace mlx_lattice::bindings {

namespace nb = nanobind;

void register_runtime(nb::module_& module);
void register_coords(nb::module_& module);
void register_entropy(nb::module_& module);
void register_convolution(nb::module_& module);
void register_pooling(nb::module_& module);

} // namespace mlx_lattice::bindings
