#include <nanobind/nanobind.h>

#include "bindings/registrations.h"

NB_MODULE(_ext, module) {
    module.doc() = "Native extension for mlx-lattice.";

    mlx_lattice::bindings::register_runtime(module);
    mlx_lattice::bindings::register_coords(module);
    mlx_lattice::bindings::register_entropy(module);
    mlx_lattice::bindings::register_convolution(module);
    mlx_lattice::bindings::register_pooling(module);
}
