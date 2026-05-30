#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "lattice/runtime.h"

namespace nb = nanobind;

NB_MODULE(_ext, m) {
  m.doc() = "Native extension for mlx-lattice.";

  m.def("version", &mlx_lattice::version);
  m.def("capabilities", []() {
    auto caps = mlx_lattice::capabilities();
    nb::dict out;
    out["cpu"] = caps.cpu;
    out["metal"] = caps.metal;
    out["cuda"] = caps.cuda;
    out["rocm"] = caps.rocm;
    return out;
  });
}
