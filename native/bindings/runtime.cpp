#include "bindings/registrations.h"

#include <nanobind/stl/string.h>

#include <string>

#include "lattice/runtime.h"

namespace mlx_lattice::bindings {

namespace {

nb::dict capabilities_dict() {
    auto caps = capabilities();
    nb::dict out;
    out["cpu"] = caps.cpu;
    out["metal"] = caps.metal;
    out["cuda"] = caps.cuda;
    out["rocm"] = caps.rocm;
    return out;
}

std::string version_string() { return std::string(version()); }

} // namespace

void register_runtime(nb::module_& module) {
    module.def(
        "version",
        &version_string,
        nb::sig("def version() -> str"),
        "Return the native mlx-lattice version."
    );
    module.def(
        "capabilities",
        &capabilities_dict,
        nb::sig("def capabilities() -> dict[str, bool]"),
        "Return compiled native backend capabilities."
    );
}

} // namespace mlx_lattice::bindings
