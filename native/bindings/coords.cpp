#include "bindings/registrations.h"

#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "ops/coords.h"

namespace mlx_lattice::bindings {

using namespace nb::literals;

namespace {

Triple triple_from_values(const std::vector<int>& values, const char* name) {
    if (values.size() != 3) {
        throw std::invalid_argument(
            std::string(name) + " must contain exactly 3 values."
        );
    }
    return {values[0], values[1], values[2]};
}

nb::tuple map_tuple(const NativeKernelMap& map) {
    return nb::make_tuple(
        map.in_rows, map.out_rows, map.kernel_ids, map.out_coords
    );
}

} // namespace

void register_coords(nb::module_& module) {
    module.def(
        "downsample_coords",
        [](const mx::array& coords, const std::vector<int>& stride) {
            return downsample_coords(
                coords, triple_from_values(stride, "stride")
            );
        },
        "coords"_a,
        "stride"_a,
        nb::sig(
            "def downsample_coords(coords: mlx.core.array, "
            "stride: collections.abc.Sequence[int]) -> mlx.core.array"
        ),
        "Downsample sparse coordinates."
    );
    module.def(
        "union_coords",
        &union_coords,
        "lhs"_a,
        "rhs"_a,
        nb::sig(
            "def union_coords(lhs: mlx.core.array, rhs: mlx.core.array) -> "
            "mlx.core.array"
        ),
        "Return the ordered union of two coordinate arrays."
    );
    module.def(
        "intersection_coords",
        &intersection_coords,
        "lhs"_a,
        "rhs"_a,
        nb::sig(
            "def intersection_coords(lhs: mlx.core.array, "
            "rhs: mlx.core.array) -> mlx.core.array"
        ),
        "Return the ordered intersection of two coordinate arrays."
    );
    module.def(
        "lookup_coords",
        &lookup_coords,
        "coords"_a,
        "queries"_a,
        nb::sig(
            "def lookup_coords(coords: mlx.core.array, "
            "queries: mlx.core.array) -> mlx.core.array"
        ),
        "Return row indices of queries in coords, or -1."
    );
    module.def(
        "build_kernel_map",
        [](const mx::array& coords,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return map_tuple(build_kernel_map(
                coords,
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding"),
                triple_from_values(dilation, "dilation")
            ));
        },
        "coords"_a,
        "kernel_size"_a,
        "stride"_a,
        "padding"_a,
        "dilation"_a,
        nb::sig(
            "def build_kernel_map(coords: mlx.core.array, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int], "
            "dilation: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a forward sparse kernel map."
    );
    module.def(
        "build_generative_map",
        [](const mx::array& coords,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride) {
            return map_tuple(build_generative_map(
                coords,
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride")
            ));
        },
        "coords"_a,
        "kernel_size"_a,
        "stride"_a,
        nb::sig(
            "def build_generative_map(coords: mlx.core.array, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a generative sparse kernel map."
    );
    module.def(
        "build_transposed_kernel_map",
        [](const mx::array& coords,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return map_tuple(build_transposed_kernel_map(
                coords,
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding"),
                triple_from_values(dilation, "dilation")
            ));
        },
        "coords"_a,
        "kernel_size"_a,
        "stride"_a,
        "padding"_a,
        "dilation"_a,
        nb::sig(
            "def build_transposed_kernel_map(coords: mlx.core.array, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int], "
            "dilation: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a transposed sparse kernel map."
    );
}

} // namespace mlx_lattice::bindings
