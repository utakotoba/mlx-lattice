#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "lattice/runtime.h"
#include "ops/coords.h"
#include "ops/exec.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// MARK: - runtime

nb::dict capabilities() {
    auto caps = mlx_lattice::capabilities();
    nb::dict out;
    out["cpu"] = caps.cpu;
    out["metal"] = caps.metal;
    out["cuda"] = caps.cuda;
    out["rocm"] = caps.rocm;
    return out;
}

std::string version() { return std::string(mlx_lattice::version()); }

// MARK: - triples

mlx_lattice::Triple
triple_from_values(const std::vector<int>& values, const char* name) {
    if (values.size() != 3) {
        throw std::invalid_argument(
            std::string(name) + " must contain exactly 3 values."
        );
    }
    return {values[0], values[1], values[2]};
}

// MARK: - maps

nb::tuple map_tuple(const mlx_lattice::NativeKernelMap& map) {
    return nb::make_tuple(
        map.in_rows, map.out_rows, map.kernel_ids, map.out_coords
    );
}

} // namespace

NB_MODULE(_ext, m) {
    m.doc() = "Native extension for mlx-lattice.";

    m.def(
        "version",
        &version,
        nb::sig("def version() -> str"),
        "Return the native mlx-lattice version."
    );
    m.def(
        "capabilities",
        &capabilities,
        nb::sig("def capabilities() -> dict[str, bool]"),
        "Return compiled native backend capabilities."
    );
    m.def(
        "downsample_coords",
        [](const mlx_lattice::mx::array& coords,
           const std::vector<int>& stride) {
            return mlx_lattice::downsample_coords(
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
    m.def(
        "union_coords",
        &mlx_lattice::union_coords,
        "lhs"_a,
        "rhs"_a,
        nb::sig(
            "def union_coords(lhs: mlx.core.array, rhs: mlx.core.array) -> "
            "mlx.core.array"
        ),
        "Return the ordered union of two coordinate arrays."
    );
    m.def(
        "intersection_coords",
        &mlx_lattice::intersection_coords,
        "lhs"_a,
        "rhs"_a,
        nb::sig(
            "def intersection_coords(lhs: mlx.core.array, "
            "rhs: mlx.core.array) -> mlx.core.array"
        ),
        "Return the ordered intersection of two coordinate arrays."
    );
    m.def(
        "lookup_coords",
        &mlx_lattice::lookup_coords,
        "coords"_a,
        "queries"_a,
        nb::sig(
            "def lookup_coords(coords: mlx.core.array, "
            "queries: mlx.core.array) -> mlx.core.array"
        ),
        "Return row indices of queries in coords, or -1."
    );
    m.def(
        "build_kernel_map",
        [](const mlx_lattice::mx::array& coords,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return map_tuple(
                mlx_lattice::build_kernel_map(
                    coords,
                    triple_from_values(kernel_size, "kernel_size"),
                    triple_from_values(stride, "stride"),
                    triple_from_values(padding, "padding"),
                    triple_from_values(dilation, "dilation")
                )
            );
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
    m.def(
        "build_generative_map",
        [](const mlx_lattice::mx::array& coords,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride) {
            return map_tuple(
                mlx_lattice::build_generative_map(
                    coords,
                    triple_from_values(kernel_size, "kernel_size"),
                    triple_from_values(stride, "stride")
                )
            );
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
    m.def(
        "build_transposed_kernel_map",
        [](const mlx_lattice::mx::array& coords,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return map_tuple(
                mlx_lattice::build_transposed_kernel_map(
                    coords,
                    triple_from_values(kernel_size, "kernel_size"),
                    triple_from_values(stride, "stride"),
                    triple_from_values(padding, "padding"),
                    triple_from_values(dilation, "dilation")
                )
            );
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
    m.def(
        "spmm_edges",
        &mlx_lattice::spmm_edges,
        "feats"_a,
        "weights"_a,
        "in_rows"_a,
        "out_rows"_a,
        "kernel_ids"_a,
        "n_out_rows"_a,
        nb::sig(
            "def spmm_edges(feats: mlx.core.array, "
            "weights: mlx.core.array, in_rows: mlx.core.array, "
            "out_rows: mlx.core.array, kernel_ids: mlx.core.array, "
            "n_out_rows: int) -> mlx.core.array"
        ),
        "Accumulate sparse edge feature products."
    );
    m.def(
        "pool_sum_edges",
        &mlx_lattice::pool_sum_edges,
        "feats"_a,
        "in_rows"_a,
        "out_rows"_a,
        "n_out_rows"_a,
        nb::sig(
            "def pool_sum_edges(feats: mlx.core.array, "
            "in_rows: mlx.core.array, out_rows: mlx.core.array, "
            "n_out_rows: int) -> mlx.core.array"
        ),
        "Sum sparse edge features by output row."
    );
    m.def(
        "pool_max_edges",
        &mlx_lattice::pool_max_edges,
        "feats"_a,
        "in_rows"_a,
        "out_rows"_a,
        "n_out_rows"_a,
        nb::sig(
            "def pool_max_edges(feats: mlx.core.array, "
            "in_rows: mlx.core.array, out_rows: mlx.core.array, "
            "n_out_rows: int) -> mlx.core.array"
        ),
        "Max-reduce sparse edge features by output row."
    );
}
