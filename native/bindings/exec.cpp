#include "bindings/registrations.h"

#include "ops/exec.h"

namespace mlx_lattice::bindings {

using namespace nb::literals;

void register_exec(nb::module_& module) {
    module.def(
        "spmm_edges",
        &spmm_edges,
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
    module.def(
        "pool_sum_edges",
        &pool_sum_edges,
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
    module.def(
        "pool_max_edges",
        &pool_max_edges,
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

} // namespace mlx_lattice::bindings
