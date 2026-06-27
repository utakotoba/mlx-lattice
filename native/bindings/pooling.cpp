#include "bindings/registrations.h"

#include "bindings/array_arg.h"

#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#include "features/pooling/api.h"

namespace mlx_lattice::bindings {

using namespace nb::literals;

void register_pooling(nb::module_& module) {
    module.def(
        "sparse_pool_features",
        [](nb::handle feats,
           nb::handle in_rows,
           nb::handle out_rows,
           nb::handle kernel_ids,
           nb::handle row_offsets,
           nb::handle counts,
           bool input_exclusive,
           const std::string& reduce,
           int out_capacity,
           int n_kernels) {
            auto op = PoolReduceOp::Sum;
            if (reduce == "max") {
                op = PoolReduceOp::Max;
            } else if (reduce == "avg") {
                op = PoolReduceOp::Avg;
            } else if (reduce != "sum") {
                throw std::invalid_argument(
                    "reduce must be 'sum', 'max', or 'avg'."
                );
            }
            return sparse_pool_features(
                op,
                array_arg(feats, "feats"),
                array_arg(in_rows, "in_rows"),
                array_arg(out_rows, "out_rows"),
                array_arg(kernel_ids, "kernel_ids"),
                array_arg(row_offsets, "row_offsets"),
                array_arg(counts, "counts"),
                out_capacity,
                n_kernels,
                input_exclusive ? PoolInputLayout::Exclusive
                                : PoolInputLayout::Overlap
            );
        },
        "feats"_a,
        "in_rows"_a,
        "out_rows"_a,
        "kernel_ids"_a,
        "row_offsets"_a,
        "counts"_a,
        "input_exclusive"_a,
        "reduce"_a,
        "out_capacity"_a,
        "n_kernels"_a,
        nb::sig(
            "def sparse_pool_features(feats: mlx.core.array, "
            "in_rows: mlx.core.array, out_rows: mlx.core.array, "
            "kernel_ids: mlx.core.array, row_offsets: mlx.core.array, "
            "counts: mlx.core.array, input_exclusive: bool, "
            "reduce: str, out_capacity: int, n_kernels: int) -> "
            "mlx.core.array"
        ),
        "Run sparse pooling feature reduction over a kernel relation."
    );
}

} // namespace mlx_lattice::bindings
