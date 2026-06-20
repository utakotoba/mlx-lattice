#include "bindings/registrations.h"

#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#include "ops/exec.h"

namespace mlx_lattice::bindings {

using namespace nb::literals;

void register_exec(nb::module_& module) {
    module.def(
        "sparse_pool_features",
        [](const mx::array& feats,
           const mx::array& in_rows,
           const mx::array& out_rows,
           const mx::array& kernel_ids,
           const mx::array& row_offsets,
           const mx::array& counts,
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
                feats,
                in_rows,
                out_rows,
                kernel_ids,
                row_offsets,
                counts,
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
    module.def(
        "sparse_conv_features",
        [](const mx::array& feats,
           const mx::array& weights,
           const mx::array& in_rows,
           const mx::array& out_rows,
           const mx::array& kernel_ids,
           const mx::array& counts,
           const mx::array& row_offsets,
           const mx::array& in_row_offsets,
           const mx::array& in_edge_ids,
           const mx::array& kernel_row_offsets,
           const mx::array& kernel_edge_ids,
           int out_capacity,
           int n_kernels) {
            return sparse_conv_features(
                feats,
                weights,
                in_rows,
                out_rows,
                kernel_ids,
                counts,
                row_offsets,
                in_row_offsets,
                in_edge_ids,
                kernel_row_offsets,
                kernel_edge_ids,
                out_capacity,
                n_kernels
            );
        },
        "feats"_a,
        "weights"_a,
        "in_rows"_a,
        "out_rows"_a,
        "kernel_ids"_a,
        "counts"_a,
        "row_offsets"_a,
        "in_row_offsets"_a,
        "in_edge_ids"_a,
        "kernel_row_offsets"_a,
        "kernel_edge_ids"_a,
        "out_capacity"_a,
        "n_kernels"_a,
        nb::sig(
            "def sparse_conv_features(feats: mlx.core.array, "
            "weights: mlx.core.array, in_rows: mlx.core.array, "
            "out_rows: mlx.core.array, kernel_ids: mlx.core.array, "
            "counts: mlx.core.array, row_offsets: mlx.core.array, "
            "in_row_offsets: mlx.core.array, in_edge_ids: mlx.core.array, "
            "kernel_row_offsets: mlx.core.array, "
            "kernel_edge_ids: mlx.core.array, "
            "out_capacity: int, "
            "n_kernels: int) -> mlx.core.array"
        ),
        "Run sparse convolution feature accumulation over a kernel relation."
    );
    module.def(
        "sparse_conv_features_sorted_implicit_gemm",
        [](const mx::array& feats,
           const mx::array& weights,
           const mx::array& sorted_out_in_map,
           const mx::array& reorder_rows,
           const mx::array& tile_masks,
           int out_capacity,
           int n_kernels) {
            return sparse_conv_features_sorted_implicit_gemm(
                feats,
                weights,
                sorted_out_in_map,
                reorder_rows,
                tile_masks,
                out_capacity,
                n_kernels
            );
        },
        "feats"_a,
        "weights"_a,
        "sorted_out_in_map"_a,
        "reorder_rows"_a,
        "tile_masks"_a,
        "out_capacity"_a,
        "n_kernels"_a,
        nb::sig(
            "def sparse_conv_features_sorted_implicit_gemm(feats: "
            "mlx.core.array, weights: mlx.core.array, "
            "sorted_out_in_map: mlx.core.array, reorder_rows: "
            "mlx.core.array, tile_masks: mlx.core.array, out_capacity: int, "
            "n_kernels: int) -> mlx.core.array"
        ),
        "Run experimental sparse convolution over a sorted implicit-GEMM "
        "relation view."
    );
}

} // namespace mlx_lattice::bindings
