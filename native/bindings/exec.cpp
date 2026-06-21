#include "bindings/registrations.h"

#include "bindings/array_arg.h"

#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#include "ops/exec.h"

namespace mlx_lattice::bindings {

using namespace nb::literals;

void register_exec(nb::module_& module) {
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
    module.def(
        "sparse_conv_features",
        [](nb::handle feats,
           nb::handle weights,
           nb::handle in_rows,
           nb::handle out_rows,
           nb::handle kernel_ids,
           nb::handle counts,
           nb::handle row_offsets,
           nb::handle in_row_offsets,
           nb::handle in_edge_ids,
           nb::handle kernel_row_offsets,
           nb::handle kernel_edge_ids,
           int out_capacity,
           int n_kernels) {
            return sparse_conv_features(
                array_arg(feats, "feats"),
                array_arg(weights, "weights"),
                array_arg(in_rows, "in_rows"),
                array_arg(out_rows, "out_rows"),
                array_arg(kernel_ids, "kernel_ids"),
                array_arg(counts, "counts"),
                array_arg(row_offsets, "row_offsets"),
                array_arg(in_row_offsets, "in_row_offsets"),
                array_arg(in_edge_ids, "in_edge_ids"),
                array_arg(kernel_row_offsets, "kernel_row_offsets"),
                array_arg(kernel_edge_ids, "kernel_edge_ids"),
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
        [](nb::handle feats,
           nb::handle weights,
           nb::handle sorted_out_in_map,
           nb::handle sorted_kv_out_in_map,
           nb::handle reorder_rows,
           nb::handle tile_masks,
           int out_capacity,
           int n_kernels,
           bool store_sorted) {
            return sparse_conv_features_sorted_implicit_gemm(
                array_arg(feats, "feats"),
                array_arg(weights, "weights"),
                array_arg(sorted_out_in_map, "sorted_out_in_map"),
                array_arg(sorted_kv_out_in_map, "sorted_kv_out_in_map"),
                array_arg(reorder_rows, "reorder_rows"),
                array_arg(tile_masks, "tile_masks"),
                out_capacity,
                n_kernels,
                store_sorted
            );
        },
        "feats"_a,
        "weights"_a,
        "sorted_out_in_map"_a,
        "sorted_kv_out_in_map"_a,
        "reorder_rows"_a,
        "tile_masks"_a,
        "out_capacity"_a,
        "n_kernels"_a,
        "store_sorted"_a = false,
        nb::sig(
            "def sparse_conv_features_sorted_implicit_gemm(feats: "
            "mlx.core.array, weights: mlx.core.array, "
            "sorted_out_in_map: mlx.core.array, "
            "sorted_kv_out_in_map: mlx.core.array, reorder_rows: "
            "mlx.core.array, tile_masks: mlx.core.array, "
            "out_capacity: int, n_kernels: int, store_sorted: bool = False) "
            "-> mlx.core.array"
        ),
        "Run sparse convolution over a sorted implicit-GEMM relation view."
    );
    module.def(
        "sparse_conv_features_sorted_direct_reference",
        [](nb::handle feats,
           nb::handle weights,
           nb::handle sorted_out_in_map,
           nb::handle reorder_rows,
           nb::handle tile_masks,
           int out_capacity,
           int n_kernels,
           bool store_sorted) {
            return sparse_conv_features_sorted_direct_reference(
                array_arg(feats, "feats"),
                array_arg(weights, "weights"),
                array_arg(sorted_out_in_map, "sorted_out_in_map"),
                array_arg(reorder_rows, "reorder_rows"),
                array_arg(tile_masks, "tile_masks"),
                out_capacity,
                n_kernels,
                store_sorted
            );
        },
        "feats"_a,
        "weights"_a,
        "sorted_out_in_map"_a,
        "reorder_rows"_a,
        "tile_masks"_a,
        "out_capacity"_a,
        "n_kernels"_a,
        "store_sorted"_a = false,
        nb::sig(
            "def sparse_conv_features_sorted_direct_reference(feats: "
            "mlx.core.array, weights: mlx.core.array, "
            "sorted_out_in_map: mlx.core.array, reorder_rows: mlx.core.array, "
            "tile_masks: mlx.core.array, "
            "out_capacity: int, n_kernels: int, store_sorted: bool = False) "
            "-> mlx.core.array"
        ),
        "Run diagnostic direct row-stationary convolution over a sorted "
        "implicit-GEMM relation view."
    );
}

} // namespace mlx_lattice::bindings
