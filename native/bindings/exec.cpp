#include "bindings/registrations.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "ops/exec.h"

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

nb::tuple sparse_tuple(const NativeSparseTensorOutput& output) {
    return nb::make_tuple(output.coords, output.feats, output.counts);
}

} // namespace

void register_exec(nb::module_& module) {
    module.def(
        "spmm_edges",
        &spmm_edges,
        "feats"_a,
        "weights"_a,
        "in_rows"_a,
        "out_rows"_a,
        "kernel_ids"_a,
        "edge_count"_a,
        "n_out_rows"_a,
        nb::sig(
            "def spmm_edges(feats: mlx.core.array, "
            "weights: mlx.core.array, in_rows: mlx.core.array, "
            "out_rows: mlx.core.array, kernel_ids: mlx.core.array, "
            "edge_count: mlx.core.array, n_out_rows: int) -> mlx.core.array"
        ),
        "Accumulate sparse edge feature products."
    );
    module.def(
        "pool_sum_edges",
        &pool_sum_edges,
        "feats"_a,
        "in_rows"_a,
        "out_rows"_a,
        "edge_count"_a,
        "n_out_rows"_a,
        nb::sig(
            "def pool_sum_edges(feats: mlx.core.array, "
            "in_rows: mlx.core.array, out_rows: mlx.core.array, "
            "edge_count: mlx.core.array, n_out_rows: int) -> mlx.core.array"
        ),
        "Sum sparse edge features by output row."
    );
    module.def(
        "pool_max_edges",
        &pool_max_edges,
        "feats"_a,
        "in_rows"_a,
        "out_rows"_a,
        "edge_count"_a,
        "n_out_rows"_a,
        nb::sig(
            "def pool_max_edges(feats: mlx.core.array, "
            "in_rows: mlx.core.array, out_rows: mlx.core.array, "
            "edge_count: mlx.core.array, n_out_rows: int) -> mlx.core.array"
        ),
        "Max-reduce sparse edge features by output row."
    );
    module.def(
        "sparse_conv",
        [](const mx::array& coords,
           const mx::array& active_rows,
           const mx::array& feats,
           const mx::array& weights,
           const std::string& map,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            auto op = SparseMapOp::Forward;
            if (map == "transposed") {
                op = SparseMapOp::Transposed;
            } else if (map == "generative") {
                op = SparseMapOp::Generative;
            } else if (map != "forward") {
                throw std::invalid_argument(
                    "map must be 'forward', 'transposed', or 'generative'."
                );
            }
            return sparse_tuple(sparse_conv(
                op,
                coords,
                active_rows,
                feats,
                weights,
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding"),
                triple_from_values(dilation, "dilation")
            ));
        },
        "coords"_a,
        "active_rows"_a,
        "feats"_a,
        "weights"_a,
        "map"_a,
        "kernel_size"_a,
        "stride"_a,
        "padding"_a,
        "dilation"_a,
        nb::sig(
            "def sparse_conv(coords: mlx.core.array, "
            "active_rows: mlx.core.array, feats: mlx.core.array, "
            "weights: mlx.core.array, map: str, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int], "
            "dilation: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array]"
        ),
        "Run fused sparse convolution over native sparse coordinates."
    );
    module.def(
        "sparse_pool",
        [](const mx::array& coords,
           const mx::array& active_rows,
           const mx::array& feats,
           const std::string& reduce,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
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
            return sparse_tuple(sparse_pool(
                op,
                coords,
                active_rows,
                feats,
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding"),
                triple_from_values(dilation, "dilation")
            ));
        },
        "coords"_a,
        "active_rows"_a,
        "feats"_a,
        "reduce"_a,
        "kernel_size"_a,
        "stride"_a,
        "padding"_a,
        "dilation"_a,
        nb::sig(
            "def sparse_pool(coords: mlx.core.array, "
            "active_rows: mlx.core.array, feats: mlx.core.array, "
            "reduce: str, kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int], "
            "dilation: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array]"
        ),
        "Run fused sparse pooling over native sparse coordinates."
    );
}

} // namespace mlx_lattice::bindings
