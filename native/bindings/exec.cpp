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
    module.def(
        "sparse_conv_features",
        [](const mx::array& feats,
           const mx::array& weights,
           const mx::array& in_rows,
           const mx::array& out_rows,
           const mx::array& kernel_ids,
           const mx::array& counts,
           int out_capacity,
           int n_kernels) {
            return sparse_conv_features(
                feats,
                weights,
                in_rows,
                out_rows,
                kernel_ids,
                counts,
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
        "out_capacity"_a,
        "n_kernels"_a,
        nb::sig(
            "def sparse_conv_features(feats: mlx.core.array, "
            "weights: mlx.core.array, in_rows: mlx.core.array, "
            "out_rows: mlx.core.array, kernel_ids: mlx.core.array, "
            "counts: mlx.core.array, out_capacity: int, "
            "n_kernels: int) -> mlx.core.array"
        ),
        "Run sparse convolution feature accumulation over a kernel relation."
    );
}

} // namespace mlx_lattice::bindings
