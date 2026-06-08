#include "ops/exec.h"

#include <vector>

#include "ops/coords.h"
#include "ops/exec/dispatch.h"
#include "ops/exec/validation.h"

namespace mlx_lattice {

namespace {

mx::array make_offsets_array(const std::vector<Triple>& offsets) {
    std::vector<int32_t> flat;
    flat.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat.insert(flat.end(), offset.begin(), offset.end());
    }
    return mx::array(
        flat.begin(), mx::Shape{int(offsets.size()), 3}, mx::int32
    );
}

} // namespace

mx::array spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    int n_out_rows
) {
    validate_spmm_edges(
        feats, weights, in_rows, out_rows, kernel_ids, edge_count, n_out_rows
    );
    return dispatch_spmm_edges(
        feats, weights, in_rows, out_rows, kernel_ids, edge_count, n_out_rows
    );
}

mx::array pool_sum_edges(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    int n_out_rows
) {
    validate_pool_edges(feats, in_rows, out_rows, edge_count, n_out_rows);
    return dispatch_pool_edges(
        PoolReduceOp::Sum, feats, in_rows, out_rows, edge_count, n_out_rows
    );
}

mx::array pool_max_edges(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    int n_out_rows
) {
    validate_pool_edges(feats, in_rows, out_rows, edge_count, n_out_rows);
    return dispatch_pool_edges(
        PoolReduceOp::Max, feats, in_rows, out_rows, edge_count, n_out_rows
    );
}

NativeSparseTensorOutput sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    validate_sparse_conv(coords, active_rows, feats, weights);
    auto offsets = op == SparseMapOp::Generative
                       ? kernel_offsets(kernel_size)
                       : kernel_offsets(kernel_size, dilation);
    if (weights.shape(0) != int(offsets.size())) {
        throw std::invalid_argument(
            "weight kernel rows must match the sparse convolution kernel."
        );
    }
    return dispatch_sparse_conv(
        op,
        coords,
        active_rows,
        feats,
        weights,
        make_offsets_array(offsets),
        stride,
        padding
    );
}

NativeSparseTensorOutput sparse_pool(
    PoolReduceOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    validate_sparse_pool(coords, active_rows, feats);
    return dispatch_sparse_pool(
        op,
        coords,
        active_rows,
        feats,
        make_offsets_array(kernel_offsets(kernel_size, dilation)),
        stride,
        padding
    );
}

} // namespace mlx_lattice
