#pragma once

#include <vector>

#include "ops/exec/types.h"

namespace mlx_lattice::exec::cpu {

void eval_spmm_edges(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_spmm_edges_input_grad(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_spmm_edges_weight_grad(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_pool_edges(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_pool_edges_grad(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_pool_max_edges_jvp(
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sparse_conv(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sparse_conv_input_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sparse_conv_weight_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sparse_pool(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sparse_pool_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sparse_pool_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::exec::cpu
