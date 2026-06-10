#pragma once

#include <vector>

#include "ops/coords/types.h"

namespace mlx_lattice::coords::cpu {

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_lookup_coords(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sparse_quantize(
    QuantizationSpec spec,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_voxelize_features(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_voxelize_feature_grad(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_generic_kernel_relation(
    CoordRelationOp op,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_generative_kernel_relation(
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_relation_grouped_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_relation_direct_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::coords::cpu
