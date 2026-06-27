#include "features/coordinates/metal/runtime.h"

#include <stdexcept>
#include <vector>

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error("Metal support is not available.");
}
} // namespace

namespace mlx_lattice::coords::metal {

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    CoordSetShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)op;
    (void)stride;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_lookup_coords(
    CoordLookupShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_morton_codes(
    CoordRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_occupancy_downsample(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_occupancy_expand(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_child_coords_from_indices(
    CoordRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_sparse_quantize(
    QuantizationSpec spec,
    int rows,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)spec;
    (void)rows;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_voxelize_features(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_voxelize_feature_grad(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_generic_kernel_relation(
    CoordRelationOp op,
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int kernel_count,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    bool direct,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)op;
    (void)rows;
    (void)kernel_count;
    (void)stride;
    (void)padding;
    (void)direct;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_target_kernel_relation(
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int target_rows,
    int kernel_count,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)rows;
    (void)target_rows;
    (void)kernel_count;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_generative_kernel_relation(
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int kernel_count,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)rows;
    (void)kernel_count;
    (void)stride;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_relation_grouped_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_relation_direct_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_relation_implicit_gemm_view(
    RelationImplicitGemmViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)op;
    (void)shape;
    (void)radius_squared;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

} // namespace mlx_lattice::coords::metal
