#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

using Triple = std::array<int, 3>;
using FloatTriple = std::array<float, 3>;

enum class CoordSetOp : std::uint8_t {
    Downsample,
    Union,
    Intersection,
};

enum class CoordRelationOp : std::uint8_t {
    Forward,
    Transposed,
};

enum CoordRelationOutputSlot : std::size_t {
    RelationInRows = 0,
    RelationOutRows,
    RelationKernelIds,
    RelationRowOffsets,
    RelationOutCoords,
    RelationCounts,
    RelationBaseOutputCount,
};

enum RelationGroupedViewOutputSlot : std::size_t {
    RelationViewRowOffsets = 0,
    RelationViewEdgeIds,
    RelationGroupedViewOutputCount,
};

enum class NeighborRelationOp : std::uint8_t {
    Knn,
    Radius,
};

enum class VoxelReduceOp : std::uint8_t {
    Sum,
    Mean,
};

enum NeighborRelationOutputSlot : std::size_t {
    NeighborQueryRows = 0,
    NeighborSourceRows,
    NeighborIds,
    NeighborDistances,
    NeighborRowOffsets,
    NeighborCounts,
    NeighborOutputCount,
};

struct NativeKernelRelation {
    // Edges are ordered by output row. Other execution views stay lazy until
    // a backward path or diagnostic consumer requests them.
    mx::array in_rows;
    mx::array out_rows;
    mx::array kernel_ids;
    mx::array row_offsets;
    mx::array out_coords;
    mx::array counts;
    mx::array in_row_offsets;
    mx::array in_edge_ids;
    mx::array kernel_row_offsets;
    mx::array kernel_edge_ids;
};

struct NativeKernelRelationViews {
    mx::array in_row_offsets;
    mx::array in_edge_ids;
    mx::array kernel_row_offsets;
    mx::array kernel_edge_ids;
};

struct NativeRelationGroupedView {
    mx::array row_offsets;
    mx::array edge_ids;
};

struct NativeRelationDirectView {
    mx::array edge_ids;
};

struct NativeNeighborRelation {
    mx::array query_rows;
    mx::array source_rows;
    mx::array neighbor_ids;
    mx::array distances;
    mx::array row_offsets;
    mx::array counts;
};

struct NativeCoordSet {
    mx::array coords;
    mx::array count;
};

struct QuantizationSpec {
    FloatTriple voxel_size;
    FloatTriple origin;
};

struct NativeSparseQuantization {
    mx::array coords;
    mx::array active_rows;
    mx::array inverse_rows;
    mx::array counts;
};

struct CoordSetShape {
    int lhs_rows;
    int rhs_rows;
};

struct CoordLookupShape {
    int rows;
    int query_rows;
};

struct NeighborRelationShape {
    int source_rows;
    int query_rows;
    int max_neighbors;
};

struct RelationGroupedViewShape {
    int edge_capacity;
    int group_count;
};

inline bool
operator==(RelationGroupedViewShape lhs, RelationGroupedViewShape rhs) {
    return lhs.edge_capacity == rhs.edge_capacity &&
           lhs.group_count == rhs.group_count;
}

inline bool
operator!=(RelationGroupedViewShape lhs, RelationGroupedViewShape rhs) {
    return !(lhs == rhs);
}

struct VoxelFeatureShape {
    int point_rows;
    int voxel_rows;
    int channels;
};

inline bool operator==(VoxelFeatureShape lhs, VoxelFeatureShape rhs) {
    return lhs.point_rows == rhs.point_rows &&
           lhs.voxel_rows == rhs.voxel_rows && lhs.channels == rhs.channels;
}

inline bool operator!=(VoxelFeatureShape lhs, VoxelFeatureShape rhs) {
    return !(lhs == rhs);
}

} // namespace mlx_lattice
