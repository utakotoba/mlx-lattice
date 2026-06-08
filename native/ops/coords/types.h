#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

using Triple = std::array<int, 3>;

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
    RelationOutCoords,
    RelationCounts,
    RelationOutputCount,
};

constexpr std::size_t DirectRelationOutputCount = RelationOutputCount - 1;

struct NativeKernelRelation {
    // Baseline edge-COO view of a logical relation. Future native execution
    // plans can lower the same relation into CSR, buckets, or implicit GEMM.
    mx::array in_rows;
    mx::array out_rows;
    mx::array kernel_ids;
    mx::array out_coords;
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

} // namespace mlx_lattice
