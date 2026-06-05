#pragma once

#include <array>
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

enum class CoordMapOp : std::uint8_t {
    Forward,
    Transposed,
};

struct NativeOutputCsrView {
    mx::array offsets;
    mx::array in_rows;
    mx::array kernel_ids;
};

struct NativeKernelBucketView {
    mx::array offsets;
    mx::array in_rows;
    mx::array out_rows;
};

struct NativeInputCsrView {
    mx::array offsets;
    mx::array out_rows;
    mx::array kernel_ids;
};

struct NativeKernelMap {
    mx::array in_rows;
    mx::array out_rows;
    mx::array kernel_ids;
    mx::array out_coords;
    mx::array kernel_offsets;
    NativeOutputCsrView output_csr;
    NativeKernelBucketView kernel_buckets;
    NativeInputCsrView input_csr;
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
