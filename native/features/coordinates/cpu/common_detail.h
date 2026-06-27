#include "features/coordinates/cpu/algorithms.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "foundation/array_utils.h"
#include "platform/cpu/schedule.h"

namespace mlx_lattice::coords::cpu {

namespace {

// MARK: - types

using Coord = std::array<int64_t, 4>;
using Edge = std::array<int32_t, 3>;

constexpr int kParallelCompactThreshold = 4096;

struct NeighborCandidate {
    int32_t source_row;
    float distance;
};

struct QuantizationInputs {
    const mx::array& points;
    const mx::array& batch_indices;
    int active_rows;
};

struct VoxelFeatureInputs {
    const mx::array& values;
    const mx::array& inverse_rows;
    const mx::array& voxel_counts;
    const mx::array& active_rows;
};

// FNV-1a hash matching Metal coord_hash_i32 for cross-platform consistency.
// See native/features/coordinates/metal/common.metal:53-60.
struct CoordHash {
    size_t operator()(const Coord& coord) const {
        uint32_t hash = 2166136261u;
        hash = (hash ^ static_cast<uint32_t>(coord[0])) * 16777619u;
        hash = (hash ^ static_cast<uint32_t>(coord[1])) * 16777619u;
        hash = (hash ^ static_cast<uint32_t>(coord[2])) * 16777619u;
        hash = (hash ^ static_cast<uint32_t>(coord[3])) * 16777619u;
        return static_cast<size_t>(hash & 0x7fffffffu);
    }
};

std::vector<Coord> read_coords(const mx::array& coords) {
    std::vector<Coord> out;
    out.reserve(coords.shape(0));
    if (coords.dtype() == mx::int32) {
        auto data = coords.data<int32_t>();
        for (int row = 0; row < coords.shape(0); ++row) {
            auto base = static_cast<ptrdiff_t>(row) * 4;
            out.push_back({
                data[base],
                data[base + 1],
                data[base + 2],
                data[base + 3],
            });
        }
        return out;
    }

    auto data = coords.data<int64_t>();
    for (int row = 0; row < coords.shape(0); ++row) {
        auto base = static_cast<ptrdiff_t>(row) * 4;
        out.push_back({
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        });
    }
    return out;
}

std::vector<Triple> read_offsets(const mx::array& offsets) {
    std::vector<Triple> out;
    out.reserve(offsets.shape(0));
    auto data = offsets.data<int32_t>();
    for (int row = 0; row < offsets.shape(0); ++row) {
        auto base = static_cast<ptrdiff_t>(row) * 3;
        out.push_back({data[base], data[base + 1], data[base + 2]});
    }
    return out;
}

int read_scalar_i32(const mx::array& value) { return value.data<int32_t>()[0]; }

void write_i32(mx::array& out, const std::vector<int32_t>& values) {
    auto data = out.data<int32_t>();
    std::fill(data, data + out.size(), 0);
    std::copy(values.begin(), values.end(), data);
}

void write_i32(
    mx::array& out,
    const std::vector<int32_t>& values,
    int32_t fill
) {
    auto data = out.data<int32_t>();
    std::fill(data, data + out.size(), fill);
    std::copy(values.begin(), values.end(), data);
}

void write_i64(mx::array& out, const std::vector<int64_t>& values) {
    auto data = out.data<int64_t>();
    std::fill(data, data + out.size(), 0);
    std::copy(values.begin(), values.end(), data);
}

void write_f32(mx::array& out, const std::vector<float>& values) {
    auto data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
    std::copy(values.begin(), values.end(), data);
}

void write_count(mx::array& out, int first, int second = 0) {
    auto data = out.data<int32_t>();
    std::fill(data, data + out.size(), 0);
    data[0] = first;
    if (out.size() > 1) {
        data[1] = second;
    }
}

void write_coords(
    mx::array& out,
    const std::vector<Coord>& coords,
    mx::Dtype dtype
) {
    if (dtype == mx::int32) {
        auto data = out.data<int32_t>();
        std::fill(data, data + out.size(), 0);
        for (int row = 0; row < int(coords.size()); ++row) {
            auto base = static_cast<ptrdiff_t>(row) * 4;
            for (int axis = 0; axis < 4; ++axis) {
                data[base + axis] = static_cast<int32_t>(coords[row][axis]);
            }
        }
        return;
    }

    auto data = out.data<int64_t>();
    std::fill(data, data + out.size(), 0);
    for (int row = 0; row < int(coords.size()); ++row) {
        auto base = static_cast<ptrdiff_t>(row) * 4;
        for (int axis = 0; axis < 4; ++axis) {
            data[base + axis] = coords[row][axis];
        }
    }
}
