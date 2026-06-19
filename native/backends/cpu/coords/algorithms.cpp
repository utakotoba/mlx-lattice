#include "backends/cpu/coords/algorithms.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"

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
// See native/backends/metal/coords/common.metal:53-60.
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

// MARK: - coords

int64_t floor_div(int64_t value, int64_t divisor) {
    auto quotient = value / divisor;
    auto remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

std::vector<Coord>
downsample_values(const std::vector<Coord>& coords, Triple stride) {
    std::vector<Coord> out;
    out.reserve(coords.size());
    std::unordered_set<Coord, CoordHash> seen;
    seen.reserve(coords.size());

    for (auto coord : coords) {
        Coord quantized = {
            coord[0],
            floor_div(coord[1], stride[0]),
            floor_div(coord[2], stride[1]),
            floor_div(coord[3], stride[2]),
        };
        if (seen.insert(quantized).second) {
            out.push_back(quantized);
        }
    }
    return out;
}

uint64_t split_morton_3(uint64_t value) {
    value &= 0x1fffffULL;
    value = (value | (value << 32)) & 0x1f00000000ffffULL;
    value = (value | (value << 16)) & 0x1f0000ff0000ffULL;
    value = (value | (value << 8)) & 0x100f00f00f00f00fULL;
    value = (value | (value << 4)) & 0x10c30c30c30c30c3ULL;
    value = (value | (value << 2)) & 0x1249249249249249ULL;
    return value;
}

int64_t morton_code(Coord coord) {
    auto code = split_morton_3(static_cast<uint64_t>(coord[1])) |
                (split_morton_3(static_cast<uint64_t>(coord[2])) << 1) |
                (split_morton_3(static_cast<uint64_t>(coord[3])) << 2);
    code += static_cast<uint64_t>(coord[0]) << 60;
    return static_cast<int64_t>(code);
}

std::vector<int64_t> morton_code_values(const mx::array& coords) {
    auto values = read_coords(coords);
    std::vector<int64_t> out;
    out.reserve(values.size());
    for (auto coord : values) {
        out.push_back(morton_code(coord));
    }
    return out;
}

int32_t child_index_for_coord(const Coord& coord) {
    auto index = int32_t(coord[1] & 1) + int32_t((coord[2] & 1) << 1) +
                 int32_t((coord[3] & 1) << 2);
    return int32_t(1 << index);
}

Coord voxel_coord_for_point(
    const float* points,
    const int32_t* batch_indices,
    int row,
    FloatTriple voxel_size,
    FloatTriple origin
) {
    auto base = static_cast<ptrdiff_t>(row) * 3;
    return {
        batch_indices[row],
        static_cast<int64_t>(
            std::floor((points[base] - origin[0]) / voxel_size[0])
        ),
        static_cast<int64_t>(
            std::floor((points[base + 1] - origin[1]) / voxel_size[1])
        ),
        static_cast<int64_t>(
            std::floor((points[base + 2] - origin[2]) / voxel_size[2])
        ),
    };
}

void write_sparse_quantization(
    std::vector<mx::array>& outputs,
    QuantizationInputs inputs,
    QuantizationSpec spec
) {
    auto point_data = inputs.points.data<float>();
    auto batch_data = inputs.batch_indices.data<int32_t>();
    auto point_count = std::min(inputs.active_rows, inputs.points.shape(0));

    std::vector<Coord> out_coords;
    std::vector<int32_t> inverse_rows(inputs.points.shape(0), -1);
    std::vector<int32_t> counts;
    std::unordered_map<Coord, int32_t, CoordHash> out_rows;
    out_coords.reserve(point_count);
    counts.reserve(point_count);
    out_rows.reserve(point_count);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        auto candidate = voxel_coord_for_point(
            point_data, batch_data, point_row, spec.voxel_size, spec.origin
        );
        auto [match, inserted] = out_rows.emplace(
            candidate, static_cast<int32_t>(out_coords.size())
        );
        if (inserted) {
            out_coords.push_back(candidate);
            counts.push_back(0);
        }
        inverse_rows[point_row] = match->second;
        counts[match->second] += 1;
    }

    write_coords(outputs[0], out_coords, mx::int32);
    write_count(outputs[1], int(out_coords.size()));
    write_i32(outputs[2], inverse_rows, -1);
    write_i32(outputs[3], counts);
}

float voxel_reduce_scale(
    VoxelReduceOp reduce,
    const int32_t* voxel_counts,
    int voxel_row
) {
    if (reduce == VoxelReduceOp::Mean) {
        return 1.0F / static_cast<float>(std::max(voxel_counts[voxel_row], 1));
    }
    return 1.0F;
}

void write_voxel_features(
    mx::array& out,
    VoxelReduceOp reduce,
    VoxelFeatureInputs inputs,
    VoxelFeatureShape shape
) {
    auto point_count =
        std::min(read_scalar_i32(inputs.active_rows), shape.point_rows);
    auto feat_data = inputs.values.data<float>();
    auto inverse_data = inputs.inverse_rows.data<int32_t>();
    auto count_data = inputs.voxel_counts.data<int32_t>();
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        auto voxel_row = inverse_data[point_row];
        if (voxel_row < 0 || voxel_row >= shape.voxel_rows) {
            continue;
        }
        auto scale = voxel_reduce_scale(reduce, count_data, voxel_row);
        for (int channel = 0; channel < shape.channels; ++channel) {
            auto point_index =
                static_cast<ptrdiff_t>(point_row) * shape.channels + channel;
            auto voxel_index =
                static_cast<ptrdiff_t>(voxel_row) * shape.channels + channel;
            out_data[voxel_index] += feat_data[point_index] * scale;
        }
    }
}

void write_voxel_feature_grad(
    mx::array& out,
    VoxelReduceOp reduce,
    VoxelFeatureInputs inputs,
    VoxelFeatureShape shape
) {
    auto point_count =
        std::min(read_scalar_i32(inputs.active_rows), shape.point_rows);
    auto cotangent_data = inputs.values.data<float>();
    auto inverse_data = inputs.inverse_rows.data<int32_t>();
    auto count_data = inputs.voxel_counts.data<int32_t>();
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        auto voxel_row = inverse_data[point_row];
        if (voxel_row < 0 || voxel_row >= shape.voxel_rows) {
            continue;
        }
        auto scale = voxel_reduce_scale(reduce, count_data, voxel_row);
        for (int channel = 0; channel < shape.channels; ++channel) {
            auto point_index =
                static_cast<ptrdiff_t>(point_row) * shape.channels + channel;
            auto voxel_index =
                static_cast<ptrdiff_t>(voxel_row) * shape.channels + channel;
            out_data[point_index] = cotangent_data[voxel_index] * scale;
        }
    }
}

std::unordered_map<Coord, int32_t, CoordHash>
first_row_map(const std::vector<Coord>& coords) {
    std::unordered_map<Coord, int32_t, CoordHash> rows;
    rows.reserve(coords.size());
    for (int row = 0; row < int(coords.size()); ++row) {
        rows.emplace(coords[row], static_cast<int32_t>(row));
    }
    return rows;
}

Coord kernel_input_coord(
    Coord out_coord,
    Triple offset,
    Triple stride,
    Triple padding
) {
    return {
        out_coord[0],
        out_coord[1] * stride[0] + offset[0] - padding[0],
        out_coord[2] * stride[1] + offset[1] - padding[1],
        out_coord[3] * stride[2] + offset[2] - padding[2],
    };
}

void write_map_rows(
    std::vector<mx::array>& outputs,
    const std::vector<Edge>& edges,
    int out_capacity
) {
    std::vector<int32_t> in_rows;
    std::vector<int32_t> out_rows;
    std::vector<int32_t> kernel_ids;
    std::vector<int32_t> row_offsets(
        static_cast<std::size_t>(out_capacity) + 1, 0
    );
    in_rows.reserve(edges.size());
    out_rows.reserve(edges.size());
    kernel_ids.reserve(edges.size());
    auto current_out = 0;
    for (auto edge : edges) {
        while (current_out <= edge[1] && current_out < out_capacity) {
            row_offsets[static_cast<std::size_t>(current_out)] =
                int32_t(in_rows.size());
            ++current_out;
        }
        in_rows.push_back(edge[0]);
        out_rows.push_back(edge[1]);
        kernel_ids.push_back(edge[2]);
    }
    while (current_out <= out_capacity) {
        row_offsets[static_cast<std::size_t>(current_out)] =
            int32_t(in_rows.size());
        ++current_out;
    }

    write_i32(outputs[RelationInRows], in_rows);
    write_i32(outputs[RelationOutRows], out_rows);
    write_i32(outputs[RelationKernelIds], kernel_ids);
    write_i32(outputs[RelationRowOffsets], row_offsets);
}

void write_map(
    std::vector<mx::array>& outputs,
    const std::vector<Edge>& edges,
    const std::vector<Coord>& out_coords,
    mx::Dtype coord_dtype,
    bool compact
) {
    auto row_major_edges = edges;
    std::stable_sort(
        row_major_edges.begin(),
        row_major_edges.end(),
        [](const Edge& lhs, const Edge& rhs) {
            if (lhs[1] != rhs[1]) {
                return lhs[1] < rhs[1];
            }
            if (lhs[2] != rhs[2]) {
                return lhs[2] < rhs[2];
            }
            return lhs[0] < rhs[0];
        }
    );
    write_map_rows(outputs, row_major_edges, int(out_coords.size()));
    write_coords(outputs[RelationOutCoords], out_coords, coord_dtype);
    if (compact) {
        write_count(
            outputs[RelationCounts],
            int(row_major_edges.size()),
            int(out_coords.size())
        );
    }
}

void write_relation_grouped_view(
    std::vector<mx::array>& outputs,
    const std::vector<mx::array>& inputs,
    RelationGroupedViewShape shape
) {
    auto edge_count =
        std::clamp(read_scalar_i32(inputs[1]), 0, shape.edge_capacity);
    auto group_ids = inputs[0].data<int32_t>();
    std::vector<int32_t> offsets(
        static_cast<std::size_t>(shape.group_count) + 1, 0
    );
    for (int edge = 0; edge < edge_count; ++edge) {
        auto group = group_ids[edge];
        if (group >= 0 && group < shape.group_count) {
            ++offsets[static_cast<std::size_t>(group) + 1];
        }
    }
    for (int group = 0; group < shape.group_count; ++group) {
        offsets[static_cast<std::size_t>(group) + 1] +=
            offsets[static_cast<std::size_t>(group)];
    }

    auto cursors = offsets;
    std::vector<int32_t> edge_ids(
        static_cast<std::size_t>(shape.edge_capacity), -1
    );
    for (int edge = 0; edge < edge_count; ++edge) {
        auto group = group_ids[edge];
        if (group >= 0 && group < shape.group_count) {
            auto slot = cursors[static_cast<std::size_t>(group)]++;
            edge_ids[static_cast<std::size_t>(slot)] = edge;
        }
    }

    write_i32(outputs[RelationViewRowOffsets], offsets);
    write_i32(outputs[RelationViewEdgeIds], edge_ids, -1);
}

void write_relation_direct_view(
    std::vector<mx::array>& outputs,
    const std::vector<mx::array>& inputs,
    RelationGroupedViewShape shape
) {
    auto edge_count =
        std::clamp(read_scalar_i32(inputs[1]), 0, shape.edge_capacity);
    auto group_ids = inputs[0].data<int32_t>();
    std::vector<int32_t> edge_ids(
        static_cast<std::size_t>(shape.group_count), -1
    );
    for (int edge = 0; edge < edge_count; ++edge) {
        auto group = group_ids[edge];
        if (group >= 0 && group < shape.group_count) {
            edge_ids[static_cast<std::size_t>(group)] = edge;
        }
    }
    write_i32(outputs[0], edge_ids, -1);
}

// MARK: - set ops

std::vector<Coord> union_values(const mx::array& lhs, const mx::array& rhs) {
    auto lhs_values = read_coords(lhs);
    auto rhs_values = read_coords(rhs);
    std::vector<Coord> out;
    out.reserve(lhs_values.size() + rhs_values.size());
    std::unordered_set<Coord, CoordHash> seen;
    seen.reserve(lhs_values.size() + rhs_values.size());

    for (auto coord : lhs_values) {
        if (seen.insert(coord).second) {
            out.push_back(coord);
        }
    }
    for (auto coord : rhs_values) {
        if (seen.insert(coord).second) {
            out.push_back(coord);
        }
    }
    return out;
}

std::vector<Coord>
intersection_values(const mx::array& lhs, const mx::array& rhs) {
    auto lhs_values = read_coords(lhs);
    auto rhs_values = read_coords(rhs);
    std::unordered_set<Coord, CoordHash> rhs_seen;
    rhs_seen.reserve(rhs_values.size());
    for (auto coord : rhs_values) {
        rhs_seen.insert(coord);
    }

    std::vector<Coord> out;
    out.reserve(std::min(lhs_values.size(), rhs_values.size()));
    std::unordered_set<Coord, CoordHash> emitted;
    emitted.reserve(lhs_values.size());
    for (auto coord : lhs_values) {
        if (rhs_seen.find(coord) != rhs_seen.end() &&
            emitted.insert(coord).second) {
            out.push_back(coord);
        }
    }
    return out;
}

std::vector<int32_t>
lookup_values(const mx::array& coords, const mx::array& queries) {
    auto rows = first_row_map(read_coords(coords));
    auto query_values = read_coords(queries);
    std::vector<int32_t> out;
    out.reserve(query_values.size());
    for (auto coord : query_values) {
        auto match = rows.find(coord);
        out.push_back(match == rows.end() ? -1 : match->second);
    }
    return out;
}

bool same_batch(Coord lhs, Coord rhs) { return lhs[0] == rhs[0]; }

float squared_spatial_distance(Coord lhs, Coord rhs) {
    auto dx = static_cast<float>(lhs[1] - rhs[1]);
    auto dy = static_cast<float>(lhs[2] - rhs[2]);
    auto dz = static_cast<float>(lhs[3] - rhs[3]);
    return dx * dx + dy * dy + dz * dz;
}

bool closer_neighbor(
    const NeighborCandidate& lhs,
    const NeighborCandidate& rhs
) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.source_row < rhs.source_row;
}

std::vector<NeighborCandidate> radius_candidates(
    const std::unordered_map<Coord, int32_t, CoordHash>& source_rows_by_coord,
    Coord query,
    float radius_squared
) {
    auto radius = static_cast<int>(std::ceil(std::sqrt(radius_squared)));
    std::vector<NeighborCandidate> candidates;
    auto reserve_radius = static_cast<std::size_t>(radius);
    candidates.reserve(reserve_radius * reserve_radius * reserve_radius);
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                auto distance = static_cast<float>(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }
                auto target = Coord{
                    query[0],
                    query[1] + dx,
                    query[2] + dy,
                    query[3] + dz,
                };
                auto match = source_rows_by_coord.find(target);
                if (match != source_rows_by_coord.end()) {
                    candidates.push_back({match->second, distance});
                }
            }
        }
    }
    return candidates;
}

void write_neighbor_relation(
    std::vector<mx::array>& outputs,
    NeighborRelationOp op,
    const mx::array& source_coords,
    int source_active_rows,
    const mx::array& query_coords,
    int query_active_rows,
    NeighborRelationShape shape,
    float radius_squared
) {
    auto source_values = read_coords(source_coords);
    source_values.resize(
        std::min(source_active_rows, int(source_values.size()))
    );
    auto query_values = read_coords(query_coords);
    query_values.resize(std::min(query_active_rows, int(query_values.size())));

    std::vector<int32_t> query_rows;
    std::vector<int32_t> source_rows;
    std::vector<int32_t> neighbor_ids;
    std::vector<int32_t> row_offsets(query_values.size() + 1, 0);
    std::vector<float> distances;
    query_rows.reserve(query_values.size() * shape.max_neighbors);
    source_rows.reserve(query_values.size() * shape.max_neighbors);
    neighbor_ids.reserve(query_values.size() * shape.max_neighbors);
    distances.reserve(query_values.size() * shape.max_neighbors);

    std::unordered_map<Coord, int32_t, CoordHash> source_rows_by_coord;
    if (op == NeighborRelationOp::Radius) {
        source_rows_by_coord.reserve(source_values.size());
        for (int source_row = 0; source_row < int(source_values.size());
             ++source_row) {
            source_rows_by_coord.emplace(
                source_values[source_row], static_cast<int32_t>(source_row)
            );
        }
    }

    std::vector<NeighborCandidate> candidates;
    candidates.reserve(source_values.size());
    for (int query_row = 0; query_row < int(query_values.size()); ++query_row) {
        row_offsets[query_row] = static_cast<int32_t>(query_rows.size());
        candidates.clear();
        auto query = query_values[query_row];
        if (op == NeighborRelationOp::Radius) {
            candidates =
                radius_candidates(source_rows_by_coord, query, radius_squared);
        } else {
            for (int source_row = 0; source_row < int(source_values.size());
                 ++source_row) {
                auto source = source_values[source_row];
                if (!same_batch(query, source)) {
                    continue;
                }
                auto distance = squared_spatial_distance(query, source);
                candidates.push_back(
                    {static_cast<int32_t>(source_row), distance}
                );
            }
        }
        std::sort(candidates.begin(), candidates.end(), closer_neighbor);
        auto limit = std::min(shape.max_neighbors, int(candidates.size()));
        for (int neighbor = 0; neighbor < limit; ++neighbor) {
            query_rows.push_back(static_cast<int32_t>(query_row));
            source_rows.push_back(candidates[neighbor].source_row);
            neighbor_ids.push_back(static_cast<int32_t>(neighbor));
            distances.push_back(candidates[neighbor].distance);
        }
    }
    row_offsets[query_values.size()] = static_cast<int32_t>(query_rows.size());

    write_i32(outputs[NeighborQueryRows], query_rows);
    write_i32(outputs[NeighborSourceRows], source_rows);
    write_i32(outputs[NeighborIds], neighbor_ids);
    write_f32(outputs[NeighborDistances], distances);
    write_i32(outputs[NeighborRowOffsets], row_offsets);
    write_count(
        outputs[NeighborCounts],
        int(query_rows.size()),
        int(query_values.size())
    );
}

// MARK: - relations

void write_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
    auto rows = first_row_map(values);
    bool identity_out = stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
    auto out_values = identity_out ? values : downsample_values(values, stride);

    std::vector<Edge> edges;
    edges.reserve(out_values.size() * offsets.size());
    for (int out_row = 0; out_row < int(out_values.size()); ++out_row) {
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            auto candidate = kernel_input_coord(
                out_values[out_row], offset, stride, padding
            );
            auto match = rows.find(candidate);
            if (match != rows.end()) {
                edges.push_back({
                    match->second,
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
    }

    write_map_rows(outputs, edges, int(out_values.size()));
    write_coords(outputs[RelationOutCoords], out_values, coords.dtype());
    write_count(
        outputs[RelationCounts], int(edges.size()), int(out_values.size())
    );
}

void write_target_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const mx::array& target_coords,
    int target_active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
    auto target_values = read_coords(target_coords);
    auto target_active =
        std::min(target_active_rows, int(target_values.size()));
    auto rows = first_row_map(values);

    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(target_active) * offsets.size());
    for (int out_row = 0; out_row < target_active; ++out_row) {
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto candidate = kernel_input_coord(
                target_values[out_row], offsets[kernel], stride, padding
            );
            auto match = rows.find(candidate);
            if (match != rows.end()) {
                edges.push_back({
                    match->second,
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
    }

    write_map_rows(outputs, edges, int(target_values.size()));
    write_count(outputs[RelationCounts], int(edges.size()), target_active);
}

void write_generative_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
    std::vector<Edge> edges;
    std::vector<Coord> out_values;
    edges.reserve(values.size() * offsets.size());
    out_values.reserve(values.size() * offsets.size());

    for (int in_row = 0; in_row < int(values.size()); ++in_row) {
        auto coord = values[in_row];
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            auto out_row = int(out_values.size());
            out_values.push_back({
                coord[0],
                coord[1] * stride[0] + offset[0],
                coord[2] * stride[1] + offset[1],
                coord[3] * stride[2] + offset[2],
            });
            edges.push_back({
                static_cast<int32_t>(in_row),
                static_cast<int32_t>(out_row),
                static_cast<int32_t>(kernel),
            });
        }
    }

    write_map_rows(outputs, edges, int(out_values.size()));
    write_coords(outputs[RelationOutCoords], out_values, coords.dtype());
    write_count(
        outputs[RelationCounts], int(edges.size()), int(out_values.size())
    );
}

void write_transposed_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding,
    bool direct
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));

    if (direct) {
        std::vector<Edge> edges;
        std::vector<Coord> out_values;
        edges.reserve(values.size() * offsets.size());
        out_values.reserve(values.size() * offsets.size());
        for (int in_row = 0; in_row < int(values.size()); ++in_row) {
            auto coord = values[in_row];
            for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
                auto out_row = in_row * int(offsets.size()) + kernel;
                out_values.push_back({
                    coord[0],
                    coord[1] * stride[0] + offsets[kernel][0] - padding[0],
                    coord[2] * stride[1] + offsets[kernel][1] - padding[1],
                    coord[3] * stride[2] + offsets[kernel][2] - padding[2],
                });
                edges.push_back({
                    static_cast<int32_t>(in_row),
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
        write_map_rows(outputs, edges, int(out_values.size()));
        write_coords(outputs[RelationOutCoords], out_values, coords.dtype());
        write_count(
            outputs[RelationCounts], int(edges.size()), int(out_values.size())
        );
        return;
    }

    std::vector<Edge> edges;
    std::vector<Coord> out_values;
    std::unordered_map<Coord, int32_t, CoordHash> out_rows;
    edges.reserve(values.size() * offsets.size());
    out_values.reserve(values.size() * offsets.size());
    out_rows.reserve(values.size() * offsets.size());

    for (int in_row = 0; in_row < int(values.size()); ++in_row) {
        auto coord = values[in_row];
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            Coord candidate = {
                coord[0],
                coord[1] * stride[0] + offset[0] - padding[0],
                coord[2] * stride[1] + offset[1] - padding[1],
                coord[3] * stride[2] + offset[2] - padding[2],
            };
            auto [match, inserted] = out_rows.emplace(
                candidate, static_cast<int32_t>(out_values.size())
            );
            if (inserted) {
                out_values.push_back(candidate);
            }
            edges.push_back({
                static_cast<int32_t>(in_row),
                match->second,
                static_cast<int32_t>(kernel),
            });
        }
    }

    write_map(outputs, edges, out_values, coords.dtype(), true);
}

} // namespace

// MARK: - primitive eval

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, stride](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            std::vector<Coord> values;
            switch (op) {
            case CoordSetOp::Downsample:
                values = downsample_values(read_coords(task_inputs[0]), stride);
                break;
            case CoordSetOp::Union:
                values = union_values(task_inputs[0], task_inputs[1]);
                break;
            case CoordSetOp::Intersection:
                values = intersection_values(task_inputs[0], task_inputs[1]);
                break;
            }

            write_coords(task_outputs[0], values, task_inputs[0].dtype());
            write_count(task_outputs[1], int(values.size()));
        }
    );
}

void eval_lookup_coords(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            write_i32(
                task_outputs[0], lookup_values(task_inputs[0], task_inputs[1])
            );
        }
    );
}

void eval_morton_codes(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            write_i64(task_outputs[0], morton_code_values(task_inputs[0]));
        }
    );
}

void eval_occupancy_downsample(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            auto coords = read_coords(task_inputs[0]);
            auto logical_rows =
                std::min(read_scalar_i32(task_inputs[1]), int(coords.size()));
            std::vector<Coord> out_coords;
            std::vector<int32_t> occupancy;
            std::unordered_map<Coord, int32_t, CoordHash> out_rows;
            out_coords.reserve(logical_rows);
            occupancy.reserve(logical_rows);
            out_rows.reserve(logical_rows);
            for (int row = 0; row < logical_rows; ++row) {
                const auto& coord = coords[row];
                Coord parent = {
                    coord[0],
                    floor_div(coord[1], 2),
                    floor_div(coord[2], 2),
                    floor_div(coord[3], 2),
                };
                auto [match, inserted] = out_rows.emplace(
                    parent, static_cast<int32_t>(out_coords.size())
                );
                if (inserted) {
                    out_coords.push_back(parent);
                    occupancy.push_back(0);
                }
                auto child = child_index_for_coord(coord);
                occupancy[match->second] |= child;
            }
            write_coords(task_outputs[0], out_coords, task_inputs[0].dtype());
            write_count(task_outputs[1], int(out_coords.size()));
            write_i32(task_outputs[2], occupancy);
        }
    );
}

void eval_occupancy_expand(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            auto coords = read_coords(task_inputs[0]);
            auto logical_rows =
                std::min(read_scalar_i32(task_inputs[1]), int(coords.size()));
            auto occupancy = task_inputs[2].data<int32_t>();
            std::vector<Coord> out_coords;
            std::vector<int32_t> parent_rows;
            std::vector<int32_t> child_indices;
            auto expanded = static_cast<size_t>(logical_rows) * 8;
            out_coords.reserve(expanded);
            parent_rows.reserve(expanded);
            child_indices.reserve(expanded);
            for (int row = 0; row < logical_rows; ++row) {
                const auto& parent = coords[row];
                auto bits = occupancy[row];
                for (int child = 0; child < 8; ++child) {
                    if ((bits & (1 << child)) == 0) {
                        continue;
                    }
                    Coord expanded = {
                        parent[0],
                        parent[1] * 2 + (child & 1),
                        parent[2] * 2 + ((child >> 1) & 1),
                        parent[3] * 2 + ((child >> 2) & 1),
                    };
                    parent_rows.push_back(row);
                    child_indices.push_back(child);
                    out_coords.push_back(expanded);
                }
            }
            write_coords(task_outputs[0], out_coords, task_inputs[0].dtype());
            write_count(task_outputs[1], int(out_coords.size()));
            write_i32(task_outputs[2], parent_rows);
            write_i32(task_outputs[3], child_indices);
        }
    );
}

void eval_child_coords_from_indices(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            auto coords = read_coords(task_inputs[0]);
            auto child_indices = task_inputs[1].data<int32_t>();
            std::vector<Coord> out_coords;
            out_coords.reserve(coords.size());
            for (int row = 0; row < int(coords.size()); ++row) {
                auto child = int(child_indices[row]);
                const auto& parent = coords[row];
                out_coords.push_back({
                    parent[0],
                    parent[1] * 2 + (child & 1),
                    parent[2] * 2 + ((child >> 1) & 1),
                    parent[3] * 2 + ((child >> 2) & 1),
                });
            }
            write_coords(task_outputs[0], out_coords, task_inputs[0].dtype());
        }
    );
}

void eval_sparse_quantize(
    QuantizationSpec spec,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [spec](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_sparse_quantization(
                task_outputs,
                QuantizationInputs{
                    task_inputs[0],
                    task_inputs[1],
                    read_scalar_i32(task_inputs[2]),
                },
                spec
            );
        }
    );
}

void eval_voxelize_features(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_voxel_features(
                task_outputs[0],
                reduce,
                VoxelFeatureInputs{
                    task_inputs[0],
                    task_inputs[1],
                    task_inputs[2],
                    task_inputs[3],
                },
                shape
            );
        }
    );
}

void eval_voxelize_feature_grad(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_voxel_feature_grad(
                task_outputs[0],
                reduce,
                VoxelFeatureInputs{
                    task_inputs[0],
                    task_inputs[1],
                    task_inputs[2],
                    task_inputs[3],
                },
                shape
            );
        }
    );
}

void eval_generic_kernel_relation(
    CoordRelationOp op,
    Triple stride,
    Triple padding,
    bool direct,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, stride, padding, direct](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            auto offsets = read_offsets(task_inputs[1]);
            auto active_rows = read_scalar_i32(task_inputs[2]);

            switch (op) {
            case CoordRelationOp::Forward:
                write_kernel_relation(
                    task_outputs,
                    task_inputs[0],
                    active_rows,
                    offsets,
                    stride,
                    padding
                );
                break;
            case CoordRelationOp::Transposed:
                write_transposed_kernel_relation(
                    task_outputs,
                    task_inputs[0],
                    active_rows,
                    offsets,
                    stride,
                    padding,
                    direct
                );
                break;
            }
        }
    );
}

void eval_target_kernel_relation(
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [stride, padding](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_target_kernel_relation(
                task_outputs,
                task_inputs[0],
                read_scalar_i32(task_inputs[2]),
                task_inputs[3],
                read_scalar_i32(task_inputs[4]),
                read_offsets(task_inputs[1]),
                stride,
                padding
            );
        }
    );
}

void eval_generative_kernel_relation(
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [stride](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            auto active_rows = read_scalar_i32(task_inputs[2]);
            write_generative_relation(
                task_outputs,
                task_inputs[0],
                active_rows,
                read_offsets(task_inputs[1]),
                stride
            );
        }
    );
}

void eval_relation_grouped_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) { write_relation_grouped_view(task_outputs, task_inputs, shape); }
    );
}

void eval_relation_direct_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) { write_relation_direct_view(task_outputs, task_inputs, shape); }
    );
}

void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, shape, radius_squared](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_neighbor_relation(
                task_outputs,
                op,
                task_inputs[0],
                read_scalar_i32(task_inputs[2]),
                task_inputs[1],
                read_scalar_i32(task_inputs[3]),
                shape,
                radius_squared
            );
        }
    );
}

} // namespace mlx_lattice::coords::cpu
