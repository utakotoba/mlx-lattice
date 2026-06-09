#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/coords/common.metal"

// MARK: - set ops

[[kernel]] void downsample_coords_i32(
    device const int* coords [[buffer(0)]],
    device int* out_coords [[buffer(1)]],
    device int* count [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& stride_x [[buffer(4)]],
    constant const int& stride_y [[buffer(5)]],
    constant const int& stride_z [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_count = 0;
    for (int row = 0; row < rows; ++row) {
        int base = row * 4;
        int candidate[4] = {
            coords[base],
            floor_div_int(coords[base + 1], stride_x),
            floor_div_int(coords[base + 2], stride_y),
            floor_div_int(coords[base + 3], stride_z),
        };
        bool seen = false;
        for (int prev = 0; prev < out_count; ++prev) {
            if (coord4_equal(candidate, out_coords, prev)) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            write_coord(out_coords, out_count, candidate);
            out_count += 1;
        }
    }
    count[0] = out_count;
}

[[kernel]] void union_coords_i32(
    device const int* lhs [[buffer(0)]],
    device const int* rhs [[buffer(1)]],
    device int* out_coords [[buffer(2)]],
    device int* count [[buffer(3)]],
    constant const int& lhs_rows [[buffer(4)]],
    constant const int& rhs_rows [[buffer(5)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_count = 0;
    for (int row = 0; row < lhs_rows + rhs_rows; ++row) {
        device const int* source = row < lhs_rows ? lhs : rhs;
        int source_row = row < lhs_rows ? row : row - lhs_rows;
        int base = source_row * 4;
        int candidate[4] = {
            source[base],
            source[base + 1],
            source[base + 2],
            source[base + 3],
        };
        bool seen = false;
        for (int prev = 0; prev < out_count; ++prev) {
            if (coord4_equal(candidate, out_coords, prev)) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            write_coord(out_coords, out_count, candidate);
            out_count += 1;
        }
    }
    count[0] = out_count;
}

[[kernel]] void intersection_coords_i32(
    device const int* lhs [[buffer(0)]],
    device const int* rhs [[buffer(1)]],
    device int* out_coords [[buffer(2)]],
    device int* count [[buffer(3)]],
    constant const int& lhs_rows [[buffer(4)]],
    constant const int& rhs_rows [[buffer(5)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_count = 0;
    for (int row = 0; row < lhs_rows; ++row) {
        int base = row * 4;
        int candidate[4] = {
            lhs[base],
            lhs[base + 1],
            lhs[base + 2],
            lhs[base + 3],
        };
        bool in_rhs = false;
        for (int rhs_row = 0; rhs_row < rhs_rows; ++rhs_row) {
            if (coord4_equal(candidate, rhs, rhs_row)) {
                in_rhs = true;
                break;
            }
        }
        if (!in_rhs) {
            continue;
        }

        bool emitted = false;
        for (int prev = 0; prev < out_count; ++prev) {
            if (coord4_equal(candidate, out_coords, prev)) {
                emitted = true;
                break;
            }
        }
        if (!emitted) {
            write_coord(out_coords, out_count, candidate);
            out_count += 1;
        }
    }
    count[0] = out_count;
}

[[kernel]] void lookup_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* queries [[buffer(1)]],
    device int* out_rows [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& query_rows [[buffer(4)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem >= uint(query_rows)) {
        return;
    }

    int query_row = int(elem);
    int out = -1;
    for (int row = 0; row < rows; ++row) {
        if (coord_equal(queries, query_row, coords, row)) {
            out = row;
            break;
        }
    }
    out_rows[query_row] = out;
}

// MARK: - quantization

[[kernel]] void sparse_quantize_f32_i32(
    device const float* points [[buffer(0)]],
    device const int* batch_indices [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* out_active_rows [[buffer(4)]],
    device int* inverse_rows [[buffer(5)]],
    device int* voxel_counts [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    constant const float& voxel_x [[buffer(8)]],
    constant const float& voxel_y [[buffer(9)]],
    constant const float& voxel_z [[buffer(10)]],
    constant const float& origin_x [[buffer(11)]],
    constant const float& origin_y [[buffer(12)]],
    constant const float& origin_z [[buffer(13)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        int coord_base = row * 4;
        out_coords[coord_base] = 0;
        out_coords[coord_base + 1] = 0;
        out_coords[coord_base + 2] = 0;
        out_coords[coord_base + 3] = 0;
        inverse_rows[row] = -1;
        voxel_counts[row] = 0;
    }

    int point_count = min(active_rows[0], rows);
    int out_count = 0;
    for (int point_row = 0; point_row < point_count; ++point_row) {
        int point_base = point_row * 3;
        int candidate[4] = {
            batch_indices[point_row],
            int(floor((points[point_base] - origin_x) / voxel_x)),
            int(floor((points[point_base + 1] - origin_y) / voxel_y)),
            int(floor((points[point_base + 2] - origin_z) / voxel_z)),
        };

        int out_row = -1;
        for (int prev = 0; prev < out_count; ++prev) {
            if (coord4_equal(candidate, out_coords, prev)) {
                out_row = prev;
                break;
            }
        }
        if (out_row < 0) {
            out_row = out_count;
            write_coord(out_coords, out_row, candidate);
            out_count += 1;
        }
        inverse_rows[point_row] = out_row;
        voxel_counts[out_row] += 1;
    }

    out_active_rows[0] = out_count;
}

inline float
voxel_reduce_scale(int reduce, device const int* voxel_counts, int voxel_row) {
    if (reduce == 1) {
        return 1.0f / float(max(voxel_counts[voxel_row], 1));
    }
    return 1.0f;
}

[[kernel]] void voxelize_features_f32_i32(
    device const float* feats [[buffer(0)]],
    device const int* inverse_rows [[buffer(1)]],
    device const int* voxel_counts [[buffer(2)]],
    device const int* active_rows [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& reduce [[buffer(5)]],
    constant const int& point_rows [[buffer(6)]],
    constant const int& voxel_rows [[buffer(7)]],
    constant const int& channels [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    for (int index = 0; index < voxel_rows * channels; ++index) {
        out[index] = 0.0f;
    }

    int point_count = min(active_rows[0], point_rows);
    for (int point_row = 0; point_row < point_count; ++point_row) {
        int voxel_row = inverse_rows[point_row];
        if (voxel_row < 0 || voxel_row >= voxel_rows) {
            continue;
        }
        float scale = voxel_reduce_scale(reduce, voxel_counts, voxel_row);
        for (int channel = 0; channel < channels; ++channel) {
            int point_index = point_row * channels + channel;
            int voxel_index = voxel_row * channels + channel;
            out[voxel_index] += feats[point_index] * scale;
        }
    }
}

[[kernel]] void voxelize_feature_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const int* inverse_rows [[buffer(1)]],
    device const int* voxel_counts [[buffer(2)]],
    device const int* active_rows [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& reduce [[buffer(5)]],
    constant const int& point_rows [[buffer(6)]],
    constant const int& voxel_rows [[buffer(7)]],
    constant const int& channels [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    for (int index = 0; index < point_rows * channels; ++index) {
        out[index] = 0.0f;
    }

    int point_count = min(active_rows[0], point_rows);
    for (int point_row = 0; point_row < point_count; ++point_row) {
        int voxel_row = inverse_rows[point_row];
        if (voxel_row < 0 || voxel_row >= voxel_rows) {
            continue;
        }
        float scale = voxel_reduce_scale(reduce, voxel_counts, voxel_row);
        for (int channel = 0; channel < channels; ++channel) {
            int point_index = point_row * channels + channel;
            int voxel_index = voxel_row * channels + channel;
            out[point_index] = cotangent[voxel_index] * scale;
        }
    }
}

// MARK: - generative relations

[[kernel]] void build_generative_kernel_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* out_coords [[buffer(6)]],
    device int* counts [[buffer(7)]],
    constant const int& rows [[buffer(8)]],
    constant const int& kernel_count [[buffer(9)]],
    constant const int& stride_x [[buffer(10)]],
    constant const int& stride_y [[buffer(11)]],
    constant const int& stride_z [[buffer(12)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    uint total = uint(logical_rows * kernel_count);
    if (elem == 0) {
        counts[0] = int(total);
        counts[1] = int(total);
    }
    if (elem >= total) {
        return;
    }

    int out_row = int(elem);
    int in_row = int(elem / uint(kernel_count));
    int kernel_index = int(elem - uint(in_row * kernel_count));
    int in_base = in_row * 4;
    int out_base = out_row * 4;
    int offset_base = kernel_index * 3;

    in_rows[out_row] = in_row;
    out_rows[out_row] = out_row;
    kernel_ids[out_row] = kernel_index;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + offsets[offset_base];
    out_coords[out_base + 2] =
        coords[in_base + 2] * stride_y + offsets[offset_base + 1];
    out_coords[out_base + 3] =
        coords[in_base + 3] * stride_z + offsets[offset_base + 2];
}

// MARK: - generic relations

[[kernel]] void relation_hash_clear_i32(
    device int* table_rows [[buffer(0)]],
    device int* counts [[buffer(1)]],
    constant const int& table_capacity [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem == 0) {
        counts[0] = 0;
        counts[1] = 0;
    }
    if (elem < uint(table_capacity)) {
        table_rows[elem] = -1;
    }
}

[[kernel]] void relation_hash_insert_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device atomic_int* table_rows [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& table_capacity [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    if (row >= uint(logical_rows)) {
        return;
    }

    int key = relation_coord_hash_i32(coords, int(row));
    int slot = key & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(
                &table_rows[slot],
                &expected,
                int(row),
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            return;
        }
        if (expected >= 0 && coord_equal(coords, expected, coords, int(row))) {
            return;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

[[kernel]] void build_identity_forward_relation_plan_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const int* table_rows [[buffer(3)]],
    device int* planned_in_rows [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& kernel_count [[buffer(7)]],
    constant const int& table_capacity [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    int coord_total = rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < logical_rows ? coords[elem] : 0;
    }

    int relation_total = rows * kernel_count;
    if (elem >= uint(relation_total)) {
        return;
    }
    int kernel_id = int(elem) / rows;
    int out_row = int(elem) - kernel_id * rows;
    if (out_row >= logical_rows) {
        planned_in_rows[elem] = -1;
        return;
    }

    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int candidate[4] = {
        coords[out_base],
        coords[out_base + 1] + kernel_offsets[offset_base],
        coords[out_base + 2] + kernel_offsets[offset_base + 1],
        coords[out_base + 3] + kernel_offsets[offset_base + 2],
    };
    planned_in_rows[elem] =
        lookup_relation_row_hash(coords, table_rows, table_capacity, candidate);
}

[[kernel]] void build_identity_forward_relation_compact_i32(
    device const int* planned_in_rows [[buffer(0)]],
    device int* in_rows [[buffer(1)]],
    device int* out_rows [[buffer(2)]],
    device int* kernel_ids [[buffer(3)]],
    device int* counts [[buffer(4)]],
    device const int* active_rows [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& kernel_count [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int edge_count = 0;
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int kernel_base = kernel_id * rows;
        for (int out_row = 0; out_row < rows; ++out_row) {
            int in_row = planned_in_rows[kernel_base + out_row];
            if (in_row < 0) {
                continue;
            }
            write_edge(
                in_rows,
                out_rows,
                kernel_ids,
                edge_count,
                in_row,
                out_row,
                kernel_id
            );
            edge_count += 1;
        }
    }
    counts[0] = edge_count;
    counts[1] = min(active_rows[0], rows);
}

[[kernel]] void build_strided_forward_output_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device int* table_rows [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& rows [[buffer(5)]],
    constant const int& table_capacity [[buffer(6)]],
    constant const int& stride_x [[buffer(7)]],
    constant const int& stride_y [[buffer(8)]],
    constant const int& stride_z [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int logical_rows = min(active_rows[0], rows);
    int out_count = 0;
    for (int row = 0; row < logical_rows; ++row) {
        int base = row * 4;
        int candidate[4] = {
            coords[base],
            floor_div_int(coords[base + 1], stride_x),
            floor_div_int(coords[base + 2], stride_y),
            floor_div_int(coords[base + 3], stride_z),
        };
        int slot = relation_coord_hash_i32(candidate) & (table_capacity - 1);
        for (int probe = 0; probe < table_capacity; ++probe) {
            int out_row = table_rows[slot];
            if (out_row < 0) {
                table_rows[slot] = out_count;
                write_coord(out_coords, out_count, candidate);
                out_count += 1;
                break;
            }
            if (coord4_equal(candidate, out_coords, out_row)) {
                break;
            }
            slot = (slot + 1) & (table_capacity - 1);
        }
    }
    counts[1] = out_count;
}

[[kernel]] void build_strided_forward_relation_plan_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* out_coords [[buffer(2)]],
    device const int* counts [[buffer(3)]],
    device const int* table_rows [[buffer(4)]],
    device int* planned_in_rows [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& kernel_count [[buffer(7)]],
    constant const int& table_capacity [[buffer(8)]],
    constant const int& stride_x [[buffer(9)]],
    constant const int& stride_y [[buffer(10)]],
    constant const int& stride_z [[buffer(11)]],
    constant const int& pad_x [[buffer(12)]],
    constant const int& pad_y [[buffer(13)]],
    constant const int& pad_z [[buffer(14)]],
    uint elem [[thread_position_in_grid]]
) {
    int relation_total = rows * kernel_count;
    if (elem >= uint(relation_total)) {
        return;
    }

    int kernel_id = int(elem) / rows;
    int out_row = int(elem) - kernel_id * rows;
    int out_count = min(counts[1], rows);
    if (out_row >= out_count) {
        planned_in_rows[elem] = -1;
        return;
    }

    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int candidate[4] = {
        out_coords[out_base],
        out_coords[out_base + 1] * stride_x + kernel_offsets[offset_base] -
            pad_x,
        out_coords[out_base + 2] * stride_y + kernel_offsets[offset_base + 1] -
            pad_y,
        out_coords[out_base + 3] * stride_z + kernel_offsets[offset_base + 2] -
            pad_z,
    };
    planned_in_rows[elem] =
        lookup_relation_row_hash(coords, table_rows, table_capacity, candidate);
}

[[kernel]] void build_strided_forward_relation_compact_i32(
    device const int* planned_in_rows [[buffer(0)]],
    device int* in_rows [[buffer(1)]],
    device int* out_rows [[buffer(2)]],
    device int* kernel_ids [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& rows [[buffer(5)]],
    constant const int& kernel_count [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int edge_count = 0;
    int out_count = min(counts[1], rows);
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int kernel_base = kernel_id * rows;
        for (int out_row = 0; out_row < out_count; ++out_row) {
            int in_row = planned_in_rows[kernel_base + out_row];
            if (in_row < 0) {
                continue;
            }
            write_edge(
                in_rows,
                out_rows,
                kernel_ids,
                edge_count,
                in_row,
                out_row,
                kernel_id
            );
            edge_count += 1;
        }
    }
    counts[0] = edge_count;
}

[[kernel]] void build_transposed_direct_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* out_coords [[buffer(6)]],
    device int* counts [[buffer(7)]],
    constant const int& rows [[buffer(8)]],
    constant const int& kernel_count [[buffer(9)]],
    constant const int& stride_x [[buffer(10)]],
    constant const int& stride_y [[buffer(11)]],
    constant const int& stride_z [[buffer(12)]],
    constant const int& pad_x [[buffer(13)]],
    constant const int& pad_y [[buffer(14)]],
    constant const int& pad_z [[buffer(15)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    int total = logical_rows * kernel_count;
    if (elem == 0) {
        counts[0] = total;
        counts[1] = total;
    }
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem);
    int in_row = int(elem) / kernel_count;
    int kernel_id = int(elem) - in_row * kernel_count;
    int in_base = in_row * 4;
    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;

    in_rows[out_row] = in_row;
    out_rows[out_row] = out_row;
    kernel_ids[out_row] = kernel_id;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + kernel_offsets[offset_base] - pad_x;
    out_coords[out_base + 2] = coords[in_base + 2] * stride_y +
                               kernel_offsets[offset_base + 1] - pad_y;
    out_coords[out_base + 3] = coords[in_base + 3] * stride_z +
                               kernel_offsets[offset_base + 2] - pad_z;
}

[[kernel]] void build_transposed_kernel_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* out_coords [[buffer(6)]],
    device int* counts [[buffer(7)]],
    constant const int& rows [[buffer(8)]],
    constant const int& kernel_count [[buffer(9)]],
    constant const int& stride_x [[buffer(10)]],
    constant const int& stride_y [[buffer(11)]],
    constant const int& stride_z [[buffer(12)]],
    constant const int& pad_x [[buffer(13)]],
    constant const int& pad_y [[buffer(14)]],
    constant const int& pad_z [[buffer(15)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int edge_count = 0;
    int out_count = 0;
    int logical_rows = min(active_rows[0], rows);
    for (int in_row = 0; in_row < logical_rows; ++in_row) {
        int in_base = in_row * 4;
        for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
            int offset_base = kernel_id * 3;
            int candidate[4] = {
                coords[in_base],
                coords[in_base + 1] * stride_x + kernel_offsets[offset_base] -
                    pad_x,
                coords[in_base + 2] * stride_y +
                    kernel_offsets[offset_base + 1] - pad_y,
                coords[in_base + 3] * stride_z +
                    kernel_offsets[offset_base + 2] - pad_z,
            };

            int out_row = -1;
            for (int prev = 0; prev < out_count; ++prev) {
                if (coord4_equal(candidate, out_coords, prev)) {
                    out_row = prev;
                    break;
                }
            }
            if (out_row < 0) {
                out_row = out_count;
                write_coord(out_coords, out_row, candidate);
                out_count += 1;
            }

            write_edge(
                in_rows,
                out_rows,
                kernel_ids,
                edge_count,
                in_row,
                out_row,
                kernel_id
            );
            edge_count += 1;
        }
    }

    counts[0] = edge_count;
    counts[1] = out_count;
}

// MARK: - neighbor relations

[[kernel]] void build_neighbor_relation_i32(
    device const int* query_active_rows [[buffer(0)]],
    device int* query_rows [[buffer(1)]],
    device int* source_rows [[buffer(2)]],
    device int* neighbor_ids [[buffer(3)]],
    device float* distances [[buffer(4)]],
    device int* counts [[buffer(5)]],
    constant const int& query_capacity [[buffer(6)]],
    constant const int& max_neighbors [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_capacity = query_capacity * max_neighbors;
    int query_count = min(query_active_rows[0], query_capacity);

    if (elem == 0) {
        counts[0] = 0;
        counts[1] = query_count;
    }

    if (elem >= uint(edge_capacity)) {
        return;
    }

    query_rows[elem] = 0;
    source_rows[elem] = -1;
    neighbor_ids[elem] = 0;
    distances[elem] = 0.0f;
}

[[kernel]] void fill_neighbor_relation_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device int* query_rows [[buffer(4)]],
    device int* source_rows [[buffer(5)]],
    device int* neighbor_ids [[buffer(6)]],
    device float* distances [[buffer(7)]],
    constant const int& op [[buffer(8)]],
    constant const int& source_capacity [[buffer(9)]],
    constant const int& query_capacity [[buffer(10)]],
    constant const int& max_neighbors [[buffer(11)]],
    constant const float& radius_squared [[buffer(12)]],
    uint query_row [[thread_position_in_grid]]
) {
    int source_count = min(source_active_rows[0], source_capacity);
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int slot_start = int(query_row) * max_neighbors;
    int selected = 0;
    for (int source_row = 0; source_row < source_count; ++source_row) {
        if (!same_batch(
                query_coords, int(query_row), source_coords, source_row
            )) {
            continue;
        }
        float distance = squared_spatial_distance(
            query_coords, int(query_row), source_coords, source_row
        );
        if (op == 1 && distance > radius_squared) {
            continue;
        }

        int insert_at = selected;
        for (int rank = 0; rank < selected; ++rank) {
            int index = slot_start + rank;
            float existing_distance = distances[index];
            int existing_source = source_rows[index];
            if (distance < existing_distance ||
                (distance == existing_distance &&
                 source_row < existing_source)) {
                insert_at = rank;
                break;
            }
        }
        if (insert_at >= max_neighbors) {
            continue;
        }

        int last = min(selected, max_neighbors - 1);
        for (int rank = last; rank > insert_at; --rank) {
            int dst = slot_start + rank;
            int src = dst - 1;
            source_rows[dst] = source_rows[src];
            distances[dst] = distances[src];
        }
        source_rows[slot_start + insert_at] = source_row;
        distances[slot_start + insert_at] = distance;
        selected = min(selected + 1, max_neighbors);
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void fill_knn_relation_topk_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device int* query_rows [[buffer(4)]],
    device int* source_rows [[buffer(5)]],
    device int* neighbor_ids [[buffer(6)]],
    device float* distances [[buffer(7)]],
    constant const int& source_capacity [[buffer(8)]],
    constant const int& query_capacity [[buffer(9)]],
    constant const int& max_neighbors [[buffer(10)]],
    uint query_row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]]
) {
    constexpr int thread_count = 128;
    constexpr int max_k = 16;
    threadgroup float group_distances[thread_count * max_k];
    threadgroup int group_sources[thread_count * max_k];

    int tid = int(thread_id);
    int source_count = min(source_active_rows[0], source_capacity);
    int query_count = min(query_active_rows[0], query_capacity);
    int k = min(max_neighbors, max_k);
    int slot_start = int(query_row) * max_neighbors;

    float local_distances[max_k];
    int local_sources[max_k];
    for (int rank = 0; rank < max_k; ++rank) {
        local_distances[rank] = 0.0f;
        local_sources[rank] = -1;
    }

    int selected = 0;
    if (query_row < uint(query_count) && max_neighbors > 0) {
        for (int source_row = tid; source_row < source_count;
             source_row += thread_count) {
            if (!same_batch(
                    query_coords, int(query_row), source_coords, source_row
                )) {
                continue;
            }
            float distance = squared_spatial_distance(
                query_coords, int(query_row), source_coords, source_row
            );
            int insert_at = selected;
            for (int rank = 0; rank < selected; ++rank) {
                if (distance < local_distances[rank] ||
                    (distance == local_distances[rank] &&
                     source_row < local_sources[rank])) {
                    insert_at = rank;
                    break;
                }
            }
            if (insert_at >= k) {
                continue;
            }
            int last = min(selected, k - 1);
            for (int rank = last; rank > insert_at; --rank) {
                local_distances[rank] = local_distances[rank - 1];
                local_sources[rank] = local_sources[rank - 1];
            }
            local_distances[insert_at] = distance;
            local_sources[insert_at] = source_row;
            selected = min(selected + 1, k);
        }
    }

    int group_base = tid * max_k;
    for (int rank = 0; rank < max_k; ++rank) {
        group_distances[group_base + rank] = local_distances[rank];
        group_sources[group_base + rank] = local_sources[rank];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0 || query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int final_selected = 0;
    for (int candidate = 0; candidate < thread_count * max_k; ++candidate) {
        int source_row = group_sources[candidate];
        if (source_row < 0) {
            continue;
        }
        float distance = group_distances[candidate];
        int insert_at = final_selected;
        for (int rank = 0; rank < final_selected; ++rank) {
            int index = slot_start + rank;
            float existing_distance = distances[index];
            int existing_source = source_rows[index];
            if (distance < existing_distance ||
                (distance == existing_distance &&
                 source_row < existing_source)) {
                insert_at = rank;
                break;
            }
        }
        if (insert_at >= k) {
            continue;
        }
        int last = min(final_selected, k - 1);
        for (int rank = last; rank > insert_at; --rank) {
            int dst = slot_start + rank;
            int src = dst - 1;
            source_rows[dst] = source_rows[src];
            distances[dst] = distances[src];
        }
        source_rows[slot_start + insert_at] = source_row;
        distances[slot_start + insert_at] = distance;
        final_selected = min(final_selected + 1, k);
    }

    for (int rank = 0; rank < final_selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void compact_neighbor_relation_i32(
    device int* query_rows [[buffer(0)]],
    device int* source_rows [[buffer(1)]],
    device int* neighbor_ids [[buffer(2)]],
    device float* distances [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& query_capacity [[buffer(5)]],
    constant const int& max_neighbors [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_edge = 0;
    int query_count = counts[1];
    for (int query_row = 0; query_row < query_count; ++query_row) {
        int slot_start = query_row * max_neighbors;
        for (int rank = 0; rank < max_neighbors; ++rank) {
            int index = slot_start + rank;
            int source_row = source_rows[index];
            if (source_row < 0) {
                break;
            }
            query_rows[out_edge] = query_row;
            source_rows[out_edge] = source_row;
            neighbor_ids[out_edge] = rank;
            distances[out_edge] = distances[index];
            out_edge += 1;
        }
    }
    counts[0] = out_edge;
    (void)query_capacity;
}
