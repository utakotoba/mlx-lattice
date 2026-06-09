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

[[kernel]] void build_forward_kernel_relation_i32(
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

    int logical_rows = min(active_rows[0], rows);
    int out_count = 0;
    bool identity_out = stride_x == 1 && stride_y == 1 && stride_z == 1 &&
                        pad_x == 0 && pad_y == 0 && pad_z == 0;

    for (int row = 0; row < logical_rows; ++row) {
        int base = row * 4;
        int candidate[4] = {
            coords[base],
            identity_out ? coords[base + 1]
                         : floor_div_int(coords[base + 1], stride_x),
            identity_out ? coords[base + 2]
                         : floor_div_int(coords[base + 2], stride_y),
            identity_out ? coords[base + 3]
                         : floor_div_int(coords[base + 3], stride_z),
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

    int edge_count = 0;
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int offset_base = kernel_id * 3;
        for (int out_row = 0; out_row < out_count; ++out_row) {
            int out_base = out_row * 4;
            int candidate[4] = {
                out_coords[out_base],
                out_coords[out_base + 1] * stride_x +
                    kernel_offsets[offset_base] - pad_x,
                out_coords[out_base + 2] * stride_y +
                    kernel_offsets[offset_base + 1] - pad_y,
                out_coords[out_base + 3] * stride_z +
                    kernel_offsets[offset_base + 2] - pad_z,
            };
            for (int in_row = 0; in_row < logical_rows; ++in_row) {
                if (coord4_equal(candidate, coords, in_row)) {
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
                    break;
                }
            }
        }
    }

    counts[0] = edge_count;
    counts[1] = out_count;
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
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device int* query_rows [[buffer(4)]],
    device int* source_rows [[buffer(5)]],
    device int* neighbor_ids [[buffer(6)]],
    device float* distances [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant const int& op [[buffer(9)]],
    constant const int& source_capacity [[buffer(10)]],
    constant const int& query_capacity [[buffer(11)]],
    constant const int& max_neighbors [[buffer(12)]],
    constant const float& radius_squared [[buffer(13)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int edge_capacity = query_capacity * max_neighbors;
    for (int edge = 0; edge < edge_capacity; ++edge) {
        query_rows[edge] = 0;
        source_rows[edge] = 0;
        neighbor_ids[edge] = 0;
        distances[edge] = 0.0f;
    }

    int source_count = min(source_active_rows[0], source_capacity);
    int query_count = min(query_active_rows[0], query_capacity);
    int edge_count = 0;
    for (int query_row = 0; query_row < query_count; ++query_row) {
        int query_start = edge_count;
        for (int neighbor = 0; neighbor < max_neighbors; ++neighbor) {
            int best_source = -1;
            float best_distance = 0.0f;
            for (int source_row = 0; source_row < source_count; ++source_row) {
                if (!same_batch(
                        query_coords, query_row, source_coords, source_row
                    )) {
                    continue;
                }
                bool used = false;
                for (int previous = query_start; previous < edge_count;
                     ++previous) {
                    if (source_rows[previous] == source_row) {
                        used = true;
                        break;
                    }
                }
                if (used) {
                    continue;
                }
                float distance = squared_spatial_distance(
                    query_coords, query_row, source_coords, source_row
                );
                if (op == 1 && distance > radius_squared) {
                    continue;
                }
                if (best_source < 0 || distance < best_distance ||
                    (distance == best_distance && source_row < best_source)) {
                    best_source = source_row;
                    best_distance = distance;
                }
            }
            if (best_source < 0) {
                break;
            }
            query_rows[edge_count] = query_row;
            source_rows[edge_count] = best_source;
            neighbor_ids[edge_count] = neighbor;
            distances[edge_count] = best_distance;
            edge_count += 1;
        }
    }

    counts[0] = edge_count;
    counts[1] = query_count;
}
