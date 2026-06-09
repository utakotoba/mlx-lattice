#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/exec/common.metal"

[[kernel]] void sparse_conv_clear_f32_i32(
    device int* out_coords [[buffer(0)]],
    device float* out_feats [[buffer(1)]],
    device int* counts [[buffer(2)]],
    constant const int& coord_total [[buffer(3)]],
    constant const int& feat_total [[buffer(4)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem == 0) {
        counts[0] = 0;
        counts[1] = 0;
    }
    if (elem < uint(coord_total)) {
        out_coords[elem] = 0;
    }
    if (elem < uint(feat_total)) {
        out_feats[elem] = 0.0f;
    }
}

[[kernel]] void sparse_conv_hash_clear_i32(
    device int* table_keys [[buffer(0)]],
    device int* table_rows [[buffer(1)]],
    device int* counts [[buffer(2)]],
    constant const int& table_capacity [[buffer(3)]],
    constant const int& empty_key [[buffer(4)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem == 0) {
        counts[0] = 0;
        counts[1] = 0;
    }
    if (elem >= uint(table_capacity)) {
        return;
    }
    table_keys[elem] = empty_key;
    table_rows[elem] = -1;
}

[[kernel]] void sparse_conv_hash_insert_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device atomic_int* table_keys [[buffer(2)]],
    device int* table_rows [[buffer(3)]],
    constant const int& n_in_rows [[buffer(4)]],
    constant const int& table_capacity [[buffer(5)]],
    constant const int& empty_key [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (row >= uint(rows)) {
        return;
    }

    int key = exec_coord_hash_i32(coords, int(row));
    int slot = key & (table_capacity - 1);
    int base = int(row) * 4;
    for (int probe = 0; probe < table_capacity; ++probe) {
        int expected = empty_key;
        if (atomic_compare_exchange_weak_explicit(
                &table_keys[slot],
                &expected,
                key,
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            table_rows[slot] = int(row);
            return;
        }
        if (expected == key) {
            int existing = table_rows[slot];
            if (existing >= 0 && coords[existing * 4] == coords[base] &&
                coords[existing * 4 + 1] == coords[base + 1] &&
                coords[existing * 4 + 2] == coords[base + 2] &&
                coords[existing * 4 + 3] == coords[base + 3]) {
                return;
            }
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

[[kernel]] void sparse_conv_forward_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& n_in_rows [[buffer(5)]],
    constant const int& n_kernels [[buffer(6)]],
    constant const int& stride_x [[buffer(7)]],
    constant const int& stride_y [[buffer(8)]],
    constant const int& stride_z [[buffer(9)]],
    constant const int& pad_x [[buffer(10)]],
    constant const int& pad_y [[buffer(11)]],
    constant const int& pad_z [[buffer(12)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        int out_count = 0;
        int edge_count = 0;
        for (int row = 0; row < rows; ++row) {
            int candidate[4];
            downsample_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            );
            if (seen_forward_coord(
                    coords, row, stride_x, stride_y, stride_z, candidate
                )) {
                continue;
            }
            out_count += 1;
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int input_coord[4] = {
                    candidate[0],
                    candidate[1] * stride_x + offsets[offset_base] - pad_x,
                    candidate[2] * stride_y + offsets[offset_base + 1] - pad_y,
                    candidate[3] * stride_z + offsets[offset_base + 2] - pad_z,
                };
                int in_row = -1;
                if (find_input_row(coords, rows, input_coord, in_row)) {
                    edge_count += 1;
                }
            }
        }
        counts[0] = edge_count;
        counts[1] = out_count;
    }

    if (elem >= uint(rows)) {
        return;
    }
    int row = int(elem);
    int candidate[4];
    downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
    if (seen_forward_coord(
            coords, row, stride_x, stride_y, stride_z, candidate
        )) {
        return;
    }
    int out_row = 0;
    for (int prev = 0; prev < row; ++prev) {
        int previous[4];
        downsample_coord(coords, prev, stride_x, stride_y, stride_z, previous);
        if (!seen_forward_coord(
                coords, prev, stride_x, stride_y, stride_z, previous
            )) {
            out_row += 1;
        }
    }
    write_coord(out_coords, out_row, candidate);
}

[[kernel]] void sparse_forward_identity_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& n_in_rows [[buffer(5)]],
    constant const int& n_kernels [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        int edge_count = 0;
        for (int out_row = 0; out_row < rows; ++out_row) {
            int out_base = out_row * 4;
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int input_coord[4] = {
                    coords[out_base],
                    coords[out_base + 1] + offsets[offset_base],
                    coords[out_base + 2] + offsets[offset_base + 1],
                    coords[out_base + 3] + offsets[offset_base + 2],
                };
                int in_row = -1;
                if (find_input_row(coords, rows, input_coord, in_row)) {
                    edge_count += 1;
                }
            }
        }
        counts[0] = edge_count;
        counts[1] = rows;
    }

    int coord_total = n_in_rows * 4;
    if (elem >= uint(coord_total)) {
        return;
    }
    int row = int(elem) / 4;
    out_coords[elem] = row < rows ? coords[elem] : 0;
}

[[kernel]] void sparse_conv_forward_plan_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* counts [[buffer(4)]],
    device int* in_rows [[buffer(5)]],
    device int* kernel_ids [[buffer(6)]],
    device int* row_offsets [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& stride_x [[buffer(11)]],
    constant const int& stride_y [[buffer(12)]],
    constant const int& stride_z [[buffer(13)]],
    constant const int& pad_x [[buffer(14)]],
    constant const int& pad_y [[buffer(15)]],
    constant const int& pad_z [[buffer(16)]],
    constant const int& identity_coords [[buffer(17)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int rows = min(active_rows[0], n_in_rows);
    int out_count = 0;
    int edge_count = 0;

    if (identity_coords != 0) {
        out_count = rows;
        for (int out_row = 0; out_row < rows; ++out_row) {
            int out_base = out_row * 4;
            int output_coord[4] = {
                coords[out_base],
                coords[out_base + 1],
                coords[out_base + 2],
                coords[out_base + 3],
            };
            write_coord(out_coords, out_row, output_coord);
            row_offsets[out_row] = edge_count;
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int input_coord[4] = {
                    coords[out_base],
                    coords[out_base + 1] + offsets[offset_base],
                    coords[out_base + 2] + offsets[offset_base + 1],
                    coords[out_base + 3] + offsets[offset_base + 2],
                };
                int in_row = -1;
                if (find_input_row(coords, rows, input_coord, in_row)) {
                    in_rows[edge_count] = in_row;
                    kernel_ids[edge_count] = kernel_id;
                    edge_count += 1;
                }
            }
        }
        row_offsets[out_count] = edge_count;
        counts[0] = edge_count;
        counts[1] = out_count;
        return;
    }

    for (int row = 0; row < rows; ++row) {
        int candidate[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
        if (seen_forward_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            )) {
            continue;
        }

        int out_row = out_count;
        write_coord(out_coords, out_row, candidate);
        row_offsets[out_row] = edge_count;
        out_count += 1;

        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int offset_base = kernel_id * 3;
            int input_coord[4] = {
                candidate[0],
                candidate[1] * stride_x + offsets[offset_base] - pad_x,
                candidate[2] * stride_y + offsets[offset_base + 1] - pad_y,
                candidate[3] * stride_z + offsets[offset_base + 2] - pad_z,
            };
            int in_row = -1;
            if (find_input_row(coords, rows, input_coord, in_row)) {
                in_rows[edge_count] = in_row;
                kernel_ids[edge_count] = kernel_id;
                edge_count += 1;
            }
        }
    }

    row_offsets[out_count] = edge_count;
    counts[0] = edge_count;
    counts[1] = out_count;
    (void)n_out_rows;
}

[[kernel]] void sparse_conv_identity_plan_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device atomic_int* counts [[buffer(4)]],
    device int* in_rows [[buffer(5)]],
    constant const int& n_in_rows [[buffer(6)]],
    constant const int& n_kernels [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        atomic_store_explicit(&counts[1], rows, memory_order_relaxed);
    }

    int coord_total = n_in_rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < rows ? coords[elem] : 0;
    }

    int relation_total = n_in_rows * n_kernels;
    if (elem >= uint(relation_total)) {
        return;
    }

    int out_row = int(elem) / n_kernels;
    int kernel_id = int(elem) - out_row * n_kernels;
    if (out_row >= rows) {
        in_rows[elem] = -1;
        return;
    }

    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int input_coord[4] = {
        coords[out_base],
        coords[out_base + 1] + offsets[offset_base],
        coords[out_base + 2] + offsets[offset_base + 1],
        coords[out_base + 3] + offsets[offset_base + 2],
    };
    int in_row = -1;
    if (find_input_row(coords, rows, input_coord, in_row)) {
        in_rows[elem] = in_row;
        atomic_fetch_add_explicit(&counts[0], 1, memory_order_relaxed);
    } else {
        in_rows[elem] = -1;
    }
}

[[kernel]] void sparse_conv_identity_hash_plan_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device const int* table_keys [[buffer(3)]],
    device const int* table_rows [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    device atomic_int* counts [[buffer(6)]],
    device int* in_rows [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_kernels [[buffer(9)]],
    constant const int& table_capacity [[buffer(10)]],
    constant const int& empty_key [[buffer(11)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        atomic_store_explicit(&counts[1], rows, memory_order_relaxed);
    }

    int coord_total = n_in_rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < rows ? coords[elem] : 0;
    }

    int relation_total = n_in_rows * n_kernels;
    if (elem >= uint(relation_total)) {
        return;
    }

    int out_row = int(elem) / n_kernels;
    int kernel_id = int(elem) - out_row * n_kernels;
    if (out_row >= rows) {
        in_rows[elem] = -1;
        return;
    }

    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int input_coord[4] = {
        coords[out_base],
        coords[out_base + 1] + offsets[offset_base],
        coords[out_base + 2] + offsets[offset_base + 1],
        coords[out_base + 3] + offsets[offset_base + 2],
    };
    int in_row = lookup_input_row_hash(
        coords, table_keys, table_rows, table_capacity, empty_key, input_coord
    );
    in_rows[elem] = in_row;
    if (in_row >= 0) {
        atomic_fetch_add_explicit(&counts[0], 1, memory_order_relaxed);
    }
}

[[kernel]] void sparse_conv_generative_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    device float* out_feats [[buffer(6)]],
    device int* counts [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& in_channels [[buffer(11)]],
    constant const int& out_channels [[buffer(12)]],
    constant const int& stride_x [[buffer(13)]],
    constant const int& stride_y [[buffer(14)]],
    constant const int& stride_z [[buffer(15)]],
    constant const int& feat_s0 [[buffer(16)]],
    constant const int& feat_s1 [[buffer(17)]],
    constant const int& weight_s0 [[buffer(18)]],
    constant const int& weight_s1 [[buffer(19)]],
    constant const int& weight_s2 [[buffer(20)]],
    constant const int& weight_s3 [[buffer(21)]],
    constant const int& weight_s4 [[buffer(22)]],
    constant const int& weight_layout [[buffer(23)]],
    constant const int& kernel_x [[buffer(24)]],
    constant const int& kernel_y [[buffer(25)]],
    constant const int& kernel_z [[buffer(26)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    int active_out_rows = rows * n_kernels;
    int coord_total = n_out_rows * 4;
    int feat_total = n_out_rows * out_channels;

    if (elem == 0) {
        counts[0] = active_out_rows;
        counts[1] = active_out_rows;
    }

    if (elem < uint(coord_total)) {
        int out_row = int(elem) / 4;
        int lane = int(elem) - out_row * 4;
        if (out_row >= active_out_rows) {
            out_coords[elem] = 0;
        } else {
            int in_row = out_row / n_kernels;
            int kernel_id = out_row - in_row * n_kernels;
            int in_base = in_row * 4;
            int offset_base = kernel_id * 3;
            if (lane == 0) {
                out_coords[elem] = coords[in_base];
            } else if (lane == 1) {
                out_coords[elem] =
                    coords[in_base + 1] * stride_x + offsets[offset_base];
            } else if (lane == 2) {
                out_coords[elem] =
                    coords[in_base + 2] * stride_y + offsets[offset_base + 1];
            } else {
                out_coords[elem] =
                    coords[in_base + 3] * stride_z + offsets[offset_base + 2];
            }
        }
    }

    if (elem >= uint(feat_total)) {
        return;
    }
    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    if (out_row >= active_out_rows) {
        out_feats[elem] = 0.0f;
        return;
    }

    int in_row = out_row / n_kernels;
    int kernel_id = out_row - in_row * n_kernels;
    float acc = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        acc += feats[in_row * feat_s0 + ci * feat_s1] * weights[weight_offset(
                                                            kernel_id,
                                                            ci,
                                                            co,
                                                            weight_layout,
                                                            kernel_x,
                                                            kernel_y,
                                                            kernel_z,
                                                            weight_s0,
                                                            weight_s1,
                                                            weight_s2,
                                                            weight_s3,
                                                            weight_s4
                                                        )];
    }
    out_feats[elem] = acc;
}

[[kernel]] void sparse_conv_identity_hash_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device const int* table_keys [[buffer(5)]],
    device const int* table_rows [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device float* out_feats [[buffer(8)]],
    device atomic_int* counts [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_out_rows [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& table_capacity [[buffer(15)]],
    constant const int& empty_key [[buffer(16)]],
    constant const int& feat_s0 [[buffer(17)]],
    constant const int& feat_s1 [[buffer(18)]],
    constant const int& weight_s0 [[buffer(19)]],
    constant const int& weight_s1 [[buffer(20)]],
    constant const int& weight_s2 [[buffer(21)]],
    constant const int& weight_s3 [[buffer(22)]],
    constant const int& weight_s4 [[buffer(23)]],
    constant const int& weight_layout [[buffer(24)]],
    constant const int& kernel_x [[buffer(25)]],
    constant const int& kernel_y [[buffer(26)]],
    constant const int& kernel_z [[buffer(27)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        atomic_store_explicit(&counts[1], rows, memory_order_relaxed);
    }

    int coord_total = n_out_rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < rows ? coords[elem] : 0;
    }

    int feat_total = n_out_rows * out_channels;
    if (elem >= uint(feat_total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    if (out_row >= rows) {
        out_feats[elem] = 0.0f;
        return;
    }

    int out_base = out_row * 4;
    float acc = 0.0f;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int offset_base = kernel_id * 3;
        int input_coord[4] = {
            coords[out_base],
            coords[out_base + 1] + offsets[offset_base],
            coords[out_base + 2] + offsets[offset_base + 1],
            coords[out_base + 3] + offsets[offset_base + 2],
        };
        int in_row = lookup_input_row_hash(
            coords,
            table_keys,
            table_rows,
            table_capacity,
            empty_key,
            input_coord
        );
        if (in_row < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            acc +=
                feats[in_row * feat_s0 + ci * feat_s1] * weights[weight_offset(
                                                             kernel_id,
                                                             ci,
                                                             co,
                                                             weight_layout,
                                                             kernel_x,
                                                             kernel_y,
                                                             kernel_z,
                                                             weight_s0,
                                                             weight_s1,
                                                             weight_s2,
                                                             weight_s3,
                                                             weight_s4
                                                         )];
        }
    }
    out_feats[elem] = acc;
}

[[kernel]] void sparse_conv_identity_hash_vec4_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device const int* table_keys [[buffer(5)]],
    device const int* table_rows [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device float* out_feats [[buffer(8)]],
    device atomic_int* counts [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_out_rows [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& table_capacity [[buffer(15)]],
    constant const int& empty_key [[buffer(16)]],
    constant const int& feat_s0 [[buffer(17)]],
    constant const int& feat_s1 [[buffer(18)]],
    constant const int& weight_s0 [[buffer(19)]],
    constant const int& weight_s1 [[buffer(20)]],
    constant const int& weight_s2 [[buffer(21)]],
    constant const int& weight_s3 [[buffer(22)]],
    constant const int& weight_s4 [[buffer(23)]],
    constant const int& weight_layout [[buffer(24)]],
    constant const int& kernel_x [[buffer(25)]],
    constant const int& kernel_y [[buffer(26)]],
    constant const int& kernel_z [[buffer(27)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        atomic_store_explicit(
            &counts[0], rows * n_kernels, memory_order_relaxed
        );
        atomic_store_explicit(&counts[1], rows, memory_order_relaxed);
    }

    int coord_total = n_out_rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < rows ? coords[elem] : 0;
    }

    int blocks = out_channels / 4;
    int feat_total = n_out_rows * blocks;
    if (elem >= uint(feat_total)) {
        return;
    }

    int out_row = int(elem) / blocks;
    int co = (int(elem) - out_row * blocks) * 4;
    int out_base = out_row * 4;
    int feat_base = out_row * out_channels + co;
    if (out_row >= rows) {
        out_feats[feat_base] = 0.0f;
        out_feats[feat_base + 1] = 0.0f;
        out_feats[feat_base + 2] = 0.0f;
        out_feats[feat_base + 3] = 0.0f;
        return;
    }

    float4 acc = float4(0.0f);
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int offset_base = kernel_id * 3;
        int input_coord[4] = {
            coords[out_base],
            coords[out_base + 1] + offsets[offset_base],
            coords[out_base + 2] + offsets[offset_base + 1],
            coords[out_base + 3] + offsets[offset_base + 2],
        };
        int in_row = lookup_input_row_hash(
            coords,
            table_keys,
            table_rows,
            table_capacity,
            empty_key,
            input_coord
        );
        if (in_row < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * feat_s0 + ci * feat_s1];
            acc += value * weight4_at(
                               weights,
                               kernel_id,
                               ci,
                               co,
                               weight_layout,
                               kernel_y,
                               kernel_z,
                               weight_s0,
                               weight_s1,
                               weight_s2,
                               weight_s3,
                               weight_s4
                           );
        }
    }
    out_feats[feat_base] = acc.x;
    out_feats[feat_base + 1] = acc.y;
    out_feats[feat_base + 2] = acc.z;
    out_feats[feat_base + 3] = acc.w;
}

[[kernel]] void sparse_conv_forward_gather_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device const int* out_coords [[buffer(5)]],
    device const int* counts [[buffer(6)]],
    device float* out_feats [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& in_channels [[buffer(11)]],
    constant const int& out_channels [[buffer(12)]],
    constant const int& stride_x [[buffer(13)]],
    constant const int& stride_y [[buffer(14)]],
    constant const int& stride_z [[buffer(15)]],
    constant const int& pad_x [[buffer(16)]],
    constant const int& pad_y [[buffer(17)]],
    constant const int& pad_z [[buffer(18)]],
    constant const int& feat_s0 [[buffer(19)]],
    constant const int& feat_s1 [[buffer(20)]],
    constant const int& weight_s0 [[buffer(21)]],
    constant const int& weight_s1 [[buffer(22)]],
    constant const int& weight_s2 [[buffer(23)]],
    constant const int& weight_s3 [[buffer(24)]],
    constant const int& weight_s4 [[buffer(25)]],
    constant const int& weight_layout [[buffer(26)]],
    constant const int& kernel_x [[buffer(27)]],
    constant const int& kernel_y [[buffer(28)]],
    constant const int& kernel_z [[buffer(29)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * out_channels;
    if (elem >= uint(total)) {
        return;
    }
    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    int out_count = counts[1];
    int rows = min(active_rows[0], n_in_rows);
    if (out_row >= out_count) {
        return;
    }

    int out_base = out_row * 4;
    float acc = 0.0f;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int offset_base = kernel_id * 3;
        int input_coord[4] = {
            out_coords[out_base],
            out_coords[out_base + 1] * stride_x + offsets[offset_base] - pad_x,
            out_coords[out_base + 2] * stride_y + offsets[offset_base + 1] -
                pad_y,
            out_coords[out_base + 3] * stride_z + offsets[offset_base + 2] -
                pad_z,
        };
        int in_row = -1;
        if (!find_input_row(coords, rows, input_coord, in_row)) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            acc +=
                feats[in_row * feat_s0 + ci * feat_s1] * weights[weight_offset(
                                                             kernel_id,
                                                             ci,
                                                             co,
                                                             weight_layout,
                                                             kernel_x,
                                                             kernel_y,
                                                             kernel_z,
                                                             weight_s0,
                                                             weight_s1,
                                                             weight_s2,
                                                             weight_s3,
                                                             weight_s4
                                                         )];
        }
    }
    out_feats[elem] = acc;
}

[[kernel]] void sparse_conv_identity_plan_gather_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* counts [[buffer(3)]],
    device float* out_feats [[buffer(4)]],
    constant const int& n_out_rows [[buffer(5)]],
    constant const int& n_kernels [[buffer(6)]],
    constant const int& in_channels [[buffer(7)]],
    constant const int& out_channels [[buffer(8)]],
    constant const int& feat_s0 [[buffer(9)]],
    constant const int& feat_s1 [[buffer(10)]],
    constant const int& weight_s0 [[buffer(11)]],
    constant const int& weight_s1 [[buffer(12)]],
    constant const int& weight_s2 [[buffer(13)]],
    constant const int& weight_s3 [[buffer(14)]],
    constant const int& weight_s4 [[buffer(15)]],
    constant const int& weight_layout [[buffer(16)]],
    constant const int& kernel_x [[buffer(17)]],
    constant const int& kernel_y [[buffer(18)]],
    constant const int& kernel_z [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    if (out_row >= counts[1]) {
        return;
    }

    float acc = 0.0f;
    int slot_base = out_row * n_kernels;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int in_row = in_rows[slot_base + kernel_id];
        if (in_row < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            acc +=
                feats[in_row * feat_s0 + ci * feat_s1] * weights[weight_offset(
                                                             kernel_id,
                                                             ci,
                                                             co,
                                                             weight_layout,
                                                             kernel_x,
                                                             kernel_y,
                                                             kernel_z,
                                                             weight_s0,
                                                             weight_s1,
                                                             weight_s2,
                                                             weight_s3,
                                                             weight_s4
                                                         )];
        }
    }
    out_feats[elem] = acc;
}

[[kernel]] void sparse_conv_identity_plan_gather_vec4_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* counts [[buffer(3)]],
    device float* out_feats [[buffer(4)]],
    constant const int& n_out_rows [[buffer(5)]],
    constant const int& n_kernels [[buffer(6)]],
    constant const int& in_channels [[buffer(7)]],
    constant const int& out_channels [[buffer(8)]],
    constant const int& feat_s0 [[buffer(9)]],
    constant const int& feat_s1 [[buffer(10)]],
    constant const int& weight_s0 [[buffer(11)]],
    constant const int& weight_s1 [[buffer(12)]],
    constant const int& weight_s2 [[buffer(13)]],
    constant const int& weight_s3 [[buffer(14)]],
    constant const int& weight_s4 [[buffer(15)]],
    constant const int& weight_layout [[buffer(16)]],
    constant const int& kernel_x [[buffer(17)]],
    constant const int& kernel_y [[buffer(18)]],
    constant const int& kernel_z [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = out_channels / 4;
    int total = n_out_rows * blocks;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / blocks;
    int co = (int(elem) - out_row * blocks) * 4;
    int feat_base = out_row * out_channels + co;
    if (out_row >= counts[1]) {
        out_feats[feat_base] = 0.0f;
        out_feats[feat_base + 1] = 0.0f;
        out_feats[feat_base + 2] = 0.0f;
        out_feats[feat_base + 3] = 0.0f;
        return;
    }

    float4 acc = float4(0.0f);
    int slot_base = out_row * n_kernels;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int in_row = in_rows[slot_base + kernel_id];
        if (in_row < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * feat_s0 + ci * feat_s1];
            acc += value * weight4_at(
                               weights,
                               kernel_id,
                               ci,
                               co,
                               weight_layout,
                               kernel_y,
                               kernel_z,
                               weight_s0,
                               weight_s1,
                               weight_s2,
                               weight_s3,
                               weight_s4
                           );
        }
    }
    out_feats[feat_base] = acc.x;
    out_feats[feat_base + 1] = acc.y;
    out_feats[feat_base + 2] = acc.z;
    out_feats[feat_base + 3] = acc.w;
}

[[kernel]] void sparse_conv_forward_plan_gather_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* kernel_ids [[buffer(3)]],
    device const int* row_offsets [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device float* out_feats [[buffer(6)]],
    constant const int& n_out_rows [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    constant const int& feat_s0 [[buffer(10)]],
    constant const int& feat_s1 [[buffer(11)]],
    constant const int& weight_s0 [[buffer(12)]],
    constant const int& weight_s1 [[buffer(13)]],
    constant const int& weight_s2 [[buffer(14)]],
    constant const int& weight_s3 [[buffer(15)]],
    constant const int& weight_s4 [[buffer(16)]],
    constant const int& weight_layout [[buffer(17)]],
    constant const int& kernel_x [[buffer(18)]],
    constant const int& kernel_y [[buffer(19)]],
    constant const int& kernel_z [[buffer(20)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    if (out_row >= counts[1]) {
        return;
    }

    float acc = 0.0f;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        for (int ci = 0; ci < in_channels; ++ci) {
            acc +=
                feats[in_row * feat_s0 + ci * feat_s1] * weights[weight_offset(
                                                             kernel_id,
                                                             ci,
                                                             co,
                                                             weight_layout,
                                                             kernel_x,
                                                             kernel_y,
                                                             kernel_z,
                                                             weight_s0,
                                                             weight_s1,
                                                             weight_s2,
                                                             weight_s3,
                                                             weight_s4
                                                         )];
        }
    }
    out_feats[elem] = acc;
}

[[kernel]] void sparse_conv_pointwise_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device int* out_coords [[buffer(4)]],
    device float* out_feats [[buffer(5)]],
    device int* counts [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    constant const int& n_out_rows [[buffer(8)]],
    constant const int& in_channels [[buffer(9)]],
    constant const int& out_channels [[buffer(10)]],
    constant const int& feat_s0 [[buffer(11)]],
    constant const int& feat_s1 [[buffer(12)]],
    constant const int& weight_s0 [[buffer(13)]],
    constant const int& weight_s1 [[buffer(14)]],
    constant const int& weight_s2 [[buffer(15)]],
    constant const int& weight_s3 [[buffer(16)]],
    constant const int& weight_s4 [[buffer(17)]],
    constant const int& weight_layout [[buffer(18)]],
    constant const int& kernel_x [[buffer(19)]],
    constant const int& kernel_y [[buffer(20)]],
    constant const int& kernel_z [[buffer(21)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    int coord_total = n_out_rows * 4;
    int feat_total = n_out_rows * out_channels;

    if (elem == 0) {
        counts[0] = rows;
        counts[1] = rows;
    }

    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        int lane = int(elem) - row * 4;
        out_coords[elem] = row < rows ? coords[row * 4 + lane] : 0;
    }

    if (elem >= uint(feat_total)) {
        return;
    }
    int row = int(elem) / out_channels;
    int co = int(elem) - row * out_channels;
    if (row >= rows) {
        out_feats[elem] = 0.0f;
        return;
    }

    float acc = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        acc += feats[row * feat_s0 + ci * feat_s1] * weights[weight_offset(
                                                         0,
                                                         ci,
                                                         co,
                                                         weight_layout,
                                                         kernel_x,
                                                         kernel_y,
                                                         kernel_z,
                                                         weight_s0,
                                                         weight_s1,
                                                         weight_s2,
                                                         weight_s3,
                                                         weight_s4
                                                     )];
    }
    out_feats[elem] = acc;
}

[[kernel]] void sparse_conv_pointwise_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* grad [[buffer(3)]],
    constant const int& n_in_rows [[buffer(4)]],
    constant const int& in_channels [[buffer(5)]],
    constant const int& out_channels [[buffer(6)]],
    constant const int& cotangent_s0 [[buffer(7)]],
    constant const int& cotangent_s1 [[buffer(8)]],
    constant const int& weight_s0 [[buffer(9)]],
    constant const int& weight_s1 [[buffer(10)]],
    constant const int& weight_s2 [[buffer(11)]],
    constant const int& weight_s3 [[buffer(12)]],
    constant const int& weight_s4 [[buffer(13)]],
    constant const int& weight_layout [[buffer(14)]],
    constant const int& kernel_x [[buffer(15)]],
    constant const int& kernel_y [[buffer(16)]],
    constant const int& kernel_z [[buffer(17)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    int total = n_in_rows * in_channels;
    if (elem >= uint(total)) {
        return;
    }
    int row = int(elem) / in_channels;
    int ci = int(elem) - row * in_channels;
    if (row >= rows) {
        grad[elem] = 0.0f;
        return;
    }

    float acc = 0.0f;
    for (int co = 0; co < out_channels; ++co) {
        acc += cotangent[row * cotangent_s0 + co * cotangent_s1] *
               weights[weight_offset(
                   0,
                   ci,
                   co,
                   weight_layout,
                   kernel_x,
                   kernel_y,
                   kernel_z,
                   weight_s0,
                   weight_s1,
                   weight_s2,
                   weight_s3,
                   weight_s4
               )];
    }
    grad[elem] = acc;
}

[[kernel]] void sparse_conv_pointwise_weight_grad_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device float* grad [[buffer(3)]],
    constant const int& n_in_rows [[buffer(4)]],
    constant const int& in_channels [[buffer(5)]],
    constant const int& out_channels [[buffer(6)]],
    constant const int& feat_s0 [[buffer(7)]],
    constant const int& feat_s1 [[buffer(8)]],
    constant const int& cotangent_s0 [[buffer(9)]],
    constant const int& cotangent_s1 [[buffer(10)]],
    constant const int& weight_layout [[buffer(11)]],
    constant const int& kernel_x [[buffer(12)]],
    constant const int& kernel_y [[buffer(13)]],
    constant const int& kernel_z [[buffer(14)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }
    int ci = int(elem) / out_channels;
    int co = int(elem) - ci * out_channels;
    int rows = min(active_rows[0], n_in_rows);

    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) {
        acc += feats[row * feat_s0 + ci * feat_s1] *
               cotangent[row * cotangent_s0 + co * cotangent_s1];
    }
    grad[dense_weight_offset(
        0,
        ci,
        co,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        in_channels,
        out_channels
    )] = acc;
}

[[kernel]] void sparse_relation_conv_clear_f32(
    device float* out [[buffer(0)]],
    constant const int& total [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(total)) {
        out[elem] = 0.0f;
    }
}

[[kernel]] void sparse_relation_conv_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device atomic_float* out [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& out_capacity [[buffer(8)]],
    constant const int& in_channels [[buffer(9)]],
    constant const int& out_channels [[buffer(10)]],
    constant const int& feat_s0 [[buffer(11)]],
    constant const int& feat_s1 [[buffer(12)]],
    constant const int& weight_s0 [[buffer(13)]],
    constant const int& weight_s1 [[buffer(14)]],
    constant const int& weight_s2 [[buffer(15)]],
    constant const int& weight_s3 [[buffer(16)]],
    constant const int& weight_s4 [[buffer(17)]],
    constant const int& weight_layout [[buffer(18)]],
    constant const int& kernel_x [[buffer(19)]],
    constant const int& kernel_y [[buffer(20)]],
    constant const int& kernel_z [[buffer(21)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int edge = int(elem) / out_channels;
    int co = int(elem) - edge * out_channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }

    float acc = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        acc += feats[in_row * feat_s0 + ci * feat_s1] * weights[weight_offset(
                                                            kernel_id,
                                                            ci,
                                                            co,
                                                            weight_layout,
                                                            kernel_x,
                                                            kernel_y,
                                                            kernel_z,
                                                            weight_s0,
                                                            weight_s1,
                                                            weight_s2,
                                                            weight_s3,
                                                            weight_s4
                                                        )];
    }
    atomic_fetch_add_explicit(
        &out[out_row * out_channels + co], acc, memory_order_relaxed
    );
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device atomic_float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& out_capacity [[buffer(8)]],
    constant const int& in_channels [[buffer(9)]],
    constant const int& out_channels [[buffer(10)]],
    constant const int& cotangent_s0 [[buffer(11)]],
    constant const int& cotangent_s1 [[buffer(12)]],
    constant const int& weight_s0 [[buffer(13)]],
    constant const int& weight_s1 [[buffer(14)]],
    constant const int& weight_s2 [[buffer(15)]],
    constant const int& weight_s3 [[buffer(16)]],
    constant const int& weight_s4 [[buffer(17)]],
    constant const int& weight_layout [[buffer(18)]],
    constant const int& kernel_x [[buffer(19)]],
    constant const int& kernel_y [[buffer(20)]],
    constant const int& kernel_z [[buffer(21)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * in_channels;
    if (elem >= uint(total)) {
        return;
    }

    int edge = int(elem) / in_channels;
    int ci = int(elem) - edge * in_channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }

    float acc = 0.0f;
    for (int co = 0; co < out_channels; ++co) {
        acc += cotangent[out_row * cotangent_s0 + co * cotangent_s1] *
               weights[weight_offset(
                   kernel_id,
                   ci,
                   co,
                   weight_layout,
                   kernel_x,
                   kernel_y,
                   kernel_z,
                   weight_s0,
                   weight_s1,
                   weight_s2,
                   weight_s3,
                   weight_s4
               )];
    }
    atomic_fetch_add_explicit(
        &grad[in_row * in_channels + ci], acc, memory_order_relaxed
    );
}

[[kernel]] void sparse_relation_conv_weight_grad_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device atomic_float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& out_capacity [[buffer(8)]],
    constant const int& in_channels [[buffer(9)]],
    constant const int& out_channels [[buffer(10)]],
    constant const int& feat_s0 [[buffer(11)]],
    constant const int& feat_s1 [[buffer(12)]],
    constant const int& cotangent_s0 [[buffer(13)]],
    constant const int& cotangent_s1 [[buffer(14)]],
    constant const int& weight_layout [[buffer(15)]],
    constant const int& kernel_x [[buffer(16)]],
    constant const int& kernel_y [[buffer(17)]],
    constant const int& kernel_z [[buffer(18)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int channel = int(elem) % (in_channels * out_channels);
    int edge = int(elem) / (in_channels * out_channels);
    int ci = channel / out_channels;
    int co = channel - ci * out_channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }

    float value = feats[in_row * feat_s0 + ci * feat_s1] *
                  cotangent[out_row * cotangent_s0 + co * cotangent_s1];
    atomic_fetch_add_explicit(
        &grad[dense_weight_offset(
            kernel_id,
            ci,
            co,
            weight_layout,
            kernel_x,
            kernel_y,
            kernel_z,
            in_channels,
            out_channels
        )],
        value,
        memory_order_relaxed
    );
}

[[kernel]] void sparse_conv_f32_i32_serial(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    device float* out_feats [[buffer(6)]],
    device int* counts [[buffer(7)]],
    constant const int& map_op [[buffer(8)]],
    constant const int& n_in_rows [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    constant const int& n_kernels [[buffer(11)]],
    constant const int& in_channels [[buffer(12)]],
    constant const int& out_channels [[buffer(13)]],
    constant const int& stride_x [[buffer(14)]],
    constant const int& stride_y [[buffer(15)]],
    constant const int& stride_z [[buffer(16)]],
    constant const int& pad_x [[buffer(17)]],
    constant const int& pad_y [[buffer(18)]],
    constant const int& pad_z [[buffer(19)]],
    constant const int& feat_s0 [[buffer(20)]],
    constant const int& feat_s1 [[buffer(21)]],
    constant const int& weight_s0 [[buffer(22)]],
    constant const int& weight_s1 [[buffer(23)]],
    constant const int& weight_s2 [[buffer(24)]],
    constant const int& weight_s3 [[buffer(25)]],
    constant const int& weight_s4 [[buffer(26)]],
    constant const int& weight_layout [[buffer(27)]],
    constant const int& kernel_x [[buffer(28)]],
    constant const int& kernel_y [[buffer(29)]],
    constant const int& kernel_z [[buffer(30)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_out_rows * 4; ++index) {
        out_coords[index] = 0;
    }
    for (int index = 0; index < n_out_rows * out_channels; ++index) {
        out_feats[index] = 0.0f;
    }

    int out_count = 0;
    int edge_count = 0;
    if (map_op == 0) {
        for (int row = 0; row < rows; ++row) {
            int candidate[4];
            downsample_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            );
            if (!seen_forward_coord(
                    coords, row, stride_x, stride_y, stride_z, candidate
                )) {
                write_coord(out_coords, out_count, candidate);
                out_count += 1;
            }
        }
        for (int out_row = 0; out_row < out_count; ++out_row) {
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int out_base = out_row * 4;
                int input_coord[4] = {
                    out_coords[out_base],
                    out_coords[out_base + 1] * stride_x + offsets[offset_base] -
                        pad_x,
                    out_coords[out_base + 2] * stride_y +
                        offsets[offset_base + 1] - pad_y,
                    out_coords[out_base + 3] * stride_z +
                        offsets[offset_base + 2] - pad_z,
                };
                int in_row = -1;
                if (!find_input_row(coords, rows, input_coord, in_row)) {
                    continue;
                }
                edge_count += 1;
                for (int ci = 0; ci < in_channels; ++ci) {
                    float value = feats[in_row * feat_s0 + ci * feat_s1];
                    int feat_base = out_row * out_channels;
                    for (int co = 0; co < out_channels; ++co) {
                        out_feats[feat_base + co] +=
                            value * weights[weight_offset(
                                        kernel_id,
                                        ci,
                                        co,
                                        weight_layout,
                                        kernel_x,
                                        kernel_y,
                                        kernel_z,
                                        weight_s0,
                                        weight_s1,
                                        weight_s2,
                                        weight_s3,
                                        weight_s4
                                    )];
                    }
                }
            }
        }
    } else {
        for (int in_row = 0; in_row < rows; ++in_row) {
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int candidate[4];
                transposed_candidate(
                    coords,
                    offsets,
                    in_row,
                    kernel_id,
                    stride_x,
                    stride_y,
                    stride_z,
                    map_op == 2 ? 0 : pad_x,
                    map_op == 2 ? 0 : pad_y,
                    map_op == 2 ? 0 : pad_z,
                    candidate
                );
                int out_row = edge_count;
                if (map_op == 1) {
                    out_row = transposed_out_row_for_coord(
                        coords,
                        offsets,
                        rows,
                        n_kernels,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        candidate
                    );
                }
                if (out_row == out_count) {
                    write_coord(out_coords, out_row, candidate);
                    out_count += 1;
                } else if (map_op == 2) {
                    write_coord(out_coords, out_row, candidate);
                    out_count += 1;
                }
                edge_count += 1;
                for (int ci = 0; ci < in_channels; ++ci) {
                    float value = feats[in_row * feat_s0 + ci * feat_s1];
                    int feat_base = out_row * out_channels;
                    for (int co = 0; co < out_channels; ++co) {
                        out_feats[feat_base + co] +=
                            value * weights[weight_offset(
                                        kernel_id,
                                        ci,
                                        co,
                                        weight_layout,
                                        kernel_x,
                                        kernel_y,
                                        kernel_z,
                                        weight_s0,
                                        weight_s1,
                                        weight_s2,
                                        weight_s3,
                                        weight_s4
                                    )];
                    }
                }
            }
        }
    }

    counts[0] = edge_count;
    counts[1] = out_count;
}

[[kernel]] void sparse_conv_input_grad_f32_i32_serial(
    device const float* cotangent [[buffer(0)]],
    device const int* coords [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& map_op [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    constant const int& n_out_rows [[buffer(8)]],
    constant const int& n_kernels [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& cotangent_s0 [[buffer(18)]],
    constant const int& cotangent_s1 [[buffer(19)]],
    constant const int& weight_s0 [[buffer(20)]],
    constant const int& weight_s1 [[buffer(21)]],
    constant const int& weight_s2 [[buffer(22)]],
    constant const int& weight_s3 [[buffer(23)]],
    constant const int& weight_s4 [[buffer(24)]],
    constant const int& weight_layout [[buffer(25)]],
    constant const int& kernel_x [[buffer(26)]],
    constant const int& kernel_y [[buffer(27)]],
    constant const int& kernel_z [[buffer(28)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_in_rows * in_channels; ++index) {
        grad[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (map_op == 0) {
                if (!valid_forward_relation_coord(
                        coords,
                        rows,
                        kernel_id,
                        offsets,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        in_row,
                        out_coord,
                        out_row
                    )) {
                    continue;
                }
            } else if (map_op == 2) {
                out_row = in_row * n_kernels + kernel_id;
            } else {
                transposed_candidate(
                    coords,
                    offsets,
                    in_row,
                    kernel_id,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
                out_row = transposed_out_row_for_coord(
                    coords,
                    offsets,
                    rows,
                    n_kernels,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
            }
            if (out_row < 0 || out_row >= n_out_rows) {
                continue;
            }
            for (int ci = 0; ci < in_channels; ++ci) {
                for (int co = 0; co < out_channels; ++co) {
                    grad[in_row * in_channels + ci] +=
                        cotangent[out_row * cotangent_s0 + co * cotangent_s1] *
                        weights[weight_offset(
                            kernel_id,
                            ci,
                            co,
                            weight_layout,
                            kernel_x,
                            kernel_y,
                            kernel_z,
                            weight_s0,
                            weight_s1,
                            weight_s2,
                            weight_s3,
                            weight_s4
                        )];
                }
            }
        }
    }
}

[[kernel]] void sparse_conv_weight_grad_f32_i32_serial(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* coords [[buffer(2)]],
    device const int* active_rows [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& map_op [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    constant const int& n_out_rows [[buffer(8)]],
    constant const int& n_kernels [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    constant const int& cotangent_s0 [[buffer(20)]],
    constant const int& cotangent_s1 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_kernels * in_channels * out_channels;
         ++index) {
        grad[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (map_op == 0) {
                if (!valid_forward_relation_coord(
                        coords,
                        rows,
                        kernel_id,
                        offsets,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        in_row,
                        out_coord,
                        out_row
                    )) {
                    continue;
                }
            } else if (map_op == 2) {
                out_row = in_row * n_kernels + kernel_id;
            } else {
                transposed_candidate(
                    coords,
                    offsets,
                    in_row,
                    kernel_id,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
                out_row = transposed_out_row_for_coord(
                    coords,
                    offsets,
                    rows,
                    n_kernels,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
            }
            if (out_row < 0 || out_row >= n_out_rows) {
                continue;
            }
            for (int ci = 0; ci < in_channels; ++ci) {
                for (int co = 0; co < out_channels; ++co) {
                    grad[dense_weight_offset(
                        kernel_id,
                        ci,
                        co,
                        weight_layout,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        in_channels,
                        out_channels
                    )] += feats[in_row * feat_s0 + ci * feat_s1] *
                          cotangent[out_row * cotangent_s0 + co * cotangent_s1];
                }
            }
        }
    }
}

[[kernel]] void sparse_pool_f32_i32_serial(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const int* offsets [[buffer(3)]],
    device int* out_coords [[buffer(4)]],
    device float* out_feats [[buffer(5)]],
    device int* counts [[buffer(6)]],
    constant const int& reduce [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_out_rows * 4; ++index) {
        out_coords[index] = 0;
    }
    for (int index = 0; index < n_out_rows * channels; ++index) {
        out_feats[index] = reduce == 1 ? -INFINITY : 0.0f;
    }

    int out_count = 0;
    for (int row = 0; row < rows; ++row) {
        int candidate[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
        if (!seen_forward_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            )) {
            write_coord(out_coords, out_count, candidate);
            out_count += 1;
        }
    }

    int edge_count = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (!valid_forward_relation_coord(
                    coords,
                    rows,
                    kernel_id,
                    offsets,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    in_row,
                    out_coord,
                    out_row
                )) {
                continue;
            }
            edge_count += 1;
            for (int channel = 0; channel < channels; ++channel) {
                int out_index = out_row * channels + channel;
                float value = feats[in_row * feat_s0 + channel * feat_s1];
                if (reduce == 1) {
                    out_feats[out_index] = max(out_feats[out_index], value);
                } else {
                    out_feats[out_index] += value;
                }
            }
        }
    }
    if (reduce == 2) {
        for (int out_row = 0; out_row < out_count; ++out_row) {
            int degree = degree_for_forward_out_row(
                coords,
                offsets,
                rows,
                n_kernels,
                out_row,
                stride_x,
                stride_y,
                stride_z,
                pad_x,
                pad_y,
                pad_z
            );
            for (int channel = 0; channel < channels; ++channel) {
                out_feats[out_row * channels + channel] /= float(degree);
            }
        }
    }
    counts[0] = edge_count;
    counts[1] = out_count;
}

[[kernel]] void sparse_pool_clear_f32_i32(
    device int* out_coords [[buffer(0)]],
    device float* out_feats [[buffer(1)]],
    device int* counts [[buffer(2)]],
    constant const int& coord_total [[buffer(3)]],
    constant const int& feat_total [[buffer(4)]],
    constant const int& reduce [[buffer(5)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem == 0) {
        counts[0] = 0;
        counts[1] = 0;
    }
    if (elem < uint(coord_total)) {
        out_coords[elem] = 0;
    }
    if (elem < uint(feat_total)) {
        out_feats[elem] = reduce == 1 ? -INFINITY : 0.0f;
    }
}

[[kernel]] void sparse_pool_forward_gather_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const int* offsets [[buffer(3)]],
    device const int* out_coords [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device float* out_feats [[buffer(6)]],
    constant const int& reduce [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    int out_count = counts[1];
    int rows = min(active_rows[0], n_in_rows);
    if (out_row >= out_count) {
        return;
    }

    int out_base = out_row * 4;
    float acc = reduce == 1 ? -INFINITY : 0.0f;
    int degree = 0;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int offset_base = kernel_id * 3;
        int input_coord[4] = {
            out_coords[out_base],
            out_coords[out_base + 1] * stride_x + offsets[offset_base] - pad_x,
            out_coords[out_base + 2] * stride_y + offsets[offset_base + 1] -
                pad_y,
            out_coords[out_base + 3] * stride_z + offsets[offset_base + 2] -
                pad_z,
        };
        int in_row = -1;
        if (!find_input_row(coords, rows, input_coord, in_row)) {
            continue;
        }

        float value = feats[in_row * feat_s0 + channel * feat_s1];
        if (reduce == 1) {
            acc = max(acc, value);
        } else {
            acc += value;
        }
        degree += 1;
    }

    if (reduce == 2) {
        acc /= float(max(degree, 1));
    }
    out_feats[elem] = acc;
}

[[kernel]] void sparse_pool_grad_f32_i32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* coords [[buffer(3)]],
    device const int* active_rows [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* grad [[buffer(6)]],
    constant const int& reduce [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& cotangent_s0 [[buffer(18)]],
    constant const int& cotangent_s1 [[buffer(19)]],
    constant const int& feat_s0 [[buffer(20)]],
    constant const int& feat_s1 [[buffer(21)]],
    constant const int& pooled_s0 [[buffer(22)]],
    constant const int& pooled_s1 [[buffer(23)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    (void)n_out_rows;
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_in_rows * channels; ++index) {
        grad[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (!valid_forward_relation_coord(
                    coords,
                    rows,
                    kernel_id,
                    offsets,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    in_row,
                    out_coord,
                    out_row
                )) {
                continue;
            }
            int degree = degree_for_forward_out_row(
                coords,
                offsets,
                rows,
                n_kernels,
                out_row,
                stride_x,
                stride_y,
                stride_z,
                pad_x,
                pad_y,
                pad_z
            );
            for (int channel = 0; channel < channels; ++channel) {
                int in_index = in_row * channels + channel;
                float feat_value = feats[in_row * feat_s0 + channel * feat_s1];
                float pooled_value =
                    pooled[out_row * pooled_s0 + channel * pooled_s1];
                float scale = 1.0f;
                if (reduce == 1) {
                    if (feat_value != pooled_value) {
                        continue;
                    }
                    int tie_count =
                        max_pool_tie_count_for_forward_out_row_channel(
                            coords,
                            offsets,
                            feats,
                            rows,
                            n_kernels,
                            out_row,
                            channel,
                            pooled_value,
                            stride_x,
                            stride_y,
                            stride_z,
                            pad_x,
                            pad_y,
                            pad_z,
                            feat_s0,
                            feat_s1
                        );
                    if (tie_count == 0) {
                        continue;
                    }
                    scale = 1.0f / float(tie_count);
                } else if (reduce == 2) {
                    scale = 1.0f / float(degree);
                }
                grad[in_index] +=
                    cotangent[out_row * cotangent_s0 + channel * cotangent_s1] *
                    scale;
            }
        }
    }
}

[[kernel]] void sparse_pool_jvp_f32_i32_serial(
    device const float* tangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* coords [[buffer(3)]],
    device const int* active_rows [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant const int& reduce [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& tangent_s0 [[buffer(18)]],
    constant const int& tangent_s1 [[buffer(19)]],
    constant const int& feat_s0 [[buffer(20)]],
    constant const int& feat_s1 [[buffer(21)]],
    constant const int& pooled_s0 [[buffer(22)]],
    constant const int& pooled_s1 [[buffer(23)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_out_rows * channels; ++index) {
        out[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (!valid_forward_relation_coord(
                    coords,
                    rows,
                    kernel_id,
                    offsets,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    in_row,
                    out_coord,
                    out_row
                )) {
                continue;
            }
            int degree = degree_for_forward_out_row(
                coords,
                offsets,
                rows,
                n_kernels,
                out_row,
                stride_x,
                stride_y,
                stride_z,
                pad_x,
                pad_y,
                pad_z
            );
            for (int channel = 0; channel < channels; ++channel) {
                float feat_value = feats[in_row * feat_s0 + channel * feat_s1];
                float pooled_value =
                    pooled[out_row * pooled_s0 + channel * pooled_s1];
                float scale = 1.0f;
                if (reduce == 1) {
                    if (feat_value != pooled_value) {
                        continue;
                    }
                    int first_rank =
                        max_pool_first_rank_for_forward_out_row_channel(
                            coords,
                            offsets,
                            feats,
                            rows,
                            n_kernels,
                            out_row,
                            channel,
                            pooled_value,
                            stride_x,
                            stride_y,
                            stride_z,
                            pad_x,
                            pad_y,
                            pad_z,
                            feat_s0,
                            feat_s1
                        );
                    if (in_row * n_kernels + kernel_id != first_rank) {
                        continue;
                    }
                } else if (reduce == 2) {
                    scale = 1.0f / float(degree);
                }
                out[out_row * channels + channel] +=
                    tangent[in_row * tangent_s0 + channel * tangent_s1] * scale;
            }
        }
    }
}
