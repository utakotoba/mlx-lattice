#include <metal_stdlib>

using namespace metal;

#include "native/features/coordinates/metal/common.metal"

// MARK: - generative relations

[[kernel]] void build_generative_kernel_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* row_offsets [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& stride_x [[buffer(11)]],
    constant const int& stride_y [[buffer(12)]],
    constant const int& stride_z [[buffer(13)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    uint total = uint(logical_rows * kernel_count);
    if (elem == 0) {
        counts[0] = int(total);
        counts[1] = int(total);
        row_offsets[total] = int(total);
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
    row_offsets[out_row] = out_row;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + offsets[offset_base];
    out_coords[out_base + 2] =
        coords[in_base + 2] * stride_y + offsets[offset_base + 1];
    out_coords[out_base + 3] =
        coords[in_base + 3] * stride_z + offsets[offset_base + 2];
}

// MARK: - generic relations

[[kernel]] void build_identity_forward_relation_slots_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const int* table_rows [[buffer(3)]],
    device int* in_rows [[buffer(4)]],
    device int* out_rows [[buffer(5)]],
    device int* kernel_ids [[buffer(6)]],
    device int* row_offsets [[buffer(7)]],
    device int* out_coords [[buffer(8)]],
    device int* counts [[buffer(9)]],
    constant const int& rows [[buffer(10)]],
    constant const int& kernel_count [[buffer(11)]],
    constant const int& table_capacity [[buffer(12)]],
    uint elem [[thread_position_in_grid]]
) {
    int out_count = min(active_rows[0], rows);
    int edge_count = out_count * kernel_count;

    if (elem == 0) {
        counts[0] = edge_count;
        counts[1] = out_count;
    }

    if (elem <= uint(rows)) {
        int row = int(elem);
        row_offsets[row] = min(row, out_count) * kernel_count;
    }

    int coord_total = rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < out_count ? coords[elem] : 0;
    }

    if (elem >= uint(edge_count)) {
        return;
    }

    int out_row = int(elem) / kernel_count;
    int kernel_id = int(elem) - out_row * kernel_count;
    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int candidate[4] = {
        coords[out_base],
        coords[out_base + 1] + kernel_offsets[offset_base],
        coords[out_base + 2] + kernel_offsets[offset_base + 1],
        coords[out_base + 3] + kernel_offsets[offset_base + 2],
    };
    int in_row =
        lookup_coord_row_hash(coords, table_rows, table_capacity, candidate);
    in_rows[elem] = in_row;
    out_rows[elem] = out_row;
    kernel_ids[elem] = in_row >= 0 ? kernel_id : -1;
}

[[kernel]] void count_identity_forward_relation_degrees_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const int* table_rows [[buffer(3)]],
    device int* row_degrees [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    device int* counts [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    constant const int& kernel_count [[buffer(8)]],
    constant const int& table_capacity [[buffer(9)]],
    uint row_id [[thread_position_in_grid]]
) {
    int out_count = min(active_rows[0], rows);
    if (row_id == 0) {
        counts[1] = out_count;
    }
    if (row_id >= uint(rows)) {
        return;
    }

    int row = int(row_id);
    int out_base = row * 4;
    if (row < out_count) {
        out_coords[out_base] = coords[out_base];
        out_coords[out_base + 1] = coords[out_base + 1];
        out_coords[out_base + 2] = coords[out_base + 2];
        out_coords[out_base + 3] = coords[out_base + 3];
    } else {
        out_coords[out_base] = 0;
        out_coords[out_base + 1] = 0;
        out_coords[out_base + 2] = 0;
        out_coords[out_base + 3] = 0;
        row_degrees[row] = 0;
        return;
    }

    int degree = 0;
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int offset_base = kernel_id * 3;
        int candidate[4] = {
            coords[out_base],
            coords[out_base + 1] + kernel_offsets[offset_base],
            coords[out_base + 2] + kernel_offsets[offset_base + 1],
            coords[out_base + 3] + kernel_offsets[offset_base + 2],
        };
        int in_row = lookup_coord_row_hash(
            coords, table_rows, table_capacity, candidate
        );
        if (in_row >= 0) {
            ++degree;
        }
    }
    row_degrees[row] = degree;
}

[[kernel]] void fill_identity_forward_relation_compact_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* table_rows [[buffer(2)]],
    device const int* row_offsets [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    device int* in_rows [[buffer(5)]],
    device int* out_rows [[buffer(6)]],
    device int* kernel_ids [[buffer(7)]],
    constant const int& rows [[buffer(8)]],
    constant const int& kernel_count [[buffer(9)]],
    constant const int& table_capacity [[buffer(10)]],
    uint row_id [[thread_position_in_grid]]
) {
    int out_count = min(counts[1], rows);
    if (row_id >= uint(out_count)) {
        return;
    }

    int out_row = int(row_id);
    int out_base = out_row * 4;
    int dst = row_offsets[out_row];
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int offset_base = kernel_id * 3;
        int candidate[4] = {
            coords[out_base],
            coords[out_base + 1] + kernel_offsets[offset_base],
            coords[out_base + 2] + kernel_offsets[offset_base + 1],
            coords[out_base + 3] + kernel_offsets[offset_base + 2],
        };
        int in_row = lookup_coord_row_hash(
            coords, table_rows, table_capacity, candidate
        );
        if (in_row < 0) {
            continue;
        }
        in_rows[dst] = in_row;
        out_rows[dst] = out_row;
        kernel_ids[dst] = kernel_id;
        ++dst;
    }
}

[[kernel]] void build_strided_forward_relation_slots_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* out_coords [[buffer(2)]],
    device int* counts [[buffer(3)]],
    device const int* table_rows [[buffer(4)]],
    device int* in_rows [[buffer(5)]],
    device int* out_rows [[buffer(6)]],
    device int* kernel_ids [[buffer(7)]],
    device int* row_offsets [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& table_capacity [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    uint elem [[thread_position_in_grid]]
) {
    int out_count = min(counts[1], rows);
    int edge_count = out_count * kernel_count;

    if (elem == 0) {
        counts[0] = edge_count;
    }

    if (elem <= uint(rows)) {
        int row = int(elem);
        row_offsets[row] = min(row, out_count) * kernel_count;
    }

    if (elem >= uint(edge_count)) {
        return;
    }

    int out_row = int(elem) / kernel_count;
    int kernel_id = int(elem) - out_row * kernel_count;
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
    int in_row =
        lookup_coord_row_hash(coords, table_rows, table_capacity, candidate);
    in_rows[elem] = in_row;
    out_rows[elem] = out_row;
    kernel_ids[elem] = in_row >= 0 ? kernel_id : -1;
}

[[kernel]] void build_target_forward_relation_slots_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* target_coords [[buffer(2)]],
    device const int* target_active_rows [[buffer(3)]],
    device const int* table_rows [[buffer(4)]],
    device int* in_rows [[buffer(5)]],
    device int* out_rows [[buffer(6)]],
    device int* kernel_ids [[buffer(7)]],
    device int* row_offsets [[buffer(8)]],
    device int* counts [[buffer(9)]],
    constant const int& target_rows [[buffer(10)]],
    constant const int& kernel_count [[buffer(11)]],
    constant const int& table_capacity [[buffer(12)]],
    constant const int& stride_x [[buffer(13)]],
    constant const int& stride_y [[buffer(14)]],
    constant const int& stride_z [[buffer(15)]],
    constant const int& pad_x [[buffer(16)]],
    constant const int& pad_y [[buffer(17)]],
    constant const int& pad_z [[buffer(18)]],
    uint elem [[thread_position_in_grid]]
) {
    int out_count = min(target_active_rows[0], target_rows);
    int edge_count = out_count * kernel_count;

    if (elem == 0) {
        counts[0] = edge_count;
        counts[1] = out_count;
    }

    if (elem <= uint(target_rows)) {
        int row = int(elem);
        row_offsets[row] = min(row, out_count) * kernel_count;
    }

    if (elem >= uint(edge_count)) {
        return;
    }

    int out_row = int(elem) / kernel_count;
    int kernel_id = int(elem) - out_row * kernel_count;
    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int candidate[4] = {
        target_coords[out_base],
        target_coords[out_base + 1] * stride_x + kernel_offsets[offset_base] -
            pad_x,
        target_coords[out_base + 2] * stride_y +
            kernel_offsets[offset_base + 1] - pad_y,
        target_coords[out_base + 3] * stride_z +
            kernel_offsets[offset_base + 2] - pad_z,
    };
    int in_row =
        lookup_coord_row_hash(coords, table_rows, table_capacity, candidate);
    in_rows[elem] = in_row;
    out_rows[elem] = out_row;
    kernel_ids[elem] = in_row >= 0 ? kernel_id : -1;
}

[[kernel]] void count_forward_relation_slot_rows_i32(
    device const int* slot_in_rows [[buffer(0)]],
    device const int* slot_kernel_ids [[buffer(1)]],
    device int* row_offsets [[buffer(2)]],
    device const int* counts [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& kernel_count [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row == 0) {
        row_offsets[0] = 0;
    }
    int out_count = min(counts[1], rows);
    if (row >= uint(rows)) {
        return;
    }
    if (row >= uint(out_count)) {
        row_offsets[int(row) + 1] = 0;
        return;
    }

    int start = int(row) * kernel_count;
    int degree = 0;
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int edge = start + kernel_id;
        if (slot_in_rows[edge] >= 0 && slot_kernel_ids[edge] >= 0) {
            ++degree;
        }
    }
    row_offsets[int(row) + 1] = degree;
}

[[kernel]] void prefix_forward_relation_slot_rows_i32(
    device int* row_offsets [[buffer(0)]],
    device int* counts [[buffer(1)]],
    constant const int& rows [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int out_count = min(counts[1], rows);
    int total = 0;
    for (int row = 0; row < out_count; ++row) {
        int degree = row_offsets[row + 1];
        row_offsets[row] = total;
        total += degree;
    }
    for (int row = out_count; row <= rows; ++row) {
        row_offsets[row] = total;
    }
    counts[0] = total;
}

[[kernel]] void count_forward_relation_slot_degrees_i32(
    device const int* slot_in_rows [[buffer(0)]],
    device const int* slot_kernel_ids [[buffer(1)]],
    device const int* counts [[buffer(2)]],
    device int* row_degrees [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& kernel_count [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    int out_count = min(counts[1], rows);
    if (row >= uint(rows)) {
        return;
    }
    if (row >= uint(out_count)) {
        row_degrees[row] = 0;
        return;
    }

    int start = int(row) * kernel_count;
    int degree = 0;
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int edge = start + kernel_id;
        if (slot_in_rows[edge] >= 0 && slot_kernel_ids[edge] >= 0) {
            ++degree;
        }
    }
    row_degrees[row] = degree;
}

[[kernel]] void scan_relation_row_degrees_blocks_i32(
    device const int* row_degrees [[buffer(0)]],
    device int* local_offsets [[buffer(1)]],
    device int* block_offsets [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup int scan[256];
    uint row = block * 256 + tid;
    scan[tid] = row < uint(rows) ? row_degrees[row] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < 256; offset <<= 1) {
        uint index = (tid + 1) * offset * 2 - 1;
        if (index < 256) {
            scan[index] += scan[index - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        block_offsets[block] = scan[255];
        scan[255] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 128; offset > 0; offset >>= 1) {
        uint index = (tid + 1) * offset * 2 - 1;
        if (index < 256) {
            int value = scan[index - offset];
            scan[index - offset] = scan[index];
            scan[index] += value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < uint(rows)) {
        local_offsets[row] = scan[tid];
    }
}

[[kernel]] void finalize_forward_relation_row_offsets_i32(
    device const int* local_offsets [[buffer(0)]],
    device const int* block_offsets [[buffer(1)]],
    device int* row_offsets [[buffer(2)]],
    device const int* counts [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row < uint(rows)) {
        row_offsets[row] = block_offsets[row / 256] + local_offsets[row];
    } else if (row == uint(rows)) {
        row_offsets[row] = counts[0];
    }
}

[[kernel]] void compact_forward_relation_slots_i32(
    device const int* slot_in_rows [[buffer(0)]],
    device const int* slot_out_rows [[buffer(1)]],
    device const int* slot_kernel_ids [[buffer(2)]],
    device const int* row_offsets [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    device int* in_rows [[buffer(5)]],
    device int* out_rows [[buffer(6)]],
    device int* kernel_ids [[buffer(7)]],
    constant const int& rows [[buffer(8)]],
    constant const int& kernel_count [[buffer(9)]],
    uint row [[thread_position_in_grid]]
) {
    int out_count = min(counts[1], rows);
    if (row >= uint(out_count)) {
        return;
    }

    int slot_start = int(row) * kernel_count;
    int dst = row_offsets[int(row)];
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int slot = slot_start + kernel_id;
        int in_row = slot_in_rows[slot];
        int slot_kernel_id = slot_kernel_ids[slot];
        if (in_row < 0 || slot_kernel_id < 0) {
            continue;
        }
        in_rows[dst] = in_row;
        out_rows[dst] = slot_out_rows[slot];
        kernel_ids[dst] = slot_kernel_id;
        ++dst;
    }
}

[[kernel]] void build_transposed_direct_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* row_offsets [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& stride_x [[buffer(11)]],
    constant const int& stride_y [[buffer(12)]],
    constant const int& stride_z [[buffer(13)]],
    constant const int& pad_x [[buffer(14)]],
    constant const int& pad_y [[buffer(15)]],
    constant const int& pad_z [[buffer(16)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    int total = logical_rows * kernel_count;
    if (elem == 0) {
        counts[0] = total;
        counts[1] = total;
        row_offsets[total] = total;
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
    row_offsets[out_row] = out_row;
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
    device int* row_offsets [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& stride_x [[buffer(11)]],
    constant const int& stride_y [[buffer(12)]],
    constant const int& stride_z [[buffer(13)]],
    constant const int& pad_x [[buffer(14)]],
    constant const int& pad_y [[buffer(15)]],
    constant const int& pad_z [[buffer(16)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

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
        }
    }

    int edge_count = 0;
    for (int out_row = 0; out_row < out_count; ++out_row) {
        row_offsets[out_row] = edge_count;
        for (int in_row = 0; in_row < logical_rows; ++in_row) {
            int in_base = in_row * 4;
            for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int candidate[4] = {
                    coords[in_base],
                    coords[in_base + 1] * stride_x +
                        kernel_offsets[offset_base] - pad_x,
                    coords[in_base + 2] * stride_y +
                        kernel_offsets[offset_base + 1] - pad_y,
                    coords[in_base + 3] * stride_z +
                        kernel_offsets[offset_base + 2] - pad_z,
                };
                if (!coord4_equal(candidate, out_coords, out_row)) {
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
    }
    for (int out_row = out_count; out_row <= rows * kernel_count; ++out_row) {
        row_offsets[out_row] = edge_count;
    }
    counts[0] = edge_count;
    counts[1] = out_count;
}

// MARK: - relation execution views

[[kernel]] void clear_relation_grouped_view_i32(
    device int* row_offsets [[buffer(0)]],
    constant const int& group_count [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem <= uint(group_count)) {
        row_offsets[elem] = 0;
    }
}

[[kernel]] void count_relation_grouped_view_i32(
    device const int* group_ids [[buffer(0)]],
    device const int* counts [[buffer(1)]],
    device atomic_int* row_offsets [[buffer(2)]],
    constant const int& edge_capacity [[buffer(3)]],
    constant const int& group_count [[buffer(4)]],
    uint edge [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    if (edge >= uint(edge_count)) {
        return;
    }

    int group = group_ids[edge];
    if (group >= 0 && group < group_count) {
        atomic_fetch_add_explicit(
            &row_offsets[group + 1], 1, memory_order_relaxed
        );
    }
}

[[kernel]] void prefix_relation_grouped_view_i32(
    device int* row_offsets [[buffer(0)]],
    device int* cursors [[buffer(1)]],
    constant const int& group_count [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int total = 0;
    row_offsets[0] = 0;
    cursors[0] = 0;
    for (int row = 1; row <= group_count; ++row) {
        int count = row_offsets[row];
        total += count;
        row_offsets[row] = total;
        cursors[row] = total;
    }
}

[[kernel]] void scan_relation_grouped_view_blocks_i32(
    device const int* counts_by_group [[buffer(0)]],
    device int* local_offsets [[buffer(1)]],
    device int* block_offsets [[buffer(2)]],
    constant const int& group_count [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup int scan[256];
    uint group = block * 256 + tid;
    scan[tid] = group < uint(group_count) ? counts_by_group[group + 1] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < 256; offset <<= 1) {
        uint index = (tid + 1) * offset * 2 - 1;
        if (index < 256) {
            scan[index] += scan[index - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        block_offsets[block] = scan[255];
        scan[255] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 128; offset > 0; offset >>= 1) {
        uint index = (tid + 1) * offset * 2 - 1;
        if (index < 256) {
            int value = scan[index - offset];
            scan[index - offset] = scan[index];
            scan[index] += value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (group < uint(group_count)) {
        local_offsets[group] = scan[tid];
    }
}

[[kernel]] void finalize_relation_grouped_view_i32(
    device const int* local_offsets [[buffer(0)]],
    device const int* block_offsets [[buffer(1)]],
    device const int* total_count [[buffer(2)]],
    device int* row_offsets [[buffer(3)]],
    device int* cursors [[buffer(4)]],
    constant const int& group_count [[buffer(5)]],
    uint group [[thread_position_in_grid]]
) {
    if (group < uint(group_count)) {
        int offset = block_offsets[group / 256] + local_offsets[group];
        row_offsets[group] = offset;
        cursors[group] = offset;
    } else if (group == uint(group_count)) {
        row_offsets[group_count] = total_count[0];
        cursors[group_count] = total_count[0];
    }
}

[[kernel]] void fill_relation_grouped_view_i32(
    device const int* group_ids [[buffer(0)]],
    device const int* counts [[buffer(1)]],
    device atomic_int* cursors [[buffer(2)]],
    device int* edge_ids [[buffer(3)]],
    constant const int& edge_capacity [[buffer(4)]],
    constant const int& group_count [[buffer(5)]],
    uint edge [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    if (edge >= uint(edge_count)) {
        return;
    }

    int group = group_ids[edge];
    if (group >= 0 && group < group_count) {
        int slot =
            atomic_fetch_add_explicit(&cursors[group], 1, memory_order_relaxed);
        edge_ids[slot] = int(edge);
    }
}

[[kernel]] void clear_relation_direct_view_i32(
    device int* edge_ids [[buffer(0)]],
    constant const int& group_count [[buffer(1)]],
    uint group [[thread_position_in_grid]]
) {
    if (group < uint(group_count)) {
        edge_ids[group] = -1;
    }
}

[[kernel]] void fill_relation_direct_view_i32(
    device const int* group_ids [[buffer(0)]],
    device const int* counts [[buffer(1)]],
    device int* edge_ids [[buffer(2)]],
    constant const int& edge_capacity [[buffer(3)]],
    constant const int& group_count [[buffer(4)]],
    uint edge [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    if (edge >= uint(edge_count)) {
        return;
    }
    int group = group_ids[edge];
    if (group >= 0 && group < group_count) {
        edge_ids[group] = int(edge);
    }
}

[[kernel]] void clear_relation_implicit_gemm_view_i32(
    device int* out_in_map [[buffer(0)]],
    device int* row_masks [[buffer(1)]],
    constant const int& total_slots [[buffer(2)]],
    constant const int& total_mask_words [[buffer(3)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(total_slots)) {
        out_in_map[elem] = -1;
    }
    if (elem < uint(total_mask_words)) {
        row_masks[elem] = 0;
    }
}

[[kernel]] void build_relation_implicit_gemm_view_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* source_active_rows [[buffer(1)]],
    device const int* output_coords [[buffer(2)]],
    device const int* output_active_rows [[buffer(3)]],
    device const int* kernel_offsets [[buffer(4)]],
    device const int* table_rows [[buffer(5)]],
    device int* out_in_map [[buffer(6)]],
    device atomic_int* row_masks [[buffer(7)]],
    constant const int& source_rows [[buffer(8)]],
    constant const int& output_rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& table_capacity [[buffer(11)]],
    constant const int& mask_words [[buffer(12)]],
    constant const int& stride_x [[buffer(13)]],
    constant const int& stride_y [[buffer(14)]],
    constant const int& stride_z [[buffer(15)]],
    constant const int& pad_x [[buffer(16)]],
    constant const int& pad_y [[buffer(17)]],
    constant const int& pad_z [[buffer(18)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = output_rows * kernel_count;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / kernel_count;
    int kernel_id = int(elem) - out_row * kernel_count;
    int output_active = min(output_active_rows[0], output_rows);
    if (out_row >= output_active) {
        return;
    }

    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int candidate[4] = {
        output_coords[out_base],
        output_coords[out_base + 1] * stride_x + kernel_offsets[offset_base] -
            pad_x,
        output_coords[out_base + 2] * stride_y +
            kernel_offsets[offset_base + 1] - pad_y,
        output_coords[out_base + 3] * stride_z +
            kernel_offsets[offset_base + 2] - pad_z,
    };
    int in_row = lookup_coord_row_hash(
        source_coords, table_rows, table_capacity, candidate
    );
    int source_active = min(source_active_rows[0], source_rows);
    if (in_row < 0 || in_row >= source_active) {
        return;
    }
    out_in_map[elem] = in_row;
    int word = kernel_id / 32;
    int bit = kernel_id - word * 32;
    atomic_fetch_or_explicit(
        &row_masks[out_row * mask_words + word],
        int(1u << uint(bit)),
        memory_order_relaxed
    );
}

// MARK: - neighbor relations
