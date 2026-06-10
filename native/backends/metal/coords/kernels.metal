#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/coords/common.metal"

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
    device int* out_coords [[buffer(9)]],
    device int* counts [[buffer(10)]],
    constant const int& target_rows [[buffer(11)]],
    constant const int& kernel_count [[buffer(12)]],
    constant const int& table_capacity [[buffer(13)]],
    constant const int& stride_x [[buffer(14)]],
    constant const int& stride_y [[buffer(15)]],
    constant const int& stride_z [[buffer(16)]],
    constant const int& pad_x [[buffer(17)]],
    constant const int& pad_y [[buffer(18)]],
    constant const int& pad_z [[buffer(19)]],
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

    int coord_total = target_rows * 4;
    if (elem < uint(coord_total)) {
        out_coords[elem] = target_coords[elem];
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

// MARK: - neighbor relations

[[kernel]] void build_neighbor_relation_i32(
    device const int* query_active_rows [[buffer(0)]],
    device int* query_rows [[buffer(1)]],
    device int* source_rows [[buffer(2)]],
    device int* neighbor_ids [[buffer(3)]],
    device float* distances [[buffer(4)]],
    device int* row_offsets [[buffer(5)]],
    device int* counts [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_capacity = query_capacity * max_neighbors;
    int query_count = min(query_active_rows[0], query_capacity);

    if (elem == 0) {
        counts[0] = 0;
        counts[1] = query_count;
    }

    if (elem <= uint(query_capacity)) {
        row_offsets[elem] = 0;
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

[[kernel]] void fill_radius_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* query_rows [[buffer(5)]],
    device int* source_rows [[buffer(6)]],
    device int* neighbor_ids [[buffer(7)]],
    device float* distances [[buffer(8)]],
    constant const int& source_capacity [[buffer(9)]],
    constant const int& query_capacity [[buffer(10)]],
    constant const int& max_neighbors [[buffer(11)]],
    constant const float& radius_squared [[buffer(12)]],
    constant const int& ceil_radius [[buffer(13)]],
    constant const int& table_capacity [[buffer(14)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = int(query_row) * max_neighbors;
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
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
        }
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void count_radius_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* row_counts [[buffer(5)]],
    constant const int& source_capacity [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    constant const float& radius_squared [[buffer(9)]],
    constant const int& ceil_radius [[buffer(10)]],
    constant const int& table_capacity [[buffer(11)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_capacity)) {
        return;
    }
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        row_counts[query_row] = 0;
        return;
    }

    int query_base = int(query_row) * 4;
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }
                selected += 1;
                if (selected >= max_neighbors) {
                    row_counts[query_row] = max_neighbors;
                    return;
                }
            }
        }
    }
    row_counts[query_row] = selected;
}

[[kernel]] void prefix_neighbor_row_offsets_i32(
    device int* row_offsets [[buffer(0)]],
    device int* counts [[buffer(1)]],
    constant const int& query_capacity [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int query_count = counts[1];
    int total = 0;
    for (int query_row = 0; query_row < query_count; ++query_row) {
        int row_count = row_offsets[query_row];
        row_offsets[query_row] = total;
        total += row_count;
    }
    row_offsets[query_count] = total;
    counts[0] = total;
    (void)query_capacity;
}

[[kernel]] void fill_radius_relation_compact_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device const int* row_offsets [[buffer(5)]],
    device int* query_rows [[buffer(6)]],
    device int* source_rows [[buffer(7)]],
    device int* neighbor_ids [[buffer(8)]],
    device float* distances [[buffer(9)]],
    constant const int& source_capacity [[buffer(10)]],
    constant const int& query_capacity [[buffer(11)]],
    constant const int& max_neighbors [[buffer(12)]],
    constant const float& radius_squared [[buffer(13)]],
    constant const int& ceil_radius [[buffer(14)]],
    constant const int& table_capacity [[buffer(15)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = row_offsets[query_row];
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
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
        }
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void count_knn_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* row_counts [[buffer(5)]],
    constant const int& source_capacity [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    constant const int& search_radius [[buffer(9)]],
    constant const int& table_capacity [[buffer(10)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_capacity)) {
        return;
    }
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        row_counts[query_row] = 0;
        return;
    }

    int query_base = int(query_row) * 4;
    int selected = 0;
    for (int dz = -search_radius; dz <= search_radius; ++dz) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }
                selected += 1;
                if (selected >= max_neighbors) {
                    row_counts[query_row] = max_neighbors;
                    return;
                }
            }
        }
    }
    row_counts[query_row] = selected;
}

[[kernel]] void fill_knn_relation_compact_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device const int* row_offsets [[buffer(5)]],
    device int* query_rows [[buffer(6)]],
    device int* source_rows [[buffer(7)]],
    device int* neighbor_ids [[buffer(8)]],
    device float* distances [[buffer(9)]],
    constant const int& source_capacity [[buffer(10)]],
    constant const int& query_capacity [[buffer(11)]],
    constant const int& max_neighbors [[buffer(12)]],
    constant const int& search_radius [[buffer(13)]],
    constant const int& table_capacity [[buffer(14)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = row_offsets[query_row];
    int selected = 0;
    for (int shell = 0; shell <= search_radius; ++shell) {
        for (int dz = -shell; dz <= shell; ++dz) {
            for (int dy = -shell; dy <= shell; ++dy) {
                for (int dx = -shell; dx <= shell; ++dx) {
                    if (max(max(abs(dx), abs(dy)), abs(dz)) != shell) {
                        continue;
                    }
                    int target[4];
                    target[0] = query_coords[query_base];
                    target[1] = query_coords[query_base + 1] + dx;
                    target[2] = query_coords[query_base + 2] + dy;
                    target[3] = query_coords[query_base + 3] + dz;
                    int source_row = lookup_coord_row_hash(
                        source_coords, source_table, table_capacity, target
                    );
                    if (source_row < 0 || source_row >= source_capacity ||
                        source_row >= source_active_rows[0]) {
                        continue;
                    }

                    float distance = float(dx * dx + dy * dy + dz * dz);
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
            }
        }
        if (selected >= max_neighbors) {
            float kth_distance = distances[slot_start + max_neighbors - 1];
            float next_min_distance = float((shell + 1) * (shell + 1));
            if (next_min_distance > kth_distance) {
                break;
            }
        }
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
    device int* row_offsets [[buffer(4)]],
    device int* counts [[buffer(5)]],
    constant const int& query_capacity [[buffer(6)]],
    constant const int& max_neighbors [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_edge = 0;
    int query_count = counts[1];
    for (int query_row = 0; query_row < query_count; ++query_row) {
        row_offsets[query_row] = out_edge;
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
    row_offsets[query_count] = out_edge;
    counts[0] = out_edge;
    (void)query_capacity;
}
