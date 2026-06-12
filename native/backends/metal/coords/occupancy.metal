#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/coords/common.metal"

inline void
occupancy_parent_coord(device const int* coords, int row, thread int* out) {
    int base = row * 4;
    out[0] = coords[base];
    out[1] = floor_div_int(coords[base + 1], 2);
    out[2] = floor_div_int(coords[base + 2], 2);
    out[3] = floor_div_int(coords[base + 3], 2);
}

inline int occupancy_child_bit(device const int* coords, int row) {
    int base = row * 4;
    int child = (coords[base + 1] & 1) | ((coords[base + 2] & 1) << 1) |
                ((coords[base + 3] & 1) << 2);
    return 1 << child;
}

inline int lookup_occupancy_parent_hash(
    device const int* coords,
    device const int* table_rows,
    int table_capacity,
    thread const int* target
) {
    int slot = coord_hash_i32(target) & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int row = table_rows[slot];
        if (row < 0) {
            return -1;
        }
        int stored[4];
        occupancy_parent_coord(coords, row, stored);
        if (target[0] == stored[0] && target[1] == stored[1] &&
            target[2] == stored[2] && target[3] == stored[3]) {
            return row;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
    return -1;
}

[[kernel]] void clear_occupancy_downsample_i32(
    device int* occupancy [[buffer(0)]],
    constant const int& rows [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(rows)) {
        occupancy[elem] = 0;
    }
}

[[kernel]] void build_occupancy_downsample_hash_i32(
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

    int candidate[4];
    occupancy_parent_coord(coords, int(row), candidate);
    int slot = coord_hash_i32(candidate) & (table_capacity - 1);
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
        int stored[4];
        occupancy_parent_coord(coords, expected, stored);
        if (candidate[0] == stored[0] && candidate[1] == stored[1] &&
            candidate[2] == stored[2] && candidate[3] == stored[3]) {
            atomic_fetch_min_explicit(
                &table_rows[slot], int(row), memory_order_relaxed
            );
            return;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

[[kernel]] void plan_occupancy_downsample_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* table_rows [[buffer(2)]],
    device int* selected [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& table_capacity [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows)) {
        return;
    }
    int logical_rows = min(active_rows[0], rows);
    if (row >= uint(logical_rows)) {
        selected[row] = 0;
        return;
    }

    int candidate[4];
    occupancy_parent_coord(coords, int(row), candidate);
    selected[row] = lookup_occupancy_parent_hash(
                        coords, table_rows, table_capacity, candidate
                    ) == int(row);
}

[[kernel]] void scatter_occupancy_downsample_i32(
    device const int* coords [[buffer(0)]],
    device const int* selected [[buffer(1)]],
    device const int* local_offsets [[buffer(2)]],
    device const int* block_offsets [[buffer(3)]],
    device int* out_coords [[buffer(4)]],
    constant const int& rows [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows) || selected[row] == 0) {
        return;
    }
    int candidate[4];
    occupancy_parent_coord(coords, int(row), candidate);
    int out_row = block_offsets[row / 256] + local_offsets[row];
    write_coord(out_coords, out_row, candidate);
}

[[kernel]] void fill_occupancy_downsample_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* table_rows [[buffer(2)]],
    device const int* local_offsets [[buffer(3)]],
    device const int* block_offsets [[buffer(4)]],
    device atomic_int* occupancy [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& table_capacity [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    if (row >= uint(logical_rows)) {
        return;
    }
    int candidate[4];
    occupancy_parent_coord(coords, int(row), candidate);
    int representative = lookup_occupancy_parent_hash(
        coords, table_rows, table_capacity, candidate
    );
    int out_row =
        block_offsets[representative / 256] + local_offsets[representative];
    atomic_fetch_or_explicit(
        &occupancy[out_row],
        occupancy_child_bit(coords, int(row)),
        memory_order_relaxed
    );
}

[[kernel]] void plan_occupancy_expand_i32(
    device const int* active_rows [[buffer(0)]],
    device const int* occupancy [[buffer(1)]],
    device int* selected [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = rows * 8;
    if (elem >= uint(total)) {
        return;
    }
    int parent = int(elem) >> 3;
    int child = int(elem) & 7;
    int logical_rows = min(active_rows[0], rows);
    selected[elem] =
        parent < logical_rows && ((occupancy[parent] & (1 << child)) != 0);
}

[[kernel]] void scatter_occupancy_expand_i32(
    device const int* coords [[buffer(0)]],
    device const int* selected [[buffer(1)]],
    device const int* local_offsets [[buffer(2)]],
    device const int* block_offsets [[buffer(3)]],
    device int* out_coords [[buffer(4)]],
    device int* parent_rows [[buffer(5)]],
    device int* child_indices [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = rows * 8;
    if (elem >= uint(total) || selected[elem] == 0) {
        return;
    }
    int parent = int(elem) >> 3;
    int child = int(elem) & 7;
    int out_row = block_offsets[elem / 256] + local_offsets[elem];
    int base = parent * 4;
    int expanded[4] = {
        coords[base],
        coords[base + 1] * 2 + (child & 1),
        coords[base + 2] * 2 + ((child >> 1) & 1),
        coords[base + 3] * 2 + ((child >> 2) & 1),
    };
    write_coord(out_coords, out_row, expanded);
    parent_rows[out_row] = parent;
    child_indices[out_row] = child;
}

[[kernel]] void child_coords_from_indices_i32(
    device const int* parent_coords [[buffer(0)]],
    device const int* child_indices [[buffer(1)]],
    device int* child_coords [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows)) {
        return;
    }
    int base = int(row) * 4;
    int child = child_indices[row] & 7;
    child_coords[base] = parent_coords[base];
    child_coords[base + 1] = parent_coords[base + 1] * 2 + (child & 1);
    child_coords[base + 2] = parent_coords[base + 2] * 2 + ((child >> 1) & 1);
    child_coords[base + 3] = parent_coords[base + 3] * 2 + ((child >> 2) & 1);
}
