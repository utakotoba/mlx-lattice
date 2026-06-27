inline bool coord_equal(
    device const int* lhs,
    int lhs_row,
    device const int* rhs,
    int rhs_row
) {
    int lhs_base = lhs_row * 4;
    int rhs_base = rhs_row * 4;
    return lhs[lhs_base] == rhs[rhs_base] &&
           lhs[lhs_base + 1] == rhs[rhs_base + 1] &&
           lhs[lhs_base + 2] == rhs[rhs_base + 2] &&
           lhs[lhs_base + 3] == rhs[rhs_base + 3];
}

inline bool
coord4_equal(thread const int* lhs, device const int* rhs, int rhs_row) {
    int rhs_base = rhs_row * 4;
    return lhs[0] == rhs[rhs_base] && lhs[1] == rhs[rhs_base + 1] &&
           lhs[2] == rhs[rhs_base + 2] && lhs[3] == rhs[rhs_base + 3];
}

inline int floor_div_int(int value, int divisor) {
    int quotient = value / divisor;
    int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        quotient -= 1;
    }
    return quotient;
}

inline void write_coord(device int* out, int row, thread const int* coord) {
    int base = row * 4;
    out[base] = coord[0];
    out[base + 1] = coord[1];
    out[base + 2] = coord[2];
    out[base + 3] = coord[3];
}

inline void write_edge(
    device int* in_rows,
    device int* out_rows,
    device int* kernel_ids,
    int row,
    int in_row,
    int out_row,
    int kernel_id
) {
    in_rows[row] = in_row;
    out_rows[row] = out_row;
    kernel_ids[row] = kernel_id;
}

inline int coord_hash_i32(int b, int x, int y, int z) {
    uint hash = 2166136261u;
    hash = (hash ^ uint(b)) * 16777619u;
    hash = (hash ^ uint(x)) * 16777619u;
    hash = (hash ^ uint(y)) * 16777619u;
    hash = (hash ^ uint(z)) * 16777619u;
    return int(hash & 0x7fffffffu);
}

inline int coord_hash_i32(device const int* coords, int row) {
    int base = row * 4;
    return coord_hash_i32(
        coords[base], coords[base + 1], coords[base + 2], coords[base + 3]
    );
}

inline int coord_hash_i32(thread const int* coord) {
    return coord_hash_i32(coord[0], coord[1], coord[2], coord[3]);
}

inline int lookup_coord_row_hash(
    device const int* coords,
    device const int* table_rows,
    int table_capacity,
    thread const int* target
) {
    int key = coord_hash_i32(target);
    int slot = key & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int row = table_rows[slot];
        if (row < 0) {
            return -1;
        }
        if (coord4_equal(target, coords, row)) {
            return row;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
    return -1;
}

inline void insert_coord_row_hash(
    device const int* coords,
    int row,
    device atomic_int* table_rows,
    int table_capacity
) {
    int slot = coord_hash_i32(coords, row) & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(
                &table_rows[slot],
                &expected,
                row,
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            return;
        }
        if (expected >= 0 && coord_equal(coords, expected, coords, row)) {
            atomic_fetch_min_explicit(
                &table_rows[slot], row, memory_order_relaxed
            );
            return;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

inline float squared_spatial_distance(
    device const int* lhs,
    int lhs_row,
    device const int* rhs,
    int rhs_row
) {
    int lhs_base = lhs_row * 4;
    int rhs_base = rhs_row * 4;
    float dx = float(lhs[lhs_base + 1] - rhs[rhs_base + 1]);
    float dy = float(lhs[lhs_base + 2] - rhs[rhs_base + 2]);
    float dz = float(lhs[lhs_base + 3] - rhs[rhs_base + 3]);
    return dx * dx + dy * dy + dz * dz;
}

inline bool same_batch(
    device const int* lhs,
    int lhs_row,
    device const int* rhs,
    int rhs_row
) {
    return lhs[lhs_row * 4] == rhs[rhs_row * 4];
}
