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
