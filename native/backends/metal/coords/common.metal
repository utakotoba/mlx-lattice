#pragma once

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

inline void build_views(
    device const int* in_rows,
    device const int* out_rows,
    device const int* kernel_ids,
    int edge_count,
    int out_count,
    int kernel_count,
    int input_count,
    device int* output_csr_offsets,
    device int* output_csr_in_rows,
    device int* output_csr_kernel_ids,
    device int* kernel_bucket_offsets,
    device int* kernel_bucket_in_rows,
    device int* kernel_bucket_out_rows,
    device int* input_csr_offsets,
    device int* input_csr_out_rows,
    device int* input_csr_kernel_ids
) {
    for (int row = 0; row <= out_count; ++row) {
        output_csr_offsets[row] = 0;
    }
    for (int row = 0; row <= kernel_count; ++row) {
        kernel_bucket_offsets[row] = 0;
    }
    for (int row = 0; row <= input_count; ++row) {
        input_csr_offsets[row] = 0;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        output_csr_offsets[out_rows[edge] + 1] += 1;
        kernel_bucket_offsets[kernel_ids[edge] + 1] += 1;
        input_csr_offsets[in_rows[edge] + 1] += 1;
    }
    for (int row = 0; row < out_count; ++row) {
        output_csr_offsets[row + 1] += output_csr_offsets[row];
    }
    for (int row = 0; row < kernel_count; ++row) {
        kernel_bucket_offsets[row + 1] += kernel_bucket_offsets[row];
    }
    for (int row = 0; row < input_count; ++row) {
        input_csr_offsets[row + 1] += input_csr_offsets[row];
    }

    for (int out_row = 0; out_row < out_count; ++out_row) {
        int cursor = output_csr_offsets[out_row];
        for (int edge = 0; edge < edge_count; ++edge) {
            if (out_rows[edge] == out_row) {
                output_csr_in_rows[cursor] = in_rows[edge];
                output_csr_kernel_ids[cursor] = kernel_ids[edge];
                cursor += 1;
            }
        }
    }
    for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
        int cursor = kernel_bucket_offsets[kernel_id];
        for (int edge = 0; edge < edge_count; ++edge) {
            if (kernel_ids[edge] == kernel_id) {
                kernel_bucket_in_rows[cursor] = in_rows[edge];
                kernel_bucket_out_rows[cursor] = out_rows[edge];
                cursor += 1;
            }
        }
    }
    for (int in_row = 0; in_row < input_count; ++in_row) {
        int cursor = input_csr_offsets[in_row];
        for (int edge = 0; edge < edge_count; ++edge) {
            if (in_rows[edge] == in_row) {
                input_csr_out_rows[cursor] = out_rows[edge];
                input_csr_kernel_ids[cursor] = kernel_ids[edge];
                cursor += 1;
            }
        }
    }
}
