#include "native/backends/metal/coords/common.metal"

inline bool coord_equal4(thread const int* lhs, thread const int* rhs) {
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] &&
           lhs[3] == rhs[3];
}

inline void downsample_coord(
    device const int* coords,
    int row,
    int stride_x,
    int stride_y,
    int stride_z,
    thread int* out
) {
    int base = row * 4;
    out[0] = coords[base];
    out[1] = floor_div_int(coords[base + 1], stride_x);
    out[2] = floor_div_int(coords[base + 2], stride_y);
    out[3] = floor_div_int(coords[base + 3], stride_z);
}

inline bool seen_forward_coord(
    device const int* coords,
    int row,
    int stride_x,
    int stride_y,
    int stride_z,
    thread const int* candidate
) {
    for (int prev = 0; prev < row; ++prev) {
        int previous[4];
        downsample_coord(coords, prev, stride_x, stride_y, stride_z, previous);
        if (coord_equal4(previous, candidate)) {
            return true;
        }
    }
    return false;
}

inline int forward_out_row_for_coord(
    device const int* coords,
    int rows,
    int stride_x,
    int stride_y,
    int stride_z,
    thread const int* target
) {
    int out_row = 0;
    for (int row = 0; row < rows; ++row) {
        int candidate[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
        if (seen_forward_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            )) {
            continue;
        }
        if (coord_equal4(candidate, target)) {
            return out_row;
        }
        out_row += 1;
    }
    return -1;
}

inline bool find_input_row(
    device const int* coords,
    int rows,
    thread const int* target,
    thread int& out_row
) {
    for (int row = 0; row < rows; ++row) {
        if (coord4_equal(target, coords, row)) {
            out_row = row;
            return true;
        }
    }
    return false;
}

inline bool valid_forward_edge_coord(
    device const int* coords,
    int rows,
    int kernel_id,
    device const int* offsets,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    int in_row,
    thread int* out_coord,
    thread int& out_row
) {
    int in_base = in_row * 4;
    int offset_base = kernel_id * 3;
    int vx = coords[in_base + 1] - offsets[offset_base] + pad_x;
    int vy = coords[in_base + 2] - offsets[offset_base + 1] + pad_y;
    int vz = coords[in_base + 3] - offsets[offset_base + 2] + pad_z;
    if (vx % stride_x != 0 || vy % stride_y != 0 || vz % stride_z != 0) {
        return false;
    }
    out_coord[0] = coords[in_base];
    out_coord[1] = vx / stride_x;
    out_coord[2] = vy / stride_y;
    out_coord[3] = vz / stride_z;
    out_row = forward_out_row_for_coord(
        coords, rows, stride_x, stride_y, stride_z, out_coord
    );
    return out_row >= 0;
}

inline void transposed_candidate(
    device const int* coords,
    device const int* offsets,
    int in_row,
    int kernel_id,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    thread int* out
) {
    int in_base = in_row * 4;
    int offset_base = kernel_id * 3;
    out[0] = coords[in_base];
    out[1] = coords[in_base + 1] * stride_x + offsets[offset_base] - pad_x;
    out[2] = coords[in_base + 2] * stride_y + offsets[offset_base + 1] - pad_y;
    out[3] = coords[in_base + 3] * stride_z + offsets[offset_base + 2] - pad_z;
}

inline int transposed_out_row_for_coord(
    device const int* coords,
    device const int* offsets,
    int rows,
    int kernels,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    thread const int* target
) {
    int out_row = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < kernels; ++kernel_id) {
            int candidate[4];
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
                candidate
            );
            bool seen = false;
            for (int prev_in = 0; prev_in <= in_row; ++prev_in) {
                int limit = prev_in == in_row ? kernel_id : kernels;
                for (int prev_kernel = 0; prev_kernel < limit; ++prev_kernel) {
                    int previous[4];
                    transposed_candidate(
                        coords,
                        offsets,
                        prev_in,
                        prev_kernel,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        previous
                    );
                    if (coord_equal4(previous, candidate)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) {
                    break;
                }
            }
            if (seen) {
                continue;
            }
            if (coord_equal4(candidate, target)) {
                return out_row;
            }
            out_row += 1;
        }
    }
    return -1;
}

inline int degree_for_forward_out_row(
    device const int* coords,
    device const int* offsets,
    int rows,
    int kernels,
    int out_row,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z
) {
    int degree = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < kernels; ++kernel_id) {
            int candidate[4];
            int edge_out = -1;
            if (valid_forward_edge_coord(
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
                    candidate,
                    edge_out
                ) &&
                edge_out == out_row) {
                degree += 1;
            }
        }
    }
    return max(degree, 1);
}

inline int weight_offset(
    int kernel_id,
    int in_channel,
    int out_channel,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    int weight_s0,
    int weight_s1,
    int weight_s2,
    int weight_s3,
    int weight_s4
) {
    if (weight_layout == 0) {
        return kernel_id * weight_s0 + in_channel * weight_s1 +
               out_channel * weight_s2;
    }

    int xy = kernel_y * kernel_z;
    int kx = kernel_id / xy;
    int rem = kernel_id % xy;
    int ky = rem / kernel_z;
    int kz = rem % kernel_z;
    (void)kernel_x;
    return out_channel * weight_s0 + kx * weight_s1 + ky * weight_s2 +
           kz * weight_s3 + in_channel * weight_s4;
}

inline int dense_weight_offset(
    int kernel_id,
    int in_channel,
    int out_channel,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    int in_channels,
    int out_channels
) {
    if (weight_layout == 0) {
        return (kernel_id * in_channels + in_channel) * out_channels +
               out_channel;
    }

    int xy = kernel_y * kernel_z;
    int kx = kernel_id / xy;
    int rem = kernel_id % xy;
    int ky = rem / kernel_z;
    int kz = rem % kernel_z;
    return (((out_channel * kernel_x + kx) * kernel_y + ky) * kernel_z + kz) *
               in_channels +
           in_channel;
}
