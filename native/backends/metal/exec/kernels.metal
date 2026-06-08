#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/coords/common.metal"

inline void read_coord(device const int* coords, int row, thread int* out) {
    int base = row * 4;
    out[0] = coords[base];
    out[1] = coords[base + 1];
    out[2] = coords[base + 2];
    out[3] = coords[base + 3];
}

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

[[kernel]] void spmm_edges_f32_serial(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* edge_count [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    constant const int& n_in_rows [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * out_channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = 0.0f;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows || kernel_id < 0 || kernel_id >= n_kernels) {
            continue;
        }

        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * in_channels + ci];
            int weight_base = (kernel_id * in_channels + ci) * out_channels;
            int out_base = out_row * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                out[out_base + co] += value * weights[weight_base + co];
            }
        }
    }
}

[[kernel]] void pool_sum_edges_f32_serial(
    device const float* feats [[buffer(0)]],
    device const int* in_rows [[buffer(1)]],
    device const int* out_rows [[buffer(2)]],
    device const int* edge_count [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& edge_capacity [[buffer(5)]],
    constant const int& channels [[buffer(6)]],
    constant const int& n_out_rows [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = 0.0f;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            out[out_base + channel] += feats[in_base + channel];
        }
    }
}

[[kernel]] void spmm_edges_input_grad_f32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* edge_count [[buffer(5)]],
    device float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_out_rows [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int grad_size = n_in_rows * in_channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows || kernel_id < 0 || kernel_id >= n_kernels) {
            continue;
        }

        int grad_base = in_row * in_channels;
        int cotangent_base = out_row * out_channels;
        int weight_base = kernel_id * in_channels * out_channels;
        for (int ci = 0; ci < in_channels; ++ci) {
            int weight_row = weight_base + ci * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                grad[grad_base + ci] +=
                    cotangent[cotangent_base + co] * weights[weight_row + co];
            }
        }
    }
}

[[kernel]] void spmm_edges_weight_grad_f32_serial(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* edge_count [[buffer(5)]],
    device float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_out_rows [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int grad_size = n_kernels * in_channels * out_channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows || kernel_id < 0 || kernel_id >= n_kernels) {
            continue;
        }

        int feat_base = in_row * in_channels;
        int cotangent_base = out_row * out_channels;
        int grad_base = kernel_id * in_channels * out_channels;
        for (int ci = 0; ci < in_channels; ++ci) {
            int grad_row = grad_base + ci * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                grad[grad_row + co] +=
                    feats[feat_base + ci] * cotangent[cotangent_base + co];
            }
        }
    }
}

[[kernel]] void pool_max_edges_f32_serial(
    device const float* feats [[buffer(0)]],
    device const int* in_rows [[buffer(1)]],
    device const int* out_rows [[buffer(2)]],
    device const int* edge_count [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& edge_capacity [[buffer(5)]],
    constant const int& channels [[buffer(6)]],
    constant const int& n_out_rows [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = -INFINITY;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            int index = out_base + channel;
            out[index] = max(out[index], feats[in_base + channel]);
        }
    }
}

[[kernel]] void pool_sum_edges_grad_f32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* edge_count [[buffer(5)]],
    device float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& channels [[buffer(8)]],
    constant const int& n_in_rows [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    (void)feats;
    (void)pooled;

    int grad_size = n_in_rows * channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int grad_base = in_row * channels;
        int cotangent_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            grad[grad_base + channel] += cotangent[cotangent_base + channel];
        }
    }
}

[[kernel]] void pool_max_edges_grad_f32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* edge_count [[buffer(5)]],
    device float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& channels [[buffer(8)]],
    constant const int& n_in_rows [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int grad_size = n_in_rows * channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            if (feats[in_base + channel] == pooled[out_base + channel]) {
                grad[in_base + channel] += cotangent[out_base + channel];
            }
        }
    }
}

[[kernel]] void pool_max_edges_jvp_f32_serial(
    device const float* tangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* edge_count [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& channels [[buffer(8)]],
    constant const int& n_in_rows [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = 0.0f;
    }

    int active_edges = min(edge_count[0], edge_capacity);
    for (int edge = 0; edge < active_edges; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            if (feats[in_base + channel] == pooled[out_base + channel]) {
                out[out_base + channel] += tangent[in_base + channel];
            }
        }
    }
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
                    float value = feats[in_row * in_channels + ci];
                    int weight_base =
                        (kernel_id * in_channels + ci) * out_channels;
                    int feat_base = out_row * out_channels;
                    for (int co = 0; co < out_channels; ++co) {
                        out_feats[feat_base + co] +=
                            value * weights[weight_base + co];
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
                    float value = feats[in_row * in_channels + ci];
                    int weight_base =
                        (kernel_id * in_channels + ci) * out_channels;
                    int feat_base = out_row * out_channels;
                    for (int co = 0; co < out_channels; ++co) {
                        out_feats[feat_base + co] +=
                            value * weights[weight_base + co];
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
                if (!valid_forward_edge_coord(
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
                int weight_base = (kernel_id * in_channels + ci) * out_channels;
                for (int co = 0; co < out_channels; ++co) {
                    grad[in_row * in_channels + ci] +=
                        cotangent[out_row * out_channels + co] *
                        weights[weight_base + co];
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
                if (!valid_forward_edge_coord(
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
                int grad_base = (kernel_id * in_channels + ci) * out_channels;
                for (int co = 0; co < out_channels; ++co) {
                    grad[grad_base + co] +=
                        feats[in_row * in_channels + ci] *
                        cotangent[out_row * out_channels + co];
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
            if (!valid_forward_edge_coord(
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
                float value = feats[in_row * channels + channel];
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
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_in_rows * channels; ++index) {
        grad[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (!valid_forward_edge_coord(
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
                int out_index = out_row * channels + channel;
                if (reduce == 1 && feats[in_index] != pooled[out_index]) {
                    continue;
                }
                float scale = reduce == 2 ? 1.0f / float(degree) : 1.0f;
                grad[in_index] += cotangent[out_index] * scale;
            }
        }
    }
}
