#include "backends/cuda/coords/kernels.cuh"

#include <cuda_runtime.h>

namespace mlx_lattice::coords::cuda {
namespace {

__device__ int elem_1d() { return int(blockIdx.x * blockDim.x + threadIdx.x); }

__device__ int floor_div_int(int value, int divisor) {
    int quotient = value / divisor;
    int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        quotient -= 1;
    }
    return quotient;
}

__device__ bool coord_equal_raw(const int* lhs, const int* rhs, int rhs_row) {
    int rhs_base = rhs_row * 4;
    return lhs[0] == rhs[rhs_base] && lhs[1] == rhs[rhs_base + 1] &&
           lhs[2] == rhs[rhs_base + 2] && lhs[3] == rhs[rhs_base + 3];
}

__device__ int find_coord(const int* coords, int rows, const int* target) {
    for (int row = 0; row < rows; ++row) {
        if (coord_equal_raw(target, coords, row)) {
            return row;
        }
    }
    return -1;
}

__device__ void write_coord(int* out, int row, int b, int x, int y, int z) {
    int base = row * 4;
    out[base] = b;
    out[base + 1] = x;
    out[base + 2] = y;
    out[base + 3] = z;
}

__device__ unsigned long long split_morton_3(unsigned long long value) {
    value &= 0x1fffffULL;
    value = (value | (value << 32)) & 0x1f00000000ffffULL;
    value = (value | (value << 16)) & 0x1f0000ff0000ffULL;
    value = (value | (value << 8)) & 0x100f00f00f00f00fULL;
    value = (value | (value << 4)) & 0x10c30c30c30c30c3ULL;
    value = (value | (value << 2)) & 0x1249249249249249ULL;
    return value;
}

__device__ int child_index(const int* coords, int row) {
    int base = row * 4;
    int index = (coords[base + 1] & 1) + ((coords[base + 2] & 1) << 1) +
                ((coords[base + 3] & 1) << 2);
    return 1 << index;
}

__device__ float
squared_distance(const int* lhs, int lhs_row, const int* rhs, int rhs_row) {
    int lhs_base = lhs_row * 4;
    int rhs_base = rhs_row * 4;
    float dx = float(lhs[lhs_base + 1] - rhs[rhs_base + 1]);
    float dy = float(lhs[lhs_base + 2] - rhs[rhs_base + 2]);
    float dz = float(lhs[lhs_base + 3] - rhs[rhs_base + 3]);
    return dx * dx + dy * dy + dz * dz;
}

__device__ bool
same_batch(const int* lhs, int lhs_row, const int* rhs, int rhs_row) {
    return lhs[lhs_row * 4] == rhs[rhs_row * 4];
}

__device__ void kernel_input_coord(
    const int* out_coord,
    const int* offset,
    TripleArgs stride,
    TripleArgs padding,
    int* candidate
) {
    candidate[0] = out_coord[0];
    candidate[1] = out_coord[1] * stride.x + offset[0] - padding.x;
    candidate[2] = out_coord[2] * stride.y + offset[1] - padding.y;
    candidate[3] = out_coord[3] * stride.z + offset[2] - padding.z;
}

__device__ void clear_i32(int* data, int count, int value) {
    for (int i = 0; i < count; ++i) {
        data[i] = value;
    }
}

} // namespace

__global__ void set_coords_i32(
    const int* lhs,
    const int* rhs,
    int* out_coords,
    int* count,
    int op,
    TripleArgs stride,
    int lhs_rows,
    int rhs_rows
) {
    if (elem_1d() != 0) {
        return;
    }
    int out_row = 0;
    if (op == 0) {
        for (int row = 0; row < lhs_rows; ++row) {
            int base = row * 4;
            int candidate[4] = {
                lhs[base],
                floor_div_int(lhs[base + 1], stride.x),
                floor_div_int(lhs[base + 2], stride.y),
                floor_div_int(lhs[base + 3], stride.z),
            };
            if (find_coord(out_coords, out_row, candidate) < 0) {
                write_coord(
                    out_coords,
                    out_row++,
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3]
                );
            }
        }
    } else if (op == 1) {
        for (int row = 0; row < lhs_rows; ++row) {
            int base = row * 4;
            int candidate[4] = {
                lhs[base], lhs[base + 1], lhs[base + 2], lhs[base + 3]
            };
            if (find_coord(out_coords, out_row, candidate) < 0) {
                write_coord(
                    out_coords,
                    out_row++,
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3]
                );
            }
        }
        for (int row = 0; row < rhs_rows; ++row) {
            int base = row * 4;
            int candidate[4] = {
                rhs[base], rhs[base + 1], rhs[base + 2], rhs[base + 3]
            };
            if (find_coord(out_coords, out_row, candidate) < 0) {
                write_coord(
                    out_coords,
                    out_row++,
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3]
                );
            }
        }
    } else {
        for (int row = 0; row < lhs_rows; ++row) {
            int base = row * 4;
            int candidate[4] = {
                lhs[base], lhs[base + 1], lhs[base + 2], lhs[base + 3]
            };
            if (find_coord(rhs, rhs_rows, candidate) >= 0 &&
                find_coord(out_coords, out_row, candidate) < 0) {
                write_coord(
                    out_coords,
                    out_row++,
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3]
                );
            }
        }
    }
    count[0] = out_row;
}

__global__ void lookup_coords_i32(
    const int* coords,
    const int* queries,
    int* out,
    int rows,
    int query_rows
) {
    int row = elem_1d();
    if (row >= query_rows) {
        return;
    }
    int base = row * 4;
    int target[4] = {
        queries[base], queries[base + 1], queries[base + 2], queries[base + 3]
    };
    out[row] = find_coord(coords, rows, target);
}

__global__ void morton_codes_i32(const int* coords, long long* out, int rows) {
    int row = elem_1d();
    if (row >= rows) {
        return;
    }
    int base = row * 4;
    unsigned long long code =
        split_morton_3(static_cast<unsigned long long>(coords[base + 1])) |
        (split_morton_3(static_cast<unsigned long long>(coords[base + 2]))
         << 1) |
        (split_morton_3(static_cast<unsigned long long>(coords[base + 3]))
         << 2);
    code += static_cast<unsigned long long>(coords[base]) << 60;
    out[row] = static_cast<long long>(code);
}

__global__ void occupancy_downsample_i32(
    const int* coords,
    const int* active_rows,
    int* out_coords,
    int* out_active_rows,
    int* occupancy,
    int rows
) {
    if (elem_1d() != 0) {
        return;
    }
    int logical = min(active_rows[0], rows);
    int out_row = 0;
    for (int row = 0; row < logical; ++row) {
        int base = row * 4;
        int parent[4] = {
            coords[base],
            floor_div_int(coords[base + 1], 2),
            floor_div_int(coords[base + 2], 2),
            floor_div_int(coords[base + 3], 2),
        };
        int slot = find_coord(out_coords, out_row, parent);
        if (slot < 0) {
            slot = out_row++;
            write_coord(
                out_coords, slot, parent[0], parent[1], parent[2], parent[3]
            );
            occupancy[slot] = 0;
        }
        occupancy[slot] |= child_index(coords, row);
    }
    out_active_rows[0] = out_row;
}

__global__ void occupancy_expand_i32(
    const int* coords,
    const int* active_rows,
    const int* occupancy,
    int* out_coords,
    int* out_active_rows,
    int* parent_rows,
    int* child_indices,
    int rows
) {
    if (elem_1d() != 0) {
        return;
    }
    int logical = min(active_rows[0], rows);
    int out_row = 0;
    for (int row = 0; row < logical; ++row) {
        int bits = occupancy[row];
        int base = row * 4;
        for (int child = 0; child < 8; ++child) {
            if ((bits & (1 << child)) == 0) {
                continue;
            }
            write_coord(
                out_coords,
                out_row,
                coords[base],
                coords[base + 1] * 2 + (child & 1),
                coords[base + 2] * 2 + ((child >> 1) & 1),
                coords[base + 3] * 2 + ((child >> 2) & 1)
            );
            parent_rows[out_row] = row;
            child_indices[out_row] = child;
            ++out_row;
        }
    }
    out_active_rows[0] = out_row;
}

__global__ void child_coords_from_indices_i32(
    const int* parent_coords,
    const int* child_indices,
    int* out,
    int rows
) {
    int row = elem_1d();
    if (row >= rows) {
        return;
    }
    int child = child_indices[row];
    int base = row * 4;
    write_coord(
        out,
        row,
        parent_coords[base],
        parent_coords[base + 1] * 2 + (child & 1),
        parent_coords[base + 2] * 2 + ((child >> 1) & 1),
        parent_coords[base + 3] * 2 + ((child >> 2) & 1)
    );
}

__global__ void sparse_quantize_i32(
    const float* points,
    const int* batch_indices,
    const int* active_rows,
    int* coords,
    int* out_active_rows,
    int* inverse_rows,
    int* counts,
    QuantizeArgs spec,
    int rows
) {
    if (elem_1d() != 0) {
        return;
    }
    int logical = min(active_rows[0], rows);
    clear_i32(inverse_rows, rows, -1);
    clear_i32(counts, rows, 0);
    int out_row = 0;
    for (int point = 0; point < logical; ++point) {
        int point_base = point * 3;
        int candidate[4] = {
            batch_indices[point],
            int(floorf((points[point_base] - spec.origin_x) / spec.voxel_x)),
            int(
                floorf((points[point_base + 1] - spec.origin_y) / spec.voxel_y)
            ),
            int(
                floorf((points[point_base + 2] - spec.origin_z) / spec.voxel_z)
            ),
        };
        int slot = find_coord(coords, out_row, candidate);
        if (slot < 0) {
            slot = out_row++;
            write_coord(
                coords,
                slot,
                candidate[0],
                candidate[1],
                candidate[2],
                candidate[3]
            );
        }
        inverse_rows[point] = slot;
        counts[slot] += 1;
    }
    out_active_rows[0] = out_row;
}

__global__ void clear_f32(float* out, int elements) {
    int elem = elem_1d();
    if (elem < elements) {
        out[elem] = 0.0f;
    }
}

__global__ void clear_i32_kernel(int* out, int elements, int value) {
    int elem = elem_1d();
    if (elem < elements) {
        out[elem] = value;
    }
}

__global__ void voxelize_features_f32(
    const float* feats,
    const int* inverse_rows,
    const int* voxel_counts,
    const int* active_rows,
    float* out,
    int reduce,
    int point_rows,
    int voxel_rows,
    int channels
) {
    int elem = elem_1d();
    int total = point_rows * channels;
    if (elem >= total) {
        return;
    }
    int point = elem / channels;
    if (point >= min(active_rows[0], point_rows)) {
        return;
    }
    int channel = elem - point * channels;
    int voxel = inverse_rows[point];
    if (voxel < 0 || voxel >= voxel_rows) {
        return;
    }
    float scale =
        reduce == 1 ? 1.0f / float(max(voxel_counts[voxel], 1)) : 1.0f;
    atomicAdd(&out[voxel * channels + channel], feats[elem] * scale);
}

__global__ void voxelize_feature_grad_f32(
    const float* cotangent,
    const int* inverse_rows,
    const int* voxel_counts,
    const int* active_rows,
    float* out,
    int reduce,
    int point_rows,
    int voxel_rows,
    int channels
) {
    int elem = elem_1d();
    int total = point_rows * channels;
    if (elem >= total) {
        return;
    }
    int point = elem / channels;
    int channel = elem - point * channels;
    if (point >= min(active_rows[0], point_rows)) {
        out[elem] = 0.0f;
        return;
    }
    int voxel = inverse_rows[point];
    if (voxel < 0 || voxel >= voxel_rows) {
        out[elem] = 0.0f;
        return;
    }
    float scale =
        reduce == 1 ? 1.0f / float(max(voxel_counts[voxel], 1)) : 1.0f;
    out[elem] = cotangent[voxel * channels + channel] * scale;
}

__global__ void generic_kernel_relation_i32(
    const int* coords,
    const int* offsets,
    const int* active_rows,
    int* in_rows,
    int* out_rows,
    int* kernel_ids,
    int* row_offsets,
    int* out_coords,
    int* counts,
    int op,
    int rows,
    int kernel_count,
    TripleArgs stride,
    TripleArgs padding,
    bool direct
) {
    if (elem_1d() != 0) {
        return;
    }
    counts[0] = 0;
    counts[1] = 0;
    int logical = min(active_rows[0], rows);
    if (op == 0) {
        int out_count = logical;
        if (!(stride.x == 1 && stride.y == 1 && stride.z == 1 &&
              padding.x == 0 && padding.y == 0 && padding.z == 0)) {
            out_count = 0;
            for (int row = 0; row < logical; ++row) {
                int base = row * 4;
                int candidate[4] = {
                    coords[base],
                    floor_div_int(coords[base + 1], stride.x),
                    floor_div_int(coords[base + 2], stride.y),
                    floor_div_int(coords[base + 3], stride.z),
                };
                if (find_coord(out_coords, out_count, candidate) < 0) {
                    write_coord(
                        out_coords,
                        out_count++,
                        candidate[0],
                        candidate[1],
                        candidate[2],
                        candidate[3]
                    );
                }
            }
        } else {
            for (int row = 0; row < logical; ++row) {
                int base = row * 4;
                write_coord(
                    out_coords,
                    row,
                    coords[base],
                    coords[base + 1],
                    coords[base + 2],
                    coords[base + 3]
                );
            }
        }
        counts[1] = out_count;
        int edge = 0;
        for (int out_row = 0; out_row < out_count; ++out_row) {
            row_offsets[out_row] = edge;
            int out_base = out_row * 4;
            for (int kernel = 0; kernel < kernel_count; ++kernel) {
                int candidate[4];
                kernel_input_coord(
                    &out_coords[out_base],
                    &offsets[kernel * 3],
                    stride,
                    padding,
                    candidate
                );
                int in_row = find_coord(coords, logical, candidate);
                if (in_row < 0) {
                    continue;
                }
                in_rows[edge] = in_row;
                out_rows[edge] = out_row;
                kernel_ids[edge] = kernel;
                ++edge;
            }
        }
        row_offsets[out_count] = edge;
        counts[0] = edge;
        return;
    }

    int edge = 0;
    int out_count = 0;
    for (int in_row = 0; in_row < logical; ++in_row) {
        int base = in_row * 4;
        for (int kernel = 0; kernel < kernel_count; ++kernel) {
            int out_row = direct ? in_row * kernel_count + kernel : out_count;
            int candidate[4] = {
                coords[base],
                coords[base + 1] * stride.x + offsets[kernel * 3] - padding.x,
                coords[base + 2] * stride.y + offsets[kernel * 3 + 1] -
                    padding.y,
                coords[base + 3] * stride.z + offsets[kernel * 3 + 2] -
                    padding.z,
            };
            if (!direct) {
                int existing = find_coord(out_coords, out_count, candidate);
                if (existing >= 0) {
                    out_row = existing;
                } else {
                    out_row = out_count++;
                    write_coord(
                        out_coords,
                        out_row,
                        candidate[0],
                        candidate[1],
                        candidate[2],
                        candidate[3]
                    );
                }
            } else {
                write_coord(
                    out_coords,
                    out_row,
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3]
                );
                out_count = max(out_count, out_row + 1);
            }
            in_rows[edge] = in_row;
            out_rows[edge] = out_row;
            kernel_ids[edge] = kernel;
            ++edge;
        }
    }
    for (int row = 0; row <= out_count; ++row) {
        row_offsets[row] = 0;
    }
    for (int e = 0; e < edge; ++e) {
        ++row_offsets[out_rows[e] + 1];
    }
    for (int row = 0; row < out_count; ++row) {
        row_offsets[row + 1] += row_offsets[row];
    }
    counts[0] = edge;
    counts[1] = out_count;
}

__global__ void target_kernel_relation_i32(
    const int* coords,
    const int* offsets,
    const int* active_rows,
    const int* target_coords,
    const int* target_active_rows,
    int* in_rows,
    int* out_rows,
    int* kernel_ids,
    int* row_offsets,
    int* out_coords,
    int* counts,
    int rows,
    int target_rows,
    int kernel_count,
    TripleArgs stride,
    TripleArgs padding
) {
    (void)out_coords;
    if (elem_1d() != 0) {
        return;
    }
    int source_count = min(active_rows[0], rows);
    int out_count = min(target_active_rows[0], target_rows);
    int edge = 0;
    for (int out_row = 0; out_row < out_count; ++out_row) {
        row_offsets[out_row] = edge;
        int out_base = out_row * 4;
        for (int kernel = 0; kernel < kernel_count; ++kernel) {
            int candidate[4];
            kernel_input_coord(
                &target_coords[out_base],
                &offsets[kernel * 3],
                stride,
                padding,
                candidate
            );
            int in_row = find_coord(coords, source_count, candidate);
            if (in_row < 0) {
                continue;
            }
            in_rows[edge] = in_row;
            out_rows[edge] = out_row;
            kernel_ids[edge] = kernel;
            ++edge;
        }
    }
    row_offsets[out_count] = edge;
    counts[0] = edge;
    counts[1] = out_count;
}

__global__ void count_target_kernel_relation_i32(
    const int* coords,
    const int* offsets,
    const int* active_rows,
    const int* target_coords,
    const int* target_active_rows,
    int* row_offsets,
    int rows,
    int target_rows,
    int kernel_count,
    TripleArgs stride,
    TripleArgs padding
) {
    int elem = elem_1d();
    int source_count = min(active_rows[0], rows);
    int out_count = min(target_active_rows[0], target_rows);
    int total = out_count * kernel_count;
    if (elem >= total) {
        return;
    }

    int out_row = elem / kernel_count;
    int kernel = elem - out_row * kernel_count;
    int candidate[4];
    kernel_input_coord(
        &target_coords[out_row * 4],
        &offsets[kernel * 3],
        stride,
        padding,
        candidate
    );
    if (find_coord(coords, source_count, candidate) >= 0) {
        atomicAdd(&row_offsets[out_row + 1], 1);
    }
}

__global__ void prefix_relation_rows_i32(
    int* row_offsets,
    int* counts,
    int row_count,
    int edge_capacity
) {
    if (elem_1d() != 0) {
        return;
    }
    int running = 0;
    for (int row = 0; row < row_count; ++row) {
        int degree = row_offsets[row + 1];
        row_offsets[row] = running;
        running += degree;
    }
    row_offsets[row_count] = running;
    counts[0] = min(running, edge_capacity);
    counts[1] = row_count;
}

__global__ void prefix_relation_rows_active_i32(
    int* row_offsets,
    int* counts,
    const int* active_rows,
    int row_capacity,
    int edge_capacity
) {
    int row_count = min(active_rows[0], row_capacity);
    if (elem_1d() != 0) {
        return;
    }
    int running = 0;
    for (int row = 0; row < row_count; ++row) {
        int degree = row_offsets[row + 1];
        row_offsets[row] = running;
        running += degree;
    }
    row_offsets[row_count] = running;
    counts[0] = min(running, edge_capacity);
    counts[1] = row_count;
}

__global__ void fill_target_kernel_relation_i32(
    const int* coords,
    const int* offsets,
    const int* active_rows,
    const int* target_coords,
    const int* target_active_rows,
    const int* row_offsets,
    int* row_cursors,
    int* in_rows,
    int* out_rows,
    int* kernel_ids,
    int rows,
    int target_rows,
    int kernel_count,
    TripleArgs stride,
    TripleArgs padding
) {
    int elem = elem_1d();
    int source_count = min(active_rows[0], rows);
    int out_count = min(target_active_rows[0], target_rows);
    int total = out_count * kernel_count;
    if (elem >= total) {
        return;
    }

    int out_row = elem / kernel_count;
    int kernel = elem - out_row * kernel_count;
    int candidate[4];
    kernel_input_coord(
        &target_coords[out_row * 4],
        &offsets[kernel * 3],
        stride,
        padding,
        candidate
    );
    int in_row = find_coord(coords, source_count, candidate);
    if (in_row < 0) {
        return;
    }

    int slot = row_offsets[out_row] + atomicAdd(&row_cursors[out_row], 1);
    in_rows[slot] = in_row;
    out_rows[slot] = out_row;
    kernel_ids[slot] = kernel;
}

__global__ void generative_kernel_relation_i32(
    const int* coords,
    const int* offsets,
    const int* active_rows,
    int* in_rows,
    int* out_rows,
    int* kernel_ids,
    int* row_offsets,
    int* out_coords,
    int* counts,
    int rows,
    int kernel_count,
    TripleArgs stride
) {
    int elem = elem_1d();
    int logical = min(active_rows[0], rows);
    int total = logical * kernel_count;
    if (elem == 0) {
        counts[0] = total;
        counts[1] = total;
        row_offsets[total] = total;
    }
    if (elem >= total) {
        return;
    }
    int in_row = elem / kernel_count;
    int kernel = elem - in_row * kernel_count;
    int in_base = in_row * 4;
    in_rows[elem] = in_row;
    out_rows[elem] = elem;
    kernel_ids[elem] = kernel;
    row_offsets[elem] = elem;
    write_coord(
        out_coords,
        elem,
        coords[in_base],
        coords[in_base + 1] * stride.x + offsets[kernel * 3],
        coords[in_base + 2] * stride.y + offsets[kernel * 3 + 1],
        coords[in_base + 3] * stride.z + offsets[kernel * 3 + 2]
    );
}

__global__ void clear_relation_grouped_view_i32(
    int* row_offsets,
    int* edge_ids,
    int edge_capacity,
    int group_count
) {
    int elem = elem_1d();
    int total = max(edge_capacity, group_count + 1);
    if (elem >= total) {
        return;
    }
    if (elem <= group_count) {
        row_offsets[elem] = 0;
    }
    if (elem < edge_capacity) {
        edge_ids[elem] = -1;
    }
}

__global__ void count_relation_grouped_view_i32(
    const int* group_ids,
    const int* counts,
    int* row_offsets,
    int edge_capacity,
    int group_count
) {
    int edge = elem_1d();
    int edge_count = min(counts[0], edge_capacity);
    if (edge >= edge_count) {
        return;
    }
    int group = group_ids[edge];
    if (group >= 0 && group < group_count) {
        atomicAdd(&row_offsets[group + 1], 1);
    }
}

__global__ void fill_relation_grouped_view_i32(
    const int* group_ids,
    const int* row_offsets,
    int* row_cursors,
    int* edge_ids,
    int edge_capacity,
    int group_count
) {
    int edge = elem_1d();
    if (edge >= edge_capacity) {
        return;
    }
    int group = group_ids[edge];
    if (group < 0 || group >= group_count) {
        return;
    }
    int slot = row_offsets[group] + atomicAdd(&row_cursors[group], 1);
    edge_ids[slot] = edge;
}

__global__ void relation_grouped_view_i32(
    const int* group_ids,
    const int* counts,
    int* row_offsets,
    int* edge_ids,
    int edge_capacity,
    int group_count
) {
    if (elem_1d() != 0) {
        return;
    }
    int edge_count = min(counts[0], edge_capacity);
    clear_i32(row_offsets, group_count + 1, 0);
    clear_i32(edge_ids, edge_capacity, -1);
    for (int edge = 0; edge < edge_count; ++edge) {
        int group = group_ids[edge];
        if (group >= 0 && group < group_count) {
            ++row_offsets[group + 1];
        }
    }
    for (int group = 0; group < group_count; ++group) {
        row_offsets[group + 1] += row_offsets[group];
    }
    for (int edge = 0; edge < edge_count; ++edge) {
        int group = group_ids[edge];
        if (group < 0 || group >= group_count) {
            continue;
        }
        int slot = row_offsets[group]++;
        edge_ids[slot] = edge;
    }
    for (int group = group_count; group > 0; --group) {
        row_offsets[group] = row_offsets[group - 1];
    }
    row_offsets[0] = 0;
}

__global__ void clear_relation_direct_view_i32(int* edge_ids, int group_count) {
    int elem = elem_1d();
    if (elem < group_count) {
        edge_ids[elem] = -1;
    }
}

__global__ void fill_relation_direct_view_i32(
    const int* group_ids,
    const int* counts,
    int* edge_ids,
    int edge_capacity,
    int group_count
) {
    int edge = elem_1d();
    int edge_count = min(counts[0], edge_capacity);
    if (edge >= edge_count) {
        return;
    }
    int group = group_ids[edge];
    if (group >= 0 && group < group_count) {
        edge_ids[group] = edge;
    }
}

__global__ void relation_direct_view_i32(
    const int* group_ids,
    const int* counts,
    int* edge_ids,
    int edge_capacity,
    int group_count
) {
    if (elem_1d() != 0) {
        return;
    }
    int edge_count = min(counts[0], edge_capacity);
    clear_i32(edge_ids, group_count, -1);
    for (int edge = 0; edge < edge_count; ++edge) {
        int group = group_ids[edge];
        if (group >= 0 && group < group_count) {
            edge_ids[group] = edge;
        }
    }
}

__global__ void neighbor_relation_i32(
    const int* source_coords,
    const int* query_coords,
    const int* source_active_rows,
    const int* query_active_rows,
    int* query_rows,
    int* source_rows,
    int* neighbor_ids,
    float* distances,
    int* row_offsets,
    int* counts,
    int op,
    int source_count,
    int query_count,
    int max_neighbors,
    float radius_squared
) {
    if (elem_1d() != 0) {
        return;
    }
    int active_sources = min(source_active_rows[0], source_count);
    int active_queries = min(query_active_rows[0], query_count);
    int edge = 0;
    for (int query = 0; query < active_queries; ++query) {
        row_offsets[query] = edge;
        for (int neighbor = 0; neighbor < max_neighbors; ++neighbor) {
            int best_row = -1;
            float best_distance = CUDART_INF_F;
            for (int source = 0; source < active_sources; ++source) {
                if (!same_batch(query_coords, query, source_coords, source)) {
                    continue;
                }
                float distance = squared_distance(
                    query_coords, query, source_coords, source
                );
                if (op == 1 && distance > radius_squared) {
                    continue;
                }
                bool already_used = false;
                for (int used = row_offsets[query]; used < edge; ++used) {
                    already_used = already_used || source_rows[used] == source;
                }
                if (already_used) {
                    continue;
                }
                if (distance < best_distance ||
                    (distance == best_distance && source < best_row)) {
                    best_distance = distance;
                    best_row = source;
                }
            }
            if (best_row < 0) {
                break;
            }
            query_rows[edge] = query;
            source_rows[edge] = best_row;
            neighbor_ids[edge] = neighbor;
            distances[edge] = best_distance;
            ++edge;
        }
    }
    row_offsets[active_queries] = edge;
    counts[0] = edge;
    counts[1] = active_queries;
}

} // namespace mlx_lattice::coords::cuda
