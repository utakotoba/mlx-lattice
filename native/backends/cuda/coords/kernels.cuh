#pragma once

namespace mlx_lattice::coords::cuda {

struct TripleArgs {
    int x;
    int y;
    int z;
};

struct QuantizeArgs {
    float voxel_x;
    float voxel_y;
    float voxel_z;
    float origin_x;
    float origin_y;
    float origin_z;
};

__global__ void set_coords_i32(
    const int* lhs,
    const int* rhs,
    int* out_coords,
    int* count,
    int op,
    TripleArgs stride,
    int lhs_rows,
    int rhs_rows
);

__global__ void lookup_coords_i32(
    const int* coords,
    const int* queries,
    int* out,
    int rows,
    int query_rows
);

__global__ void morton_codes_i32(const int* coords, long long* out, int rows);

__global__ void occupancy_downsample_i32(
    const int* coords,
    const int* active_rows,
    int* out_coords,
    int* out_active_rows,
    int* occupancy,
    int rows
);

__global__ void occupancy_expand_i32(
    const int* coords,
    const int* active_rows,
    const int* occupancy,
    int* out_coords,
    int* out_active_rows,
    int* parent_rows,
    int* child_indices,
    int rows
);

__global__ void child_coords_from_indices_i32(
    const int* parent_coords,
    const int* child_indices,
    int* out,
    int rows
);

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
);

__global__ void clear_f32(float* out, int elements);
__global__ void clear_i32_kernel(int* out, int elements, int value);

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
);

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
);

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
);

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
);

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
);

__global__ void prefix_relation_rows_i32(
    int* row_offsets,
    int* counts,
    int row_count,
    int edge_capacity
);

__global__ void prefix_relation_rows_active_i32(
    int* row_offsets,
    int* counts,
    const int* active_rows,
    int row_capacity,
    int edge_capacity
);

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
);

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
);

__global__ void clear_relation_grouped_view_i32(
    int* row_offsets,
    int* edge_ids,
    int edge_capacity,
    int group_count
);

__global__ void count_relation_grouped_view_i32(
    const int* group_ids,
    const int* counts,
    int* row_offsets,
    int edge_capacity,
    int group_count
);

__global__ void fill_relation_grouped_view_i32(
    const int* group_ids,
    const int* row_offsets,
    int* row_cursors,
    int* edge_ids,
    int edge_capacity,
    int group_count
);

__global__ void relation_grouped_view_i32(
    const int* group_ids,
    const int* counts,
    int* row_offsets,
    int* edge_ids,
    int edge_capacity,
    int group_count
);

__global__ void clear_relation_direct_view_i32(int* edge_ids, int group_count);

__global__ void fill_relation_direct_view_i32(
    const int* group_ids,
    const int* counts,
    int* edge_ids,
    int edge_capacity,
    int group_count
);

__global__ void relation_direct_view_i32(
    const int* group_ids,
    const int* counts,
    int* edge_ids,
    int edge_capacity,
    int group_count
);

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
);

} // namespace mlx_lattice::coords::cuda
