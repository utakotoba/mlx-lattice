#include <metal_stdlib>

using namespace metal;

[[kernel]] void fill_i32(
    device int* out [[buffer(0)]],
    constant const int& value [[buffer(1)]],
    constant const int& size [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(size)) {
        out[elem] = value;
    }
}

[[kernel]] void fill_linear_i32(
    device int* out [[buffer(0)]],
    constant const int& step [[buffer(1)]],
    constant const int& size [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(size)) {
        out[elem] = int(elem) * step;
    }
}

[[kernel]] void build_subm_kernel_map_i32(
    device const int* coords [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device int* maps [[buffer(2)]],
    device atomic_int* sizes [[buffer(3)]],
    device int* kernels [[buffer(4)]],
    device int* residual_maps [[buffer(5)]],
    device int* residual_kernels [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    constant const int& kernel_count [[buffer(8)]],
    constant const int& center_kernel [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(rows * kernel_count);
    if (elem >= total) {
        return;
    }

    int kernel_index = int(elem / uint(rows));
    int out_row = int(elem - uint(kernel_index * rows));
    int base = out_row * 4;
    int target_b = coords[base];
    int target_x = coords[base + 1] + offsets[kernel_index * 3];
    int target_y = coords[base + 2] + offsets[kernel_index * 3 + 1];
    int target_z = coords[base + 3] + offsets[kernel_index * 3 + 2];

    for (int in_row = 0; in_row < rows; ++in_row) {
        int input = in_row * 4;
        if (coords[input] == target_b && coords[input + 1] == target_x &&
            coords[input + 2] == target_y && coords[input + 3] == target_z) {
            maps[elem * 2] = in_row;
            maps[elem * 2 + 1] = out_row;
            kernels[elem] = kernel_index;
            atomic_fetch_add_explicit(
                &sizes[kernel_index], 1, memory_order_relaxed
            );
            if (kernel_index != center_kernel) {
                int slot = kernel_index < center_kernel ? kernel_index
                                                        : kernel_index - 1;
                int residual = out_row * (kernel_count - 1) + slot;
                residual_maps[residual * 2] = in_row;
                residual_maps[residual * 2 + 1] = out_row;
                residual_kernels[residual] = kernel_index;
            }
            return;
        }
    }
}
