#include "ops/coords/dispatch.h"

#include "backends/cpu/coords.h"
#include "backends/metal/coords.h"
#include "mlx/device.h"

namespace mlx_lattice {

namespace {

bool should_use_metal_coords(const mx::array& coords) {
#if MLX_LATTICE_HAS_METAL
    return coords.dtype() == mx::int32 && mx::is_available(mx::Device::gpu) &&
           mx::default_device() == mx::Device(mx::Device::gpu);
#else
    (void)coords;
    return false;
#endif
}

} // namespace

// MARK: - set ops

mx::array dispatch_downsample_coords(const mx::array& coords, Triple stride) {
    if (should_use_metal_coords(coords)) {
        return metal::downsample_coords(coords, stride);
    }
    return cpu::downsample_coords(coords, stride);
}

mx::array dispatch_union_coords(const mx::array& lhs, const mx::array& rhs) {
    if (should_use_metal_coords(lhs)) {
        return metal::union_coords(lhs, rhs);
    }
    return cpu::union_coords(lhs, rhs);
}

mx::array
dispatch_intersection_coords(const mx::array& lhs, const mx::array& rhs) {
    if (should_use_metal_coords(lhs)) {
        return metal::intersection_coords(lhs, rhs);
    }
    return cpu::intersection_coords(lhs, rhs);
}

mx::array
dispatch_lookup_coords(const mx::array& coords, const mx::array& queries) {
    if (should_use_metal_coords(coords)) {
        return metal::lookup_coords(coords, queries);
    }
    return cpu::lookup_coords(coords, queries);
}

// MARK: - maps

NativeKernelMap dispatch_build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    if (should_use_metal_coords(coords)) {
        return metal::build_kernel_map(
            coords, kernel_size, stride, padding, dilation
        );
    }
    return cpu::build_kernel_map(
        coords, kernel_size, stride, padding, dilation
    );
}

NativeKernelMap dispatch_build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
) {
    if (should_use_metal_coords(coords)) {
        return metal::build_generative_map(coords, kernel_size, stride);
    }
    return cpu::build_generative_map(coords, kernel_size, stride);
}

NativeKernelMap dispatch_build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    if (should_use_metal_coords(coords)) {
        return metal::build_transposed_kernel_map(
            coords, kernel_size, stride, padding, dilation
        );
    }
    return cpu::build_transposed_kernel_map(
        coords, kernel_size, stride, padding, dilation
    );
}

} // namespace mlx_lattice
