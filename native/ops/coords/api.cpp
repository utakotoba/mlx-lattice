#include "ops/coords.h"

#include "ops/coords/dispatch.h"
#include "ops/coords/validation.h"

namespace mlx_lattice {

// MARK: - set ops

mx::array downsample_coords(const mx::array& coords, Triple stride) {
    validate_coords(coords);
    validate_positive(stride, "stride");
    return dispatch_downsample_coords(coords, stride);
}

mx::array union_coords(const mx::array& lhs, const mx::array& rhs) {
    validate_coord_pair(lhs, rhs);
    return dispatch_union_coords(lhs, rhs);
}

mx::array intersection_coords(const mx::array& lhs, const mx::array& rhs) {
    validate_coord_pair(lhs, rhs);
    return dispatch_intersection_coords(lhs, rhs);
}

mx::array lookup_coords(const mx::array& coords, const mx::array& queries) {
    validate_coord_pair(coords, queries);
    return dispatch_lookup_coords(coords, queries);
}

// MARK: - relations

NativeKernelRelation build_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    validate_coords(coords);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    validate_nonnegative(padding, "padding");
    validate_positive(dilation, "dilation");
    return dispatch_build_kernel_relation(
        coords, active_rows, kernel_size, stride, padding, dilation
    );
}

NativeKernelRelation build_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride
) {
    validate_coords(coords);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    return dispatch_build_generative_relation(
        coords, active_rows, kernel_size, stride
    );
}

NativeKernelRelation build_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    validate_coords(coords);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    validate_nonnegative(padding, "padding");
    validate_positive(dilation, "dilation");
    return dispatch_build_transposed_kernel_relation(
        coords, active_rows, kernel_size, stride, padding, dilation
    );
}

} // namespace mlx_lattice
