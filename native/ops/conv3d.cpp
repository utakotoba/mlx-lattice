#include "ops/conv3d.h"

#include <stdexcept>

#include "mlx/ops.h"
#include "ops/conv3d/primitives.h"
#include "ops/conv3d/validation.h"

namespace mlx_lattice {

// MARK: - api

mx::array conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows,
    mx::StreamOrDevice stream
) {
    validate_conv3d_feats(feats, weight, maps, kernels, out_rows);

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{out_rows, weight.shape(2)},
        mx::float32,
        make_conv3d_feats_primitive(
            s, out_rows, feats.shape(1), weight.shape(2)
        ),
        {feats_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous}
    );
}

mx::array conv3d_subm_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int center_kernel,
    mx::StreamOrDevice stream
) {
    validate_conv3d_feats(feats, weight, maps, kernels, feats.shape(0));
    if (center_kernel < 0 || center_kernel >= weight.shape(0)) {
        throw std::invalid_argument("center_kernel is out of bounds.");
    }

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{feats.shape(0), weight.shape(2)},
        mx::float32,
        make_conv3d_subm_feats_primitive(
            s, feats.shape(0), feats.shape(1), weight.shape(2), center_kernel
        ),
        {feats_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous}
    );
}

mx::array conv3d_residual_feats(
    const mx::array& base,
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    mx::StreamOrDevice stream
) {
    validate_conv3d_residual_feats(base, feats, weight, maps, kernels, offsets);

    auto s = to_stream(stream);
    auto base_contiguous = mx::contiguous(base, false, s);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);
    auto offsets_contiguous = mx::contiguous(offsets, false, s);

    return mx::array(
        base.shape(),
        mx::float32,
        make_conv3d_residual_feats_primitive(
            s, base.shape(0), feats.shape(1), weight.shape(2)
        ),
        {base_contiguous,
         feats_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous,
         offsets_contiguous}
    );
}

mx::array pool3d_feats(
    const mx::array& feats,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    int out_rows,
    mx::StreamOrDevice stream
) {
    validate_pool3d_feats(feats, maps, kernels, offsets, out_rows);

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);
    auto offsets_contiguous = mx::contiguous(offsets, false, s);

    return mx::array(
        mx::Shape{out_rows, feats.shape(1)},
        mx::float32,
        make_pool3d_feats_primitive(s, out_rows, feats.shape(1)),
        {feats_contiguous,
         maps_contiguous,
         kernels_contiguous,
         offsets_contiguous}
    );
}

mx::array max_pool3d_feats(
    const mx::array& feats,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    int out_rows,
    mx::StreamOrDevice stream
) {
    validate_max_pool3d_feats(feats, maps, kernels, offsets, out_rows);

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);
    auto offsets_contiguous = mx::contiguous(offsets, false, s);

    return mx::array(
        mx::Shape{out_rows, feats.shape(1)},
        mx::float32,
        make_max_pool3d_feats_primitive(s, out_rows, feats.shape(1)),
        {feats_contiguous,
         maps_contiguous,
         kernels_contiguous,
         offsets_contiguous}
    );
}

mx::array pool3d_feats_grad(
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int rows,
    mx::StreamOrDevice stream
) {
    validate_pool3d_feats_grad(grad, maps, kernels, rows);

    auto s = to_stream(stream);
    auto grad_contiguous = mx::contiguous(grad, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{rows, grad.shape(1)},
        mx::float32,
        make_pool3d_feats_grad_primitive(s, rows, grad.shape(1)),
        {grad_contiguous, maps_contiguous, kernels_contiguous}
    );
}

mx::array conv3d_feats_grad(
    const mx::array& grad,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int rows,
    mx::StreamOrDevice stream
) {
    validate_conv3d_feats_grad(grad, weight, maps, kernels, rows);

    auto s = to_stream(stream);
    auto grad_contiguous = mx::contiguous(grad, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{rows, weight.shape(1)},
        mx::float32,
        make_conv3d_feats_grad_primitive(
            s, rows, weight.shape(1), weight.shape(2)
        ),
        {grad_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous}
    );
}

mx::array conv3d_weight_grad(
    const mx::array& feats,
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int kernel_count,
    mx::StreamOrDevice stream
) {
    validate_conv3d_weight_grad(feats, grad, maps, kernels, kernel_count);

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto grad_contiguous = mx::contiguous(grad, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{kernel_count, feats.shape(1), grad.shape(1)},
        mx::float32,
        make_conv3d_weight_grad_primitive(
            s, kernel_count, feats.shape(1), grad.shape(1)
        ),
        {feats_contiguous, grad_contiguous, maps_contiguous, kernels_contiguous}
    );
}

} // namespace mlx_lattice
