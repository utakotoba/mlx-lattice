#include "ops/conv3d/validation.h"

#include <stdexcept>

namespace mlx_lattice {

namespace {

void validate_maps_and_kernels(
    const mx::array& maps,
    const mx::array& kernels
) {
    if (maps.ndim() != 2 || maps.shape(1) != 2) {
        throw std::invalid_argument("maps must have shape (M, 2).");
    }
    if (kernels.ndim() != 1 || kernels.shape(0) != maps.shape(0)) {
        throw std::invalid_argument("kernels must have shape (M,).");
    }
    if (maps.dtype() != mx::int32 || kernels.dtype() != mx::int32) {
        throw std::invalid_argument("maps and kernels must be int32.");
    }
}

} // namespace

void validate_conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows
) {
    if (out_rows < 0) {
        throw std::invalid_argument("out_rows must be non-negative.");
    }
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, Cin).");
    }
    if (weight.ndim() != 3) {
        throw std::invalid_argument("weight must have shape (K, Cin, Cout).");
    }
    validate_maps_and_kernels(maps, kernels);
    if (feats.dtype() != mx::float32 || weight.dtype() != mx::float32) {
        throw std::invalid_argument("conv3d_feats supports float32 tensors.");
    }
    if (weight.shape(1) != feats.shape(1)) {
        throw std::invalid_argument("weight input channels must match feats.");
    }
}

void validate_conv3d_residual_feats(
    const mx::array& base,
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets
) {
    validate_conv3d_feats(feats, weight, maps, kernels, base.shape(0));
    if (base.ndim() != 2 || base.dtype() != mx::float32) {
        throw std::invalid_argument("base must be a float32 matrix.");
    }
    if (base.shape(1) != weight.shape(2)) {
        throw std::invalid_argument("base output channels must match weight.");
    }
    if (offsets.ndim() != 1 || offsets.shape(0) != base.shape(0) + 1) {
        throw std::invalid_argument("offsets must have shape (rows + 1,).");
    }
    if (offsets.dtype() != mx::int32) {
        throw std::invalid_argument("offsets must be int32.");
    }
}

void validate_pool3d_feats(
    const mx::array& feats,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    int out_rows
) {
    if (out_rows < 0) {
        throw std::invalid_argument("out_rows must be non-negative.");
    }
    if (feats.ndim() != 2 || feats.dtype() != mx::float32) {
        throw std::invalid_argument("feats must be a float32 matrix.");
    }
    validate_maps_and_kernels(maps, kernels);
    if (offsets.ndim() != 1 || offsets.shape(0) != out_rows + 1) {
        throw std::invalid_argument("offsets must have shape (rows + 1,).");
    }
    if (offsets.dtype() != mx::int32) {
        throw std::invalid_argument("offsets must be int32.");
    }
}

void validate_pool3d_feats_grad(
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int rows
) {
    if (rows < 0) {
        throw std::invalid_argument("rows must be non-negative.");
    }
    if (grad.ndim() != 2 || grad.dtype() != mx::float32) {
        throw std::invalid_argument("grad must be a float32 matrix.");
    }
    validate_maps_and_kernels(maps, kernels);
}

void validate_conv3d_feats_grad(
    const mx::array& grad,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int rows
) {
    if (rows < 0) {
        throw std::invalid_argument("rows must be non-negative.");
    }
    if (grad.ndim() != 2 || weight.ndim() != 3) {
        throw std::invalid_argument("grad must be 2D and weight must be 3D.");
    }
    if (grad.dtype() != mx::float32 || weight.dtype() != mx::float32) {
        throw std::invalid_argument(
            "conv3d gradients support float32 tensors."
        );
    }
    if (grad.shape(1) != weight.shape(2)) {
        throw std::invalid_argument("grad output channels must match weight.");
    }
    validate_maps_and_kernels(maps, kernels);
}

void validate_conv3d_weight_grad(
    const mx::array& feats,
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int kernel_count
) {
    if (kernel_count < 0) {
        throw std::invalid_argument("kernel_count must be non-negative.");
    }
    if (feats.ndim() != 2 || grad.ndim() != 2) {
        throw std::invalid_argument("feats and grad must be matrices.");
    }
    if (feats.dtype() != mx::float32 || grad.dtype() != mx::float32) {
        throw std::invalid_argument(
            "conv3d gradients support float32 tensors."
        );
    }
    validate_maps_and_kernels(maps, kernels);
}

} // namespace mlx_lattice
