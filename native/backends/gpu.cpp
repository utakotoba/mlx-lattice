#include "backends/gpu.h"

#include <stdexcept>
#include <utility>

#include "backends/cuda/conv/runtime.h"
#include "backends/cuda/coords/runtime.h"
#include "backends/cuda/pool/runtime.h"
#include "backends/metal/conv/runtime.h"
#include "backends/metal/coords/runtime.h"
#include "backends/metal/pool/runtime.h"

#if MLX_LATTICE_HAS_CUDA
#include "mlx/backend/cuda/cuda.h"
#endif

namespace mlx_lattice::backend::gpu {
namespace {

[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "No compiled native GPU backend is available for the selected MLX GPU "
        "device."
    );
}

} // namespace

NativeGpuBackend current_backend(const mx::Stream& stream) {
    (void)stream;
#if MLX_LATTICE_HAS_METAL
    return NativeGpuBackend::Metal;
#elif MLX_LATTICE_HAS_CUDA
    return NativeGpuBackend::Cuda;
#else
    unavailable();
#endif
}

template <typename MetalFn, typename CudaFn>
void dispatch(const mx::Stream& stream, MetalFn&& metal_fn, CudaFn&& cuda_fn) {
    switch (current_backend(stream)) {
    case NativeGpuBackend::Metal:
        return metal_fn();
    case NativeGpuBackend::Cuda:
        return cuda_fn();
    }
}

namespace conv {

void eval(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() { backend::metal::conv::eval(shape, stream, inputs, outputs); },
        [&]() { backend::cuda::conv::eval(shape, stream, inputs, outputs); }
    );
}

void eval_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::conv::eval_input_grad(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::conv::eval_input_grad(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::conv::eval_weight_grad(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::conv::eval_weight_grad(
                shape, stream, inputs, outputs
            );
        }
    );
}

} // namespace conv

namespace pool {

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::pool::eval(reduce, shape, stream, inputs, outputs);
        },
        [&]() {
            backend::cuda::pool::eval(reduce, shape, stream, inputs, outputs);
        }
    );
}

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::pool::eval_grad(
                reduce, shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::pool::eval_grad(
                reduce, shape, stream, inputs, outputs
            );
        }
    );
}

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::pool::eval_jvp(
                reduce, shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::pool::eval_jvp(
                reduce, shape, stream, inputs, outputs
            );
        }
    );
}

} // namespace pool

namespace coords {

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    CoordSetShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_set_coords(
                op, stride, shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_set_coords(
                op, stride, shape, stream, inputs, outputs
            );
        }
    );
}

void eval_lookup_coords(
    CoordLookupShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_lookup_coords(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_lookup_coords(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_morton_codes(
    CoordRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_morton_codes(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_morton_codes(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_occupancy_downsample(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_occupancy_downsample(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_occupancy_downsample(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_occupancy_expand(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_occupancy_expand(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_occupancy_expand(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_child_coords_from_indices(
    CoordRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_child_coords_from_indices(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_child_coords_from_indices(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_sparse_quantize(
    QuantizationSpec spec,
    int rows,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_sparse_quantize(
                spec, rows, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_sparse_quantize(
                spec, rows, stream, inputs, outputs
            );
        }
    );
}

void eval_voxelize_features(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_voxelize_features(
                reduce, shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_voxelize_features(
                reduce, shape, stream, inputs, outputs
            );
        }
    );
}

void eval_voxelize_feature_grad(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_voxelize_feature_grad(
                reduce, shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_voxelize_feature_grad(
                reduce, shape, stream, inputs, outputs
            );
        }
    );
}

void eval_generic_kernel_relation(
    CoordRelationOp op,
    int rows,
    int kernel_count,
    Triple stride,
    Triple padding,
    bool direct,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_generic_kernel_relation(
                op,
                rows,
                kernel_count,
                stride,
                padding,
                direct,
                stream,
                inputs,
                outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_generic_kernel_relation(
                op,
                rows,
                kernel_count,
                stride,
                padding,
                direct,
                stream,
                inputs,
                outputs
            );
        }
    );
}

void eval_target_kernel_relation(
    int rows,
    int target_rows,
    int kernel_count,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_target_kernel_relation(
                rows,
                target_rows,
                kernel_count,
                stride,
                padding,
                stream,
                inputs,
                outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_target_kernel_relation(
                rows,
                target_rows,
                kernel_count,
                stride,
                padding,
                stream,
                inputs,
                outputs
            );
        }
    );
}

void eval_generative_kernel_relation(
    int rows,
    int kernel_count,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_generative_kernel_relation(
                rows, kernel_count, stride, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_generative_kernel_relation(
                rows, kernel_count, stride, stream, inputs, outputs
            );
        }
    );
}

void eval_relation_grouped_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_relation_grouped_view(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_relation_grouped_view(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_relation_direct_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_relation_direct_view(
                shape, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_relation_direct_view(
                shape, stream, inputs, outputs
            );
        }
    );
}

void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    dispatch(
        stream,
        [&]() {
            backend::metal::coords::eval_neighbor_relation(
                op, shape, radius_squared, stream, inputs, outputs
            );
        },
        [&]() {
            backend::cuda::coords::eval_neighbor_relation(
                op, shape, radius_squared, stream, inputs, outputs
            );
        }
    );
}

} // namespace coords

} // namespace mlx_lattice::backend::gpu
