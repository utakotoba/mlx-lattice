#include "features/convolution/metal/runtime.h"

#include <algorithm>
#include <stdexcept>

#include "features/convolution/metal/tensor_ops/input_grad/runtime.h"
#include "features/convolution/metal/tensor_ops/weight_grad/runtime.h"
#include "foundation/array_utils.h"
#include "platform/metal/runtime_utils.h"

namespace mlx_lattice::backend::metal::conv {
namespace {

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

#ifdef _METAL_
template <typename Encoder>
void bind_common_shape(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    SparseConvShape shape,
    int first_index
) {
    auto edge_capacity = static_cast<int>(inputs[2].shape(0));
    set_bytes_range(
        encoder,
        first_index,
        edge_capacity,
        shape.out_capacity,
        shape.in_channels,
        shape.out_channels
    );
}

template <typename Encoder>
void bind_weight_shape(
    Encoder& encoder,
    const mx::array& weights,
    SparseConvShape shape,
    int first_index
) {
    set_bytes_range(
        encoder,
        first_index,
        stride_at(weights, 0),
        stride_at(weights, 1),
        stride_at(weights, 2),
        weights.ndim() == 5 ? stride_at(weights, 3) : 0,
        weights.ndim() == 5 ? stride_at(weights, 4) : 0,
        shape.weight_layout,
        shape.kernel_x,
        shape.kernel_y,
        shape.kernel_z
    );
}

template <typename Encoder, typename Library>
void clear_output(
    Encoder& encoder,
    mx::metal::Device& device,
    Library library,
    mx::array& out
) {
    auto clear = device.get_kernel("sparse_relation_conv_clear_f32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(out, 0);
    auto total = static_cast<int>(out.size());
    set_bytes_range(encoder, 1, total);
    dispatch_1d(encoder, clear, static_cast<size_t>(total));
}

bool is_float16(const mx::array& array) { return array.dtype() == mx::float16; }

const char*
typed_kernel_name(const char* fp32_kernel, const char* fp16_kernel, bool fp16) {
    return fp16 ? fp16_kernel : fp32_kernel;
}

bool is_dense_5d_c_weight(const mx::array& weights, SparseConvShape shape) {
    auto supported_channels = [](int channels) {
        return channels == 16 || channels == 32 || channels == 64;
    };
    return shape.weight_layout == 1 && supported_channels(shape.in_channels) &&
           supported_channels(shape.out_channels) && shape.n_kernels >= 16 &&
           weights.ndim() == 5 && stride_at(weights, 4) == 1 &&
           stride_at(weights, 3) == shape.in_channels &&
           stride_at(weights, 2) == shape.kernel_z * shape.in_channels &&
           stride_at(weights, 1) ==
               shape.kernel_y * shape.kernel_z * shape.in_channels &&
           stride_at(weights, 0) == shape.kernel_x * shape.kernel_y *
                                        shape.kernel_z * shape.in_channels;
}

bool is_dense_5d_square_c_shape(SparseConvShape shape) {
    return shape.weight_layout == 1 &&
           shape.in_channels == shape.out_channels &&
           (shape.in_channels == 16 || shape.in_channels == 32 ||
            shape.in_channels == 64) &&
           shape.n_kernels >= 16;
}

bool supports_dense_input_grad_shape(SparseConvShape shape) {
    if (shape.in_channels == shape.out_channels) {
        return shape.in_channels == 16 || shape.in_channels == 32 ||
               shape.in_channels == 64;
    }
    return (shape.in_channels == 16 && shape.out_channels == 32) ||
           (shape.in_channels == 16 && shape.out_channels == 64) ||
           (shape.in_channels == 64 && shape.out_channels == 16);
}

struct TypedKernelName {
    const char* fp32;
    const char* fp16;
};

struct ChannelKernelName {
    int in_channels;
    int out_channels;
    TypedKernelName kernel;
};

const char* find_channel_kernel_name(
    const ChannelKernelName* names,
    size_t count,
    SparseConvShape shape,
    bool fp16
) {
    for (size_t index = 0; index < count; ++index) {
        const auto& item = names[index];
        if (shape.in_channels == item.in_channels &&
            shape.out_channels == item.out_channels) {
            return typed_kernel_name(item.kernel.fp32, item.kernel.fp16, fp16);
        }
    }
    throw std::runtime_error("Unsupported dense convolution channel shape.");
}

const char* dense_forward_kernel_name(SparseConvShape shape, bool fp16) {
    static constexpr ChannelKernelName kDenseForwardKernels[] = {
        {16,
         16,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin16_cout16",
          "sparse_relation_conv_f16_i32_cout16_dense_cin16_cout16"}},
        {16,
         32,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin16_cout32",
          "sparse_relation_conv_f16_i32_cout16_dense_cin16_cout32"}},
        {16,
         64,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin16_cout64",
          "sparse_relation_conv_f16_i32_cout16_dense_cin16_cout64"}},
        {32,
         16,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin32_cout16",
          "sparse_relation_conv_f16_i32_cout16_dense_cin32_cout16"}},
        {32,
         32,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin32_cout32",
          "sparse_relation_conv_f16_i32_cout16_dense_cin32_cout32"}},
        {32,
         64,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin32_cout64",
          "sparse_relation_conv_f16_i32_cout16_dense_cin32_cout64"}},
        {64,
         16,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin64_cout16",
          "sparse_relation_conv_f16_i32_cout16_dense_cin64_cout16"}},
        {64,
         32,
         {"sparse_relation_conv_f32_i32_cout16_dense_ci4_cin64_cout32",
          "sparse_relation_conv_f16_i32_cout16_dense_cin64_cout32"}},
        {64,
         64,
         {"sparse_relation_conv_f32_i32_cout16_dense_cin64_cout64",
          "sparse_relation_conv_f16_i32_cout16_dense_cin64_cout64"}},
    };
    return find_channel_kernel_name(
        kDenseForwardKernels, std::size(kDenseForwardKernels), shape, fp16
    );
}

const char* dense_input_grad_kernel_name(SparseConvShape shape, bool fp16) {
    static constexpr ChannelKernelName kDenseInputGradKernels[] = {
        {16,
         16,
         {"sparse_relation_conv_input_grad_f32_i32_cin16_dense_c16",
          "sparse_relation_conv_input_grad_f16_i32_cin16_dense_c16"}},
        {16,
         32,
         {"sparse_relation_conv_input_grad_f32_i32_cin16_dense_cin16_"
          "cout32",
          "sparse_relation_conv_input_grad_f16_i32_cin16_dense_cin16_"
          "cout32"}},
        {16,
         64,
         {"sparse_relation_conv_input_grad_f32_i32_cin16_dense_cin16_"
          "cout64",
          "sparse_relation_conv_input_grad_f16_i32_cin16_dense_cin16_"
          "cout64"}},
        {32,
         32,
         {"sparse_relation_conv_input_grad_f32_i32_cin4_dense_c32",
          "sparse_relation_conv_input_grad_f16_i32_cin4_dense_c32"}},
        {64,
         16,
         {"sparse_relation_conv_input_grad_f32_i32_cin16_dense_cin64_"
          "cout16",
          "sparse_relation_conv_input_grad_f16_i32_cin16_dense_cin64_"
          "cout16"}},
        {64,
         64,
         {"sparse_relation_conv_input_grad_f32_i32_cin4_dense_c64",
          "sparse_relation_conv_input_grad_f16_i32_cin4_dense_c64"}},
    };
    return find_channel_kernel_name(
        kDenseInputGradKernels, std::size(kDenseInputGradKernels), shape, fp16
    );
}

const char*
dense_input_grad_grouped_kernel_name(SparseConvShape shape, bool fp16) {
    static constexpr ChannelKernelName kGroupedDenseInputGradKernels[] = {
        {32,
         32,
         {"sparse_relation_conv_input_grad_f32_i32_cin16_dense_c32",
          "sparse_relation_conv_input_grad_f16_i32_cin16_dense_c32"}},
        {64,
         64,
         {"sparse_relation_conv_input_grad_f16_i32_cin16_dense_c64",
          "sparse_relation_conv_input_grad_f16_i32_cin16_dense_c64"}},
    };
    return find_channel_kernel_name(
        kGroupedDenseInputGradKernels,
        std::size(kGroupedDenseInputGradKernels),
        shape,
        fp16
    );
}

const char* dense_weight_grad_kernel_name(SparseConvShape shape, bool fp16) {
    static constexpr ChannelKernelName kDenseWeightGradKernels[] = {
        {16,
         16,
         {"sparse_relation_conv_weight_grad_f32_i32_c4_dense_c16",
          "sparse_relation_conv_weight_grad_f16_i32_c4_dense_c16"}},
        {32,
         32,
         {"sparse_relation_conv_weight_grad_f32_i32_c4_dense_c32",
          "sparse_relation_conv_weight_grad_f16_i32_c4_dense_c32"}},
        {64,
         64,
         {"sparse_relation_conv_weight_grad_f32_i32_c4_dense_c64",
          "sparse_relation_conv_weight_grad_f16_i32_c4_dense_c64"}},
    };
    return find_channel_kernel_name(
        kDenseWeightGradKernels, std::size(kDenseWeightGradKernels), shape, fp16
    );
}

void encode_weight_grad_classic(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto fp16 = is_float16(inputs[0]);
    auto use_block4 = shape.in_channels % 4 == 0 &&
                      shape.out_channels % 4 == 0 && shape.n_kernels >= 16 &&
                      shape.in_capacity >= 50000;
    auto edge_count = static_cast<int>(inputs[2].shape(0));
    auto use_cout16 = shape.out_channels == 16 && shape.n_kernels != 1 &&
                      (shape.n_kernels >= 16 || edge_count >= 50000);
    auto use_dense_c = is_dense_5d_square_c_shape(shape) &&
                       (shape.in_channels > 16 || fp16) &&
                       (shape.in_capacity >= 4096 || edge_count >= 50000);
    auto use_gather = fp16 || shape.n_kernels >= 16;
    if (!use_gather && !use_block4 && !use_cout16 && !use_dense_c) {
        clear_output(encoder, device, library, out);
    }
    auto kernel = device.get_kernel(
        use_dense_c
            ? dense_weight_grad_kernel_name(shape, fp16)
            : (use_cout16
                   ? typed_kernel_name(
                         "sparse_relation_conv_weight_grad_cout16_f32_i32",
                         "sparse_relation_conv_weight_grad_cout16_f16_i32",
                         fp16
                     )
               : use_block4
                   ? typed_kernel_name(
                         "sparse_relation_conv_weight_grad_block4_f32_i32",
                         "sparse_relation_conv_weight_grad_block4_f16_i32",
                         fp16
                     )
                   : (fp16 ? "sparse_relation_conv_weight_grad_f16_i32"
                           : (use_gather
                                  ? "sparse_relation_conv_weight_grad_f32_i32"
                                  : "sparse_relation_conv_weight_grad_atomic_"
                                    "f32_i32"))),
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 9);
    set_bytes_range(
        encoder,
        10,
        edge_count,
        shape.out_capacity,
        shape.n_kernels,
        shape.in_channels,
        shape.out_channels,
        stride_at(inputs[0], 0),
        stride_at(inputs[0], 1),
        stride_at(inputs[1], 0),
        stride_at(inputs[1], 1),
        shape.weight_layout,
        shape.kernel_x,
        shape.kernel_y,
        shape.kernel_z
    );
    if (use_dense_c) {
        auto blocks = static_cast<size_t>(shape.in_channels / 4);
        auto total_tiles =
            static_cast<size_t>(shape.n_kernels) * blocks * blocks;
        encoder.dispatch_threadgroups(
            MTL::Size(total_tiles, 1, 1), MTL::Size(64, 1, 1)
        );
    } else if (use_block4) {
        auto total_tiles = static_cast<size_t>(shape.n_kernels) *
                           static_cast<size_t>(shape.in_channels / 4);
        encoder.dispatch_threadgroups(
            MTL::Size(total_tiles, 1, 1), MTL::Size(64, 1, 1)
        );
    } else if (use_cout16) {
        auto total_pairs = static_cast<size_t>(shape.n_kernels) *
                           static_cast<size_t>(shape.in_channels);
        encoder.dispatch_threadgroups(
            MTL::Size(total_pairs, 1, 1), MTL::Size(256, 1, 1)
        );
    } else {
        dispatch_1d(
            encoder,
            kernel,
            (use_gather ? static_cast<size_t>(shape.n_kernels)
                        : static_cast<size_t>(inputs[2].shape(0))) *
                static_cast<size_t>(shape.in_channels) *
                static_cast<size_t>(shape.out_channels)
        );
    }
}

#endif

} // namespace

void eval(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto fp16 = is_float16(inputs[0]);
    auto use_cout16 = shape.out_channels == 16 &&
                      ((shape.n_kernels >= 16 && shape.out_capacity >= 4096) ||
                       shape.out_capacity >= 50000);
    auto use_dense_c =
        shape.out_capacity >= 4096 && is_dense_5d_c_weight(inputs[1], shape);
    auto use_vec4 = !fp16 && shape.out_channels % 4 == 0;
    auto use_gather = fp16 || use_vec4 || shape.n_kernels == 1;
    if (!use_gather) {
        clear_output(encoder, device, library, out);
    }
    const char* kernel_name = "sparse_relation_conv_atomic_f32_i32";
    if (use_dense_c) {
        if (fp16 && shape.in_channels == 64 && shape.out_channels == 64 &&
            shape.out_capacity >= 50000 && stride_at(inputs[0], 1) == 1) {
            kernel_name =
                "sparse_relation_conv_f16_i32_cout16_dense_contiguous_cin64_"
                "cout64";
        } else {
            kernel_name = dense_forward_kernel_name(shape, fp16);
        }
    } else if (use_cout16) {
        kernel_name = typed_kernel_name(
            "sparse_relation_conv_f32_i32_cout16",
            "sparse_relation_conv_f16_i32_cout16",
            fp16
        );
    } else if (use_vec4) {
        kernel_name = "sparse_relation_conv_f32_i32_vec4";
    } else if (fp16) {
        kernel_name = "sparse_relation_conv_f16_i32";
    } else if (use_gather) {
        kernel_name = "sparse_relation_conv_f32_i32";
    }
    auto kernel = device.get_kernel(kernel_name, library);
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < 7; ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 7);
    bind_common_shape(encoder, inputs, shape, 8);
    set_bytes_range(
        encoder, 12, stride_at(inputs[0], 0), stride_at(inputs[0], 1)
    );
    bind_weight_shape(encoder, inputs[1], shape, 14);
    auto work_items =
        use_dense_c && shape.out_channels > 16
            ? static_cast<size_t>(shape.out_capacity) *
                  static_cast<size_t>(shape.out_channels / 16)
        : use_cout16 || (use_dense_c && shape.out_channels == 16)
            ? static_cast<size_t>(shape.out_capacity)
        : use_vec4 ? static_cast<size_t>(shape.out_capacity) *
                         static_cast<size_t>(shape.out_channels / 4)
                   : static_cast<size_t>(
                         use_gather ? shape.out_capacity
                                    : static_cast<int>(inputs[2].shape(0))
                     ) * static_cast<size_t>(shape.out_channels);
    dispatch_1d(encoder, kernel, work_items);
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sorted_implicit_gemm(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto supported_channels =
        shape.in_channels == shape.out_channels &&
        (shape.in_channels == 32 || shape.in_channels == 64);
    if (inputs[0].dtype() != mx::float16 || inputs[1].dtype() != mx::float16 ||
        inputs[2].dtype() != mx::int32 || inputs[3].dtype() != mx::int32 ||
        inputs[4].dtype() != mx::int32 || inputs[5].dtype() != mx::int32 ||
        shape.weight_layout != 0 || !supported_channels ||
        shape.n_kernels != 27 || inputs[1].ndim() != 3 ||
        stride_at(inputs[0], 1) != 1 || stride_at(inputs[1], 2) != 1 ||
        stride_at(inputs[1], 1) != shape.out_channels ||
        stride_at(inputs[1], 0) != shape.in_channels * shape.out_channels) {
        throw std::invalid_argument(
            "sorted implicit GEMM conv supports only contiguous "
            "fp16 mapped weights with K=27 and Cin=Cout in {32, 64}."
        );
    }

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel(
        shape.in_channels == 32
            ? "row_stationary_tensor_coop_devicew_rowfill4x2_kvmap_f16acc_"
              "sorted_c32_m64"
            : "row_stationary_tensor_coop_devicew_full64_kvmap_f16acc_sorted_"
              "c64_m64",
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_input_array(inputs[5], 4);
    encoder.set_output_array(out, 5);
    set_bytes_range(encoder, 6, shape.out_capacity, shape.store_sorted);
    auto groups = static_cast<size_t>((shape.out_capacity + 63) / 64);
    encoder.dispatch_threadgroups(
        MTL::Size(groups, 1, 1), MTL::Size(128, 1, 1)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sorted_direct_reference(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto supported_channels =
        shape.in_channels == shape.out_channels &&
        (shape.in_channels == 32 || shape.in_channels == 64);
    if (inputs[0].dtype() != mx::float16 || inputs[1].dtype() != mx::float16 ||
        inputs[2].dtype() != mx::int32 || inputs[3].dtype() != mx::int32 ||
        inputs[4].dtype() != mx::int32 || shape.weight_layout != 0 ||
        !supported_channels || shape.n_kernels != 27 || inputs[1].ndim() != 3 ||
        stride_at(inputs[0], 1) != 1 || stride_at(inputs[1], 2) != 1 ||
        stride_at(inputs[1], 1) != shape.out_channels ||
        stride_at(inputs[1], 0) != shape.in_channels * shape.out_channels) {
        throw std::invalid_argument(
            "sorted direct conv reference supports only contiguous fp16 "
            "mapped weights with K=27 and Cin=Cout in {32, 64}."
        );
    }

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel(
        shape.in_channels == 32 ? "row_stationary_direct_packedw_c32_m64"
                                : "row_stationary_direct_packedw_c64_m64",
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(inputs[3], 3);
    encoder.set_input_array(inputs[4], 4);
    encoder.set_output_array(out, 5);
    set_bytes_range(encoder, 6, shape.out_capacity, shape.store_sorted);
    auto groups = static_cast<size_t>((shape.out_capacity + 63) / 64);
    encoder.dispatch_threadgroups(
        MTL::Size(groups, 1, 1), MTL::Size(128, 1, 1)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto fp16 = is_float16(inputs[0]);
    auto use_dense_c = shape.in_capacity >= 4096 &&
                       is_dense_5d_c_weight(inputs[1], shape) &&
                       supports_dense_input_grad_shape(shape);
    auto use_grouped_dense_c =
        use_dense_c && shape.in_capacity >= 100000 &&
        (shape.in_channels == 32 ||
         (fp16 && shape.in_channels == 64 && shape.out_channels == 64));
    if (!use_dense_c && !fp16 &&
        tensor_ops::conv::input_grad::is_preferred(shape, stream)) {
        tensor_ops::conv::input_grad::encode(shape, stream, inputs, out);
        return;
    }
    auto use_cin16 = shape.in_channels == 16 && shape.in_capacity >= 4096;
    auto use_vec4 = !fp16 && shape.in_channels % 4 == 0;
    auto kernel = device.get_kernel(
        use_grouped_dense_c ? dense_input_grad_grouped_kernel_name(shape, fp16)
        : use_dense_c
            ? dense_input_grad_kernel_name(shape, fp16)
            : (use_cin16
                   ? typed_kernel_name(
                         "sparse_relation_conv_input_grad_f32_i32_cin16",
                         "sparse_relation_conv_input_grad_f16_i32_cin16",
                         fp16
                     )
                   : (use_vec4
                          ? "sparse_relation_conv_input_grad_f32_i32_vec4"
                          : (fp16
                                 ? "sparse_relation_conv_input_grad_f16_i32"
                                 : "sparse_relation_conv_input_grad_f32_i32"))),
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 9);
    set_bytes_range(
        encoder,
        10,
        static_cast<int>(inputs[2].shape(0)),
        shape.out_capacity,
        shape.in_capacity,
        shape.in_channels,
        shape.out_channels,
        stride_at(inputs[0], 0),
        stride_at(inputs[0], 1)
    );
    bind_weight_shape(encoder, inputs[1], shape, 17);
    auto work_items =
        use_grouped_dense_c ? static_cast<size_t>(shape.in_capacity) *
                                  static_cast<size_t>(shape.in_channels / 16)
        : use_dense_c && shape.in_channels > 16
            ? static_cast<size_t>(shape.in_capacity) *
                  static_cast<size_t>(shape.in_channels / 4)
        : use_cin16 || (use_dense_c && shape.in_channels == 16)
            ? static_cast<size_t>(shape.in_capacity)
        : use_vec4 ? static_cast<size_t>(shape.in_capacity) *
                         static_cast<size_t>(shape.in_channels / 4)
                   : static_cast<size_t>(shape.in_capacity) *
                         static_cast<size_t>(shape.in_channels);
    dispatch_1d(encoder, kernel, work_items);
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    if (tensor_ops::conv::weight_grad::is_preferred(shape, stream)) {
        tensor_ops::conv::weight_grad::encode(shape, stream, inputs, out);
        return;
    }
    encode_weight_grad_classic(shape, stream, inputs, out);
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::conv
