#include "backends/metal/exec/runtime.h"

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/device.h"
#include "mlx/stream.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::exec::metal {

namespace {

std::string binary_dir() {
    static std::string dir = [] {
        Dl_info info;
        if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
            throw std::runtime_error("Unable to resolve native module path.");
        }
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }();
    return dir;
}

void allocate(mx::array& output) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
}

} // namespace

bool supports(
    const mx::array& feats, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids
) {
#if MLX_LATTICE_HAS_METAL
    return feats.dtype() == mx::float32 && weights.dtype() == mx::float32 &&
           in_rows.dtype() == mx::int32 && out_rows.dtype() == mx::int32 &&
           kernel_ids.dtype() == mx::int32 &&
           mx::is_available(mx::Device::gpu) &&
           mx::default_device() == mx::Device(mx::Device::gpu);
#else
    (void)feats;
    (void)weights;
    (void)in_rows;
    (void)out_rows;
    (void)kernel_ids;
    return false;
#endif
}

bool supports_pool(
    const mx::array& feats, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& in_rows,
    const mx::array& out_rows
) {
#if MLX_LATTICE_HAS_METAL
    return feats.dtype() == mx::float32 && in_rows.dtype() == mx::int32 &&
           out_rows.dtype() == mx::int32 && mx::is_available(mx::Device::gpu) &&
           mx::default_device() == mx::Device(mx::Device::gpu);
#else
    (void)feats;
    (void)in_rows;
    (void)out_rows;
    return false;
#endif
}

void eval_spmm_edges(
    SpmmEdgesShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("spmm_edges_f32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 5);
    encoder.set_bytes(shape.edge_count, 6);
    encoder.set_bytes(shape.in_channels, 7);
    encoder.set_bytes(shape.out_channels, 8);
    encoder.set_bytes(shape.n_out_rows, 9);
    encoder.set_bytes(inputs[0].shape(0), 10);
    encoder.set_bytes(inputs[1].shape(0), 11);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_pool_edges(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel_name = op == PoolReduceOp::Sum ? "pool_sum_edges_f32_serial"
                                               : "pool_max_edges_f32_serial";
    auto kernel = device.get_kernel(kernel_name, library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 3);
    encoder.set_bytes(shape.edge_count, 4);
    encoder.set_bytes(shape.channels, 5);
    encoder.set_bytes(shape.n_out_rows, 6);
    encoder.set_bytes(inputs[0].shape(0), 7);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::exec::metal
