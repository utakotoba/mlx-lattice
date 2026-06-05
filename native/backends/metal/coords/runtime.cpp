#include "backends/metal/coords/runtime.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::coords::metal {

namespace {

// MARK: - library

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

const char* set_kernel_name(CoordSetOp op) {
    switch (op) {
    case CoordSetOp::Downsample:
        return "downsample_coords_i32";
    case CoordSetOp::Union:
        return "union_coords_i32";
    case CoordSetOp::Intersection:
        return "intersection_coords_i32";
    }
}

const char* map_kernel_name(CoordMapOp op) {
    switch (op) {
    case CoordMapOp::Forward:
        return "build_forward_kernel_map_i32";
    case CoordMapOp::Transposed:
        return "build_transposed_kernel_map_i32";
    }
}

// MARK: - guards

void require_i32_inputs(
    const std::vector<mx::array>& inputs,
    std::initializer_list<const char*> names
) {
    int index = 0;
    for (auto name : names) {
        if (inputs[index++].dtype() != mx::int32) {
            throw std::invalid_argument(
                std::string("Metal coordinate kernels require int32 ") + name +
                "."
            );
        }
    }
}

void allocate(mx::array& output) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
}

void allocate_all(std::vector<mx::array>& outputs) {
    for (auto& output : outputs) {
        allocate(output);
    }
}

} // namespace

// MARK: - availability

bool supports(const mx::array& coords) {
#if MLX_LATTICE_HAS_METAL
    return coords.dtype() == mx::int32 && mx::is_available(mx::Device::gpu) &&
           mx::default_device() == mx::Device(mx::Device::gpu);
#else
    (void)coords;
    return false;
#endif
}

// MARK: - set ops

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    CoordSetShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords"});
    if (op != CoordSetOp::Downsample) {
        require_i32_inputs(inputs, {"coords", "rhs coords"});
    }

#ifdef _METAL_
    auto& out_coords = outputs[0];
    auto& count = outputs[1];
    allocate(out_coords);
    allocate(count);

    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel(set_kernel_name(op), library);

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    if (op != CoordSetOp::Downsample) {
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out_coords, 2);
        encoder.set_output_array(count, 3);
        encoder.set_bytes(shape.lhs_rows, 4);
        encoder.set_bytes(shape.rhs_rows, 5);
    } else {
        encoder.set_output_array(out_coords, 1);
        encoder.set_output_array(count, 2);
        encoder.set_bytes(shape.lhs_rows, 3);
        encoder.set_bytes(stride[0], 4);
        encoder.set_bytes(stride[1], 5);
        encoder.set_bytes(stride[2], 6);
    }
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)stride;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_lookup_coords(
    CoordLookupShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "queries"});

#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("lookup_coords_i32", library);
    auto group = std::min(
        static_cast<size_t>(std::max(shape.query_rows, 1)),
        kernel->maxTotalThreadsPerThreadgroup()
    );

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(shape.rows, 3);
    encoder.set_bytes(shape.query_rows, 4);
    encoder.dispatch_threads(
        MTL::Size(static_cast<size_t>(std::max(shape.query_rows, 1)), 1, 1),
        MTL::Size(group, 1, 1)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

// MARK: - maps

void eval_generic_kernel_map(
    CoordMapOp op,
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int kernel_count,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "kernel offsets"});

#ifdef _METAL_
    allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel(map_kernel_name(op), library);

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    for (int i = 0; i < int(outputs.size()); ++i) {
        encoder.set_output_array(outputs[i], i + 2);
    }
    encoder.set_bytes(rows, 16);
    encoder.set_bytes(kernel_count, 17);
    encoder.set_bytes(stride[0], 18);
    encoder.set_bytes(stride[1], 19);
    encoder.set_bytes(stride[2], 20);
    encoder.set_bytes(padding[0], 21);
    encoder.set_bytes(padding[1], 22);
    encoder.set_bytes(padding[2], 23);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)rows;
    (void)kernel_count;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_generative_kernel_map(
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int kernel_count,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "kernel offsets"});

#ifdef _METAL_
    allocate_all(outputs);

    auto pair_count = rows * kernel_count;
    auto thread_count = std::max({pair_count + 1, rows + 1, kernel_count + 1});

    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("build_generative_kernel_map_i32", library);
    auto group = std::min(
        static_cast<size_t>(thread_count),
        kernel->maxTotalThreadsPerThreadgroup()
    );

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    for (int i = 0; i < int(outputs.size()); ++i) {
        encoder.set_output_array(outputs[i], i + 2);
    }
    encoder.set_bytes(rows, 15);
    encoder.set_bytes(kernel_count, 16);
    encoder.set_bytes(stride[0], 17);
    encoder.set_bytes(stride[1], 18);
    encoder.set_bytes(stride[2], 19);
    encoder.dispatch_threads(
        MTL::Size(static_cast<size_t>(thread_count), 1, 1),
        MTL::Size(group, 1, 1)
    );
#else
    (void)rows;
    (void)kernel_count;
    (void)stride;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::coords::metal
