#include "backends/metal/coords.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::metal {

namespace {

// MARK: - types

enum class SetOp : std::uint8_t {
    Downsample,
    Union,
    Intersection,
};

enum class MapOp : std::uint8_t {
    Forward,
    Transposed,
};

struct CompactCoordBuffers {
    mx::array coords;
    mx::array count;
};

struct LookupShape {
    int rows;
    int query_rows;
};

struct SetShape {
    int lhs_rows;
    int rhs_rows;
};

// MARK: - arrays

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

mx::array make_offsets_array(const std::vector<Triple>& offsets) {
    std::vector<int32_t> flat;
    flat.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat.insert(flat.end(), offset.begin(), offset.end());
    }
    return mx::array(
        flat.begin(), mx::Shape{int(offsets.size()), 3}, mx::int32
    );
}

int scalar_i32(const mx::array& value) {
    auto cpu_value = mx::contiguous(value, false, mx::Device::cpu);
    cpu_value.eval();
    cpu_value.wait();
    return cpu_value.data<int32_t>()[0];
}

mx::array compact_coords(const CompactCoordBuffers& buffers) {
    auto rows = scalar_i32(buffers.count);
    return mx::slice(buffers.coords, {0, 0}, {rows, 4});
}

mx::array compact_rows(const mx::array& rows, int count) {
    return mx::slice(rows, {0}, {count});
}

mx::array compact_offsets(const mx::array& offsets, int count) {
    return mx::slice(offsets, {0}, {count + 1});
}

NativeKernelMap compact_map(
    const std::vector<mx::array>& outputs,
    const mx::array& kernel_offsets
) {
    auto count_values = mx::contiguous(outputs[4], false, mx::Device::cpu);
    count_values.eval();
    count_values.wait();
    auto counts = count_values.data<int32_t>();
    auto edge_count = counts[0];
    auto out_count = counts[1];
    auto kernel_count = kernel_offsets.shape(0);
    return {
        compact_rows(outputs[0], edge_count),
        compact_rows(outputs[1], edge_count),
        compact_rows(outputs[2], edge_count),
        mx::slice(outputs[3], {0, 0}, {out_count, 4}),
        kernel_offsets,
        {compact_offsets(outputs[5], out_count),
         compact_rows(outputs[6], edge_count),
         compact_rows(outputs[7], edge_count)},
        {compact_offsets(outputs[8], kernel_count),
         compact_rows(outputs[9], edge_count),
         compact_rows(outputs[10], edge_count)},
        {compact_offsets(outputs[11], outputs[11].shape(0) - 1),
         compact_rows(outputs[12], edge_count),
         compact_rows(outputs[13], edge_count)},
    };
}

const char* set_kernel_name(SetOp op) {
    switch (op) {
    case SetOp::Downsample:
        return "downsample_coords_i32";
    case SetOp::Union:
        return "union_coords_i32";
    case SetOp::Intersection:
        return "intersection_coords_i32";
    }
}

const char* map_kernel_name(MapOp op) {
    switch (op) {
    case MapOp::Forward:
        return "build_forward_kernel_map_i32";
    case MapOp::Transposed:
        return "build_transposed_kernel_map_i32";
    }
}

// MARK: - primitives

class SetCoords : public mx::Primitive {
  public:
    SetCoords(mx::Stream stream, SetOp op, Triple stride, SetShape shape)
        : mx::Primitive(stream), op_(op), stride_(stride), shape_(shape) {}

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("SetCoords has no CPU implementation.");
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
#ifdef _METAL_
        auto& out_coords = outputs[0];
        auto& count = outputs[1];
        out_coords.set_data(mx::allocator::malloc(out_coords.nbytes()));
        count.set_data(mx::allocator::malloc(count.nbytes()));

        auto& stream = this->stream();
        auto& device = mx::metal::device(stream.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(stream);
        auto kernel = device.get_kernel(set_kernel_name(op_), library);

        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        if (op_ != SetOp::Downsample) {
            encoder.set_input_array(inputs[1], 1);
            encoder.set_output_array(out_coords, 2);
            encoder.set_output_array(count, 3);
            encoder.set_bytes(shape_.lhs_rows, 4);
            encoder.set_bytes(shape_.rhs_rows, 5);
        } else {
            encoder.set_output_array(out_coords, 1);
            encoder.set_output_array(count, 2);
            encoder.set_bytes(shape_.lhs_rows, 3);
            encoder.set_bytes(stride_[0], 4);
            encoder.set_bytes(stride_[1], 5);
            encoder.set_bytes(stride_[2], 6);
        }
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
        throw std::runtime_error("Metal support is not available.");
#endif
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("SetCoords has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("SetCoords has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("SetCoords has no vmap implementation.");
    }

    const char* name() const override { return "SetCoords"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SetCoords&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               shape_.lhs_rows == op.shape_.lhs_rows &&
               shape_.rhs_rows == op.shape_.rhs_rows;
    }

  private:
    SetOp op_;
    Triple stride_;
    SetShape shape_;
};

class LookupCoords : public mx::Primitive {
  public:
    LookupCoords(mx::Stream stream, LookupShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("LookupCoords has no CPU implementation.");
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
#ifdef _METAL_
        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));

        auto& stream = this->stream();
        auto& device = mx::metal::device(stream.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(stream);
        auto kernel = device.get_kernel("lookup_coords_i32", library);
        auto group = std::min(
            static_cast<size_t>(std::max(shape_.query_rows, 1)),
            kernel->maxTotalThreadsPerThreadgroup()
        );

        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        encoder.set_bytes(shape_.rows, 3);
        encoder.set_bytes(shape_.query_rows, 4);
        encoder.dispatch_threads(
            MTL::Size(
                static_cast<size_t>(std::max(shape_.query_rows, 1)), 1, 1
            ),
            MTL::Size(group, 1, 1)
        );
#else
        throw std::runtime_error("Metal support is not available.");
#endif
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("LookupCoords has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("LookupCoords has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("LookupCoords has no vmap implementation.");
    }

    const char* name() const override { return "LookupCoords"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const LookupCoords&>(other);
        return shape_.rows == op.shape_.rows &&
               shape_.query_rows == op.shape_.query_rows;
    }

  private:
    LookupShape shape_;
};

class GenericKernelMap : public mx::Primitive {
  public:
    GenericKernelMap(
        mx::Stream stream,
        MapOp op,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int kernel_count,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), op_(op), rows_(rows),
          kernel_count_(kernel_count), stride_(stride), padding_(padding) {}

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("GenericKernelMap has no CPU implementation.");
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
#ifdef _METAL_
        for (auto& output : outputs) {
            output.set_data(mx::allocator::malloc(output.nbytes()));
        }

        auto& stream = this->stream();
        auto& device = mx::metal::device(stream.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(stream);
        auto kernel = device.get_kernel(map_kernel_name(op_), library);

        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        for (int i = 0; i < int(outputs.size()); ++i) {
            encoder.set_output_array(outputs[i], i + 2);
        }
        encoder.set_bytes(rows_, 16);
        encoder.set_bytes(kernel_count_, 17);
        encoder.set_bytes(stride_[0], 18);
        encoder.set_bytes(stride_[1], 19);
        encoder.set_bytes(stride_[2], 20);
        encoder.set_bytes(padding_[0], 21);
        encoder.set_bytes(padding_[1], 22);
        encoder.set_bytes(padding_[2], 23);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
        throw std::runtime_error("Metal support is not available.");
#endif
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("GenericKernelMap has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("GenericKernelMap has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error(
            "GenericKernelMap has no vmap implementation."
        );
    }

    const char* name() const override { return "GenericKernelMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const GenericKernelMap&>(other);
        return op_ == map.op_ && rows_ == map.rows_ &&
               kernel_count_ == map.kernel_count_ && stride_ == map.stride_ &&
               padding_ == map.padding_;
    }

  private:
    MapOp op_;
    int rows_;
    int kernel_count_;
    Triple stride_;
    Triple padding_;
};

class GenerativeKernelMap : public mx::Primitive {
  public:
    GenerativeKernelMap(
        mx::Stream stream,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int kernel_count,
        Triple stride
    )
        : mx::Primitive(stream), rows_(rows), kernel_count_(kernel_count),
          stride_(stride) {}

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error(
            "GenerativeKernelMap has no CPU implementation."
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
#ifdef _METAL_
        const auto& coords = inputs[0];
        const auto& offsets = inputs[1];
        auto& in_rows = outputs[0];
        auto& out_rows = outputs[1];
        auto& kernel_ids = outputs[2];
        auto& out_coords = outputs[3];
        auto& output_csr_offsets = outputs[4];
        auto& output_csr_in_rows = outputs[5];
        auto& output_csr_kernel_ids = outputs[6];
        auto& kernel_bucket_offsets = outputs[7];
        auto& kernel_bucket_in_rows = outputs[8];
        auto& kernel_bucket_out_rows = outputs[9];
        auto& input_csr_offsets = outputs[10];
        auto& input_csr_out_rows = outputs[11];
        auto& input_csr_kernel_ids = outputs[12];

        in_rows.set_data(mx::allocator::malloc(in_rows.nbytes()));
        out_rows.set_data(mx::allocator::malloc(out_rows.nbytes()));
        kernel_ids.set_data(mx::allocator::malloc(kernel_ids.nbytes()));
        out_coords.set_data(mx::allocator::malloc(out_coords.nbytes()));
        output_csr_offsets.set_data(
            mx::allocator::malloc(output_csr_offsets.nbytes())
        );
        output_csr_in_rows.set_data(
            mx::allocator::malloc(output_csr_in_rows.nbytes())
        );
        output_csr_kernel_ids.set_data(
            mx::allocator::malloc(output_csr_kernel_ids.nbytes())
        );
        kernel_bucket_offsets.set_data(
            mx::allocator::malloc(kernel_bucket_offsets.nbytes())
        );
        kernel_bucket_in_rows.set_data(
            mx::allocator::malloc(kernel_bucket_in_rows.nbytes())
        );
        kernel_bucket_out_rows.set_data(
            mx::allocator::malloc(kernel_bucket_out_rows.nbytes())
        );
        input_csr_offsets.set_data(
            mx::allocator::malloc(input_csr_offsets.nbytes())
        );
        input_csr_out_rows.set_data(
            mx::allocator::malloc(input_csr_out_rows.nbytes())
        );
        input_csr_kernel_ids.set_data(
            mx::allocator::malloc(input_csr_kernel_ids.nbytes())
        );

        auto pair_count = rows_ * kernel_count_;
        auto thread_count =
            std::max({pair_count + 1, rows_ + 1, kernel_count_ + 1});

        auto& stream = this->stream();
        auto& device = mx::metal::device(stream.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(stream);
        auto kernel =
            device.get_kernel("build_generative_kernel_map_i32", library);
        auto group = std::min(
            static_cast<size_t>(thread_count),
            kernel->maxTotalThreadsPerThreadgroup()
        );

        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(coords, 0);
        encoder.set_input_array(offsets, 1);
        encoder.set_output_array(in_rows, 2);
        encoder.set_output_array(out_rows, 3);
        encoder.set_output_array(kernel_ids, 4);
        encoder.set_output_array(out_coords, 5);
        encoder.set_output_array(output_csr_offsets, 6);
        encoder.set_output_array(output_csr_in_rows, 7);
        encoder.set_output_array(output_csr_kernel_ids, 8);
        encoder.set_output_array(kernel_bucket_offsets, 9);
        encoder.set_output_array(kernel_bucket_in_rows, 10);
        encoder.set_output_array(kernel_bucket_out_rows, 11);
        encoder.set_output_array(input_csr_offsets, 12);
        encoder.set_output_array(input_csr_out_rows, 13);
        encoder.set_output_array(input_csr_kernel_ids, 14);
        encoder.set_bytes(rows_, 15);
        encoder.set_bytes(kernel_count_, 16);
        encoder.set_bytes(stride_[0], 17);
        encoder.set_bytes(stride_[1], 18);
        encoder.set_bytes(stride_[2], 19);
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(thread_count), 1, 1),
            MTL::Size(group, 1, 1)
        );
#else
        throw std::runtime_error("Metal support is not available.");
#endif
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error(
            "GenerativeKernelMap has no jvp implementation."
        );
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error(
            "GenerativeKernelMap has no vjp implementation."
        );
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error(
            "GenerativeKernelMap has no vmap implementation."
        );
    }

    const char* name() const override { return "GenerativeKernelMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const GenerativeKernelMap&>(other);
        return rows_ == map.rows_ && kernel_count_ == map.kernel_count_ &&
               stride_ == map.stride_;
    }

  private:
    int rows_;
    int kernel_count_;
    Triple stride_;
};

} // namespace

// MARK: - api

mx::array downsample_coords(const mx::array& coords, Triple stride) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal downsample_coords requires int32 coords."
        );
    }

    auto rows = coords.shape(0);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{rows, 4}, mx::Shape{1}},
        {mx::int32, mx::int32},
        std::make_shared<SetCoords>(
            mx::default_stream(mx::Device::gpu),
            SetOp::Downsample,
            stride,
            SetShape{rows, 0}
        ),
        {mx::contiguous(coords, false, mx::Device::gpu)}
    );
    return compact_coords({outputs[0], outputs[1]});
}

mx::array union_coords(const mx::array& lhs, const mx::array& rhs) {
    if (lhs.dtype() != mx::int32 || rhs.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal union_coords requires int32 coords."
        );
    }

    auto lhs_rows = lhs.shape(0);
    auto rhs_rows = rhs.shape(0);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{lhs_rows + rhs_rows, 4}, mx::Shape{1}},
        {mx::int32, mx::int32},
        std::make_shared<SetCoords>(
            mx::default_stream(mx::Device::gpu),
            SetOp::Union,
            Triple{1, 1, 1},
            SetShape{lhs_rows, rhs_rows}
        ),
        {mx::contiguous(lhs, false, mx::Device::gpu),
         mx::contiguous(rhs, false, mx::Device::gpu)}
    );
    return compact_coords({outputs[0], outputs[1]});
}

mx::array intersection_coords(const mx::array& lhs, const mx::array& rhs) {
    if (lhs.dtype() != mx::int32 || rhs.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal intersection_coords requires int32 coords."
        );
    }

    auto lhs_rows = lhs.shape(0);
    auto rhs_rows = rhs.shape(0);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{lhs_rows, 4}, mx::Shape{1}},
        {mx::int32, mx::int32},
        std::make_shared<SetCoords>(
            mx::default_stream(mx::Device::gpu),
            SetOp::Intersection,
            Triple{1, 1, 1},
            SetShape{lhs_rows, rhs_rows}
        ),
        {mx::contiguous(lhs, false, mx::Device::gpu),
         mx::contiguous(rhs, false, mx::Device::gpu)}
    );
    return compact_coords({outputs[0], outputs[1]});
}

mx::array lookup_coords(const mx::array& coords, const mx::array& queries) {
    if (coords.dtype() != mx::int32 || queries.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal lookup_coords requires int32 coords."
        );
    }

    auto rows = coords.shape(0);
    auto query_rows = queries.shape(0);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{query_rows}},
        {mx::int32},
        std::make_shared<LookupCoords>(
            mx::default_stream(mx::Device::gpu), LookupShape{rows, query_rows}
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(queries, false, mx::Device::gpu)}
    );
    return outputs[0];
}

NativeKernelMap build_kernel_map(
    const mx::array& coords,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal build_kernel_map requires int32 coords."
        );
    }

    auto offsets = kernel_offsets(kernel_size, dilation);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto max_out_rows = rows;
    auto max_edges = max_out_rows * kernel_count;
    auto outputs = mx::array::make_arrays(
        {mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_out_rows, 4},
         mx::Shape{2},
         mx::Shape{max_out_rows + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{kernel_count + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{rows + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges}},
        {mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<GenericKernelMap>(
            mx::default_stream(mx::Device::gpu),
            MapOp::Forward,
            rows,
            kernel_count,
            stride,
            padding
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );
    return compact_map(outputs, offset_values);
}

NativeKernelMap build_generative_map(
    const mx::array& coords,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride
) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal generative maps require int32 coords."
        );
    }

    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto pair_count = rows * kernel_count;
    auto outputs = mx::array::make_arrays(
        {mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{pair_count, 4},
         mx::Shape{pair_count + 1},
         mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{kernel_count + 1},
         mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{rows + 1},
         mx::Shape{pair_count},
         mx::Shape{pair_count}},
        {mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<GenerativeKernelMap>(
            mx::default_stream(mx::Device::gpu), rows, kernel_count, stride
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );

    return {
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        offset_values,
        {outputs[4], outputs[5], outputs[6]},
        {outputs[7], outputs[8], outputs[9]},
        {outputs[10], outputs[11], outputs[12]},
    };
}

NativeKernelMap build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal build_transposed_kernel_map requires int32 coords."
        );
    }

    auto offsets = kernel_offsets(kernel_size, dilation);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto max_edges = rows * kernel_count;
    auto outputs = mx::array::make_arrays(
        {mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_edges, 4},
         mx::Shape{2},
         mx::Shape{max_edges + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{kernel_count + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{rows + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges}},
        {mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<GenericKernelMap>(
            mx::default_stream(mx::Device::gpu),
            MapOp::Transposed,
            rows,
            kernel_count,
            stride,
            padding
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );
    return compact_map(outputs, offset_values);
}

} // namespace mlx_lattice::metal
