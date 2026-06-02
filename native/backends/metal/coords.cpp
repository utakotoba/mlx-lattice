#include "backends/metal/coords.h"

#include <dlfcn.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/coords.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::metal {

namespace {

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

// MARK: - primitives

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

        in_rows.set_data(mx::allocator::malloc(in_rows.nbytes()));
        out_rows.set_data(mx::allocator::malloc(out_rows.nbytes()));
        kernel_ids.set_data(mx::allocator::malloc(kernel_ids.nbytes()));
        out_coords.set_data(mx::allocator::malloc(out_coords.nbytes()));

        auto pair_count = rows_ * kernel_count_;
        if (pair_count == 0) {
            return;
        }

        auto& stream = this->stream();
        auto& device = mx::metal::device(stream.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(stream);
        auto kernel =
            device.get_kernel("build_generative_kernel_map_i32", library);
        auto group = std::min(
            static_cast<size_t>(pair_count),
            kernel->maxTotalThreadsPerThreadgroup()
        );

        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(coords, 0);
        encoder.set_input_array(offsets, 1);
        encoder.set_output_array(in_rows, 2);
        encoder.set_output_array(out_rows, 3);
        encoder.set_output_array(kernel_ids, 4);
        encoder.set_output_array(out_coords, 5);
        encoder.set_bytes(rows_, 6);
        encoder.set_bytes(kernel_count_, 7);
        encoder.set_bytes(stride_[0], 8);
        encoder.set_bytes(stride_[1], 9);
        encoder.set_bytes(stride_[2], 10);
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(pair_count), 1, 1),
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

NativeKernelMap build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    return cpu::build_kernel_map(
        coords, kernel_size, stride, padding, dilation
    );
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
         mx::Shape{pair_count, 4}},
        {mx::int32, mx::int32, mx::int32, mx::int32},
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
    };
}

NativeKernelMap build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    return cpu::build_transposed_kernel_map(
        coords, kernel_size, stride, padding, dilation
    );
}

} // namespace mlx_lattice::metal
