#include "backends/cuda/coords.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx_lattice::cuda {

namespace {

namespace mx = mlx::core;

// MARK: - helpers

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

mx::array make_i32_array(std::vector<int32_t> data, mx::Shape shape) {
    return mx::array(data.begin(), std::move(shape), mx::int32);
}

int next_power_of_two(int value) {
    int out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

dim3 grid_for(size_t elements, int block = 256) {
    return dim3(static_cast<unsigned int>((elements + block - 1) / block));
}

template <typename Kernel, typename... Args>
void launch(
    mx::cu::CommandEncoder& encoder,
    Kernel kernel,
    size_t elements,
    Args... args
) {
    if (elements == 0) {
        return;
    }
    constexpr int block = 256;
    encoder.add_kernel_node(
        kernel, grid_for(elements, block), dim3(block), args...
    );
}

// MARK: - kernels

__global__ void fill_i32(int* out, int value, int size) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem < size) {
        out[elem] = value;
    }
}

__global__ void fill_linear_i32(int* out, int step, int size) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem < size) {
        out[elem] = elem * step;
    }
}

__device__ int coord_hash_i32(int b, int x, int y, int z) {
    uint32_t h = 2166136261u;
    h = (h ^ uint32_t(b)) * 16777619u;
    h = (h ^ uint32_t(x)) * 16777619u;
    h = (h ^ uint32_t(y)) * 16777619u;
    h = (h ^ uint32_t(z)) * 16777619u;
    int out = int(h & 0x7fffffffu);
    return out == int(0x7fffffff) ? out - 1 : out;
}

__device__ bool same_coord(
    const int* __restrict__ coords,
    int row,
    int b,
    int x,
    int y,
    int z
) {
    int base = row * 4;
    return coords[base] == b && coords[base + 1] == x &&
           coords[base + 2] == y && coords[base + 3] == z;
}

__global__ void insert_coord_hash_i32(
    const int* __restrict__ coords,
    int* __restrict__ table_keys,
    int* __restrict__ table_rows,
    int rows,
    int table_capacity,
    int empty_key
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int base = row * 4;
    int b = coords[base];
    int x = coords[base + 1];
    int y = coords[base + 2];
    int z = coords[base + 3];
    int key = coord_hash_i32(b, x, y, z);
    int slot = key & (table_capacity - 1);

    for (int probe = 0; probe < table_capacity; ++probe) {
        int old = atomicCAS(&table_keys[slot], empty_key, key);
        if (old == empty_key) {
            table_rows[slot] = row;
            return;
        }
        if (old == key) {
            int existing = table_rows[slot];
            if (existing >= 0 && same_coord(coords, existing, b, x, y, z)) {
                return;
            }
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

__global__ void build_subm_kernel_map_i32(
    const int* __restrict__ coords,
    const int* __restrict__ offsets,
    const int* __restrict__ table_keys,
    const int* __restrict__ table_rows,
    int* __restrict__ maps,
    int* __restrict__ sizes,
    int* __restrict__ kernels,
    int* __restrict__ residual_maps,
    int* __restrict__ residual_kernels,
    int rows,
    int kernel_count,
    int center_kernel,
    int table_capacity,
    int empty_key
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * kernel_count;
    if (elem >= total) {
        return;
    }

    int kernel_index = elem / rows;
    int out_row = elem - kernel_index * rows;
    int base = out_row * 4;
    int target_b = coords[base];
    int target_x = coords[base + 1] + offsets[kernel_index * 3];
    int target_y = coords[base + 2] + offsets[kernel_index * 3 + 1];
    int target_z = coords[base + 3] + offsets[kernel_index * 3 + 2];
    int key = coord_hash_i32(target_b, target_x, target_y, target_z);
    int slot = key & (table_capacity - 1);

    for (int probe = 0; probe < table_capacity; ++probe) {
        int found_key = table_keys[slot];
        if (found_key == empty_key) {
            return;
        }
        if (found_key == key) {
            int in_row = table_rows[slot];
            if (in_row < 0 ||
                !same_coord(
                    coords, in_row, target_b, target_x, target_y, target_z
                )) {
                slot = (slot + 1) & (table_capacity - 1);
                continue;
            }

            maps[elem * 2] = in_row;
            maps[elem * 2 + 1] = out_row;
            kernels[elem] = kernel_index;
            atomicAdd(&sizes[kernel_index], 1);
            if (kernel_index != center_kernel) {
                int local_slot = kernel_index < center_kernel
                                     ? kernel_index
                                     : kernel_index - 1;
                int residual = out_row * (kernel_count - 1) + local_slot;
                residual_maps[residual * 2] = in_row;
                residual_maps[residual * 2 + 1] = out_row;
                residual_kernels[residual] = kernel_index;
            }
            return;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

__global__ void build_generative_map_i32(
    const int* __restrict__ coords,
    const int* __restrict__ offsets,
    int* __restrict__ maps,
    int* __restrict__ kernels,
    int* __restrict__ out_coords,
    int rows,
    int kernel_count,
    int stride_x,
    int stride_y,
    int stride_z
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * kernel_count;
    if (elem >= total) {
        return;
    }

    int in_row = elem / kernel_count;
    int kernel_index = elem - in_row * kernel_count;
    int out_row = elem;
    int in_base = in_row * 4;
    int out_base = out_row * 4;
    maps[out_row * 2] = in_row;
    maps[out_row * 2 + 1] = out_row;
    kernels[out_row] = kernel_index;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + offsets[kernel_index * 3];
    out_coords[out_base + 2] =
        coords[in_base + 2] * stride_y + offsets[kernel_index * 3 + 1];
    out_coords[out_base + 3] =
        coords[in_base + 3] * stride_z + offsets[kernel_index * 3 + 2];
}

// MARK: - primitives

class SubmKernelMap : public mx::Primitive {
  public:
    SubmKernelMap(
        mx::Stream stream,
        int rows,
        int kernels,
        int center_kernel,
        int table_capacity
    )
        : mx::Primitive(stream), rows_(rows), kernels_(kernels),
          center_kernel_(center_kernel), table_capacity_(table_capacity) {}

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("SubmKernelMap has no CPU implementation.");
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        const auto& coords = inputs[0];
        const auto& offsets = inputs[1];
        auto& maps = outputs[0];
        auto& sizes = outputs[1];
        auto& kernels = outputs[2];
        auto& residual_maps = outputs[3];
        auto& residual_kernels = outputs[4];
        auto& residual_offsets = outputs[5];
        auto& table_keys = outputs[6];
        auto& table_rows = outputs[7];

        maps.set_data(mx::allocator::malloc(maps.nbytes()));
        sizes.set_data(mx::allocator::malloc(sizes.nbytes()));
        kernels.set_data(mx::allocator::malloc(kernels.nbytes()));
        residual_maps.set_data(mx::allocator::malloc(residual_maps.nbytes()));
        residual_kernels.set_data(
            mx::allocator::malloc(residual_kernels.nbytes())
        );
        residual_offsets.set_data(
            mx::allocator::malloc(residual_offsets.nbytes())
        );
        table_keys.set_data(mx::allocator::malloc(table_keys.nbytes()));
        table_rows.set_data(mx::allocator::malloc(table_rows.nbytes()));

        auto& encoder = mx::cu::get_command_encoder(stream());
        encoder.set_input_array(coords);
        encoder.set_input_array(offsets);
        encoder.set_output_array(maps);
        encoder.set_output_array(sizes);
        encoder.set_output_array(kernels);
        encoder.set_output_array(residual_maps);
        encoder.set_output_array(residual_kernels);
        encoder.set_output_array(residual_offsets);
        encoder.set_output_array(table_keys);
        encoder.set_output_array(table_rows);

        int pair_slots = rows_ * kernels_;
        int residual_slots = rows_ * std::max(kernels_ - 1, 0);
        int residual_offset_count = rows_ + 1;
        int invalid = -1;
        int empty_key = 0x7fffffff;

        launch(
            encoder,
            fill_i32,
            kernels_,
            mx::gpu_ptr<int32_t>(sizes),
            0,
            kernels_
        );
        launch(
            encoder,
            fill_i32,
            pair_slots,
            mx::gpu_ptr<int32_t>(kernels),
            invalid,
            pair_slots
        );
        launch(
            encoder,
            fill_i32,
            residual_slots,
            mx::gpu_ptr<int32_t>(residual_kernels),
            invalid,
            residual_slots
        );
        launch(
            encoder,
            fill_i32,
            table_capacity_,
            mx::gpu_ptr<int32_t>(table_keys),
            empty_key,
            table_capacity_
        );
        launch(
            encoder,
            fill_i32,
            table_capacity_,
            mx::gpu_ptr<int32_t>(table_rows),
            invalid,
            table_capacity_
        );
        launch(
            encoder,
            fill_linear_i32,
            residual_offset_count,
            mx::gpu_ptr<int32_t>(residual_offsets),
            std::max(kernels_ - 1, 0),
            residual_offset_count
        );
        if (pair_slots == 0) {
            return;
        }

        launch(
            encoder,
            insert_coord_hash_i32,
            rows_,
            mx::gpu_ptr<int32_t>(coords),
            mx::gpu_ptr<int32_t>(table_keys),
            mx::gpu_ptr<int32_t>(table_rows),
            rows_,
            table_capacity_,
            empty_key
        );
        launch(
            encoder,
            build_subm_kernel_map_i32,
            pair_slots,
            mx::gpu_ptr<int32_t>(coords),
            mx::gpu_ptr<int32_t>(offsets),
            mx::gpu_ptr<int32_t>(table_keys),
            mx::gpu_ptr<int32_t>(table_rows),
            mx::gpu_ptr<int32_t>(maps),
            mx::gpu_ptr<int32_t>(sizes),
            mx::gpu_ptr<int32_t>(kernels),
            mx::gpu_ptr<int32_t>(residual_maps),
            mx::gpu_ptr<int32_t>(residual_kernels),
            rows_,
            kernels_,
            center_kernel_,
            table_capacity_,
            empty_key
        );
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("SubmKernelMap has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("SubmKernelMap has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("SubmKernelMap has no vmap implementation.");
    }

    const char* name() const override { return "CudaSubmKernelMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const SubmKernelMap&>(other);
        return rows_ == map.rows_ && kernels_ == map.kernels_ &&
               center_kernel_ == map.center_kernel_ &&
               table_capacity_ == map.table_capacity_;
    }

  private:
    int rows_;
    int kernels_;
    int center_kernel_;
    int table_capacity_;
};

class GenerativeMap : public mx::Primitive {
  public:
    GenerativeMap(mx::Stream stream, int rows, int kernels, Triple stride)
        : mx::Primitive(stream), rows_(rows), kernels_(kernels),
          stride_(stride) {}

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("GenerativeMap has no CPU implementation.");
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        const auto& coords = inputs[0];
        const auto& offsets = inputs[1];
        auto& maps = outputs[0];
        auto& kernels = outputs[1];
        auto& out_coords = outputs[2];

        maps.set_data(mx::allocator::malloc(maps.nbytes()));
        kernels.set_data(mx::allocator::malloc(kernels.nbytes()));
        out_coords.set_data(mx::allocator::malloc(out_coords.nbytes()));

        auto& encoder = mx::cu::get_command_encoder(stream());
        encoder.set_input_array(coords);
        encoder.set_input_array(offsets);
        encoder.set_output_array(maps);
        encoder.set_output_array(kernels);
        encoder.set_output_array(out_coords);

        int pair_count = rows_ * kernels_;
        launch(
            encoder,
            build_generative_map_i32,
            pair_count,
            mx::gpu_ptr<int32_t>(coords),
            mx::gpu_ptr<int32_t>(offsets),
            mx::gpu_ptr<int32_t>(maps),
            mx::gpu_ptr<int32_t>(kernels),
            mx::gpu_ptr<int32_t>(out_coords),
            rows_,
            kernels_,
            stride_[0],
            stride_[1],
            stride_[2]
        );
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("GenerativeMap has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("GenerativeMap has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("GenerativeMap has no vmap implementation.");
    }

    const char* name() const override { return "CudaGenerativeMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const GenerativeMap&>(other);
        return rows_ == map.rows_ && kernels_ == map.kernels_ &&
               stride_ == map.stride_;
    }

  private:
    int rows_;
    int kernels_;
    Triple stride_;
};

} // namespace

// MARK: - api

KernelMapData
build_subm_kernel_map(const mx::array& coords, Triple kernel_size) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "CUDA coordinate maps require int32 coords."
        );
    }

    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto center = std::find(offsets.begin(), offsets.end(), Triple{0, 0, 0});
    if (center == offsets.end()) {
        throw std::invalid_argument(
            "submanifold maps require a kernel with center offset."
        );
    }

    int center_kernel = int(std::distance(offsets.begin(), center));
    auto rows = coords.shape(0);
    auto pair_slots = rows * int(offsets.size());
    auto residual_slots = rows * std::max(int(offsets.size()) - 1, 0);
    auto table_capacity = next_power_of_two(std::max(rows * 4, 1));
    auto outputs = mx::array::make_arrays(
        {mx::Shape{pair_slots, 2},
         mx::Shape{int(offsets.size())},
         mx::Shape{pair_slots},
         mx::Shape{residual_slots, 2},
         mx::Shape{residual_slots},
         mx::Shape{rows + 1},
         mx::Shape{table_capacity},
         mx::Shape{table_capacity}},
        {mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<SubmKernelMap>(
            mx::default_stream(mx::Device::gpu),
            rows,
            int(offsets.size()),
            center_kernel,
            table_capacity
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );

    return {
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        coords,
        offset_values,
    };
}

KernelMapData build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "CUDA generative maps require int32 coords."
        );
    }

    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto pair_count = rows * int(offsets.size());
    auto sizes =
        mx::full({int(offsets.size())}, rows, mx::int32, mx::Device::gpu);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{pair_count, 2},
         mx::Shape{pair_count},
         mx::Shape{pair_count, 4}},
        {mx::int32, mx::int32, mx::int32},
        std::make_shared<GenerativeMap>(
            mx::default_stream(mx::Device::gpu),
            rows,
            int(offsets.size()),
            stride
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );

    return {
        outputs[0],
        sizes,
        outputs[1],
        make_i32_array({}, mx::Shape{0, 2}),
        make_i32_array({}, mx::Shape{0}),
        make_i32_array({0}, mx::Shape{1}),
        outputs[2],
        offset_values,
    };
}

} // namespace mlx_lattice::cuda
