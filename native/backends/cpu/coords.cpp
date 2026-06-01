#include "backends/cpu/coords.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "mlx/device.h"
#include "mlx/ops.h"

namespace mlx_lattice::cpu {

namespace {

// MARK: - types

using Coord = std::array<int64_t, 4>;

struct CoordHash {
    size_t operator()(const Coord& coord) const {
        size_t seed = 0;
        for (auto value : coord) {
            auto part = std::hash<int64_t>{}(value);
            seed ^= part + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// MARK: - arrays

std::vector<Coord> read_coords(const mx::array& coords) {
    auto cpu_coords = mx::contiguous(coords, false, mx::Device::cpu);
    cpu_coords.eval();
    cpu_coords.wait();

    std::vector<Coord> out;
    out.reserve(cpu_coords.shape(0));
    if (cpu_coords.dtype() == mx::int32) {
        auto data = cpu_coords.data<int32_t>();
        for (int row = 0; row < cpu_coords.shape(0); ++row) {
            out.push_back({
                data[row * 4],
                data[row * 4 + 1],
                data[row * 4 + 2],
                data[row * 4 + 3],
            });
        }
        return out;
    }

    auto data = cpu_coords.data<int64_t>();
    for (int row = 0; row < cpu_coords.shape(0); ++row) {
        out.push_back({
            data[row * 4],
            data[row * 4 + 1],
            data[row * 4 + 2],
            data[row * 4 + 3],
        });
    }
    return out;
}

mx::array make_coords_array(const std::vector<Coord>& coords, mx::Dtype dtype) {
    if (dtype == mx::int32) {
        std::vector<int32_t> data;
        data.reserve(coords.size() * 4);
        for (auto coord : coords) {
            for (auto value : coord) {
                data.push_back(static_cast<int32_t>(value));
            }
        }
        return mx::array(data.begin(), mx::Shape{int(coords.size()), 4}, dtype);
    }

    std::vector<int64_t> data;
    data.reserve(coords.size() * 4);
    for (auto coord : coords) {
        data.insert(data.end(), coord.begin(), coord.end());
    }
    return mx::array(data.begin(), mx::Shape{int(coords.size()), 4}, dtype);
}

mx::array make_i32_array(std::vector<int32_t> data, mx::Shape shape) {
    return mx::array(data.begin(), std::move(shape), mx::int32);
}

// MARK: - coordinates

int64_t floor_div(int64_t value, int64_t divisor) {
    auto quotient = value / divisor;
    auto remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

std::vector<Coord>
downsample_values(const std::vector<Coord>& coords, Triple stride) {
    std::vector<Coord> out;
    out.reserve(coords.size());
    std::unordered_set<Coord, CoordHash> seen;
    seen.reserve(coords.size());

    for (auto coord : coords) {
        Coord quantized = {
            coord[0],
            floor_div(coord[1], stride[0]),
            floor_div(coord[2], stride[1]),
            floor_div(coord[3], stride[2]),
        };
        if (seen.insert(quantized).second) {
            out.push_back(quantized);
        }
    }
    return out;
}

Coord min_coord(const std::vector<Coord>& values) {
    auto out = values.front();
    for (auto coord : values) {
        for (int axis = 0; axis < 4; ++axis) {
            out[axis] = std::min(out[axis], coord[axis]);
        }
    }
    return out;
}

Coord max_coord(const std::vector<Coord>& values) {
    auto out = values.front();
    for (auto coord : values) {
        for (int axis = 0; axis < 4; ++axis) {
            out[axis] = std::max(out[axis], coord[axis]);
        }
    }
    return out;
}

int64_t encode(Coord coord, Coord mins, Coord dims) {
    Coord shifted = {
        coord[0] - mins[0],
        coord[1] - mins[1],
        coord[2] - mins[2],
        coord[3] - mins[3],
    };
    return (
        ((shifted[0] * dims[1] + shifted[1]) * dims[2] + shifted[2]) * dims[3] +
        shifted[3]
    );
}

// MARK: - maps

KernelMapData
make_empty_map(const mx::array& coords, const std::vector<Triple>& offsets) {
    std::vector<int32_t> flat_offsets;
    flat_offsets.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat_offsets.insert(flat_offsets.end(), offset.begin(), offset.end());
    }

    return {
        make_i32_array({}, mx::Shape{0, 2}),
        mx::zeros({int(offsets.size())}, mx::int32, mx::Device::cpu),
        make_i32_array({}, mx::Shape{0}),
        make_i32_array({}, mx::Shape{0, 2}),
        make_i32_array({}, mx::Shape{0}),
        make_i32_array({0}, mx::Shape{1}),
        make_coords_array({}, coords.dtype()),
        make_i32_array(flat_offsets, mx::Shape{int(offsets.size()), 3}),
    };
}

int center_kernel(const std::vector<Triple>& offsets) {
    auto center = std::find(offsets.begin(), offsets.end(), Triple{0, 0, 0});
    if (center == offsets.end()) {
        return -1;
    }
    return int(std::distance(offsets.begin(), center));
}

} // namespace

// MARK: - api

mx::array downsample_coords(const mx::array& coords, Triple stride) {
    return make_coords_array(
        downsample_values(read_coords(coords), stride), coords.dtype()
    );
}

KernelMapData build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding
) {
    auto offsets = mlx_lattice::kernel_offsets(kernel_size);
    auto values = read_coords(coords);
    if (values.empty()) {
        return make_empty_map(coords, offsets);
    }

    bool unit_stride = stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
    auto out = unit_stride ? values : downsample_values(values, stride);

    Coord offset_min = {
        0, offsets.front()[0], offsets.front()[1], offsets.front()[2]
    };
    Coord offset_max = offset_min;
    for (auto offset : offsets) {
        offset_min[1] = std::min<int64_t>(offset_min[1], offset[0]);
        offset_min[2] = std::min<int64_t>(offset_min[2], offset[1]);
        offset_min[3] = std::min<int64_t>(offset_min[3], offset[2]);
        offset_max[1] = std::max<int64_t>(offset_max[1], offset[0]);
        offset_max[2] = std::max<int64_t>(offset_max[2], offset[1]);
        offset_max[3] = std::max<int64_t>(offset_max[3], offset[2]);
    }

    Coord mins = min_coord(values);
    Coord maxs = max_coord(values);
    for (auto coord : out) {
        Coord low = {
            coord[0],
            coord[1] * stride[0] + offset_min[1] - padding[0],
            coord[2] * stride[1] + offset_min[2] - padding[1],
            coord[3] * stride[2] + offset_min[3] - padding[2],
        };
        Coord high = {
            coord[0],
            coord[1] * stride[0] + offset_max[1] - padding[0],
            coord[2] * stride[1] + offset_max[2] - padding[1],
            coord[3] * stride[2] + offset_max[3] - padding[2],
        };
        for (int axis = 0; axis < 4; ++axis) {
            mins[axis] = std::min(mins[axis], low[axis]);
            maxs[axis] = std::max(maxs[axis], high[axis]);
        }
    }

    Coord dims = {
        maxs[0] - mins[0] + 1,
        maxs[1] - mins[1] + 1,
        maxs[2] - mins[2] + 1,
        maxs[3] - mins[3] + 1,
    };

    std::vector<int64_t> input_keys;
    input_keys.reserve(values.size());
    for (auto coord : values) {
        input_keys.push_back(encode(coord, mins, dims));
    }

    std::vector<int> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return input_keys[lhs] < input_keys[rhs];
    });

    std::vector<int64_t> sorted_keys;
    sorted_keys.reserve(order.size());
    for (auto row : order) {
        sorted_keys.push_back(input_keys[row]);
    }

    std::vector<int64_t> out_keys;
    out_keys.reserve(out.size());
    for (auto coord : out) {
        out_keys.push_back(encode(coord, mins, dims));
    }

    std::vector<int32_t> maps;
    std::vector<int32_t> kernels;
    std::vector<int32_t> sizes;
    std::vector<std::vector<std::array<int32_t, 2>>> residual_rows(out.size());
    std::vector<std::vector<int32_t>> residual_kernel_rows(out.size());
    sizes.reserve(offsets.size());
    int center = center_kernel(offsets);

    for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
        auto offset = offsets[kernel];
        int32_t count = 0;
        int64_t delta =
            offset[0] * dims[2] * dims[3] + offset[1] * dims[3] + offset[2];
        for (int out_row = 0; out_row < int(out.size()); ++out_row) {
            int64_t key = 0;
            if (unit_stride) {
                key = out_keys[out_row] + delta;
            } else {
                Coord candidate = {
                    out[out_row][0],
                    out[out_row][1] * stride[0] + offset[0] - padding[0],
                    out[out_row][2] * stride[1] + offset[1] - padding[1],
                    out[out_row][3] * stride[2] + offset[2] - padding[2],
                };
                key = encode(candidate, mins, dims);
            }

            auto pos =
                std::lower_bound(sorted_keys.begin(), sorted_keys.end(), key);
            if (pos != sorted_keys.end() && *pos == key) {
                auto in_row = order[std::distance(sorted_keys.begin(), pos)];
                maps.push_back(static_cast<int32_t>(in_row));
                maps.push_back(static_cast<int32_t>(out_row));
                kernels.push_back(kernel);
                if (kernel != center) {
                    residual_rows[out_row].push_back(
                        {static_cast<int32_t>(in_row),
                         static_cast<int32_t>(out_row)}
                    );
                    residual_kernel_rows[out_row].push_back(kernel);
                }
                ++count;
            }
        }
        sizes.push_back(count);
    }

    std::vector<int32_t> residual_maps;
    std::vector<int32_t> residual_kernels;
    std::vector<int32_t> residual_offsets;
    residual_offsets.reserve(out.size() + 1);
    residual_offsets.push_back(0);
    for (int out_row = 0; out_row < int(out.size()); ++out_row) {
        for (int idx = 0; idx < int(residual_rows[out_row].size()); ++idx) {
            auto pair = residual_rows[out_row][idx];
            residual_maps.push_back(pair[0]);
            residual_maps.push_back(pair[1]);
            residual_kernels.push_back(residual_kernel_rows[out_row][idx]);
        }
        residual_offsets.push_back(int32_t(residual_kernels.size()));
    }

    std::vector<int32_t> flat_offsets;
    flat_offsets.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat_offsets.insert(flat_offsets.end(), offset.begin(), offset.end());
    }

    return {
        make_i32_array(maps, mx::Shape{int(maps.size() / 2), 2}),
        make_i32_array(sizes, mx::Shape{int(sizes.size())}),
        make_i32_array(kernels, mx::Shape{int(kernels.size())}),
        make_i32_array(
            residual_maps, mx::Shape{int(residual_maps.size() / 2), 2}
        ),
        make_i32_array(
            residual_kernels, mx::Shape{int(residual_kernels.size())}
        ),
        make_i32_array(
            residual_offsets, mx::Shape{int(residual_offsets.size())}
        ),
        make_coords_array(out, coords.dtype()),
        make_i32_array(flat_offsets, mx::Shape{int(offsets.size()), 3}),
    };
}

KernelMapData build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
) {
    auto offsets = mlx_lattice::kernel_offsets(kernel_size);
    auto values = read_coords(coords);
    std::vector<Coord> out;
    std::vector<int32_t> maps;
    std::vector<int32_t> kernels;
    std::vector<int32_t> sizes(offsets.size(), 0);
    out.reserve(values.size() * offsets.size());
    maps.reserve(values.size() * offsets.size() * 2);
    kernels.reserve(values.size() * offsets.size());

    for (int in_row = 0; in_row < int(values.size()); ++in_row) {
        auto coord = values[in_row];
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            int out_row = int(out.size());
            out.push_back({
                coord[0],
                coord[1] * stride[0] + offset[0],
                coord[2] * stride[1] + offset[1],
                coord[3] * stride[2] + offset[2],
            });
            maps.push_back(in_row);
            maps.push_back(out_row);
            kernels.push_back(kernel);
            ++sizes[kernel];
        }
    }

    std::vector<int32_t> flat_offsets;
    flat_offsets.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat_offsets.insert(flat_offsets.end(), offset.begin(), offset.end());
    }

    return {
        make_i32_array(maps, mx::Shape{int(maps.size() / 2), 2}),
        make_i32_array(sizes, mx::Shape{int(sizes.size())}),
        make_i32_array(kernels, mx::Shape{int(kernels.size())}),
        make_i32_array({}, mx::Shape{0, 2}),
        make_i32_array({}, mx::Shape{0}),
        make_i32_array({0}, mx::Shape{1}),
        make_coords_array(out, coords.dtype()),
        make_i32_array(flat_offsets, mx::Shape{int(offsets.size()), 3}),
    };
}

} // namespace mlx_lattice::cpu
