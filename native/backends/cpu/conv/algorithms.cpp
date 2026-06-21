#include "backends/cpu/conv/algorithms.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"
#include "mlx/types/half_types.h"

namespace mlx_lattice::backend::cpu::conv {
namespace {

constexpr int kParallelGrain = 4096;

enum class WeightAccess : std::uint8_t {
    Generic,
    MappedContiguous,
    Dense5dContiguous,
};

template <typename T> void fill_zero(mx::array& out) {
    auto* data = out.data<T>();
    std::fill(data, data + out.size(), 0.0F);
}

template <typename Fn> void parallel_for(int begin, int end, Fn fn) {
    auto count = end - begin;
    auto hardware = std::thread::hardware_concurrency();
    if (count <= kParallelGrain || hardware <= 1) {
        for (int index = begin; index < end; ++index) {
            fn(index);
        }
        return;
    }

    auto workers = std::min<int>(
        static_cast<int>(hardware),
        std::max(1, (count + kParallelGrain - 1) / kParallelGrain)
    );
    auto block = (count + workers - 1) / workers;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    for (int worker = 1; worker < workers; ++worker) {
        auto first = begin + worker * block;
        auto last = std::min(end, first + block);
        if (first >= last) {
            continue;
        }
        threads.emplace_back([first, last, &fn]() {
            for (int index = first; index < last; ++index) {
                fn(index);
            }
        });
    }

    auto first_last = std::min(end, begin + block);
    for (int index = begin; index < first_last; ++index) {
        fn(index);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

bool is_mapped_contiguous(const mx::array& weights, SparseConvShape shape) {
    return shape.weight_layout == 0 && weights.ndim() == 3 &&
           weights.strides(2) == 1 &&
           weights.strides(1) == shape.out_channels &&
           weights.strides(0) ==
               static_cast<std::ptrdiff_t>(shape.in_channels) *
                   shape.out_channels;
}

bool is_dense_5d_contiguous(const mx::array& weights, SparseConvShape shape) {
    return shape.weight_layout == 1 && weights.ndim() == 5 &&
           weights.strides(4) == 1 && weights.strides(3) == shape.in_channels &&
           weights.strides(2) == static_cast<std::ptrdiff_t>(shape.kernel_z) *
                                     shape.in_channels &&
           weights.strides(1) == static_cast<std::ptrdiff_t>(shape.kernel_y) *
                                     shape.kernel_z * shape.in_channels &&
           weights.strides(0) == static_cast<std::ptrdiff_t>(shape.kernel_x) *
                                     shape.kernel_y * shape.kernel_z *
                                     shape.in_channels;
}

WeightAccess weight_access(const mx::array& weights, SparseConvShape shape) {
    if (is_mapped_contiguous(weights, shape)) {
        return WeightAccess::MappedContiguous;
    }
    if (is_dense_5d_contiguous(weights, shape)) {
        return WeightAccess::Dense5dContiguous;
    }
    return WeightAccess::Generic;
}

std::ptrdiff_t weight_offset(
    const mx::array& weights,
    const SparseConvShape& shape,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return static_cast<std::ptrdiff_t>(kernel) * weights.strides(0) +
               static_cast<std::ptrdiff_t>(in_channel) * weights.strides(1) +
               static_cast<std::ptrdiff_t>(out_channel) * weights.strides(2);
    }

    auto xy = shape.kernel_y * shape.kernel_z;
    auto kx = kernel / xy;
    auto rem = kernel % xy;
    auto ky = rem / shape.kernel_z;
    auto kz = rem % shape.kernel_z;
    return static_cast<std::ptrdiff_t>(out_channel) * weights.strides(0) +
           static_cast<std::ptrdiff_t>(kx) * weights.strides(1) +
           static_cast<std::ptrdiff_t>(ky) * weights.strides(2) +
           static_cast<std::ptrdiff_t>(kz) * weights.strides(3) +
           static_cast<std::ptrdiff_t>(in_channel) * weights.strides(4);
}

std::ptrdiff_t dense_weight_offset(
    const SparseConvShape& shape,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return (static_cast<std::ptrdiff_t>(kernel) * shape.in_channels +
                in_channel) *
                   shape.out_channels +
               out_channel;
    }

    auto xy = shape.kernel_y * shape.kernel_z;
    auto kx = kernel / xy;
    auto rem = kernel % xy;
    auto ky = rem / shape.kernel_z;
    auto kz = rem % shape.kernel_z;
    return (((static_cast<std::ptrdiff_t>(out_channel) * shape.kernel_x + kx) *
                 shape.kernel_y +
             ky) *
                shape.kernel_z +
            kz) *
               shape.in_channels +
           in_channel;
}

template <WeightAccess Access>
std::ptrdiff_t fast_weight_offset(
    const mx::array& weights,
    const SparseConvShape& shape,
    int kernel,
    int in_channel,
    int out_channel
) {
    if constexpr (Access == WeightAccess::MappedContiguous) {
        return (static_cast<std::ptrdiff_t>(kernel) * shape.in_channels +
                in_channel) *
                   shape.out_channels +
               out_channel;
    } else if constexpr (Access == WeightAccess::Dense5dContiguous) {
        return dense_weight_offset(shape, kernel, in_channel, out_channel);
    } else {
        return weight_offset(weights, shape, kernel, in_channel, out_channel);
    }
}

int edge_count(const mx::array& rows, const mx::array& counts) {
    return std::min(counts.data<int32_t>()[0], static_cast<int>(rows.shape(0)));
}

int out_count(const mx::array& counts, SparseConvShape shape) {
    return std::min(counts.data<int32_t>()[1], shape.out_capacity);
}

template <typename T, WeightAccess Access>
void eval_typed(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    const auto& feats = ready[0];
    const auto& weights = ready[1];
    const auto& in_rows = ready[2];
    const auto& out_rows = ready[3];
    const auto& kernel_ids = ready[4];
    const auto& row_offsets = ready[6];

    auto& out = task_outputs[0];
    auto* out_data = out.data<T>();
    const auto* feat_data = feats.data<T>();
    const auto* weight_data = weights.data<T>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();
    const auto* offset_data = row_offsets.data<int32_t>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);
    const auto edge_total = edge_count(in_rows, ready[5]);
    const auto active_out = out_count(ready[5], shape);

    (void)out_rows;
    parallel_for(0, shape.out_capacity, [&](int out_row) {
        auto* out_row_data = out_data + static_cast<std::ptrdiff_t>(out_row) *
                                            shape.out_channels;
        std::fill(out_row_data, out_row_data + shape.out_channels, T(0.0F));
        if (out_row >= active_out) {
            return;
        }
        for (int edge = offset_data[out_row]; edge < offset_data[out_row + 1];
             ++edge) {
            if (edge < 0 || edge >= edge_total) {
                continue;
            }
            auto in_row = in_data[edge];
            auto kernel = kernel_data[edge];
            if (in_row < 0 || kernel < 0) {
                continue;
            }
            for (int ci = 0; ci < shape.in_channels; ++ci) {
                auto value = static_cast<float>(
                    feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(ci) * feat_s1]
                );
                for (int co = 0; co < shape.out_channels; ++co) {
                    auto acc = static_cast<float>(out_row_data[co]);
                    acc += value *
                           static_cast<float>(weight_data[fast_weight_offset<
                               Access>(weights, shape, kernel, ci, co)]);
                    out_row_data[co] = T(acc);
                }
            }
        }
    });
}

template <typename T, WeightAccess Access, int InChannels, int OutChannels>
void eval_dense_channels_typed(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    const auto& feats = ready[0];
    const auto& weights = ready[1];
    const auto& in_rows = ready[2];
    const auto& out_rows = ready[3];
    const auto& kernel_ids = ready[4];
    const auto& row_offsets = ready[6];

    auto& out = task_outputs[0];
    auto* out_data = out.data<T>();
    const auto* feat_data = feats.data<T>();
    const auto* weight_data = weights.data<T>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();
    const auto* offset_data = row_offsets.data<int32_t>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);
    const auto edge_total = edge_count(in_rows, ready[5]);
    const auto active_out = out_count(ready[5], shape);

    (void)out_rows;
    parallel_for(0, shape.out_capacity, [&](int out_row) {
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * OutChannels;
        std::array<float, OutChannels> acc{};
        if (out_row < active_out) {
            for (int edge = offset_data[out_row];
                 edge < offset_data[out_row + 1];
                 ++edge) {
                if (edge < 0 || edge >= edge_total) {
                    continue;
                }
                auto in_row = in_data[edge];
                auto kernel = kernel_data[edge];
                if (in_row < 0 || kernel < 0) {
                    continue;
                }
                for (int ci = 0; ci < InChannels; ++ci) {
                    auto value = static_cast<float>(
                        feat_data
                            [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                             static_cast<std::ptrdiff_t>(ci) * feat_s1]
                    );
                    for (int co = 0; co < OutChannels; ++co) {
                        acc[co] +=
                            value *
                            static_cast<float>(weight_data[fast_weight_offset<
                                Access>(weights, shape, kernel, ci, co)]);
                    }
                }
            }
        }
        for (int co = 0; co < OutChannels; ++co) {
            out_row_data[co] = T(acc[co]);
        }
    });
}

template <typename T, WeightAccess Access>
void eval_sorted_implicit_gemm_typed(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    const auto& feats = ready[0];
    const auto& weights = ready[1];
    const auto& sorted_out_in_map = ready[2];
    const auto& reorder_rows = ready[4];

    auto& out = task_outputs[0];
    auto* out_data = out.data<T>();
    const auto* feat_data = feats.data<T>();
    const auto* weight_data = weights.data<T>();
    const auto* map_data = sorted_out_in_map.data<int32_t>();
    const auto* reorder_data = reorder_rows.data<int32_t>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);

    parallel_for(0, shape.out_capacity, [&](int sorted_row) {
        auto out_row =
            shape.store_sorted != 0 ? sorted_row : reorder_data[sorted_row];
        auto* out_row_data = out_data + static_cast<std::ptrdiff_t>(out_row) *
                                            shape.out_channels;
        std::fill(out_row_data, out_row_data + shape.out_channels, T(0.0F));
        for (int kernel = 0; kernel < shape.n_kernels; ++kernel) {
            auto in_row = map_data
                [static_cast<std::ptrdiff_t>(sorted_row) * shape.n_kernels +
                 kernel];
            if (in_row < 0 || in_row >= shape.in_capacity) {
                continue;
            }
            for (int ci = 0; ci < shape.in_channels; ++ci) {
                auto value = static_cast<float>(
                    feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(ci) * feat_s1]
                );
                for (int co = 0; co < shape.out_channels; ++co) {
                    auto acc = static_cast<float>(out_row_data[co]);
                    acc += value *
                           static_cast<float>(weight_data[fast_weight_offset<
                               Access>(weights, shape, kernel, ci, co)]);
                    out_row_data[co] = T(acc);
                }
            }
        }
    });
}

template <typename T, WeightAccess Access>
void eval_input_grad_typed(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    const auto& cotangent = ready[0];
    const auto& weights = ready[1];
    const auto& in_rows = ready[2];
    const auto& out_rows = ready[3];
    const auto& kernel_ids = ready[4];
    const auto& in_row_offsets = ready[7];
    const auto& in_edge_ids = ready[8];

    auto& grad = task_outputs[0];
    fill_zero<T>(grad);
    auto* grad_data = grad.data<T>();
    const auto* cotangent_data = cotangent.data<T>();
    const auto* weight_data = weights.data<T>();
    const auto* out_data = out_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();
    const auto* offset_data = in_row_offsets.data<int32_t>();
    const auto* edge_data = in_edge_ids.data<int32_t>();
    const auto cotangent_s0 = cotangent.strides(0);
    const auto cotangent_s1 = cotangent.strides(1);
    const auto edge_total = edge_count(in_rows, ready[5]);

    (void)in_rows;
    parallel_for(0, shape.in_capacity, [&](int in_row) {
        auto* grad_row =
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
        for (int ci = 0; ci < shape.in_channels; ++ci) {
            auto acc = 0.0F;
            for (int cursor = offset_data[in_row];
                 cursor < offset_data[in_row + 1];
                 ++cursor) {
                auto edge = edge_data[cursor];
                if (edge < 0 || edge >= edge_total) {
                    continue;
                }
                auto out_row = out_data[edge];
                auto kernel = kernel_data[edge];
                if (out_row < 0 || kernel < 0 ||
                    out_row >= shape.out_capacity) {
                    continue;
                }
                for (int co = 0; co < shape.out_channels; ++co) {
                    acc +=
                        static_cast<float>(
                            cotangent_data
                                [static_cast<std::ptrdiff_t>(out_row) *
                                     cotangent_s0 +
                                 static_cast<std::ptrdiff_t>(co) * cotangent_s1]
                        ) *
                        static_cast<float>(weight_data[fast_weight_offset<
                            Access>(weights, shape, kernel, ci, co)]);
                }
            }
            grad_row[ci] = T(acc);
        }
    });
}

template <typename T, WeightAccess Access>
void eval_weight_grad_typed(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    const auto& feats = ready[0];
    const auto& cotangent = ready[1];
    const auto& in_rows = ready[2];
    const auto& out_rows = ready[3];
    const auto& kernel_ids = ready[4];
    const auto& kernel_row_offsets = ready[7];
    const auto& kernel_edge_ids = ready[8];

    auto& grad = task_outputs[0];
    fill_zero<T>(grad);
    auto* grad_data = grad.data<T>();
    const auto* feat_data = feats.data<T>();
    const auto* cotangent_data = cotangent.data<T>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_data = out_rows.data<int32_t>();
    const auto* offset_data = kernel_row_offsets.data<int32_t>();
    const auto* edge_data = kernel_edge_ids.data<int32_t>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);
    const auto cotangent_s0 = cotangent.strides(0);
    const auto cotangent_s1 = cotangent.strides(1);
    const auto edge_total = edge_count(in_rows, ready[5]);

    (void)kernel_ids;
    auto total = shape.n_kernels * shape.in_channels * shape.out_channels;
    parallel_for(0, total, [&](int index) {
        auto co = index % shape.out_channels;
        auto ci = (index / shape.out_channels) % shape.in_channels;
        auto kernel = index / (shape.in_channels * shape.out_channels);
        auto acc = 0.0F;
        for (int cursor = offset_data[kernel]; cursor < offset_data[kernel + 1];
             ++cursor) {
            auto edge = edge_data[cursor];
            if (edge < 0 || edge >= edge_total) {
                continue;
            }
            auto in_row = in_data[edge];
            auto out_row = out_data[edge];
            if (in_row < 0 || out_row < 0 || out_row >= shape.out_capacity) {
                continue;
            }
            auto feat_value = static_cast<float>(
                feat_data
                    [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                     static_cast<std::ptrdiff_t>(ci) * feat_s1]
            );
            acc +=
                feat_value *
                static_cast<float>(
                    cotangent_data
                        [static_cast<std::ptrdiff_t>(out_row) * cotangent_s0 +
                         static_cast<std::ptrdiff_t>(co) * cotangent_s1]
                );
        }
        grad_data[fast_weight_offset<Access>(grad, shape, kernel, ci, co)] =
            T(acc);
    });
}

template <typename T, WeightAccess Access>
void eval_route(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    if (shape.in_channels == 16 && shape.out_channels == 16) {
        eval_dense_channels_typed<T, Access, 16, 16>(
            shape, ready, task_outputs
        );
        return;
    }
    if (shape.in_channels == 32 && shape.out_channels == 32) {
        eval_dense_channels_typed<T, Access, 32, 32>(
            shape, ready, task_outputs
        );
        return;
    }
    if (shape.in_channels == 64 && shape.out_channels == 64) {
        eval_dense_channels_typed<T, Access, 64, 64>(
            shape, ready, task_outputs
        );
        return;
    }
    eval_typed<T, Access>(shape, ready, task_outputs);
}

template <typename T, WeightAccess Access>
void eval_input_grad_route(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    eval_input_grad_typed<T, Access>(shape, ready, task_outputs);
}

template <typename T, WeightAccess Access>
void eval_weight_grad_route(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    eval_weight_grad_typed<T, Access>(shape, ready, task_outputs);
}

void eval_dispatch(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    if (ready[0].dtype() == mx::float32) {
        switch (weight_access(ready[1], shape)) {
        case WeightAccess::MappedContiguous:
            eval_route<float, WeightAccess::MappedContiguous>(
                shape, ready, task_outputs
            );
            return;
        case WeightAccess::Dense5dContiguous:
            eval_route<float, WeightAccess::Dense5dContiguous>(
                shape, ready, task_outputs
            );
            return;
        case WeightAccess::Generic:
            eval_route<float, WeightAccess::Generic>(
                shape, ready, task_outputs
            );
            return;
        }
    }

    switch (weight_access(ready[1], shape)) {
    case WeightAccess::MappedContiguous:
        eval_route<mx::float16_t, WeightAccess::MappedContiguous>(
            shape, ready, task_outputs
        );
        return;
    case WeightAccess::Dense5dContiguous:
        eval_route<mx::float16_t, WeightAccess::Dense5dContiguous>(
            shape, ready, task_outputs
        );
        return;
    case WeightAccess::Generic:
        eval_route<mx::float16_t, WeightAccess::Generic>(
            shape, ready, task_outputs
        );
        return;
    }
}

void eval_sorted_implicit_gemm_dispatch(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    if (ready[0].dtype() == mx::float32) {
        switch (weight_access(ready[1], shape)) {
        case WeightAccess::MappedContiguous:
            eval_sorted_implicit_gemm_typed<
                float,
                WeightAccess::MappedContiguous>(shape, ready, task_outputs);
            return;
        case WeightAccess::Dense5dContiguous:
            eval_sorted_implicit_gemm_typed<
                float,
                WeightAccess::Dense5dContiguous>(shape, ready, task_outputs);
            return;
        case WeightAccess::Generic:
            eval_sorted_implicit_gemm_typed<float, WeightAccess::Generic>(
                shape, ready, task_outputs
            );
            return;
        }
    }

    switch (weight_access(ready[1], shape)) {
    case WeightAccess::MappedContiguous:
        eval_sorted_implicit_gemm_typed<
            mx::float16_t,
            WeightAccess::MappedContiguous>(shape, ready, task_outputs);
        return;
    case WeightAccess::Dense5dContiguous:
        eval_sorted_implicit_gemm_typed<
            mx::float16_t,
            WeightAccess::Dense5dContiguous>(shape, ready, task_outputs);
        return;
    case WeightAccess::Generic:
        eval_sorted_implicit_gemm_typed<mx::float16_t, WeightAccess::Generic>(
            shape, ready, task_outputs
        );
        return;
    }
}

void eval_input_grad_dispatch(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    if (ready[0].dtype() == mx::float32) {
        switch (weight_access(ready[1], shape)) {
        case WeightAccess::MappedContiguous:
            eval_input_grad_route<float, WeightAccess::MappedContiguous>(
                shape, ready, task_outputs
            );
            return;
        case WeightAccess::Dense5dContiguous:
            eval_input_grad_route<float, WeightAccess::Dense5dContiguous>(
                shape, ready, task_outputs
            );
            return;
        case WeightAccess::Generic:
            eval_input_grad_route<float, WeightAccess::Generic>(
                shape, ready, task_outputs
            );
            return;
        }
    }

    switch (weight_access(ready[1], shape)) {
    case WeightAccess::MappedContiguous:
        eval_input_grad_route<mx::float16_t, WeightAccess::MappedContiguous>(
            shape, ready, task_outputs
        );
        return;
    case WeightAccess::Dense5dContiguous:
        eval_input_grad_route<mx::float16_t, WeightAccess::Dense5dContiguous>(
            shape, ready, task_outputs
        );
        return;
    case WeightAccess::Generic:
        eval_input_grad_route<mx::float16_t, WeightAccess::Generic>(
            shape, ready, task_outputs
        );
        return;
    }
}

void eval_weight_grad_dispatch(
    SparseConvShape shape,
    const std::vector<mx::array>& ready,
    std::vector<mx::array>& task_outputs
) {
    if (ready[0].dtype() == mx::float32) {
        switch (weight_access(ready[1], shape)) {
        case WeightAccess::MappedContiguous:
            eval_weight_grad_route<float, WeightAccess::MappedContiguous>(
                shape, ready, task_outputs
            );
            return;
        case WeightAccess::Dense5dContiguous:
            eval_weight_grad_route<float, WeightAccess::Dense5dContiguous>(
                shape, ready, task_outputs
            );
            return;
        case WeightAccess::Generic:
            eval_weight_grad_route<float, WeightAccess::Generic>(
                shape, ready, task_outputs
            );
            return;
        }
    }

    switch (weight_access(ready[1], shape)) {
    case WeightAccess::MappedContiguous:
        eval_weight_grad_route<mx::float16_t, WeightAccess::MappedContiguous>(
            shape, ready, task_outputs
        );
        return;
    case WeightAccess::Dense5dContiguous:
        eval_weight_grad_route<mx::float16_t, WeightAccess::Dense5dContiguous>(
            shape, ready, task_outputs
        );
        return;
    case WeightAccess::Generic:
        eval_weight_grad_route<mx::float16_t, WeightAccess::Generic>(
            shape, ready, task_outputs
        );
        return;
    }
}

} // namespace

void eval(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) { eval_dispatch(shape, ready, task_outputs); }
    );
}

void eval_sorted_implicit_gemm(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) { eval_sorted_implicit_gemm_dispatch(shape, ready, task_outputs); }
    );
}

void eval_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) { eval_input_grad_dispatch(shape, ready, task_outputs); }
    );
}

void eval_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) { eval_weight_grad_dispatch(shape, ready, task_outputs); }
    );
}

} // namespace mlx_lattice::backend::cpu::conv
