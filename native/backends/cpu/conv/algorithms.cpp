#include "backends/cpu/conv/algorithms.h"

#include <algorithm>
#include <cstddef>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"

namespace mlx_lattice::backend::cpu::conv {
namespace {

void fill_zero(mx::array& out) {
    auto* data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
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

int edge_count(const mx::array& rows, const mx::array& counts) {
    return std::min(counts.data<int32_t>()[0], static_cast<int>(rows.shape(0)));
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
        ) {
            const auto& feats = ready[0];
            const auto& weights = ready[1];
            const auto& in_rows = ready[2];
            const auto& out_rows = ready[3];
            const auto& kernel_ids = ready[4];
            const auto& row_offsets = ready[6];

            auto& out = task_outputs[0];
            auto* out_data = out.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* weight_data = weights.data<float>();
            const auto* in_data = in_rows.data<int32_t>();
            const auto* kernel_data = kernel_ids.data<int32_t>();
            const auto* offset_data = row_offsets.data<int32_t>();
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            const auto edge_total = edge_count(in_rows, ready[5]);
            const auto out_count =
                std::min(ready[5].data<int32_t>()[1], shape.out_capacity);

            (void)out_rows;
            for (int out_row = 0; out_row < shape.out_capacity; ++out_row) {
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.out_channels;
                std::fill(
                    out_row_data, out_row_data + shape.out_channels, 0.0F
                );
                if (out_row >= out_count) {
                    continue;
                }
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
                    for (int ci = 0; ci < shape.in_channels; ++ci) {
                        auto value = feat_data
                            [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                             static_cast<std::ptrdiff_t>(ci) * feat_s1];
                        for (int co = 0; co < shape.out_channels; ++co) {
                            out_row_data[co] +=
                                value * weight_data[weight_offset(
                                            weights, shape, kernel, ci, co
                                        )];
                        }
                    }
                }
            }
        }
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
        ) {
            const auto& cotangent = ready[0];
            const auto& weights = ready[1];
            const auto& in_rows = ready[2];
            const auto& out_rows = ready[3];
            const auto& kernel_ids = ready[4];
            const auto& in_row_offsets = ready[7];
            const auto& in_edge_ids = ready[8];

            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* weight_data = weights.data<float>();
            const auto* out_data = out_rows.data<int32_t>();
            const auto* kernel_data = kernel_ids.data<int32_t>();
            const auto* offset_data = in_row_offsets.data<int32_t>();
            const auto* edge_data = in_edge_ids.data<int32_t>();
            const auto cotangent_s0 = cotangent.strides(0);
            const auto cotangent_s1 = cotangent.strides(1);
            const auto edge_total = edge_count(in_rows, ready[5]);

            (void)in_rows;
            for (int in_row = 0; in_row < shape.in_capacity; ++in_row) {
                auto* grad_row =
                    grad_data +
                    static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
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
                            acc += cotangent_data
                                       [static_cast<std::ptrdiff_t>(out_row) *
                                            cotangent_s0 +
                                        static_cast<std::ptrdiff_t>(co) *
                                            cotangent_s1] *
                                   weight_data[weight_offset(
                                       weights, shape, kernel, ci, co
                                   )];
                        }
                    }
                    grad_row[ci] = acc;
                }
            }
        }
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
        ) {
            const auto& feats = ready[0];
            const auto& cotangent = ready[1];
            const auto& in_rows = ready[2];
            const auto& out_rows = ready[3];
            const auto& kernel_ids = ready[4];
            const auto& kernel_row_offsets = ready[7];
            const auto& kernel_edge_ids = ready[8];

            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
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
            for (int kernel = 0; kernel < shape.n_kernels; ++kernel) {
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    for (int co = 0; co < shape.out_channels; ++co) {
                        auto acc = 0.0F;
                        for (int cursor = offset_data[kernel];
                             cursor < offset_data[kernel + 1];
                             ++cursor) {
                            auto edge = edge_data[cursor];
                            if (edge < 0 || edge >= edge_total) {
                                continue;
                            }
                            auto in_row = in_data[edge];
                            auto out_row = out_data[edge];
                            if (in_row < 0 || out_row < 0 ||
                                out_row >= shape.out_capacity) {
                                continue;
                            }
                            auto feat_value = feat_data
                                [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                                 static_cast<std::ptrdiff_t>(ci) * feat_s1];
                            acc += feat_value *
                                   cotangent_data
                                       [static_cast<std::ptrdiff_t>(out_row) *
                                            cotangent_s0 +
                                        static_cast<std::ptrdiff_t>(co) *
                                            cotangent_s1];
                        }
                        grad_data[dense_weight_offset(shape, kernel, ci, co)] =
                            acc;
                    }
                }
            }
        }
    );
}

} // namespace mlx_lattice::backend::cpu::conv
