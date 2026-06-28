#include "features/convolution/cpu/algorithms.h"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "foundation/array_utils.h"
#include "mlx/types/half_types.h"
#include "platform/cpu/schedule.h"

namespace mlx_lattice::backend::cpu::conv {
namespace {

template <typename Fn> void parallel_rows(int rows, Fn fn) {
    auto hardware = std::max(1u, std::thread::hardware_concurrency());
    auto workers = std::min(rows, static_cast<int>(hardware));
    if (workers <= 1 || rows < 1024) {
        for (int row = 0; row < rows; ++row) {
            fn(row);
        }
        return;
    }
    auto block = (rows + workers - 1) / workers;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    for (int worker = 1; worker < workers; ++worker) {
        auto begin = worker * block;
        auto end = std::min(rows, begin + block);
        if (begin < end) {
            threads.emplace_back([begin, end, &fn]() {
                for (int row = begin; row < end; ++row) {
                    fn(row);
                }
            });
        }
    }
    for (int row = 0; row < std::min(rows, block); ++row) {
        fn(row);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

struct PackedWeightIndex {
    int weight_row;
    int channel;
    int bits;
    int out_channels;
    int out_channel;
};

inline std::uint32_t
unpack(const std::uint32_t* weights, const PackedWeightIndex& index) {
    auto bit = index.channel * index.bits;
    auto mask = (std::uint32_t{1} << index.bits) - 1;
    auto word = weights
        [(index.weight_row * index.out_channels) +
         (bit >> 5) * index.out_channels + index.out_channel];
    return (word >> (bit & 31)) & mask;
}

template <typename T>
void eval_typed(
    QuantizedSparseConvShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto* feats = inputs[0].data<T>();
    const auto* weights = inputs[1].data<std::uint32_t>();
    const auto* scales = inputs[2].data<T>();
    const auto* biases = inputs[3].data<T>();
    const auto* in_rows = inputs[4].data<std::int32_t>();
    const auto* kernel_ids = inputs[6].data<std::int32_t>();
    const auto* counts = inputs[7].data<std::int32_t>();
    const auto* row_offsets = inputs[8].data<std::int32_t>();
    auto* out = outputs[0].data<T>();

    auto out_count = std::min(counts[1], shape.out_capacity);
    auto edge_count = std::min(counts[0], static_cast<int>(inputs[4].shape(0)));
    auto packed_words = shape.storage_in_channels * shape.bits / 32;
    auto groups = shape.storage_in_channels / shape.group_size;
    auto feat_s0 = inputs[0].strides(0);
    auto feat_s1 = inputs[0].strides(1);
    parallel_rows(shape.out_capacity, [&](int out_row) {
        auto out_base = out_row * shape.out_channels;
        if (out_row >= out_count) {
            std::fill(
                out + out_base, out + out_base + shape.out_channels, T(0.0F)
            );
            return;
        }
        for (int co = 0; co < shape.out_channels; ++co) {
            float acc = 0.0F;
            for (int edge = row_offsets[out_row];
                 edge < row_offsets[out_row + 1];
                 ++edge) {
                if (edge < 0 || edge >= edge_count) {
                    continue;
                }
                auto in_row = in_rows[edge];
                auto kernel = kernel_ids[edge];
                if (in_row < 0 || kernel < 0 || kernel >= shape.n_kernels) {
                    continue;
                }
                auto weight_row = kernel * packed_words;
                auto quant_row = kernel * groups;
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    auto group = ci / shape.group_size;
                    auto value =
                        static_cast<float>(unpack(
                            weights,
                            {
                                .weight_row = weight_row,
                                .channel = ci,
                                .bits = shape.bits,
                                .out_channels = shape.out_channels,
                                .out_channel = co,
                            }
                        )) *
                            static_cast<float>(
                                scales
                                    [(quant_row + group) * shape.out_channels +
                                     co]
                            ) +
                        static_cast<float>(
                            biases
                                [(quant_row + group) * shape.out_channels + co]
                        );
                    acc += static_cast<float>(
                               feats[in_row * feat_s0 + ci * feat_s1]
                           ) *
                           value;
                }
            }
            out[out_base + co] = static_cast<T>(acc);
        }
    });
}

} // namespace

void eval_quantized(
    QuantizedSparseConvShape shape,
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
            if (ready[0].dtype() == mx::float32) {
                eval_typed<float>(shape, ready, task_outputs);
            } else {
                eval_typed<mx::float16_t>(shape, ready, task_outputs);
            }
        }
    );
}

} // namespace mlx_lattice::backend::cpu::conv
