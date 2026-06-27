#pragma once

#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/stream.h"

namespace mlx_lattice::backend {

namespace mx = mlx::core;

template <typename Fn>
void schedule_cpu(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    Fn fn
) {
    auto& encoder = mx::cpu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    for (auto& output : outputs) {
        encoder.set_output_array(output);
    }
    encoder.dispatch([task_inputs = inputs,
                      task_outputs = outputs,
                      fn = std::move(fn)]() mutable {
        fn(task_inputs, task_outputs);
    });
}

} // namespace mlx_lattice::backend
