#include "features/coordinates/cpu/algorithm_details.h"

namespace mlx_lattice::coords::cpu {
void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, stride](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            std::vector<Coord> values;
            switch (op) {
            case CoordSetOp::Downsample:
                values = downsample_values(read_coords(task_inputs[0]), stride);
                break;
            case CoordSetOp::Union:
                values = union_values(task_inputs[0], task_inputs[1]);
                break;
            case CoordSetOp::Intersection:
                values = intersection_values(task_inputs[0], task_inputs[1]);
                break;
            }

            write_coords(task_outputs[0], values, task_inputs[0].dtype());
            write_count(task_outputs[1], int(values.size()));
        }
    );
}

void eval_lookup_coords(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            write_i32(
                task_outputs[0], lookup_values(task_inputs[0], task_inputs[1])
            );
        }
    );
}

void eval_morton_codes(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            write_i64(task_outputs[0], morton_code_values(task_inputs[0]));
        }
    );
}

} // namespace mlx_lattice::coords::cpu
