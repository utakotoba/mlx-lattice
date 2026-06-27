#include "features/coordinates/cpu/algorithm_details.h"

namespace mlx_lattice::coords::cpu {
void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, shape, radius_squared](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_neighbor_relation(
                task_outputs,
                op,
                task_inputs[0],
                read_scalar_i32(task_inputs[2]),
                task_inputs[1],
                read_scalar_i32(task_inputs[3]),
                shape,
                radius_squared
            );
        }
    );
}

} // namespace mlx_lattice::coords::cpu
