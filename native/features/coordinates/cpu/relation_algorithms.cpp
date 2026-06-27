#include "features/coordinates/cpu/algorithm_details.h"

namespace mlx_lattice::coords::cpu {
void eval_generic_kernel_relation(
    CoordRelationOp op,
    Triple stride,
    Triple padding,
    bool direct,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, stride, padding, direct](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            auto offsets = read_offsets(task_inputs[1]);
            auto active_rows = read_scalar_i32(task_inputs[2]);

            switch (op) {
            case CoordRelationOp::Forward:
                write_kernel_relation(
                    task_outputs,
                    task_inputs[0],
                    active_rows,
                    offsets,
                    stride,
                    padding
                );
                break;
            case CoordRelationOp::Transposed:
                write_transposed_kernel_relation(
                    task_outputs,
                    task_inputs[0],
                    active_rows,
                    offsets,
                    stride,
                    padding,
                    direct
                );
                break;
            }
        }
    );
}

void eval_target_kernel_relation(
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [stride, padding](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_target_kernel_relation(
                task_outputs,
                task_inputs[0],
                read_scalar_i32(task_inputs[2]),
                task_inputs[3],
                read_scalar_i32(task_inputs[4]),
                read_offsets(task_inputs[1]),
                stride,
                padding
            );
        }
    );
}

void eval_generative_kernel_relation(
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
        [stride](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            auto active_rows = read_scalar_i32(task_inputs[2]);
            write_generative_relation(
                task_outputs,
                task_inputs[0],
                active_rows,
                read_offsets(task_inputs[1]),
                stride
            );
        }
    );
}

void eval_relation_grouped_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) { write_relation_grouped_view(task_outputs, task_inputs, shape); }
    );
}

void eval_relation_direct_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) { write_relation_direct_view(task_outputs, task_inputs, shape); }
    );
}

void eval_relation_implicit_gemm_view(
    RelationImplicitGemmViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_relation_implicit_gemm_view(task_outputs, task_inputs, shape);
        }
    );
}

} // namespace mlx_lattice::coords::cpu
