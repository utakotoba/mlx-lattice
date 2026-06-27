#include "features/coordinates/cpu/algorithm_details.h"

namespace mlx_lattice::coords::cpu {
void eval_occupancy_downsample(
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
            auto coords = read_coords(task_inputs[0]);
            auto logical_rows =
                std::min(read_scalar_i32(task_inputs[1]), int(coords.size()));
            std::vector<Coord> out_coords;
            std::vector<int32_t> occupancy;
            std::unordered_map<Coord, int32_t, CoordHash> out_rows;
            out_coords.reserve(logical_rows);
            occupancy.reserve(logical_rows);
            out_rows.reserve(logical_rows);
            for (int row = 0; row < logical_rows; ++row) {
                const auto& coord = coords[row];
                Coord parent = {
                    coord[0],
                    floor_div(coord[1], 2),
                    floor_div(coord[2], 2),
                    floor_div(coord[3], 2),
                };
                auto [match, inserted] = out_rows.emplace(
                    parent, static_cast<int32_t>(out_coords.size())
                );
                if (inserted) {
                    out_coords.push_back(parent);
                    occupancy.push_back(0);
                }
                auto child = child_index_for_coord(coord);
                occupancy[match->second] |= child;
            }
            write_coords(task_outputs[0], out_coords, task_inputs[0].dtype());
            write_count(task_outputs[1], int(out_coords.size()));
            write_i32(task_outputs[2], occupancy);
        }
    );
}

void eval_occupancy_expand(
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
            auto coords = read_coords(task_inputs[0]);
            auto logical_rows =
                std::min(read_scalar_i32(task_inputs[1]), int(coords.size()));
            auto occupancy = task_inputs[2].data<int32_t>();
            std::vector<Coord> out_coords;
            std::vector<int32_t> parent_rows;
            std::vector<int32_t> child_indices;
            auto expanded = static_cast<size_t>(logical_rows) * 8;
            out_coords.reserve(expanded);
            parent_rows.reserve(expanded);
            child_indices.reserve(expanded);
            for (int row = 0; row < logical_rows; ++row) {
                const auto& parent = coords[row];
                auto bits = occupancy[row];
                for (int child = 0; child < 8; ++child) {
                    if ((bits & (1 << child)) == 0) {
                        continue;
                    }
                    Coord expanded = {
                        parent[0],
                        parent[1] * 2 + (child & 1),
                        parent[2] * 2 + ((child >> 1) & 1),
                        parent[3] * 2 + ((child >> 2) & 1),
                    };
                    parent_rows.push_back(row);
                    child_indices.push_back(child);
                    out_coords.push_back(expanded);
                }
            }
            write_coords(task_outputs[0], out_coords, task_inputs[0].dtype());
            write_count(task_outputs[1], int(out_coords.size()));
            write_i32(task_outputs[2], parent_rows);
            write_i32(task_outputs[3], child_indices);
        }
    );
}

void eval_child_coords_from_indices(
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
            auto coords = read_coords(task_inputs[0]);
            auto child_indices = task_inputs[1].data<int32_t>();
            std::vector<Coord> out_coords;
            out_coords.reserve(coords.size());
            for (int row = 0; row < int(coords.size()); ++row) {
                auto child = int(child_indices[row]);
                const auto& parent = coords[row];
                out_coords.push_back({
                    parent[0],
                    parent[1] * 2 + (child & 1),
                    parent[2] * 2 + ((child >> 1) & 1),
                    parent[3] * 2 + ((child >> 2) & 1),
                });
            }
            write_coords(task_outputs[0], out_coords, task_inputs[0].dtype());
        }
    );
}

} // namespace mlx_lattice::coords::cpu
