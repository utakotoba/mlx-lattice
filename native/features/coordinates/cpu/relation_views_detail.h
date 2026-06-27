Coord kernel_input_coord(
    Coord out_coord,
    Triple offset,
    Triple stride,
    Triple padding
) {
    return {
        out_coord[0],
        out_coord[1] * stride[0] + offset[0] - padding[0],
        out_coord[2] * stride[1] + offset[1] - padding[1],
        out_coord[3] * stride[2] + offset[2] - padding[2],
    };
}

void write_map_rows(
    std::vector<mx::array>& outputs,
    const std::vector<Edge>& edges,
    int out_capacity
) {
    std::vector<int32_t> in_rows;
    std::vector<int32_t> out_rows;
    std::vector<int32_t> kernel_ids;
    std::vector<int32_t> row_offsets(
        static_cast<std::size_t>(out_capacity) + 1, 0
    );
    in_rows.reserve(edges.size());
    out_rows.reserve(edges.size());
    kernel_ids.reserve(edges.size());
    auto current_out = 0;
    for (auto edge : edges) {
        while (current_out <= edge[1] && current_out < out_capacity) {
            row_offsets[static_cast<std::size_t>(current_out)] =
                int32_t(in_rows.size());
            ++current_out;
        }
        in_rows.push_back(edge[0]);
        out_rows.push_back(edge[1]);
        kernel_ids.push_back(edge[2]);
    }
    while (current_out <= out_capacity) {
        row_offsets[static_cast<std::size_t>(current_out)] =
            int32_t(in_rows.size());
        ++current_out;
    }

    write_i32(outputs[RelationInRows], in_rows);
    write_i32(outputs[RelationOutRows], out_rows);
    write_i32(outputs[RelationKernelIds], kernel_ids);
    write_i32(outputs[RelationRowOffsets], row_offsets);
}

void write_map(
    std::vector<mx::array>& outputs,
    const std::vector<Edge>& edges,
    const std::vector<Coord>& out_coords,
    mx::Dtype coord_dtype,
    bool compact
) {
    auto row_major_edges = edges;
    std::stable_sort(
        row_major_edges.begin(),
        row_major_edges.end(),
        [](const Edge& lhs, const Edge& rhs) {
            if (lhs[1] != rhs[1]) {
                return lhs[1] < rhs[1];
            }
            if (lhs[2] != rhs[2]) {
                return lhs[2] < rhs[2];
            }
            return lhs[0] < rhs[0];
        }
    );
    write_map_rows(outputs, row_major_edges, int(out_coords.size()));
    write_coords(outputs[RelationOutCoords], out_coords, coord_dtype);
    if (compact) {
        write_count(
            outputs[RelationCounts],
            int(row_major_edges.size()),
            int(out_coords.size())
        );
    }
}

void write_relation_grouped_view(
    std::vector<mx::array>& outputs,
    const std::vector<mx::array>& inputs,
    RelationGroupedViewShape shape
) {
    auto edge_count =
        std::clamp(read_scalar_i32(inputs[1]), 0, shape.edge_capacity);
    auto group_ids = inputs[0].data<int32_t>();
    std::vector<int32_t> offsets(
        static_cast<std::size_t>(shape.group_count) + 1, 0
    );
    for (int edge = 0; edge < edge_count; ++edge) {
        auto group = group_ids[edge];
        if (group >= 0 && group < shape.group_count) {
            ++offsets[static_cast<std::size_t>(group) + 1];
        }
    }
    for (int group = 0; group < shape.group_count; ++group) {
        offsets[static_cast<std::size_t>(group) + 1] +=
            offsets[static_cast<std::size_t>(group)];
    }

    auto cursors = offsets;
    std::vector<int32_t> edge_ids(
        static_cast<std::size_t>(shape.edge_capacity), -1
    );
    for (int edge = 0; edge < edge_count; ++edge) {
        auto group = group_ids[edge];
        if (group >= 0 && group < shape.group_count) {
            auto slot = cursors[static_cast<std::size_t>(group)]++;
            edge_ids[static_cast<std::size_t>(slot)] = edge;
        }
    }

    write_i32(outputs[RelationViewRowOffsets], offsets);
    write_i32(outputs[RelationViewEdgeIds], edge_ids, -1);
}

void write_relation_implicit_gemm_view(
    std::vector<mx::array>& outputs,
    const std::vector<mx::array>& inputs,
    RelationImplicitGemmViewShape shape
) {
    auto source_values = read_coords(inputs[0]);
    auto source_active =
        std::clamp(read_scalar_i32(inputs[1]), 0, shape.source_rows);
    auto output_values = read_coords(inputs[2]);
    auto output_active =
        std::clamp(read_scalar_i32(inputs[3]), 0, shape.output_rows);
    auto offsets = read_offsets(inputs[4]);
    auto source_rows = first_row_map(source_values);

    std::vector<int32_t> out_in_map(
        static_cast<std::size_t>(shape.output_rows) * shape.kernel_count, -1
    );
    std::vector<int32_t> row_masks(
        static_cast<std::size_t>(shape.output_rows) * shape.mask_words, 0
    );

    for (int out_row = 0; out_row < output_active; ++out_row) {
        auto out_coord = output_values[static_cast<std::size_t>(out_row)];
        for (int kernel = 0; kernel < shape.kernel_count; ++kernel) {
            auto candidate = kernel_input_coord(
                out_coord,
                offsets[static_cast<std::size_t>(kernel)],
                shape.stride,
                shape.padding
            );
            auto found = source_rows.find(candidate);
            if (found == source_rows.end() || found->second >= source_active) {
                continue;
            }
            out_in_map
                [static_cast<std::size_t>(out_row) * shape.kernel_count +
                 kernel] = found->second;
            auto word = kernel / 32;
            auto bit = kernel % 32;
            row_masks
                [static_cast<std::size_t>(out_row) * shape.mask_words + word] |=
                int32_t(uint32_t(1) << bit);
        }
    }

    write_i32(outputs[RelationImplicitGemmOutInMap], out_in_map, -1);
    write_i32(outputs[RelationImplicitGemmRowMasks], row_masks);
}

void write_relation_direct_view(
    std::vector<mx::array>& outputs,
    const std::vector<mx::array>& inputs,
    RelationGroupedViewShape shape
) {
    auto edge_count =
        std::clamp(read_scalar_i32(inputs[1]), 0, shape.edge_capacity);
    auto group_ids = inputs[0].data<int32_t>();
    std::vector<int32_t> edge_ids(
        static_cast<std::size_t>(shape.group_count), -1
    );
    for (int edge = 0; edge < edge_count; ++edge) {
        auto group = group_ids[edge];
        if (group >= 0 && group < shape.group_count) {
            edge_ids[static_cast<std::size_t>(group)] = edge;
        }
    }
    write_i32(outputs[0], edge_ids, -1);
}
