// MARK: - relations

void write_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
    auto rows = first_row_map(values);
    bool identity_out = stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
    auto out_values = identity_out ? values : downsample_values(values, stride);

    std::vector<Edge> edges;
    edges.reserve(out_values.size() * offsets.size());
    for (int out_row = 0; out_row < int(out_values.size()); ++out_row) {
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            auto candidate = kernel_input_coord(
                out_values[out_row], offset, stride, padding
            );
            auto match = rows.find(candidate);
            if (match != rows.end()) {
                edges.push_back({
                    match->second,
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
    }

    write_map_rows(outputs, edges, int(out_values.size()));
    write_coords(outputs[RelationOutCoords], out_values, coords.dtype());
    write_count(
        outputs[RelationCounts], int(edges.size()), int(out_values.size())
    );
}

void write_target_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const mx::array& target_coords,
    int target_active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
    auto target_values = read_coords(target_coords);
    auto target_active =
        std::min(target_active_rows, int(target_values.size()));
    auto rows = first_row_map(values);

    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(target_active) * offsets.size());
    for (int out_row = 0; out_row < target_active; ++out_row) {
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto candidate = kernel_input_coord(
                target_values[out_row], offsets[kernel], stride, padding
            );
            auto match = rows.find(candidate);
            if (match != rows.end()) {
                edges.push_back({
                    match->second,
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
    }

    write_map_rows(outputs, edges, int(target_values.size()));
    write_count(outputs[RelationCounts], int(edges.size()), target_active);
}

void write_generative_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
    std::vector<Edge> edges;
    std::vector<Coord> out_values;
    edges.reserve(values.size() * offsets.size());
    out_values.reserve(values.size() * offsets.size());

    for (int in_row = 0; in_row < int(values.size()); ++in_row) {
        auto coord = values[in_row];
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            auto out_row = int(out_values.size());
            out_values.push_back({
                coord[0],
                coord[1] * stride[0] + offset[0],
                coord[2] * stride[1] + offset[1],
                coord[3] * stride[2] + offset[2],
            });
            edges.push_back({
                static_cast<int32_t>(in_row),
                static_cast<int32_t>(out_row),
                static_cast<int32_t>(kernel),
            });
        }
    }

    write_map_rows(outputs, edges, int(out_values.size()));
    write_coords(outputs[RelationOutCoords], out_values, coords.dtype());
    write_count(
        outputs[RelationCounts], int(edges.size()), int(out_values.size())
    );
}

void write_transposed_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding,
    bool direct
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));

    if (direct) {
        std::vector<Edge> edges;
        std::vector<Coord> out_values;
        edges.reserve(values.size() * offsets.size());
        out_values.reserve(values.size() * offsets.size());
        for (int in_row = 0; in_row < int(values.size()); ++in_row) {
            auto coord = values[in_row];
            for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
                auto out_row = in_row * int(offsets.size()) + kernel;
                out_values.push_back({
                    coord[0],
                    coord[1] * stride[0] + offsets[kernel][0] - padding[0],
                    coord[2] * stride[1] + offsets[kernel][1] - padding[1],
                    coord[3] * stride[2] + offsets[kernel][2] - padding[2],
                });
                edges.push_back({
                    static_cast<int32_t>(in_row),
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
        write_map_rows(outputs, edges, int(out_values.size()));
        write_coords(outputs[RelationOutCoords], out_values, coords.dtype());
        write_count(
            outputs[RelationCounts], int(edges.size()), int(out_values.size())
        );
        return;
    }

    std::vector<Edge> edges;
    std::vector<Coord> out_values;
    std::unordered_map<Coord, int32_t, CoordHash> out_rows;
    edges.reserve(values.size() * offsets.size());
    out_values.reserve(values.size() * offsets.size());
    out_rows.reserve(values.size() * offsets.size());

    for (int in_row = 0; in_row < int(values.size()); ++in_row) {
        auto coord = values[in_row];
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            Coord candidate = {
                coord[0],
                coord[1] * stride[0] + offset[0] - padding[0],
                coord[2] * stride[1] + offset[1] - padding[1],
                coord[3] * stride[2] + offset[2] - padding[2],
            };
            auto [match, inserted] = out_rows.emplace(
                candidate, static_cast<int32_t>(out_values.size())
            );
            if (inserted) {
                out_values.push_back(candidate);
            }
            edges.push_back({
                static_cast<int32_t>(in_row),
                match->second,
                static_cast<int32_t>(kernel),
            });
        }
    }

    write_map(outputs, edges, out_values, coords.dtype(), true);
}
