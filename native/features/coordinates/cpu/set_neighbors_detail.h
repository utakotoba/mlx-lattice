std::vector<Coord> union_values(const mx::array& lhs, const mx::array& rhs) {
    auto lhs_values = read_coords(lhs);
    auto rhs_values = read_coords(rhs);
    std::vector<Coord> out;
    out.reserve(lhs_values.size() + rhs_values.size());
    std::unordered_set<Coord, CoordHash> seen;
    seen.reserve(lhs_values.size() + rhs_values.size());

    for (auto coord : lhs_values) {
        if (seen.insert(coord).second) {
            out.push_back(coord);
        }
    }
    for (auto coord : rhs_values) {
        if (seen.insert(coord).second) {
            out.push_back(coord);
        }
    }
    return out;
}

std::vector<Coord>
intersection_values(const mx::array& lhs, const mx::array& rhs) {
    auto lhs_values = read_coords(lhs);
    auto rhs_values = read_coords(rhs);
    std::unordered_set<Coord, CoordHash> rhs_seen;
    rhs_seen.reserve(rhs_values.size());
    for (auto coord : rhs_values) {
        rhs_seen.insert(coord);
    }

    std::vector<Coord> out;
    out.reserve(std::min(lhs_values.size(), rhs_values.size()));
    std::unordered_set<Coord, CoordHash> emitted;
    emitted.reserve(lhs_values.size());
    for (auto coord : lhs_values) {
        if (rhs_seen.find(coord) != rhs_seen.end() &&
            emitted.insert(coord).second) {
            out.push_back(coord);
        }
    }
    return out;
}

std::vector<int32_t>
lookup_values(const mx::array& coords, const mx::array& queries) {
    auto rows = first_row_map(read_coords(coords));
    auto query_values = read_coords(queries);
    std::vector<int32_t> out;
    out.reserve(query_values.size());
    for (auto coord : query_values) {
        auto match = rows.find(coord);
        out.push_back(match == rows.end() ? -1 : match->second);
    }
    return out;
}

struct AlignmentRow {
    Coord coord;
    int32_t lhs_row;
    int32_t rhs_row;
};

std::vector<AlignmentRow>
alignment_values(SparseJoinOp join, SparseAlignmentInputs inputs) {
    auto lhs_values = read_coords(inputs.lhs_coords);
    auto rhs_values = read_coords(inputs.rhs_coords);
    auto lhs_count = std::min(inputs.lhs_active_rows, int(lhs_values.size()));
    auto rhs_count = std::min(inputs.rhs_active_rows, int(rhs_values.size()));
    lhs_values.resize(lhs_count);
    rhs_values.resize(rhs_count);

    auto lhs_rows = first_row_map(lhs_values);
    auto rhs_rows = first_row_map(rhs_values);
    std::vector<AlignmentRow> out;
    out.reserve(lhs_values.size() + rhs_values.size());

    if (join == SparseJoinOp::Right) {
        for (int rhs_row = 0; rhs_row < int(rhs_values.size()); ++rhs_row) {
            auto match = lhs_rows.find(rhs_values[rhs_row]);
            out.push_back({
                rhs_values[rhs_row],
                match == lhs_rows.end() ? -1 : match->second,
                static_cast<int32_t>(rhs_row),
            });
        }
        return out;
    }

    for (int lhs_row = 0; lhs_row < int(lhs_values.size()); ++lhs_row) {
        auto match = rhs_rows.find(lhs_values[lhs_row]);
        if (join == SparseJoinOp::Inner && match == rhs_rows.end()) {
            continue;
        }
        out.push_back({
            lhs_values[lhs_row],
            static_cast<int32_t>(lhs_row),
            match == rhs_rows.end() ? -1 : match->second,
        });
    }

    if (join != SparseJoinOp::Outer) {
        return out;
    }

    for (int rhs_row = 0; rhs_row < int(rhs_values.size()); ++rhs_row) {
        if (lhs_rows.find(rhs_values[rhs_row]) != lhs_rows.end()) {
            continue;
        }
        out.push_back({rhs_values[rhs_row], -1, static_cast<int32_t>(rhs_row)});
    }
    return out;
}

void write_sparse_alignment(
    std::vector<mx::array>& outputs,
    SparseJoinOp join,
    SparseAlignmentInputs inputs
) {
    auto rows = alignment_values(join, inputs);
    std::vector<Coord> coords;
    std::vector<int32_t> lhs_rows;
    std::vector<int32_t> rhs_rows;
    coords.reserve(rows.size());
    lhs_rows.reserve(rows.size());
    rhs_rows.reserve(rows.size());
    for (auto row : rows) {
        coords.push_back(row.coord);
        lhs_rows.push_back(row.lhs_row);
        rhs_rows.push_back(row.rhs_row);
    }
    write_coords(outputs[0], coords, inputs.lhs_coords.dtype());
    write_count(outputs[1], int(coords.size()));
    write_i32(outputs[2], lhs_rows, -1);
    write_i32(outputs[3], rhs_rows, -1);
}

bool same_batch(Coord lhs, Coord rhs) { return lhs[0] == rhs[0]; }

float squared_spatial_distance(Coord lhs, Coord rhs) {
    auto dx = static_cast<float>(lhs[1] - rhs[1]);
    auto dy = static_cast<float>(lhs[2] - rhs[2]);
    auto dz = static_cast<float>(lhs[3] - rhs[3]);
    return dx * dx + dy * dy + dz * dz;
}

bool closer_neighbor(
    const NeighborCandidate& lhs,
    const NeighborCandidate& rhs
) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.source_row < rhs.source_row;
}

std::vector<NeighborCandidate> radius_candidates(
    const std::unordered_map<Coord, int32_t, CoordHash>& source_rows_by_coord,
    Coord query,
    float radius_squared
) {
    auto radius = static_cast<int>(std::ceil(std::sqrt(radius_squared)));
    std::vector<NeighborCandidate> candidates;
    auto reserve_radius = static_cast<std::size_t>(radius);
    candidates.reserve(reserve_radius * reserve_radius * reserve_radius);
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                auto distance = static_cast<float>(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }
                auto target = Coord{
                    query[0],
                    query[1] + dx,
                    query[2] + dy,
                    query[3] + dz,
                };
                auto match = source_rows_by_coord.find(target);
                if (match != source_rows_by_coord.end()) {
                    candidates.push_back({match->second, distance});
                }
            }
        }
    }
    return candidates;
}

void write_neighbor_relation(
    std::vector<mx::array>& outputs,
    NeighborRelationOp op,
    const mx::array& source_coords,
    int source_active_rows,
    const mx::array& query_coords,
    int query_active_rows,
    NeighborRelationShape shape,
    float radius_squared
) {
    auto source_values = read_coords(source_coords);
    source_values.resize(
        std::min(source_active_rows, int(source_values.size()))
    );
    auto query_values = read_coords(query_coords);
    query_values.resize(std::min(query_active_rows, int(query_values.size())));

    std::vector<int32_t> query_rows;
    std::vector<int32_t> source_rows;
    std::vector<int32_t> neighbor_ids;
    std::vector<int32_t> row_offsets(query_values.size() + 1, 0);
    std::vector<float> distances;
    query_rows.reserve(query_values.size() * shape.max_neighbors);
    source_rows.reserve(query_values.size() * shape.max_neighbors);
    neighbor_ids.reserve(query_values.size() * shape.max_neighbors);
    distances.reserve(query_values.size() * shape.max_neighbors);

    std::unordered_map<Coord, int32_t, CoordHash> source_rows_by_coord;
    if (op == NeighborRelationOp::Radius) {
        source_rows_by_coord.reserve(source_values.size());
        for (int source_row = 0; source_row < int(source_values.size());
             ++source_row) {
            source_rows_by_coord.emplace(
                source_values[source_row], static_cast<int32_t>(source_row)
            );
        }
    }

    std::vector<NeighborCandidate> candidates;
    candidates.reserve(source_values.size());
    for (int query_row = 0; query_row < int(query_values.size()); ++query_row) {
        row_offsets[query_row] = static_cast<int32_t>(query_rows.size());
        candidates.clear();
        auto query = query_values[query_row];
        if (op == NeighborRelationOp::Radius) {
            candidates =
                radius_candidates(source_rows_by_coord, query, radius_squared);
        } else {
            for (int source_row = 0; source_row < int(source_values.size());
                 ++source_row) {
                auto source = source_values[source_row];
                if (!same_batch(query, source)) {
                    continue;
                }
                auto distance = squared_spatial_distance(query, source);
                candidates.push_back(
                    {static_cast<int32_t>(source_row), distance}
                );
            }
        }
        std::sort(candidates.begin(), candidates.end(), closer_neighbor);
        auto limit = std::min(shape.max_neighbors, int(candidates.size()));
        for (int neighbor = 0; neighbor < limit; ++neighbor) {
            query_rows.push_back(static_cast<int32_t>(query_row));
            source_rows.push_back(candidates[neighbor].source_row);
            neighbor_ids.push_back(static_cast<int32_t>(neighbor));
            distances.push_back(candidates[neighbor].distance);
        }
    }
    row_offsets[query_values.size()] = static_cast<int32_t>(query_rows.size());

    write_i32(outputs[NeighborQueryRows], query_rows);
    write_i32(outputs[NeighborSourceRows], source_rows);
    write_i32(outputs[NeighborIds], neighbor_ids);
    write_f32(outputs[NeighborDistances], distances);
    write_i32(outputs[NeighborRowOffsets], row_offsets);
    write_count(
        outputs[NeighborCounts],
        int(query_rows.size()),
        int(query_values.size())
    );
}
