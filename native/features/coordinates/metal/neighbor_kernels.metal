#include <metal_stdlib>

using namespace metal;

#include "native/features/coordinates/metal/common.metal"

// MARK: - generative relations

[[kernel]] void build_neighbor_relation_i32(
    device const int* query_active_rows [[buffer(0)]],
    device int* query_rows [[buffer(1)]],
    device int* source_rows [[buffer(2)]],
    device int* neighbor_ids [[buffer(3)]],
    device float* distances [[buffer(4)]],
    device int* row_offsets [[buffer(5)]],
    device int* counts [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_capacity = query_capacity * max_neighbors;
    int query_count = min(query_active_rows[0], query_capacity);

    if (elem == 0) {
        counts[0] = 0;
        counts[1] = query_count;
    }

    if (elem <= uint(query_capacity)) {
        row_offsets[elem] = 0;
    }

    if (elem >= uint(edge_capacity)) {
        return;
    }

    query_rows[elem] = 0;
    source_rows[elem] = -1;
    neighbor_ids[elem] = 0;
    distances[elem] = 0.0f;
}

[[kernel]] void fill_neighbor_relation_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device int* query_rows [[buffer(4)]],
    device int* source_rows [[buffer(5)]],
    device int* neighbor_ids [[buffer(6)]],
    device float* distances [[buffer(7)]],
    constant const int& op [[buffer(8)]],
    constant const int& source_capacity [[buffer(9)]],
    constant const int& query_capacity [[buffer(10)]],
    constant const int& max_neighbors [[buffer(11)]],
    constant const float& radius_squared [[buffer(12)]],
    uint query_row [[thread_position_in_grid]]
) {
    int source_count = min(source_active_rows[0], source_capacity);
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int slot_start = int(query_row) * max_neighbors;
    int selected = 0;
    for (int source_row = 0; source_row < source_count; ++source_row) {
        if (!same_batch(
                query_coords, int(query_row), source_coords, source_row
            )) {
            continue;
        }
        float distance = squared_spatial_distance(
            query_coords, int(query_row), source_coords, source_row
        );
        if (op == 1 && distance > radius_squared) {
            continue;
        }

        int insert_at = selected;
        for (int rank = 0; rank < selected; ++rank) {
            int index = slot_start + rank;
            float existing_distance = distances[index];
            int existing_source = source_rows[index];
            if (distance < existing_distance ||
                (distance == existing_distance &&
                 source_row < existing_source)) {
                insert_at = rank;
                break;
            }
        }
        if (insert_at >= max_neighbors) {
            continue;
        }

        int last = min(selected, max_neighbors - 1);
        for (int rank = last; rank > insert_at; --rank) {
            int dst = slot_start + rank;
            int src = dst - 1;
            source_rows[dst] = source_rows[src];
            distances[dst] = distances[src];
        }
        source_rows[slot_start + insert_at] = source_row;
        distances[slot_start + insert_at] = distance;
        selected = min(selected + 1, max_neighbors);
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void count_radius_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* row_counts [[buffer(5)]],
    constant const int& source_capacity [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    constant const float& radius_squared [[buffer(9)]],
    constant const int& ceil_radius [[buffer(10)]],
    constant const int& table_capacity [[buffer(11)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_capacity)) {
        return;
    }
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        row_counts[query_row] = 0;
        return;
    }

    int query_base = int(query_row) * 4;
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }
                selected += 1;
                if (selected >= max_neighbors) {
                    row_counts[query_row] = max_neighbors;
                    return;
                }
            }
        }
    }
    row_counts[query_row] = selected;
}

[[kernel]] void prefix_neighbor_row_offsets_i32(
    device int* row_offsets [[buffer(0)]],
    device int* counts [[buffer(1)]],
    constant const int& query_capacity [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int query_count = counts[1];
    int total = 0;
    for (int query_row = 0; query_row < query_count; ++query_row) {
        int row_count = row_offsets[query_row];
        row_offsets[query_row] = total;
        total += row_count;
    }
    row_offsets[query_count] = total;
    counts[0] = total;
    (void)query_capacity;
}

[[kernel]] void fill_radius_relation_compact_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device const int* row_offsets [[buffer(5)]],
    device int* query_rows [[buffer(6)]],
    device int* source_rows [[buffer(7)]],
    device int* neighbor_ids [[buffer(8)]],
    device float* distances [[buffer(9)]],
    constant const int& source_capacity [[buffer(10)]],
    constant const int& query_capacity [[buffer(11)]],
    constant const int& max_neighbors [[buffer(12)]],
    constant const float& radius_squared [[buffer(13)]],
    constant const int& ceil_radius [[buffer(14)]],
    constant const int& table_capacity [[buffer(15)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = row_offsets[query_row];
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }

                int insert_at = selected;
                for (int rank = 0; rank < selected; ++rank) {
                    int index = slot_start + rank;
                    float existing_distance = distances[index];
                    int existing_source = source_rows[index];
                    if (distance < existing_distance ||
                        (distance == existing_distance &&
                         source_row < existing_source)) {
                        insert_at = rank;
                        break;
                    }
                }
                if (insert_at >= max_neighbors) {
                    continue;
                }

                int last = min(selected, max_neighbors - 1);
                for (int rank = last; rank > insert_at; --rank) {
                    int dst = slot_start + rank;
                    int src = dst - 1;
                    source_rows[dst] = source_rows[src];
                    distances[dst] = distances[src];
                }
                source_rows[slot_start + insert_at] = source_row;
                distances[slot_start + insert_at] = distance;
                selected = min(selected + 1, max_neighbors);
            }
        }
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void count_knn_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* row_counts [[buffer(5)]],
    constant const int& source_capacity [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    constant const int& search_radius [[buffer(9)]],
    constant const int& table_capacity [[buffer(10)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_capacity)) {
        return;
    }
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        row_counts[query_row] = 0;
        return;
    }

    int query_base = int(query_row) * 4;
    int selected = 0;
    for (int dz = -search_radius; dz <= search_radius; ++dz) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }
                selected += 1;
                if (selected >= max_neighbors) {
                    row_counts[query_row] = max_neighbors;
                    return;
                }
            }
        }
    }
    row_counts[query_row] = selected;
}

[[kernel]] void fill_knn_relation_compact_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device const int* row_offsets [[buffer(5)]],
    device int* query_rows [[buffer(6)]],
    device int* source_rows [[buffer(7)]],
    device int* neighbor_ids [[buffer(8)]],
    device float* distances [[buffer(9)]],
    constant const int& source_capacity [[buffer(10)]],
    constant const int& query_capacity [[buffer(11)]],
    constant const int& max_neighbors [[buffer(12)]],
    constant const int& search_radius [[buffer(13)]],
    constant const int& table_capacity [[buffer(14)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = row_offsets[query_row];
    int selected = 0;
    for (int shell = 0; shell <= search_radius; ++shell) {
        for (int dz = -shell; dz <= shell; ++dz) {
            for (int dy = -shell; dy <= shell; ++dy) {
                for (int dx = -shell; dx <= shell; ++dx) {
                    if (max(max(abs(dx), abs(dy)), abs(dz)) != shell) {
                        continue;
                    }
                    int target[4];
                    target[0] = query_coords[query_base];
                    target[1] = query_coords[query_base + 1] + dx;
                    target[2] = query_coords[query_base + 2] + dy;
                    target[3] = query_coords[query_base + 3] + dz;
                    int source_row = lookup_coord_row_hash(
                        source_coords, source_table, table_capacity, target
                    );
                    if (source_row < 0 || source_row >= source_capacity ||
                        source_row >= source_active_rows[0]) {
                        continue;
                    }

                    float distance = float(dx * dx + dy * dy + dz * dz);
                    int insert_at = selected;
                    for (int rank = 0; rank < selected; ++rank) {
                        int index = slot_start + rank;
                        float existing_distance = distances[index];
                        int existing_source = source_rows[index];
                        if (distance < existing_distance ||
                            (distance == existing_distance &&
                             source_row < existing_source)) {
                            insert_at = rank;
                            break;
                        }
                    }
                    if (insert_at >= max_neighbors) {
                        continue;
                    }

                    int last = min(selected, max_neighbors - 1);
                    for (int rank = last; rank > insert_at; --rank) {
                        int dst = slot_start + rank;
                        int src = dst - 1;
                        source_rows[dst] = source_rows[src];
                        distances[dst] = distances[src];
                    }
                    source_rows[slot_start + insert_at] = source_row;
                    distances[slot_start + insert_at] = distance;
                    selected = min(selected + 1, max_neighbors);
                }
            }
        }
        if (selected >= max_neighbors) {
            float kth_distance = distances[slot_start + max_neighbors - 1];
            float next_min_distance = float((shell + 1) * (shell + 1));
            if (next_min_distance > kth_distance) {
                break;
            }
        }
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void fill_knn_relation_topk_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device int* query_rows [[buffer(4)]],
    device int* source_rows [[buffer(5)]],
    device int* neighbor_ids [[buffer(6)]],
    device float* distances [[buffer(7)]],
    constant const int& source_capacity [[buffer(8)]],
    constant const int& query_capacity [[buffer(9)]],
    constant const int& max_neighbors [[buffer(10)]],
    uint query_row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]]
) {
    constexpr int thread_count = 128;
    constexpr int max_k = 16;
    threadgroup float group_distances[thread_count * max_k];
    threadgroup int group_sources[thread_count * max_k];

    int tid = int(thread_id);
    int source_count = min(source_active_rows[0], source_capacity);
    int query_count = min(query_active_rows[0], query_capacity);
    int k = min(max_neighbors, max_k);
    int slot_start = int(query_row) * max_neighbors;

    float local_distances[max_k];
    int local_sources[max_k];
    for (int rank = 0; rank < max_k; ++rank) {
        local_distances[rank] = 0.0f;
        local_sources[rank] = -1;
    }

    int selected = 0;
    if (query_row < uint(query_count) && max_neighbors > 0) {
        for (int source_row = tid; source_row < source_count;
             source_row += thread_count) {
            if (!same_batch(
                    query_coords, int(query_row), source_coords, source_row
                )) {
                continue;
            }
            float distance = squared_spatial_distance(
                query_coords, int(query_row), source_coords, source_row
            );
            int insert_at = selected;
            for (int rank = 0; rank < selected; ++rank) {
                if (distance < local_distances[rank] ||
                    (distance == local_distances[rank] &&
                     source_row < local_sources[rank])) {
                    insert_at = rank;
                    break;
                }
            }
            if (insert_at >= k) {
                continue;
            }
            int last = min(selected, k - 1);
            for (int rank = last; rank > insert_at; --rank) {
                local_distances[rank] = local_distances[rank - 1];
                local_sources[rank] = local_sources[rank - 1];
            }
            local_distances[insert_at] = distance;
            local_sources[insert_at] = source_row;
            selected = min(selected + 1, k);
        }
    }

    int group_base = tid * max_k;
    for (int rank = 0; rank < max_k; ++rank) {
        group_distances[group_base + rank] = local_distances[rank];
        group_sources[group_base + rank] = local_sources[rank];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0 || query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int final_selected = 0;
    for (int candidate = 0; candidate < thread_count * max_k; ++candidate) {
        int source_row = group_sources[candidate];
        if (source_row < 0) {
            continue;
        }
        float distance = group_distances[candidate];
        int insert_at = final_selected;
        for (int rank = 0; rank < final_selected; ++rank) {
            int index = slot_start + rank;
            float existing_distance = distances[index];
            int existing_source = source_rows[index];
            if (distance < existing_distance ||
                (distance == existing_distance &&
                 source_row < existing_source)) {
                insert_at = rank;
                break;
            }
        }
        if (insert_at >= k) {
            continue;
        }
        int last = min(final_selected, k - 1);
        for (int rank = last; rank > insert_at; --rank) {
            int dst = slot_start + rank;
            int src = dst - 1;
            source_rows[dst] = source_rows[src];
            distances[dst] = distances[src];
        }
        source_rows[slot_start + insert_at] = source_row;
        distances[slot_start + insert_at] = distance;
        final_selected = min(final_selected + 1, k);
    }

    for (int rank = 0; rank < final_selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void compact_neighbor_relation_i32(
    device int* query_rows [[buffer(0)]],
    device int* source_rows [[buffer(1)]],
    device int* neighbor_ids [[buffer(2)]],
    device float* distances [[buffer(3)]],
    device int* row_offsets [[buffer(4)]],
    device int* counts [[buffer(5)]],
    constant const int& query_capacity [[buffer(6)]],
    constant const int& max_neighbors [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_edge = 0;
    int query_count = counts[1];
    for (int query_row = 0; query_row < query_count; ++query_row) {
        row_offsets[query_row] = out_edge;
        int slot_start = query_row * max_neighbors;
        for (int rank = 0; rank < max_neighbors; ++rank) {
            int index = slot_start + rank;
            int source_row = source_rows[index];
            if (source_row < 0) {
                break;
            }
            query_rows[out_edge] = query_row;
            source_rows[out_edge] = source_row;
            neighbor_ids[out_edge] = rank;
            distances[out_edge] = distances[index];
            out_edge += 1;
        }
    }
    row_offsets[query_count] = out_edge;
    counts[0] = out_edge;
    (void)query_capacity;
}
