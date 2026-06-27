#include <metal_stdlib>

using namespace metal;

#include "native/features/coordinates/metal/common.metal"

[[kernel]] void coord_hash_clear_i32(
    device int* table_rows [[buffer(0)]],
    constant const int& table_capacity [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(table_capacity)) {
        table_rows[elem] = -1;
    }
}

[[kernel]] void coord_hash_insert_rows_i32(
    device const int* coords [[buffer(0)]],
    device atomic_int* table_rows [[buffer(1)]],
    constant const int& rows [[buffer(2)]],
    constant const int& table_capacity [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row < uint(rows)) {
        insert_coord_row_hash(coords, int(row), table_rows, table_capacity);
    }
}

[[kernel]] void coord_hash_insert_active_rows_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device atomic_int* table_rows [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& table_capacity [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    if (row < uint(logical_rows)) {
        insert_coord_row_hash(coords, int(row), table_rows, table_capacity);
    }
}
