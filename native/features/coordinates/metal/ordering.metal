#include <metal_stdlib>

using namespace metal;

inline ulong split_morton_3(ulong value) {
    value &= 0x1ffffful;
    value = (value | (value << 32)) & 0x1f00000000fffful;
    value = (value | (value << 16)) & 0x1f0000ff0000fful;
    value = (value | (value << 8)) & 0x100f00f00f00f00ful;
    value = (value | (value << 4)) & 0x10c30c30c30c30c3ul;
    value = (value | (value << 2)) & 0x1249249249249249ul;
    return value;
}

[[kernel]] void morton_codes_i32(
    device const int* coords [[buffer(0)]],
    device long* codes [[buffer(1)]],
    constant const int& rows [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows)) {
        return;
    }

    int base = int(row) * 4;
    ulong code = split_morton_3(ulong(coords[base + 1])) |
                 (split_morton_3(ulong(coords[base + 2])) << 1) |
                 (split_morton_3(ulong(coords[base + 3])) << 2);
    code += ulong(coords[base]) << 60;
    codes[row] = long(code);
}
