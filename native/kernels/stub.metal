#include <metal_stdlib>

using namespace metal;

[[kernel]] void mlx_lattice_stub(device uint* out [[buffer(0)]],
                                 uint tid [[thread_position_in_grid]]) {
  if (tid == 0) {
    out[0] = 0;
  }
}
