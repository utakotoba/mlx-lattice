#include "lattice/runtime.h"

#ifndef MLX_LATTICE_VERSION
#define MLX_LATTICE_VERSION "0.0.0"
#endif

#ifndef MLX_LATTICE_HAS_CPU
#define MLX_LATTICE_HAS_CPU 1
#endif

#ifndef MLX_LATTICE_HAS_METAL
#define MLX_LATTICE_HAS_METAL 0
#endif

#ifndef MLX_LATTICE_HAS_CUDA
#define MLX_LATTICE_HAS_CUDA 0
#endif

#ifndef MLX_LATTICE_HAS_ROCM
#define MLX_LATTICE_HAS_ROCM 0
#endif

namespace mlx_lattice {

std::string version() {
  return MLX_LATTICE_VERSION;
}

Capabilities capabilities() {
  return {
      static_cast<bool>(MLX_LATTICE_HAS_CPU),
      static_cast<bool>(MLX_LATTICE_HAS_METAL),
      static_cast<bool>(MLX_LATTICE_HAS_CUDA),
      static_cast<bool>(MLX_LATTICE_HAS_ROCM),
  };
}

}  // namespace mlx_lattice
