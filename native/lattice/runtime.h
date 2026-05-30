#pragma once

#include <string>

namespace mlx_lattice {

struct Capabilities {
  bool cpu;
  bool metal;
  bool cuda;
  bool rocm;
};

std::string version();
Capabilities capabilities();

}  // namespace mlx_lattice
