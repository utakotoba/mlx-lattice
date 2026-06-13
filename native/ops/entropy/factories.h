#pragma once

#include <string>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

mx::array make_normalized_cdf(const mx::array& prob);

std::string make_range_encode(const mx::array& cdf, const mx::array& symbols);

mx::array make_range_decode(const mx::array& cdf, const std::string& stream);

} // namespace mlx_lattice
