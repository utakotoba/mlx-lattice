#pragma once

#include <string>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

mx::array normalized_cdf(const mx::array& prob);

std::string range_encode(const mx::array& cdf, const mx::array& symbols);

mx::array range_decode(const mx::array& cdf, const std::string& stream);

std::string
range_encode_from_prob(const mx::array& prob, const mx::array& symbols);

mx::array
range_decode_from_prob(const mx::array& prob, const std::string& stream);

std::string
rans_encode_from_prob(const mx::array& prob, const mx::array& symbols);

mx::array
rans_decode_from_prob(const mx::array& prob, const std::string& stream);

} // namespace mlx_lattice
