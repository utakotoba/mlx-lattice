#include "features/entropy/api.h"

#include <stdexcept>
#include <string>
#include <string_view>

#include "features/entropy/factories.h"

namespace mlx_lattice {

mx::array normalized_cdf(const mx::array& prob) {
    if (prob.ndim() != 2 || prob.shape(1) < 2) {
        throw std::invalid_argument(
            "normalized_cdf expects a two-dimensional probability array."
        );
    }
    if (prob.dtype() != mx::float32 && prob.dtype() != mx::float16) {
        throw std::invalid_argument(
            "normalized_cdf supports float32 and float16 probabilities."
        );
    }
    return make_normalized_cdf(prob);
}

void validate_prob(const mx::array& prob, std::string_view op_name) {
    if (prob.ndim() != 2 || prob.shape(1) < 2) {
        throw std::invalid_argument(
            std::string(op_name) +
            " expects a two-dimensional probability array."
        );
    }
    if (prob.dtype() != mx::float32 && prob.dtype() != mx::float16) {
        throw std::invalid_argument(
            std::string(op_name) +
            " supports float32 and float16 probabilities."
        );
    }
}

std::string range_encode(const mx::array& cdf, const mx::array& symbols) {
    if (cdf.ndim() != 2 || cdf.dtype() != mx::int16) {
        throw std::invalid_argument(
            "range_encode expects a two-dimensional int16 CDF array."
        );
    }
    if (symbols.ndim() != 1 || symbols.dtype() != mx::int32) {
        throw std::invalid_argument(
            "range_encode expects one-dimensional int32 symbols."
        );
    }
    if (cdf.shape(0) != symbols.shape(0)) {
        throw std::invalid_argument(
            "range_encode CDF rows must match symbol count."
        );
    }
    return make_range_encode(cdf, symbols);
}

mx::array range_decode(const mx::array& cdf, const std::string& stream) {
    if (cdf.ndim() != 2 || cdf.dtype() != mx::int16) {
        throw std::invalid_argument(
            "range_decode expects a two-dimensional int16 CDF array."
        );
    }
    return make_range_decode(cdf, stream);
}

std::string
range_encode_from_prob(const mx::array& prob, const mx::array& symbols) {
    validate_prob(prob, "range_encode_from_prob");
    if (symbols.ndim() != 1 || symbols.dtype() != mx::int32) {
        throw std::invalid_argument(
            "range_encode_from_prob expects one-dimensional int32 symbols."
        );
    }
    if (prob.shape(0) != symbols.shape(0)) {
        throw std::invalid_argument(
            "range_encode_from_prob probability rows must match symbol count."
        );
    }
    return make_range_encode_from_prob(prob, symbols);
}

mx::array
range_decode_from_prob(const mx::array& prob, const std::string& stream) {
    validate_prob(prob, "range_decode_from_prob");
    return make_range_decode_from_prob(prob, stream);
}

std::string
rans_encode_from_prob(const mx::array& prob, const mx::array& symbols) {
    validate_prob(prob, "rans_encode_from_prob");
    if (symbols.ndim() != 1 || symbols.dtype() != mx::int32) {
        throw std::invalid_argument(
            "rans_encode_from_prob expects one-dimensional int32 symbols."
        );
    }
    if (prob.shape(0) != symbols.shape(0)) {
        throw std::invalid_argument(
            "rans_encode_from_prob probability rows must match symbol count."
        );
    }
    return make_rans_encode_from_prob(prob, symbols);
}

mx::array
rans_decode_from_prob(const mx::array& prob, const std::string& stream) {
    validate_prob(prob, "rans_decode_from_prob");
    return make_rans_decode_from_prob(prob, stream);
}

} // namespace mlx_lattice
