#include "bindings/registrations.h"

#include "bindings/array_arg.h"

#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#include "features/entropy/api.h"

namespace mlx_lattice::bindings {

using namespace nb::literals;

void register_entropy(nb::module_& module) {
    module.def(
        "normalized_cdf",
        [](nb::handle prob) {
            return mlx_lattice::normalized_cdf(array_arg(prob, "prob"));
        },
        "prob"_a,
        nb::sig("def normalized_cdf(prob: mlx.core.array) -> mlx.core.array"),
        "Convert probability rows to int16 normalized CDF rows."
    );
    module.def(
        "range_encode",
        [](nb::handle cdf, nb::handle symbols) {
            auto encoded = mlx_lattice::range_encode(
                array_arg(cdf, "cdf"), array_arg(symbols, "symbols")
            );
            return nb::bytes(encoded.data(), encoded.size());
        },
        "cdf"_a,
        "symbols"_a,
        nb::sig(
            "def range_encode(cdf: mlx.core.array, "
            "symbols: mlx.core.array) -> bytes"
        ),
        "Encode int32 symbols with int16 normalized CDF rows."
    );
    module.def(
        "range_decode",
        [](nb::handle cdf, const nb::bytes& stream) {
            char* data = nullptr;
            Py_ssize_t size = 0;
            if (PyBytes_AsStringAndSize(stream.ptr(), &data, &size) != 0) {
                throw std::invalid_argument("stream must be bytes.");
            }
            return mlx_lattice::range_decode(
                array_arg(cdf, "cdf"),
                std::string(data, static_cast<size_t>(size))
            );
        },
        "cdf"_a,
        "stream"_a,
        nb::sig(
            "def range_decode(cdf: mlx.core.array, stream: bytes) -> "
            "mlx.core.array"
        ),
        "Decode int32 symbols from a range-coded stream."
    );
    module.def(
        "range_encode_from_prob",
        [](nb::handle prob, nb::handle symbols) {
            auto encoded = mlx_lattice::range_encode_from_prob(
                array_arg(prob, "prob"), array_arg(symbols, "symbols")
            );
            return nb::bytes(encoded.data(), encoded.size());
        },
        "prob"_a,
        "symbols"_a,
        nb::sig(
            "def range_encode_from_prob(prob: mlx.core.array, "
            "symbols: mlx.core.array) -> bytes"
        ),
        "Encode int32 symbols directly from probability rows."
    );
    module.def(
        "range_decode_from_prob",
        [](nb::handle prob, const nb::bytes& stream) {
            char* data = nullptr;
            Py_ssize_t size = 0;
            if (PyBytes_AsStringAndSize(stream.ptr(), &data, &size) != 0) {
                throw std::invalid_argument("stream must be bytes.");
            }
            return mlx_lattice::range_decode_from_prob(
                array_arg(prob, "prob"),
                std::string(data, static_cast<size_t>(size))
            );
        },
        "prob"_a,
        "stream"_a,
        nb::sig(
            "def range_decode_from_prob(prob: mlx.core.array, stream: bytes) "
            "-> "
            "mlx.core.array"
        ),
        "Decode int32 symbols directly from probability rows."
    );
    module.def(
        "rans_encode_from_prob",
        [](nb::handle prob, nb::handle symbols) {
            auto encoded = mlx_lattice::rans_encode_from_prob(
                array_arg(prob, "prob"), array_arg(symbols, "symbols")
            );
            return nb::bytes(encoded.data(), encoded.size());
        },
        "prob"_a,
        "symbols"_a,
        nb::sig(
            "def rans_encode_from_prob(prob: mlx.core.array, "
            "symbols: mlx.core.array) -> bytes"
        ),
        "Encode int32 symbols with byte-oriented rANS from probability rows."
    );
    module.def(
        "rans_decode_from_prob",
        [](nb::handle prob, const nb::bytes& stream) {
            char* data = nullptr;
            Py_ssize_t size = 0;
            if (PyBytes_AsStringAndSize(stream.ptr(), &data, &size) != 0) {
                throw std::invalid_argument("stream must be bytes.");
            }
            return mlx_lattice::rans_decode_from_prob(
                array_arg(prob, "prob"),
                std::string(data, static_cast<size_t>(size))
            );
        },
        "prob"_a,
        "stream"_a,
        nb::sig(
            "def rans_decode_from_prob(prob: mlx.core.array, stream: bytes) -> "
            "mlx.core.array"
        ),
        "Decode int32 symbols with byte-oriented rANS from probability rows."
    );
}

} // namespace mlx_lattice::bindings
