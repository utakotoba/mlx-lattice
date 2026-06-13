#include "ops/entropy/factories.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/types/half_types.h"

namespace mlx_lattice {

namespace {

constexpr std::string_view kRangeMagic = "MLXGERNG01";
constexpr std::string_view kRansMagic = "MLXGERNS01";
constexpr uint32_t kRangeTotal = 1u << 16;
constexpr uint32_t kCodeHalf = 0x80000000u;
constexpr uint32_t kCodeFirstQuarter = 0x40000000u;
constexpr uint32_t kCodeThirdQuarter = 0xc0000000u;
constexpr uint32_t kRansLowerBound = 1u << 23;
constexpr uint32_t kRansMask = kRangeTotal - 1u;

uint16_t read_cdf_value(const int16_t* cdf, int alphabet, int row, int col) {
    if (col == alphabet) {
        return 0;
    }
    return static_cast<uint16_t>(
        cdf[static_cast<ptrdiff_t>(row) * (alphabet + 1) + col]
    );
}

uint32_t cdf_value(const int16_t* cdf, int alphabet, int row, int col) {
    if (col == alphabet) {
        return kRangeTotal;
    }
    return read_cdf_value(cdf, alphabet, row, col);
}

void validate_cdf_row(const int16_t* cdf, int alphabet, int row) {
    auto previous = uint32_t{0};
    if (cdf_value(cdf, alphabet, row, 0) != 0) {
        throw std::invalid_argument("CDF rows must start at zero.");
    }
    for (int col = 1; col <= alphabet; ++col) {
        auto current = cdf_value(cdf, alphabet, row, col);
        if (current <= previous || current > kRangeTotal) {
            throw std::invalid_argument(
                "CDF rows must be strictly increasing and normalized."
            );
        }
        previous = current;
    }
}

void append_u32(std::string& out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<char>((value >> shift) & 0xffu));
    }
}

uint32_t read_u32(std::string_view stream, size_t offset) {
    if (offset + 4 > stream.size()) {
        throw std::invalid_argument("Truncated range stream header.");
    }
    auto value = uint32_t{0};
    for (int byte = 0; byte < 4; ++byte) {
        value |= static_cast<uint32_t>(
                     static_cast<unsigned char>(stream[offset + byte])
                 )
                 << (8 * byte);
    }
    return value;
}

void validate_header(std::string_view stream, int rows, int alphabet) {
    if (stream.size() < kRangeMagic.size() ||
        stream.substr(0, kRangeMagic.size()) != kRangeMagic) {
        throw std::invalid_argument("Invalid mlx-lattice range stream.");
    }
    auto stream_rows = static_cast<int>(read_u32(stream, kRangeMagic.size()));
    auto stream_alphabet =
        static_cast<int>(read_u32(stream, kRangeMagic.size() + 4));
    if (stream_rows != rows || stream_alphabet != alphabet) {
        throw std::invalid_argument("Range stream shape does not match CDF.");
    }
}

void write_header(std::string& out, int rows, int alphabet) {
    out.append(kRangeMagic);
    append_u32(out, static_cast<uint32_t>(rows));
    append_u32(out, static_cast<uint32_t>(alphabet));
}

void write_rans_header(
    std::string& out,
    int rows,
    int alphabet,
    uint32_t state
) {
    out.append(kRansMagic);
    append_u32(out, static_cast<uint32_t>(rows));
    append_u32(out, static_cast<uint32_t>(alphabet));
    append_u32(out, state);
}

void validate_rans_header(std::string_view stream, int rows, int alphabet) {
    if (stream.size() < kRansMagic.size() + 12 ||
        stream.substr(0, kRansMagic.size()) != kRansMagic) {
        throw std::invalid_argument("Invalid mlx-lattice rANS stream.");
    }
    auto stream_rows = static_cast<int>(read_u32(stream, kRansMagic.size()));
    auto stream_alphabet =
        static_cast<int>(read_u32(stream, kRansMagic.size() + 4));
    if (stream_rows != rows || stream_alphabet != alphabet) {
        throw std::invalid_argument(
            "rANS stream shape does not match probabilities."
        );
    }
}

struct CdfShape {
    int rows;
    int alphabet;
};

struct SymbolRange {
    uint32_t lo;
    uint32_t hi;
};

class BitWriter {
  public:
    explicit BitWriter(std::string& out) : out_(out) {}

    void write(bool bit) {
        current_ = static_cast<uint8_t>((current_ << 1) | uint8_t(bit));
        ++bits_;
        if (bits_ == 8) {
            out_.push_back(static_cast<char>(current_));
            current_ = 0;
            bits_ = 0;
        }
    }

    void flush() {
        if (bits_ == 0) {
            return;
        }
        current_ = static_cast<uint8_t>(current_ << (8 - bits_));
        out_.push_back(static_cast<char>(current_));
        current_ = 0;
        bits_ = 0;
    }

  private:
    std::string& out_;
    uint8_t current_{0};
    int bits_{0};
};

class BitReader {
  public:
    explicit BitReader(std::string_view stream)
        : stream_(stream), cursor_(kRangeMagic.size() + 8) {}

    uint32_t read() {
        if (bits_ == 0) {
            if (cursor_ >= stream_.size()) {
                current_ = 0;
            } else {
                current_ = static_cast<uint8_t>(
                    static_cast<unsigned char>(stream_[cursor_++])
                );
            }
            bits_ = 8;
        }
        auto bit = uint32_t(current_ >> 7);
        current_ = static_cast<uint8_t>(current_ << 1);
        --bits_;
        return bit;
    }

  private:
    std::string_view stream_;
    size_t cursor_;
    uint8_t current_{0};
    int bits_{0};
};

class ArithmeticEncoder {
  public:
    explicit ArithmeticEncoder(std::string& out) : writer_(out) {}

    void encode(SymbolRange symbol) {
        auto range = uint64_t{high_} - low_ + 1u;
        high_ = low_ +
                static_cast<uint32_t>((range * symbol.hi) / kRangeTotal - 1u);
        low_ = low_ + static_cast<uint32_t>((range * symbol.lo) / kRangeTotal);

        while (true) {
            if (high_ < kCodeHalf) {
                emit(false);
            } else if (low_ >= kCodeHalf) {
                emit(true);
                low_ -= kCodeHalf;
                high_ -= kCodeHalf;
            } else if (low_ >= kCodeFirstQuarter && high_ < kCodeThirdQuarter) {
                ++pending_;
                low_ -= kCodeFirstQuarter;
                high_ -= kCodeFirstQuarter;
            } else {
                break;
            }
            low_ <<= 1;
            high_ = (high_ << 1) | 1u;
        }
    }

    void finish() {
        ++pending_;
        emit(low_ >= kCodeFirstQuarter);
        writer_.flush();
    }

  private:
    void emit(bool bit) {
        writer_.write(bit);
        while (pending_ > 0) {
            writer_.write(!bit);
            --pending_;
        }
    }

    BitWriter writer_;
    uint32_t low_{0};
    uint32_t high_{0xffffffffu};
    uint32_t pending_{0};
};

class ArithmeticDecoder {
  public:
    explicit ArithmeticDecoder(std::string_view stream) : reader_(stream) {
        for (int bit = 0; bit < 32; ++bit) {
            code_ = (code_ << 1) | reader_.read();
        }
    }

    uint32_t scaled() const {
        auto range = uint64_t{high_} - low_ + 1u;
        return static_cast<uint32_t>(
            ((uint64_t{code_} - low_ + 1u) * kRangeTotal - 1u) / range
        );
    }

    void decode(SymbolRange symbol) {
        auto range = uint64_t{high_} - low_ + 1u;
        high_ = low_ +
                static_cast<uint32_t>((range * symbol.hi) / kRangeTotal - 1u);
        low_ = low_ + static_cast<uint32_t>((range * symbol.lo) / kRangeTotal);

        while (true) {
            if (high_ < kCodeHalf) {
            } else if (low_ >= kCodeHalf) {
                code_ -= kCodeHalf;
                low_ -= kCodeHalf;
                high_ -= kCodeHalf;
            } else if (low_ >= kCodeFirstQuarter && high_ < kCodeThirdQuarter) {
                code_ -= kCodeFirstQuarter;
                low_ -= kCodeFirstQuarter;
                high_ -= kCodeFirstQuarter;
            } else {
                break;
            }
            low_ <<= 1;
            high_ = (high_ << 1) | 1u;
            code_ = (code_ << 1) | reader_.read();
        }
    }

  private:
    BitReader reader_;
    uint32_t low_{0};
    uint32_t high_{0xffffffffu};
    uint32_t code_{0};
};

std::string encode_symbols(
    const int16_t* cdf,
    const int32_t* symbols,
    int rows,
    int alphabet
) {
    std::string out;
    out.reserve(kRangeMagic.size() + 8 + static_cast<size_t>(rows));
    write_header(out, rows, alphabet);
    ArithmeticEncoder encoder(out);

    for (int row = 0; row < rows; ++row) {
        validate_cdf_row(cdf, alphabet, row);
        auto symbol = symbols[row];
        if (symbol < 0 || symbol >= alphabet) {
            throw std::invalid_argument("Range symbol is outside the CDF.");
        }
        auto lo = cdf_value(cdf, alphabet, row, symbol);
        auto hi = cdf_value(cdf, alphabet, row, symbol + 1);
        encoder.encode(SymbolRange{lo, hi});
    }
    encoder.finish();
    return out;
}

std::vector<int32_t> decode_symbols(
    const int16_t* cdf,
    std::string_view stream,
    int rows,
    int alphabet
) {
    validate_header(stream, rows, alphabet);

    ArithmeticDecoder decoder(stream);

    std::vector<int32_t> symbols(rows);
    for (int row = 0; row < rows; ++row) {
        validate_cdf_row(cdf, alphabet, row);
        auto scaled = decoder.scaled();
        auto symbol = 0;
        for (; symbol < alphabet; ++symbol) {
            auto lo = cdf_value(cdf, alphabet, row, symbol);
            auto hi = cdf_value(cdf, alphabet, row, symbol + 1);
            if (lo <= scaled && scaled < hi) {
                break;
            }
        }
        if (symbol == alphabet) {
            throw std::invalid_argument("Range stream symbol decode failed.");
        }
        symbols[row] = symbol;

        auto lo = cdf_value(cdf, alphabet, row, symbol);
        auto hi = cdf_value(cdf, alphabet, row, symbol + 1);
        decoder.decode(SymbolRange{lo, hi});
    }
    return symbols;
}

template <typename T> float read_prob_value(const T* values, ptrdiff_t index) {
    return static_cast<float>(values[index]);
}

template <>
float read_prob_value<mx::float16_t>(
    const mx::float16_t* values,
    ptrdiff_t index
) {
    return static_cast<float>(values[index]);
}

template <typename T>
void normalize_cdf_row(
    const T* prob_data,
    int row,
    CdfShape shape,
    std::vector<uint32_t>& cdf
) {
    cdf.resize(static_cast<size_t>(shape.alphabet) + 1);
    cdf[0] = 0;
    auto running = 0.0F;
    auto base = static_cast<ptrdiff_t>(row) * shape.alphabet;
    auto scale = static_cast<float>(kRangeTotal - shape.alphabet);
    for (int col = 1; col < shape.alphabet; ++col) {
        auto prob_value = read_prob_value(prob_data, base + col - 1);
        if (!std::isfinite(prob_value)) {
            throw std::invalid_argument(
                "Probability rows must contain finite values."
            );
        }
        running += std::clamp(prob_value, 0.0F, 1.0F);
        auto cdf_value =
            static_cast<int32_t>(std::lround(running * scale)) + col;
        cdf_value =
            std::clamp(cdf_value, col, int(kRangeTotal - shape.alphabet + col));
        cdf[static_cast<size_t>(col)] = static_cast<uint32_t>(cdf_value);
    }
    cdf[static_cast<size_t>(shape.alphabet)] = kRangeTotal;
}

template <typename T>
void normalize_cdf_typed(
    const T* prob_data,
    int16_t* out_data,
    CdfShape shape
) {
    std::vector<uint32_t> cdf;
    for (int row = 0; row < shape.rows; ++row) {
        normalize_cdf_row(prob_data, row, shape, cdf);
        auto out_base = static_cast<ptrdiff_t>(row) * (shape.alphabet + 1);
        for (int col = 0; col < shape.alphabet; ++col) {
            out_data[out_base + col] =
                static_cast<int16_t>(cdf[static_cast<size_t>(col)]);
        }
        out_data[out_base + shape.alphabet] = 0;
    }
}

template <typename T>
std::string encode_symbols_from_prob(
    const T* prob,
    const int32_t* symbols,
    CdfShape shape
) {
    std::string out;
    out.reserve(kRangeMagic.size() + 8 + static_cast<size_t>(shape.rows));
    write_header(out, shape.rows, shape.alphabet);
    ArithmeticEncoder encoder(out);
    std::vector<uint32_t> cdf;
    for (int row = 0; row < shape.rows; ++row) {
        normalize_cdf_row(prob, row, shape, cdf);
        auto symbol = symbols[row];
        if (symbol < 0 || symbol >= shape.alphabet) {
            throw std::invalid_argument("Range symbol is outside the CDF.");
        }
        auto index = static_cast<size_t>(symbol);
        encoder.encode(SymbolRange{cdf[index], cdf[index + 1]});
    }
    encoder.finish();
    return out;
}

template <typename T>
std::vector<int32_t> decode_symbols_from_prob(
    const T* prob,
    std::string_view stream,
    CdfShape shape
) {
    validate_header(stream, shape.rows, shape.alphabet);
    ArithmeticDecoder decoder(stream);
    std::vector<int32_t> symbols(shape.rows);
    std::vector<uint32_t> cdf;
    for (int row = 0; row < shape.rows; ++row) {
        normalize_cdf_row(prob, row, shape, cdf);
        auto scaled = decoder.scaled();
        auto symbol = 0;
        for (; symbol < shape.alphabet; ++symbol) {
            auto index = static_cast<size_t>(symbol);
            auto lo = cdf[index];
            auto hi = cdf[index + 1];
            if (lo <= scaled && scaled < hi) {
                break;
            }
        }
        if (symbol == shape.alphabet) {
            throw std::invalid_argument("Range stream symbol decode failed.");
        }
        symbols[row] = symbol;
        auto index = static_cast<size_t>(symbol);
        decoder.decode(SymbolRange{cdf[index], cdf[index + 1]});
    }
    return symbols;
}

template <typename T>
std::string rans_encode_symbols_from_prob(
    const T* prob,
    const int32_t* symbols,
    CdfShape shape
) {
    auto state = kRansLowerBound;
    std::vector<uint8_t> payload;
    payload.reserve(static_cast<size_t>(shape.rows));
    std::vector<uint32_t> cdf;
    for (int row = shape.rows - 1; row >= 0; --row) {
        normalize_cdf_row(prob, row, shape, cdf);
        auto symbol = symbols[row];
        if (symbol < 0 || symbol >= shape.alphabet) {
            throw std::invalid_argument("rANS symbol is outside the CDF.");
        }
        auto index = static_cast<size_t>(symbol);
        auto lo = cdf[index];
        auto freq = cdf[index + 1] - lo;
        auto max_state = ((kRansLowerBound >> 16) << 8) * freq;
        while (state >= max_state) {
            payload.push_back(static_cast<uint8_t>(state & 0xffu));
            state >>= 8;
        }
        state = ((state / freq) << 16) + (state % freq) + lo;
    }

    std::string out;
    out.reserve(kRansMagic.size() + 12 + payload.size());
    write_rans_header(out, shape.rows, shape.alphabet, state);
    for (auto it = payload.rbegin(); it != payload.rend(); ++it) {
        out.push_back(static_cast<char>(*it));
    }
    return out;
}

template <typename T>
std::vector<int32_t> rans_decode_symbols_from_prob(
    const T* prob,
    std::string_view stream,
    CdfShape shape
) {
    validate_rans_header(stream, shape.rows, shape.alphabet);
    auto state = read_u32(stream, kRansMagic.size() + 8);
    auto cursor = kRansMagic.size() + 12;
    std::vector<int32_t> symbols(shape.rows);
    std::vector<uint32_t> cdf;
    for (int row = 0; row < shape.rows; ++row) {
        normalize_cdf_row(prob, row, shape, cdf);
        auto slot = state & kRansMask;
        auto symbol = 0;
        for (; symbol < shape.alphabet; ++symbol) {
            auto index = static_cast<size_t>(symbol);
            auto lo = cdf[index];
            auto hi = cdf[index + 1];
            if (lo <= slot && slot < hi) {
                break;
            }
        }
        if (symbol == shape.alphabet) {
            throw std::invalid_argument("rANS stream symbol decode failed.");
        }
        symbols[row] = symbol;
        auto index = static_cast<size_t>(symbol);
        auto lo = cdf[index];
        auto freq = cdf[index + 1] - lo;
        state = freq * (state >> 16) + (slot - lo);
        while (state < kRansLowerBound) {
            if (cursor >= stream.size()) {
                throw std::invalid_argument(
                    "Truncated mlx-lattice rANS stream."
                );
            }
            state = (state << 8) | static_cast<unsigned char>(stream[cursor++]);
        }
    }
    if (cursor != stream.size()) {
        throw std::invalid_argument("Malformed mlx-lattice rANS stream.");
    }
    return symbols;
}

} // namespace

mx::array make_normalized_cdf(const mx::array& prob) {
    auto rows = prob.shape(0);
    auto alphabet = prob.shape(1);
    if (alphabet <= 0 || alphabet >= int(kRangeTotal)) {
        throw std::invalid_argument("Probability alphabet is out of range.");
    }
    if (prob.dtype() != mx::float32 && prob.dtype() != mx::float16) {
        throw std::invalid_argument(
            "Probability rows must be float32 or float16."
        );
    }
    auto device = mx::Device::cpu;
    auto ready_prob = mx::contiguous(prob, false, device);
    mx::eval(ready_prob);

    auto bytes = static_cast<size_t>(rows) * (alphabet + 1) * sizeof(int16_t);
    auto buffer = mx::allocator::malloc(bytes);
    auto out = mx::array(buffer, mx::Shape{rows, alphabet + 1}, mx::int16);
    auto* out_data = out.data<int16_t>();
    auto shape = CdfShape{rows, alphabet};
    if (prob.dtype() == mx::float32) {
        normalize_cdf_typed<float>(ready_prob.data<float>(), out_data, shape);
    } else {
        normalize_cdf_typed<mx::float16_t>(
            ready_prob.data<mx::float16_t>(), out_data, shape
        );
    }
    return out;
}

std::string make_range_encode(const mx::array& cdf, const mx::array& symbols) {
    auto device = mx::Device::cpu;
    auto ready_cdf = mx::contiguous(cdf, false, device);
    auto ready_symbols = mx::contiguous(symbols, false, device);
    mx::eval(ready_cdf, ready_symbols);
    return encode_symbols(
        ready_cdf.data<int16_t>(),
        ready_symbols.data<int32_t>(),
        ready_cdf.shape(0),
        ready_cdf.shape(1) - 1
    );
}

mx::array make_range_decode(const mx::array& cdf, const std::string& stream) {
    auto device = mx::Device::cpu;
    auto ready_cdf = mx::contiguous(cdf, false, device);
    mx::eval(ready_cdf);
    auto rows = ready_cdf.shape(0);
    auto decoded = decode_symbols(
        ready_cdf.data<int16_t>(), stream, rows, ready_cdf.shape(1) - 1
    );
    auto bytes = decoded.size() * sizeof(int32_t);
    auto buffer = mx::allocator::malloc(bytes);
    std::memcpy(buffer.raw_ptr(), decoded.data(), bytes);
    auto out = mx::array(buffer, mx::Shape{rows}, mx::int32);
    auto copied = mx::copy(out, mx::default_device());
    mx::eval(copied);
    return copied;
}

std::string
make_range_encode_from_prob(const mx::array& prob, const mx::array& symbols) {
    auto device = mx::Device::cpu;
    auto ready_prob = mx::contiguous(prob, false, device);
    auto ready_symbols = mx::contiguous(symbols, false, device);
    mx::eval(ready_prob, ready_symbols);
    auto shape = CdfShape{ready_prob.shape(0), ready_prob.shape(1)};
    if (prob.dtype() == mx::float32) {
        return encode_symbols_from_prob<float>(
            ready_prob.data<float>(), ready_symbols.data<int32_t>(), shape
        );
    }
    return encode_symbols_from_prob<mx::float16_t>(
        ready_prob.data<mx::float16_t>(), ready_symbols.data<int32_t>(), shape
    );
}

mx::array
make_range_decode_from_prob(const mx::array& prob, const std::string& stream) {
    auto device = mx::Device::cpu;
    auto ready_prob = mx::contiguous(prob, false, device);
    mx::eval(ready_prob);
    auto shape = CdfShape{ready_prob.shape(0), ready_prob.shape(1)};
    std::vector<int32_t> decoded;
    if (prob.dtype() == mx::float32) {
        decoded = decode_symbols_from_prob<float>(
            ready_prob.data<float>(), stream, shape
        );
    } else {
        decoded = decode_symbols_from_prob<mx::float16_t>(
            ready_prob.data<mx::float16_t>(), stream, shape
        );
    }
    auto bytes = decoded.size() * sizeof(int32_t);
    auto buffer = mx::allocator::malloc(bytes);
    std::memcpy(buffer.raw_ptr(), decoded.data(), bytes);
    auto out = mx::array(buffer, mx::Shape{shape.rows}, mx::int32);
    auto copied = mx::copy(out, mx::default_device());
    mx::eval(copied);
    return copied;
}

std::string
make_rans_encode_from_prob(const mx::array& prob, const mx::array& symbols) {
    auto device = mx::Device::cpu;
    auto ready_prob = mx::contiguous(prob, false, device);
    auto ready_symbols = mx::contiguous(symbols, false, device);
    mx::eval(ready_prob, ready_symbols);
    auto shape = CdfShape{ready_prob.shape(0), ready_prob.shape(1)};
    if (prob.dtype() == mx::float32) {
        return rans_encode_symbols_from_prob<float>(
            ready_prob.data<float>(), ready_symbols.data<int32_t>(), shape
        );
    }
    return rans_encode_symbols_from_prob<mx::float16_t>(
        ready_prob.data<mx::float16_t>(), ready_symbols.data<int32_t>(), shape
    );
}

mx::array
make_rans_decode_from_prob(const mx::array& prob, const std::string& stream) {
    auto device = mx::Device::cpu;
    auto ready_prob = mx::contiguous(prob, false, device);
    mx::eval(ready_prob);
    auto shape = CdfShape{ready_prob.shape(0), ready_prob.shape(1)};
    std::vector<int32_t> decoded;
    if (prob.dtype() == mx::float32) {
        decoded = rans_decode_symbols_from_prob<float>(
            ready_prob.data<float>(), stream, shape
        );
    } else {
        decoded = rans_decode_symbols_from_prob<mx::float16_t>(
            ready_prob.data<mx::float16_t>(), stream, shape
        );
    }
    auto bytes = decoded.size() * sizeof(int32_t);
    auto buffer = mx::allocator::malloc(bytes);
    std::memcpy(buffer.raw_ptr(), decoded.data(), bytes);
    auto out = mx::array(buffer, mx::Shape{shape.rows}, mx::int32);
    auto copied = mx::copy(out, mx::default_device());
    mx::eval(copied);
    return copied;
}

} // namespace mlx_lattice
