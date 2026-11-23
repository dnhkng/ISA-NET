#define TERNLOG_STORY_NO_MAIN
#include "ternlog_vocab_story_m4.cpp"

#include <arm_acle.h>
#include <zlib.h>

#include <fstream>
#include <future>
#include <stdexcept>

#if defined(TERNLOG_FFN_BATCH4_HOOK)
#include "m4_vocab_ffn_batch4_hook.h"
#endif

namespace {

constexpr std::uint32_t kFrequencyBudget = (1U << 16) - 256U;
constexpr std::uint32_t kRangeTop = 1U << 24;
constexpr std::size_t kHeaderBytes = 106;
constexpr std::size_t kRawHeaderBytes = 16;
constexpr int kMatchBuckets[4] = {6, 16, 32, 64};

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("unable to open " + path);
    const std::streamsize size = stream.tellg();
    stream.seekg(0);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (size && !stream.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("unable to read " + path);
    }
    return data;
}

void write_binary(const std::string& path, const std::vector<std::uint8_t>& data) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(
            reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("unable to write " + path);
    }
}

template <typename T>
void append_scalar(std::vector<std::uint8_t>& output, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T read_scalar(const std::vector<std::uint8_t>& input, std::size_t& offset) {
    if (offset + sizeof(T) > input.size()) throw std::runtime_error("truncated stream");
    T value;
    std::memcpy(&value, input.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

std::uint32_t crc32_bytes(const std::uint8_t* data, std::size_t size) {
    std::uint32_t result = 0;
    while (size != 0) {
        const uInt chunk = static_cast<uInt>(std::min<std::size_t>(
            size, std::numeric_limits<uInt>::max()
        ));
        result = static_cast<std::uint32_t>(crc32(result, data, chunk));
        data += chunk;
        size -= chunk;
    }
    return result;
}

std::vector<std::uint8_t> compress_raw(
    const std::vector<std::uint8_t>& input
) {
    if (input.size() > std::numeric_limits<std::size_t>::max() - kRawHeaderBytes) {
        throw std::runtime_error("raw input is too large");
    }
    std::vector<std::uint8_t> result;
    result.reserve(kRawHeaderBytes + input.size());
    result.insert(result.end(), {'T', 'L', 'R', '0'});
    append_scalar(result, static_cast<std::uint64_t>(input.size()));
    append_scalar(result, crc32_bytes(input.data(), input.size()));
    result.insert(result.end(), input.begin(), input.end());
    if (result.size() != kRawHeaderBytes + input.size()) {
        throw std::runtime_error("internal raw stream size error");
    }
    return result;
}

std::vector<std::uint8_t> decompress_raw(
    const std::vector<std::uint8_t>& stream
) {
    if (stream.size() < kRawHeaderBytes
        || std::memcmp(stream.data(), "TLR0", 4) != 0) {
        throw std::runtime_error("not a TLR0 stream");
    }
    std::size_t offset = 4;
    const std::uint64_t original_bytes = read_scalar<std::uint64_t>(stream, offset);
    const std::uint32_t expected_crc = read_scalar<std::uint32_t>(stream, offset);
    if (original_bytes > std::numeric_limits<std::size_t>::max()
        || static_cast<std::size_t>(original_bytes) != stream.size() - offset) {
        throw std::runtime_error("raw payload length mismatch");
    }
    std::vector<std::uint8_t> result(stream.begin() + offset, stream.end());
    if (crc32_bytes(result.data(), result.size()) != expected_crc) {
        throw std::runtime_error("checksum mismatch");
    }
    return result;
}

class RangeEncoderNative {
  public:
    void encode(std::uint32_t start, std::uint32_t size, std::uint32_t total) {
        const std::uint32_t scale = range_ / total;
        low_ += static_cast<std::uint64_t>(start) * scale;
        range_ = size * scale;
        while (range_ < kRangeTop) {
            range_ <<= 8;
            shift_low();
        }
    }

    std::vector<std::uint8_t> finish() {
        for (int index = 0; index < 5; ++index) shift_low();
        return std::move(output_);
    }

  private:
    void shift_low() {
        const std::uint32_t low32 = static_cast<std::uint32_t>(low_);
        if (low32 < 0xFF000000U || (low_ >> 32) != 0) {
            const std::uint8_t carry = static_cast<std::uint8_t>(low_ >> 32);
            std::uint8_t pending = cache_;
            do {
                output_.push_back(static_cast<std::uint8_t>(pending + carry));
                pending = 0xFFU;
            } while (--cache_size_ != 0);
            cache_ = static_cast<std::uint8_t>(low32 >> 24);
        }
        ++cache_size_;
        low_ = static_cast<std::uint64_t>(low32 << 8);
    }

    std::uint64_t low_ = 0;
    std::uint32_t range_ = 0xFFFFFFFFU;
    std::uint8_t cache_ = 0;
    std::size_t cache_size_ = 1;
    std::vector<std::uint8_t> output_;
};

class RangeDecoderNative {
  public:
    RangeDecoderNative(const std::vector<std::uint8_t>& input, std::size_t offset)
        : input_(input), position_(offset) {
        for (int index = 0; index < 5; ++index) {
            code_ = (code_ << 8) | next_byte();
        }
    }

    std::uint32_t frequency(std::uint32_t total) {
        scale_ = range_ / total;
        return std::min(code_ / scale_, total - 1);
    }

    void consume(std::uint32_t start, std::uint32_t size) {
        code_ -= start * scale_;
        range_ = size * scale_;
        while (range_ < kRangeTop) {
            code_ = (code_ << 8) | next_byte();
            range_ <<= 8;
        }
    }

  private:
    std::uint32_t next_byte() {
        return position_ < input_.size() ? input_[position_++] : 0;
    }

    const std::vector<std::uint8_t>& input_;
    std::size_t position_;
    std::uint32_t range_ = 0xFFFFFFFFU;
    std::uint32_t code_ = 0;
    std::uint32_t scale_ = 0;
};

struct MatchParameters {
    std::uint32_t history = 0;
    std::uint8_t table_bits = 0;
    std::uint8_t context = 0;
    std::array<double, 4> weights{};
    std::array<double, 4> confidences{};
};

class MatchModelNative {
  public:
    explicit MatchModelNative(const MatchParameters& parameters)
        : parameters_(parameters),
          history_(parameters.history),
          table_(parameters.history ? (1U << parameters.table_bits) : 0,
                 std::numeric_limits<std::uint32_t>::max()),
          mask_(table_.empty() ? 0 : static_cast<std::uint32_t>(table_.size() - 1)) {}

    std::pair<int, int> predict() const {
        return pointer_ == invalid_position()
            ? std::pair<int, int>{-1, 0}
            : std::pair<int, int>{history_[pointer_ % history_.size()], length_};
    }

    void update(std::uint8_t byte) {
        if (history_.empty()) return;
        if (pointer_ != invalid_position()) {
            if (history_[pointer_ % history_.size()] == byte) {
                ++pointer_;
                ++length_;
            } else {
                pointer_ = invalid_position();
                length_ = 0;
            }
        }
        history_[position_ % history_.size()] = byte;
        ++position_;
        if (position_ < parameters_.context) return;
        std::array<std::uint8_t, 16> key{};
        for (std::uint32_t index = 0; index < parameters_.context; ++index) {
            key[index] = history_[
                (position_ - parameters_.context + index) % history_.size()
            ];
        }
        std::uint32_t crc = 0xFFFFFFFFU;
        for (std::uint32_t index = 0; index < parameters_.context; ++index) {
            crc = __crc32b(crc, key[index]);
        }
        const std::uint32_t slot = (crc ^ 0xFFFFFFFFU) & mask_;
        if (pointer_ == invalid_position()) {
            const std::uint32_t candidate = table_[slot];
            if (candidate != invalid_position()
                && position_ - candidate <= parameters_.history
                && candidate >= parameters_.context) {
                bool equal = true;
                for (std::uint32_t index = 0; index < parameters_.context; ++index) {
                    equal &= history_[
                        (candidate - parameters_.context + index) % history_.size()
                    ] == key[index];
                }
                if (equal) {
                    pointer_ = candidate;
                    length_ = parameters_.context;
                }
            }
        }
        table_[slot] = position_;
    }

  private:
    static constexpr std::uint32_t invalid_position() {
        return std::numeric_limits<std::uint32_t>::max();
    }
    MatchParameters parameters_;
    std::vector<std::uint8_t> history_;
    std::vector<std::uint32_t> table_;
    std::uint32_t mask_ = 0;
    std::uint32_t position_ = 0;
    std::uint32_t pointer_ = invalid_position();
    int length_ = 0;
};

float32x4_t exp2_quartic(float32x4_t exponent) {
    const float32x4_t clipped = vmaxq_f32(
        vdupq_n_f32(-16.0F), vminq_f32(vdupq_n_f32(0.0F), exponent)
    );
    const float32x4_t base2 = vmulq_n_f32(clipped, 1.4426950408889634F);
    const int32x4_t integer = vcvtmq_s32_f32(base2);
    const float32x4_t fraction = vsubq_f32(base2, vcvtq_f32_s32(integer));
    const float32x4_t power = vreinterpretq_f32_s32(
        vshlq_n_s32(vaddq_s32(integer, vdupq_n_s32(127)), 23)
    );
    float32x4_t polynomial = vfmaq_n_f32(
        vdupq_n_f32(0.05550410866482158F),
        fraction, 0.009618129107628477F
    );
    polynomial = vfmaq_f32(
        vdupq_n_f32(0.2402265069591007F), fraction, polynomial
    );
    polynomial = vfmaq_f32(
        vdupq_n_f32(0.6931471805599453F), fraction, polynomial
    );
    polynomial = vfmaq_f32(vdupq_n_f32(1.0F), fraction, polynomial);
    return vmulq_f32(power, polynomial);
}

struct Frequencies {
    alignas(64) std::array<std::uint32_t, 257> cumulative{};
    alignas(64) std::array<double, 256> expert_probabilities{};
    alignas(64) std::array<std::uint32_t, 256> expert_frequencies{};
    std::uint32_t expert_total = 0;
};

class AdaptiveUniformMixer {
  public:
    void reset() { log_odds_ = 0.0; }
    double model_weight() const {
        if (log_odds_ >= 36.0) return 1.0;
        if (log_odds_ <= -36.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-log_odds_));
    }

    void update(double model_probability) {
        log_odds_ += std::log(std::max(model_probability, 1.0e-300) * 256.0);
    }

  private:
    double log_odds_ = 0.0;
};

class AdaptiveIntegerMixer {
  public:
    static constexpr std::uint32_t kOne = 1U << 16;

    void reset() { model_weight_ = kOne / 2; }
    std::uint32_t model_weight() const { return model_weight_; }

    void update(std::uint32_t model_frequency, std::uint32_t model_total) {
        const std::uint64_t model_term =
            static_cast<std::uint64_t>(model_weight_)
            * model_frequency * 256U;
        const std::uint64_t uniform_term =
            static_cast<std::uint64_t>(kOne - model_weight_) * model_total;
        const std::uint64_t total = model_term + uniform_term;
        if (total == 0) {
            model_weight_ = kOne / 2;
            return;
        }
        // Keep both experts recoverable. Quantizing an exact Bayesian
        // posterior to either endpoint would make that endpoint absorbing,
        // unlike the legacy log-odds update's much wider dynamic range.
        model_weight_ = std::clamp<std::uint32_t>(
            static_cast<std::uint32_t>(
                (model_term * kOne + total / 2) / total
            ),
            1U, kOne - 1U
        );
    }

  private:
    std::uint32_t model_weight_ = kOne / 2;
};

void uniform_frequencies(Frequencies& result) {
    for (std::uint32_t symbol = 0; symbol <= 256; ++symbol) {
        result.cumulative[symbol] = symbol << 8;
    }
}

void make_frequencies(
    Frequencies& result,
    const float* logits,
    const MatchParameters& match,
    int predicted,
    int match_length,
    double model_weight = 1.0
) {
    float32x4_t maximum = vld1q_f32(logits);
    for (int symbol = 4; symbol < 256; symbol += 4) {
        maximum = vmaxq_f32(maximum, vld1q_f32(logits + symbol));
    }
    const float scalar_maximum = vmaxvq_f32(maximum);
    alignas(64) std::array<float, 256> weights{};
#if defined(TERNLOG_CODEC_EXACT_EXP)
    double exact_sum = 0.0;
    for (int symbol = 0; symbol < 256; ++symbol) {
        weights[symbol] = static_cast<float>(std::exp(
            static_cast<double>(logits[symbol] - scalar_maximum)
        ));
        exact_sum += weights[symbol];
    }
    const float inverse = static_cast<float>(1.0 / exact_sum);
#else
    float32x4_t sum = vdupq_n_f32(0.0F);
    for (int symbol = 0; symbol < 256; symbol += 4) {
        const float32x4_t values = exp2_quartic(
            vsubq_f32(vld1q_f32(logits + symbol), vdupq_n_f32(scalar_maximum))
        );
        vst1q_f32(weights.data() + symbol, values);
        sum = vaddq_f32(sum, values);
    }
    const float inverse = 1.0F / vaddvq_f32(sum);
#endif
    int bucket = -1;
    for (int index = 0; index < 4; ++index) {
        if (match_length >= kMatchBuckets[index]) bucket = index;
    }
    const bool active = predicted >= 0 && bucket >= 0
        && match.weights[bucket] != 0.0;
    const double weight = active ? match.weights[bucket] : 0.0;
    const double confidence = active ? match.confidences[bucket] : 0.0;
    const double other = active ? (1.0 - confidence) / 255.0 : 0.0;
    std::uint32_t cumulative = 0;
    result.cumulative[0] = 0;
    std::uint32_t largest_frequency = 0;
    int largest_symbol = 0;
    for (int symbol = 0; symbol < 256; ++symbol) {
        double probability = static_cast<double>(weights[symbol] * inverse);
        if (active) {
            probability = (1.0 - weight) * probability + weight * other;
            if (symbol == predicted) probability += weight * (confidence - other);
        }
        result.expert_probabilities[symbol] = probability;
        probability = model_weight * probability
            + (1.0 - model_weight) / 256.0;
        const std::uint32_t frequency = static_cast<std::uint32_t>(
            std::floor(probability * kFrequencyBudget)
        ) + 1U;
        cumulative += frequency;
        result.cumulative[symbol + 1] = cumulative;
        if (frequency > largest_frequency) {
            largest_frequency = frequency;
            largest_symbol = symbol;
        }
    }
    if (cumulative > 65536U) {
        const std::uint32_t excess = cumulative - 65536U;
        for (int symbol = largest_symbol + 1; symbol <= 256; ++symbol) {
            result.cumulative[symbol] -= excess;
        }
    }
}

void mix_coded_frequencies_with_uniform(
    Frequencies& result,
    double model_weight
) {
    // Adapt against the exact discrete neural distribution seen by the range
    // coder, not the pre-quantization float probabilities. Otherwise the
    // posterior and coded expert silently diverge on low-probability symbols,
    // defeating the Bayesian mixture's bounded-regret property.
    const std::uint32_t model_total = result.cumulative[256];
    std::array<std::uint32_t, 256> model_frequencies{};
    for (int symbol = 0; symbol < 256; ++symbol) {
        model_frequencies[symbol] =
            result.cumulative[symbol + 1] - result.cumulative[symbol];
    }
    std::uint32_t cumulative = 0;
    result.cumulative[0] = 0;
    std::uint32_t largest_frequency = 0;
    int largest_symbol = 0;
    for (int symbol = 0; symbol < 256; ++symbol) {
        const double neural_probability =
            static_cast<double>(model_frequencies[symbol]) / model_total;
        result.expert_probabilities[symbol] = neural_probability;
        const double probability = model_weight * neural_probability
            + (1.0 - model_weight) / 256.0;
        const std::uint32_t frequency = static_cast<std::uint32_t>(
            std::floor(probability * kFrequencyBudget)
        ) + 1U;
        cumulative += frequency;
        result.cumulative[symbol + 1] = cumulative;
        if (frequency > largest_frequency) {
            largest_frequency = frequency;
            largest_symbol = symbol;
        }
    }
    if (cumulative > 65536U) {
        const std::uint32_t excess = cumulative - 65536U;
        for (int symbol = largest_symbol + 1; symbol <= 256; ++symbol) {
            result.cumulative[symbol] -= excess;
        }
    }
}

void mix_integer_frequencies_with_uniform(
    Frequencies& result,
    std::uint32_t model_weight
) {
    constexpr int shift = 24;  // Q16 posterior times the 256-way uniform total.
    const std::uint32_t model_total = result.cumulative[256];
    result.expert_total = model_total;
    for (int symbol = 0; symbol < 256; ++symbol) {
        result.expert_frequencies[symbol] =
            result.cumulative[symbol + 1] - result.cumulative[symbol];
    }
    std::uint32_t cumulative = 0;
    result.cumulative[0] = 0;
#if defined(TERNLOG_INTEGER_MIX_NEON)
    // Exact Q24 arithmetic, four symbols at a time. Integer prefix sums retain
    // the scalar cumulative distribution, including minimum-frequency clamping.
    const uint64x2_t uniform = vdupq_n_u64(
        static_cast<std::uint64_t>(AdaptiveIntegerMixer::kOne - model_weight)
        * model_total);
    const uint32x2_t scale = vdup_n_u32(model_weight * 256U);
    for (int symbol = 0; symbol < 256; symbol += 4) {
        const uint32x4_t expert = vld1q_u32(result.expert_frequencies.data() + symbol);
        const uint64x2_t lo = vaddq_u64(vmull_u32(vget_low_u32(expert), scale), uniform);
        const uint64x2_t hi = vaddq_u64(vmull_u32(vget_high_u32(expert), scale), uniform);
        uint32x4_t f = vmaxq_u32(vcombine_u32(vshrn_n_u64(lo, shift),
                                            vshrn_n_u64(hi, shift)), vdupq_n_u32(1));
        const uint32x4_t zero = vdupq_n_u32(0);
        f = vaddq_u32(f, vextq_u32(zero, f, 3));
        f = vaddq_u32(f, vextq_u32(zero, f, 2));
        f = vaddq_u32(f, vdupq_n_u32(cumulative));
        vst1q_u32(result.cumulative.data() + symbol + 1, f);
        cumulative = vgetq_lane_u32(f, 3);
    }
#else
    for (int symbol = 0; symbol < 256; ++symbol) {
        const std::uint32_t model_frequency = result.expert_frequencies[symbol];
        const std::uint64_t numerator =
            static_cast<std::uint64_t>(model_weight)
                * model_frequency * 256U
            + static_cast<std::uint64_t>(
                AdaptiveIntegerMixer::kOne - model_weight
            ) * model_total;
        const std::uint32_t frequency = std::max<std::uint32_t>(
            1U, static_cast<std::uint32_t>(numerator >> shift)
        );
        cumulative += frequency;
        result.cumulative[symbol + 1] = cumulative;
    }
#endif
}

MatchParameters embedded_match(bool enabled) {
    MatchParameters result;
    if (!enabled) return result;
    result.history = layout::kMatchHistory;
    result.table_bits = static_cast<std::uint8_t>(layout::kMatchTableBits);
    result.context = static_cast<std::uint8_t>(layout::kMatchContext);
    std::copy_n(layout::kMatchWeights, 4, result.weights.begin());
    std::copy_n(layout::kMatchConfidences, 4, result.confidences.begin());
    return result;
}

constexpr std::uint64_t fnv1a_64(std::string_view text) {
    std::uint64_t value = 1469598103934665603ULL;
    for (char byte : text) {
        value ^= static_cast<std::uint8_t>(byte);
        value *= 1099511628211ULL;
    }
    return value;
}

constexpr std::uint64_t tli_arithmetic_tag() {
    // Bump the explicit revision whenever TLI frequency semantics change.
    // Revision 2 makes every four-lane payload length explicit.  This prevents
    // arithmetic-coder termination redundancy from hiding a truncated final
    // lane while preserving byte-exact decoded output.
    std::uint64_t tag = 0x544c493200000002ULL;
#if defined(__clang_version__)
    tag ^= fnv1a_64(__clang_version__);
#elif defined(__VERSION__)
    tag ^= fnv1a_64(__VERSION__);
#endif
#if defined(__FAST_MATH__)
    tag ^= 0x4f6f624d0cc10b4dULL;
#endif
#if defined(TERNLOG_CODEC_EXACT_EXP)
    tag ^= 0x8ed722e463dbe831ULL;
#endif
    return tag;
}

std::array<std::uint8_t, 16> stream_model_fingerprint(
    bool adaptive_integer
) {
    std::array<std::uint8_t, 16> result{};
    std::copy_n(layout::kModelFingerprint, result.size(), result.begin());
    if (!adaptive_integer) return result;
    const std::uint64_t tag = tli_arithmetic_tag();
    const std::uint64_t second = (tag << 29) | (tag >> 35);
    for (int index = 0; index < 8; ++index) {
        result[index] ^= static_cast<std::uint8_t>(tag >> (index * 8));
        result[index + 8] ^= static_cast<std::uint8_t>(second >> (index * 8));
    }
    return result;
}

std::vector<std::uint8_t> make_header(
    std::size_t original_bytes,
    std::uint32_t original_crc,
    const MatchParameters& match,
    bool adaptive_integer = false
) {
    std::vector<std::uint8_t> result;
    result.reserve(kHeaderBytes);
    result.insert(result.end(), {'T', 'L', 'Z', '2'});
    append_scalar(result, static_cast<std::uint64_t>(original_bytes));
    append_scalar(result, static_cast<std::uint32_t>(layout::kContext));
    const auto fingerprint = stream_model_fingerprint(adaptive_integer);
    result.insert(result.end(), fingerprint.begin(), fingerprint.end());
    append_scalar(result, original_crc);
    append_scalar(result, match.history);
    append_scalar(result, match.table_bits);
    append_scalar(result, match.context);
    for (double value : match.weights) append_scalar(result, value);
    for (double value : match.confidences) append_scalar(result, value);
    if (result.size() != kHeaderBytes) throw std::runtime_error("internal header size error");
    return result;
}

struct ParsedHeader {
    std::uint64_t original_bytes;
    std::uint32_t block_size;
    std::uint32_t crc;
    MatchParameters match;
    bool adaptive_uniform = false;
    bool adaptive_coded = false;
    bool adaptive_carry = false;
    bool adaptive_integer = false;
};

ParsedHeader parse_header(const std::vector<std::uint8_t>& stream) {
    const bool adaptive_legacy = stream.size() >= 4
        && std::memcmp(stream.data(), "TLA2", 4) == 0;
    const bool adaptive_coded = stream.size() >= 4
        && std::memcmp(stream.data(), "TLC2", 4) == 0;
    const bool adaptive_carry = stream.size() >= 4
        && std::memcmp(stream.data(), "TLG2", 4) == 0;
    const bool adaptive_integer = stream.size() >= 4
        && std::memcmp(stream.data(), "TLI2", 4) == 0;
    const bool adaptive_uniform =
        adaptive_legacy || adaptive_coded || adaptive_carry || adaptive_integer;
    if (stream.size() < kHeaderBytes
        || (!adaptive_uniform && std::memcmp(stream.data(), "TLZ2", 4) != 0)) {
        throw std::runtime_error("not a TLZ2 stream");
    }
    std::size_t offset = 4;
    ParsedHeader result{};
    result.adaptive_uniform = adaptive_uniform;
    result.adaptive_coded = adaptive_coded || adaptive_carry || adaptive_integer;
    result.adaptive_carry = adaptive_carry;
    result.adaptive_integer = adaptive_integer;
    result.original_bytes = read_scalar<std::uint64_t>(stream, offset);
    result.block_size = read_scalar<std::uint32_t>(stream, offset);
    const auto expected_fingerprint = stream_model_fingerprint(adaptive_integer);
    if (std::memcmp(
            stream.data() + offset, expected_fingerprint.data(),
            expected_fingerprint.size()
        ) != 0) {
        throw std::runtime_error(
            "model or arithmetic-runtime fingerprint does not match stream"
        );
    }
    offset += 16;
    result.crc = read_scalar<std::uint32_t>(stream, offset);
    result.match.history = read_scalar<std::uint32_t>(stream, offset);
    result.match.table_bits = read_scalar<std::uint8_t>(stream, offset);
    result.match.context = read_scalar<std::uint8_t>(stream, offset);
    for (double& value : result.match.weights) value = read_scalar<double>(stream, offset);
    for (double& value : result.match.confidences) value = read_scalar<double>(stream, offset);
    if (result.block_size != static_cast<std::uint32_t>(layout::kContext)) {
        throw std::runtime_error("stream block size does not match model");
    }
    return result;
}

enum class AdaptiveCodingMode {
    None,
    LegacyFloatReset,
    CodedReset,
    CodedCarry,
    IntegerReset,
};

std::vector<std::uint8_t> compress_native(
    const std::vector<std::uint8_t>& input,
    bool use_match,
    AdaptiveCodingMode adaptive_mode = AdaptiveCodingMode::None
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
    , bool known_input_tiles = false
#endif
) {
    const bool adaptive_uniform = adaptive_mode != AdaptiveCodingMode::None;
    const bool adaptive_coded = adaptive_mode == AdaptiveCodingMode::CodedReset
        || adaptive_mode == AdaptiveCodingMode::CodedCarry
        || adaptive_mode == AdaptiveCodingMode::IntegerReset;
    const bool adaptive_carry = adaptive_mode == AdaptiveCodingMode::CodedCarry;
    const bool adaptive_integer = adaptive_mode == AdaptiveCodingMode::IntegerReset;
    const MatchParameters match = embedded_match(use_match);
    MatchModelNative matcher(match);
    Transformer model;
#if defined(TERNLOG_PROFILE_PHASES)
    model.reset_profile();
#endif
    RangeEncoderNative encoder;
    Frequencies frequencies;
    AdaptiveUniformMixer adaptive_mixer;
    AdaptiveIntegerMixer adaptive_integer_mixer;
    alignas(64) float logits[256];
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
    alignas(64) float tile_logits[32 * 256];
#endif
#if defined(TERNLOG_CODEC_PROFILE)
    std::uint64_t model_nanoseconds = 0;
    std::uint64_t frequency_nanoseconds = 0;
    std::uint64_t coder_match_nanoseconds = 0;
#endif
    for (std::size_t base = 0; base < input.size(); base += layout::kContext) {
        model.reset();
        if (!adaptive_carry || base == 0) adaptive_mixer.reset();
        adaptive_integer_mixer.reset();
        const std::size_t length = std::min<std::size_t>(
            layout::kContext, input.size() - base
        );
        for (std::size_t index = 0; index < length; ++index) {
#if defined(TERNLOG_CODEC_PROFILE)
            std::chrono::steady_clock::time_point coder_start;
#endif
            const auto [predicted, match_length] = matcher.predict();
            if (index == 0) {
                uniform_frequencies(frequencies);
            } else {
#if defined(TERNLOG_CODEC_PROFILE)
                const auto model_start = std::chrono::steady_clock::now();
#endif
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
                if (known_input_tiles) {
                    const auto offset = (index - 1) % 32;
                    if (offset == 0) {
                        model.forward_known_tile(input.data() + base + index - 1,
                            std::min<std::size_t>(32, length - index), tile_logits);
                    }
                    std::memcpy(logits, tile_logits + offset * 256, sizeof(logits));
                } else
#endif
                model.forward(input[base + index - 1], logits);
#if defined(TERNLOG_CODEC_PROFILE)
                const auto frequency_start = std::chrono::steady_clock::now();
                model_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    frequency_start - model_start
                ).count();
#endif
                make_frequencies(
                    frequencies, logits, match, predicted, match_length,
                    adaptive_uniform && !adaptive_coded
                        ? adaptive_mixer.model_weight()
                        : 1.0
                );
                if (adaptive_integer) {
                    mix_integer_frequencies_with_uniform(
                        frequencies, adaptive_integer_mixer.model_weight()
                    );
                } else if (adaptive_coded) {
                    mix_coded_frequencies_with_uniform(
                        frequencies, adaptive_mixer.model_weight()
                    );
                }
#if defined(TERNLOG_CODEC_PROFILE)
                coder_start = std::chrono::steady_clock::now();
                frequency_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    coder_start - frequency_start
                ).count();
#endif
            }
            const std::uint8_t symbol = input[base + index];
            const std::uint32_t start = frequencies.cumulative[symbol];
            const std::uint32_t size = frequencies.cumulative[symbol + 1] - start;
            encoder.encode(start, size, frequencies.cumulative[256]);
            if (adaptive_uniform && index != 0) {
                if (adaptive_integer) {
                    adaptive_integer_mixer.update(
                        frequencies.expert_frequencies[symbol],
                        frequencies.expert_total
                    );
                } else {
                    adaptive_mixer.update(frequencies.expert_probabilities[symbol]);
                }
            }
            matcher.update(symbol);
#if defined(TERNLOG_CODEC_PROFILE)
            if (index != 0) {
                coder_match_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - coder_start
                ).count();
            }
#endif
        }
    }
#if defined(TERNLOG_PROFILE_PHASES)
    model.print_profile();
#endif
#if defined(TERNLOG_CODEC_PROFILE)
    const double counted = static_cast<double>(
        model_nanoseconds + frequency_nanoseconds + coder_match_nanoseconds
    );
    std::fprintf(stderr,
        "{\"model_ns\":%llu,\"frequency_ns\":%llu,\"coder_match_ns\":%llu,"
        "\"model_share\":%.6f,\"frequency_share\":%.6f,\"coder_match_share\":%.6f}\n",
        static_cast<unsigned long long>(model_nanoseconds),
        static_cast<unsigned long long>(frequency_nanoseconds),
        static_cast<unsigned long long>(coder_match_nanoseconds),
        model_nanoseconds / counted, frequency_nanoseconds / counted,
        coder_match_nanoseconds / counted);
#endif
    std::vector<std::uint8_t> result = make_header(
        input.size(), static_cast<std::uint32_t>(
            crc32(0, input.data(), static_cast<uInt>(input.size()))
        ), match, adaptive_integer
    );
    if (adaptive_uniform) {
        const char* magic = adaptive_carry
            ? "TLG2"
            : adaptive_integer ? "TLI2"
            : adaptive_coded ? "TLC2" : "TLA2";
        std::memcpy(result.data(), magic, 4);
    }
    std::vector<std::uint8_t> payload = encoder.finish();
    result.insert(result.end(), payload.begin(), payload.end());
    return result;
}

std::vector<std::uint8_t> decompress_native(
    const std::vector<std::uint8_t>& stream,
    bool verify_crc = true
) {
    const ParsedHeader header = parse_header(stream);
    MatchModelNative matcher(header.match);
    Transformer model;
    RangeDecoderNative decoder(stream, kHeaderBytes);
    Frequencies frequencies;
    AdaptiveUniformMixer adaptive_mixer;
    AdaptiveIntegerMixer adaptive_integer_mixer;
    alignas(64) float logits[256];
    std::vector<std::uint8_t> output;
    output.reserve(static_cast<std::size_t>(header.original_bytes));
    while (output.size() < header.original_bytes) {
        const std::size_t base = output.size();
        const std::size_t length = std::min<std::size_t>(
            header.block_size, header.original_bytes - base
        );
        model.reset();
        if (!header.adaptive_carry || base == 0) adaptive_mixer.reset();
        adaptive_integer_mixer.reset();
        for (std::size_t index = 0; index < length; ++index) {
            const auto [predicted, match_length] = matcher.predict();
            if (index == 0) {
                uniform_frequencies(frequencies);
            } else {
                model.forward(output.back(), logits);
                make_frequencies(
                    frequencies, logits, header.match, predicted, match_length,
                    header.adaptive_uniform && !header.adaptive_coded
                        ? adaptive_mixer.model_weight()
                        : 1.0
                );
                if (header.adaptive_integer) {
                    mix_integer_frequencies_with_uniform(
                        frequencies, adaptive_integer_mixer.model_weight()
                    );
                } else if (header.adaptive_coded) {
                    mix_coded_frequencies_with_uniform(
                        frequencies, adaptive_mixer.model_weight()
                    );
                }
            }
            const std::uint32_t value = decoder.frequency(
                frequencies.cumulative[256]
            );
            const auto found = std::upper_bound(
                frequencies.cumulative.begin() + 1,
                frequencies.cumulative.end(), value
            );
            const std::uint8_t symbol = static_cast<std::uint8_t>(
                found - frequencies.cumulative.begin() - 1
            );
            const std::uint32_t start = frequencies.cumulative[symbol];
            decoder.consume(
                start, frequencies.cumulative[symbol + 1] - start
            );
            output.push_back(symbol);
            if (header.adaptive_uniform && index != 0) {
                if (header.adaptive_integer) {
                    adaptive_integer_mixer.update(
                        frequencies.expert_frequencies[symbol],
                        frequencies.expert_total
                    );
                } else {
                    adaptive_mixer.update(frequencies.expert_probabilities[symbol]);
                }
            }
            matcher.update(symbol);
        }
    }
    const std::uint32_t actual_crc = static_cast<std::uint32_t>(
        crc32(0, output.data(), static_cast<uInt>(output.size()))
    );
    if (verify_crc && actual_crc != header.crc) {
        throw std::runtime_error("checksum mismatch");
    }
    return output;
}

constexpr int kCodecLanes = 4;

// Weights are immutable and shared; each lane owns its model/KV, matcher and
// coder. Join every worker before propagating exceptions or releasing buffers.
template <typename Function>
void run_codec_lanes(int workers, Function function) {
    if (workers < 1 || workers > kCodecLanes)
        throw std::runtime_error("threads must be in 1..4");
    if (workers == 1) {
        for (int lane = 0; lane < kCodecLanes; ++lane) function(lane);
        return;
    }
    std::vector<std::future<void>> tasks;
    for (int worker = 0; worker < workers; ++worker) {
        tasks.push_back(std::async(std::launch::async, [&, worker] {
            for (int lane = worker; lane < kCodecLanes; lane += workers)
                function(lane);
        }));
    }
    std::exception_ptr failure;
    for (auto& task : tasks) {
        try { task.get(); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
}

std::array<std::size_t, kCodecLanes + 1> lane_boundaries(std::size_t size) {
    const std::size_t blocks = (size + layout::kContext - 1) / layout::kContext;
    std::array<std::size_t, kCodecLanes + 1> result{};
    result[0] = 0;
    for (int lane = 1; lane < kCodecLanes; ++lane) {
        result[lane] = std::min(
            size,
            (blocks * static_cast<std::size_t>(lane) / kCodecLanes)
                * layout::kContext
        );
    }
    result[kCodecLanes] = size;
    return result;
}

std::vector<std::uint8_t> compress_four_lane(
    const std::vector<std::uint8_t>& input,
    bool use_match,
    AdaptiveCodingMode adaptive_mode = AdaptiveCodingMode::None,
    int workers = 1
) {
    if (adaptive_mode != AdaptiveCodingMode::None
        && adaptive_mode != AdaptiveCodingMode::CodedReset
        && adaptive_mode != AdaptiveCodingMode::IntegerReset) {
        throw std::runtime_error("four-lane codec supports coded-reset adaptation only");
    }
    const auto edges = lane_boundaries(input.size());
    std::array<std::vector<std::uint8_t>, kCodecLanes> streams;
    run_codec_lanes(workers, [&](int lane) {
        streams[lane] = compress_native(
            std::vector<std::uint8_t>(
                input.begin() + edges[lane], input.begin() + edges[lane + 1]
            ),
            use_match,
            adaptive_mode
        );
    });
    const MatchParameters match = embedded_match(use_match);
    std::vector<std::uint8_t> result = make_header(
        input.size(),
        static_cast<std::uint32_t>(crc32(
            0, input.data(), static_cast<uInt>(input.size())
        )),
        match, adaptive_mode == AdaptiveCodingMode::IntegerReset
    );
    const char* magic = adaptive_mode == AdaptiveCodingMode::IntegerReset
        ? "TLI4"
        : adaptive_mode == AdaptiveCodingMode::CodedReset ? "TLC4" : "TLZ4";
    std::memcpy(result.data(), magic, 4);
    // Store every length so truncation or trailing bytes fail closed.  Inferring
    // the final length from EOF allowed a redundant arithmetic-coder tail byte
    // to be removed without changing the decoded CRC.
    for (int lane = 0; lane < kCodecLanes; ++lane) {
        append_scalar(
            result,
            static_cast<std::uint64_t>(streams[lane].size() - kHeaderBytes)
        );
    }
    for (const auto& stream : streams) {
        result.insert(result.end(), stream.begin() + kHeaderBytes, stream.end());
    }
    return result;
}

std::vector<std::uint8_t> decompress_four_lane(
    const std::vector<std::uint8_t>& stream,
    int workers = 1
) {
    constexpr std::size_t directory_bytes = kCodecLanes * sizeof(std::uint64_t);
    const bool adaptive_coded = stream.size() >= 4
        && std::memcmp(stream.data(), "TLC4", 4) == 0;
    const bool adaptive_integer = stream.size() >= 4
        && std::memcmp(stream.data(), "TLI4", 4) == 0;
    const bool adaptive = adaptive_coded || adaptive_integer;
    if (stream.size() < kHeaderBytes + directory_bytes
        || (!adaptive && std::memcmp(stream.data(), "TLZ4", 4) != 0)) {
        throw std::runtime_error("not a TLZ4 stream");
    }
    std::vector<std::uint8_t> common(
        stream.begin(), stream.begin() + kHeaderBytes
    );
    std::memcpy(
        common.data(),
        adaptive_integer ? "TLI2" : adaptive_coded ? "TLC2" : "TLZ2", 4
    );
    const ParsedHeader header = parse_header(common);
    const auto edges = lane_boundaries(static_cast<std::size_t>(header.original_bytes));
    std::array<std::size_t, kCodecLanes> payload_bytes{};
    std::size_t offset = kHeaderBytes;
    std::size_t known_payload = 0;
    for (int lane = 0; lane < kCodecLanes; ++lane) {
        payload_bytes[lane] = read_scalar<std::uint64_t>(stream, offset);
        if (payload_bytes[lane] > stream.size() - known_payload)
            throw std::runtime_error("four-lane payload length overflow");
        known_payload += payload_bytes[lane];
    }
    if (known_payload > stream.size() - offset
        || offset + known_payload != stream.size()) {
        throw std::runtime_error("four-lane payload length mismatch");
    }

    std::vector<std::uint8_t> result;
    result.reserve(static_cast<std::size_t>(header.original_bytes));
    std::array<std::vector<std::uint8_t>, kCodecLanes> nested_streams;
    std::array<std::vector<std::uint8_t>, kCodecLanes> decoded_lanes;
    for (int lane = 0; lane < kCodecLanes; ++lane) {
        const std::size_t lane_bytes = edges[lane + 1] - edges[lane];
        std::vector<std::uint8_t> nested = make_header(
            lane_bytes, 0, header.match, adaptive_integer
        );
        if (adaptive) {
            std::memcpy(
                nested.data(), adaptive_integer ? "TLI2" : "TLC2", 4
            );
        }
        if (offset + payload_bytes[lane] > stream.size()) {
            throw std::runtime_error("truncated TLZ4 payload");
        }
        nested.insert(
            nested.end(), stream.begin() + offset,
            stream.begin() + offset + payload_bytes[lane]
        );
        offset += payload_bytes[lane];
        nested_streams[lane] = std::move(nested);
    }
    run_codec_lanes(workers, [&](int lane) {
        decoded_lanes[lane] = decompress_native(nested_streams[lane], false);
    });
    for (const auto& decoded : decoded_lanes) {
        result.insert(result.end(), decoded.begin(), decoded.end());
    }
    const std::uint32_t actual_crc = static_cast<std::uint32_t>(
        crc32(0, result.data(), static_cast<uInt>(result.size()))
    );
    if (actual_crc != header.crc) throw std::runtime_error("checksum mismatch");
    return result;
}

#if defined(TERNLOG_QKV_BATCH4_HOOK)

#ifndef TERNLOG_QKV_AUDIT_BATCHES
#define TERNLOG_QKV_AUDIT_BATCHES 6
#endif

#ifndef TERNLOG_FFN_AUDIT_BATCHES
#define TERNLOG_FFN_AUDIT_BATCHES 6
#endif

#if defined(TERNLOG_FFN_BATCH4_HOOK)
extern "C" void ternlog_vocab_ffn_sdot_batch4_raw_ptrs_2row(
    std::int32_t* const*, const std::int8_t* const*, const std::int8_t*
);
extern "C" void ternlog_vocab_ffn_sdot_1row(
    std::int32_t*, const std::int8_t*, const std::int8_t*
);
extern "C" void ternlog_vocab_ffn_batch4_postprocess_ptrs(
    float* const*, const std::uint8_t* const*, const std::int32_t* const*,
    const float*, const float*, const float*
);
extern "C" void ternlog_vocab_ffn_scalar_postprocess(
    float*, const std::uint8_t*, const std::int32_t*,
    const float*, const float*, const float*
);
#endif

struct alignas(16) FiberContext {
    std::uint64_t x19_x28[10]{};
    std::uint64_t frame_pointer = 0;
    std::uint64_t link_register = 0;
    std::uint64_t stack_pointer = 0;
    std::uint64_t d8_d15[8]{};
};

static_assert(offsetof(FiberContext, frame_pointer) == 80);
static_assert(offsetof(FiberContext, link_register) == 88);
static_assert(offsetof(FiberContext, stack_pointer) == 96);
static_assert(offsetof(FiberContext, d8_d15) == 104);

extern "C" void ternlog_m4_fiber_switch(FiberContext*, FiberContext*);
extern "C" void ternlog_m4_fiber_trampoline();

class QkvFiberScheduler;

struct QkvFiberStart {
    QkvFiberScheduler* scheduler = nullptr;
    int lane = 0;
};

thread_local QkvFiberScheduler* active_qkv_scheduler = nullptr;

enum class FiberWaitKind : std::uint8_t {
    none,
    qkv,
    ffn,
};

using QkvRawSums = std::array<
    std::array<std::int32_t, layout::kQkvRows>, kCodecLanes
>;

void qkv_tickets_batch4_fused_value(
    const std::array<QkvBatchRequest, kCodecLanes>& requests,
    QkvRawSums* audit_raw_sums = nullptr
) {
    constexpr int row_block = 8;
    constexpr int row_group = 4;
    constexpr int row_blocks = layout::kQkvRows / row_block;
    constexpr int value_offset =
        (layout::kHeads + layout::kKeyHeads) * kQkHeadDim;
    static_assert(layout::kQkvRows - value_offset == kHeadDim);

    const QkvBatchRequest& common = requests[0];
    for (int block = 0; block < row_blocks; ++block) {
        for (int group = 0; group < row_block; group += row_group) {
            int32x4_t sums[kCodecLanes][row_group];
            for (int lane = 0; lane < kCodecLanes; ++lane) {
                for (int row = 0; row < row_group; ++row) {
                    sums[lane][row] = vdupq_n_s32(0);
                }
            }
            for (int ticket = common.ticket_offsets[block];
                 ticket < common.ticket_offsets[block + 1]; ++ticket) {
                int8x16_t values[kCodecLanes];
                const int column = 16 * common.ticket_columns[ticket];
                for (int lane = 0; lane < kCodecLanes; ++lane) {
                    values[lane] = vld1q_s8(requests[lane].activation + column);
                }
                const std::int8_t* weights = common.packed_weights
                    + static_cast<std::size_t>(ticket) * row_block * 16
                    + group * 16;
                for (int row = 0; row < row_group; ++row) {
                    const int8x16_t weight = vld1q_s8(weights + row * 16);
                    for (int lane = 0; lane < kCodecLanes; ++lane) {
                        sums[lane][row] = vdotq_s32(
                            sums[lane][row], weight, values[lane]
                        );
                    }
                }
            }

            const int first_output_row = block * row_block + group;
            for (int lane = 0; lane < kCodecLanes; ++lane) {
                const QkvBatchRequest& request = requests[lane];
                const float inverse_value_scale = 1.0F / request.value_scale;
                alignas(16) std::int32_t raw_sums[row_group];
                for (int row = 0; row < row_group; ++row) {
                    raw_sums[row] = vaddvq_s32(sums[lane][row]);
                    if (audit_raw_sums != nullptr) {
                        (*audit_raw_sums)[lane][first_output_row + row] =
                            raw_sums[row];
                    }
                }
                if (first_output_row >= value_offset) {
                    store_qkv_value_cache_scalar_group(
                        raw_sums, row_group, request.activation_scale,
                        request.weight_scales + first_output_row,
                        inverse_value_scale, *request.value_cache,
                        request.layer_index, request.value_slot,
                        first_output_row - value_offset
                    );
                } else {
#if defined(TERNLOG_QKV_VECTOR_SCALE)
                    static_assert(row_group == 4);
                    const float32x4_t raw = vcvtq_f32_s32(vld1q_s32(raw_sums));
                    const float32x4_t act = vdupq_n_f32(request.activation_scale);
                    const float32x4_t ws = vld1q_f32(request.weight_scales + first_output_row);
                    float32x4_t scaled, output;
                    // Two separately rounded FP32 multiplies, exactly like the
                    // volatile scalar boundary; forbid fast-math reassociation.
                    asm("fmul %0.4s, %1.4s, %2.4s" : "=w"(scaled) : "w"(raw), "w"(act));
                    asm("fmul %0.4s, %1.4s, %2.4s" : "=w"(output) : "w"(scaled), "w"(ws));
                    vst1q_f32(request.output + first_output_row, output);
#else
                    for (int row = 0; row < row_group; ++row) {
                        const int output_row = first_output_row + row;
                        volatile float scaled = static_cast<float>(raw_sums[row])
                            * request.activation_scale;
                        request.output[output_row] = scaled
                            * request.weight_scales[output_row];
                    }
#endif
                }
            }
        }
    }
}

class QkvFiberScheduler {
  public:
    explicit QkvFiberScheduler(const std::vector<std::uint8_t>& stream) {
        constexpr std::size_t directory_bytes =
            kCodecLanes * sizeof(std::uint64_t);
        const bool adaptive_coded = stream.size() >= 4
            && std::memcmp(stream.data(), "TLC4", 4) == 0;
        const bool adaptive_integer = stream.size() >= 4
            && std::memcmp(stream.data(), "TLI4", 4) == 0;
        const bool adaptive = adaptive_coded || adaptive_integer;
        if (stream.size() < kHeaderBytes + directory_bytes
            || (!adaptive && std::memcmp(stream.data(), "TLZ4", 4) != 0)) {
            throw std::runtime_error("not a TLZ4 stream");
        }
        std::vector<std::uint8_t> common(
            stream.begin(), stream.begin() + kHeaderBytes
        );
        std::memcpy(
            common.data(),
            adaptive_integer ? "TLI2" : adaptive_coded ? "TLC2" : "TLZ2", 4
        );
        header_ = parse_header(common);
        edges_ = lane_boundaries(static_cast<std::size_t>(header_.original_bytes));

        std::array<std::size_t, kCodecLanes> payload_bytes{};
        std::size_t offset = kHeaderBytes;
        std::size_t known_payload = 0;
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            payload_bytes[lane] = read_scalar<std::uint64_t>(stream, offset);
            known_payload += payload_bytes[lane];
        }
        if (known_payload > stream.size() - offset
            || offset + known_payload != stream.size()) {
            throw std::runtime_error("four-lane payload length mismatch");
        }

        for (int lane = 0; lane < kCodecLanes; ++lane) {
            const std::size_t lane_bytes = edges_[lane + 1] - edges_[lane];
            nested_[lane] = make_header(
                lane_bytes, 0, header_.match, header_.adaptive_integer
            );
            if (offset + payload_bytes[lane] > stream.size()) {
                throw std::runtime_error("truncated TLZ4 payload");
            }
            nested_[lane].insert(
                nested_[lane].end(), stream.begin() + offset,
                stream.begin() + offset + payload_bytes[lane]
            );
            offset += payload_bytes[lane];
            decoders_[lane] = std::make_unique<RangeDecoderNative>(
                nested_[lane], kHeaderBytes
            );
            initialize_fiber(lane, lane_bytes);
        }
    }

    QkvFiberScheduler(
        const std::vector<std::uint8_t>& input,
        bool use_match,
        AdaptiveCodingMode adaptive_mode = AdaptiveCodingMode::None
    ) : encoding_input_(&input), encoding_(true) {
        if (adaptive_mode != AdaptiveCodingMode::None
            && adaptive_mode != AdaptiveCodingMode::CodedReset
            && adaptive_mode != AdaptiveCodingMode::IntegerReset) {
            throw std::runtime_error(
                "batch four-lane codec supports reset adaptation only"
            );
        }
        header_.original_bytes = input.size();
        header_.block_size = layout::kContext;
        header_.crc = static_cast<std::uint32_t>(
            crc32(0, input.data(), static_cast<uInt>(input.size()))
        );
        header_.match = embedded_match(use_match);
        header_.adaptive_uniform = adaptive_mode != AdaptiveCodingMode::None;
        header_.adaptive_coded = adaptive_mode != AdaptiveCodingMode::None;
        header_.adaptive_integer =
            adaptive_mode == AdaptiveCodingMode::IntegerReset;
        edges_ = lane_boundaries(input.size());
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            const std::size_t lane_bytes = edges_[lane + 1] - edges_[lane];
            encoders_[lane] = std::make_unique<RangeEncoderNative>();
            initialize_fiber(lane, lane_bytes);
        }
    }

    std::vector<std::uint8_t> run() {
#if defined(TERNLOG_BATCH4_PROFILE)
        const auto run_started = std::chrono::steady_clock::now();
#endif
        if (active_qkv_scheduler != nullptr
            || ternlog_qkv_batch_submit != nullptr
#if defined(TERNLOG_FFN_BATCH4_HOOK)
            || ternlog_ffn_batch_submit != nullptr
#endif
        ) {
            throw std::runtime_error("nested QKV fiber scheduler");
        }
        active_qkv_scheduler = this;
        ternlog_qkv_batch_submit = &submit_request;
#if defined(TERNLOG_FFN_BATCH4_HOOK)
        ternlog_ffn_batch_submit = &submit_ffn_request;
#endif
        try {
            while (active_lanes_ != 0) {
                bool progressed = false;
                for (int lane = 0; lane < kCodecLanes; ++lane) {
                    if (done_[lane] || waiting_[lane]) continue;
                    resume(lane);
                    progressed = true;
                    if (failure_) std::rethrow_exception(failure_);
                }
                if (waiting_count_ == kCodecLanes) {
                    execute_batch();
                    progressed = true;
                } else if (active_lanes_ < kCodecLanes && waiting_count_ != 0) {
                    execute_scalar_waiters();
                    progressed = true;
                }
                if (!progressed && active_lanes_ != 0) {
                    throw std::runtime_error("QKV fiber scheduler deadlock");
                }
            }
        } catch (...) {
            ternlog_qkv_batch_submit = nullptr;
#if defined(TERNLOG_FFN_BATCH4_HOOK)
            ternlog_ffn_batch_submit = nullptr;
#endif
            active_qkv_scheduler = nullptr;
            throw;
        }
        ternlog_qkv_batch_submit = nullptr;
#if defined(TERNLOG_FFN_BATCH4_HOOK)
        ternlog_ffn_batch_submit = nullptr;
#endif
        active_qkv_scheduler = nullptr;

        if (encoding_) {
            std::vector<std::uint8_t> result = finish_encoding();
#if defined(TERNLOG_PROFILE_PHASES)
            for (const auto& model : models_) model->print_profile();
#endif
#if defined(TERNLOG_BATCH4_PROFILE)
            print_batch_profile(run_started);
#endif
            return result;
        }

        std::vector<std::uint8_t> result;
        result.reserve(static_cast<std::size_t>(header_.original_bytes));
        for (const auto& output : outputs_) {
            result.insert(result.end(), output.begin(), output.end());
        }
        const std::uint32_t actual_crc = static_cast<std::uint32_t>(
            crc32(0, result.data(), static_cast<uInt>(result.size()))
        );
        if (actual_crc != header_.crc) {
            throw std::runtime_error("checksum mismatch");
        }
#if defined(TERNLOG_PROFILE_PHASES)
        for (const auto& model : models_) model->print_profile();
#endif
#if defined(TERNLOG_BATCH4_PROFILE)
        print_batch_profile(run_started);
#endif
        return result;
    }

    void fiber_entry(int lane) noexcept {
        try {
            if (encoding_) {
                encode_lane(lane);
            } else {
                decode_lane(lane);
            }
        } catch (...) {
            failure_ = std::current_exception();
        }
        done_[lane] = true;
        --active_lanes_;
        yield(lane);
        std::abort();
    }

  private:
    static constexpr std::size_t kFiberStackBytes = 256 * 1024;

    void initialize_fiber(int lane, std::size_t lane_bytes) {
        outputs_[lane].reserve(lane_bytes);
        matchers_[lane] = std::make_unique<MatchModelNative>(header_.match);
        models_[lane] = std::make_unique<Transformer>();
        starts_[lane] = QkvFiberStart{this, lane};
        stacks_[lane].resize(kFiberStackBytes);
        const auto stack_top = reinterpret_cast<std::uintptr_t>(
            stacks_[lane].data() + stacks_[lane].size()
        ) & ~static_cast<std::uintptr_t>(15);
        contexts_[lane].x19_x28[0] = reinterpret_cast<std::uintptr_t>(
            &starts_[lane]
        );
        contexts_[lane].link_register = reinterpret_cast<std::uintptr_t>(
            &ternlog_m4_fiber_trampoline
        );
        contexts_[lane].stack_pointer = stack_top;
    }

    static bool submit_request(const QkvBatchRequest& request) {
        QkvFiberScheduler* scheduler = active_qkv_scheduler;
        if (scheduler == nullptr || scheduler->active_lanes_ != kCodecLanes) {
            return false;
        }
        const int lane = scheduler->current_lane_;
        if (lane < 0 || scheduler->waiting_[lane]) {
            std::abort();
        }
        scheduler->requests_[lane] = request;
        scheduler->wait_kinds_[lane] = FiberWaitKind::qkv;
        scheduler->waiting_[lane] = true;
        ++scheduler->waiting_count_;
        scheduler->yield(lane);
        return true;
    }

#if defined(TERNLOG_FFN_BATCH4_HOOK)
    static bool submit_ffn_request(const TernlogFfnBatchRequest& request) {
        QkvFiberScheduler* scheduler = active_qkv_scheduler;
        if (scheduler == nullptr || scheduler->active_lanes_ != kCodecLanes) {
            return false;
        }
        const int lane = scheduler->current_lane_;
        if (lane < 0 || scheduler->waiting_[lane]) {
            std::abort();
        }
        scheduler->ffn_requests_[lane] = request;
        scheduler->wait_kinds_[lane] = FiberWaitKind::ffn;
        scheduler->waiting_[lane] = true;
        ++scheduler->waiting_count_;
        scheduler->yield(lane);
        return true;
    }
#endif

    void execute_batch() {
        const FiberWaitKind kind = wait_kinds_[0];
        for (int lane = 1; lane < kCodecLanes; ++lane) {
            if (wait_kinds_[lane] != kind) {
                throw std::runtime_error("fiber lanes reached different wait kinds");
            }
        }
        if (kind == FiberWaitKind::qkv) {
            execute_qkv_batch();
            return;
        }
#if defined(TERNLOG_FFN_BATCH4_HOOK)
        if (kind == FiberWaitKind::ffn) {
            execute_ffn_batch();
            return;
        }
#endif
        throw std::runtime_error("invalid fiber wait kind");
    }

    void execute_qkv_batch() {
        const QkvBatchRequest& first = requests_[0];
        for (int lane = 1; lane < kCodecLanes; ++lane) {
            const QkvBatchRequest& request = requests_[lane];
            if (request.packed_weights != first.packed_weights
                || request.weight_scales != first.weight_scales
                || request.ticket_offsets != first.ticket_offsets
                || request.ticket_columns != first.ticket_columns
                || request.layer_index != first.layer_index
                || request.value_slot != first.value_slot
                || request.value_scale != first.value_scale) {
                throw std::runtime_error("QKV lanes reached different phases");
            }
        }
        const bool audit = batches_executed_ < TERNLOG_QKV_AUDIT_BATCHES;
        if (audit) audit_first_batch();
#if defined(TERNLOG_BATCH4_PROFILE)
        const auto batch_started = std::chrono::steady_clock::now();
#endif
        qkv_tickets_batch4_fused_value(
            requests_, audit ? &audit_raw_sums_ : nullptr
        );
#if defined(TERNLOG_BATCH4_PROFILE)
        batch_qkv_nanoseconds_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - batch_started
            ).count()
        );
        ++batch_qkv_calls_;
#endif
        if (audit) verify_first_batch_audit();
        ++batches_executed_;
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            waiting_[lane] = false;
            wait_kinds_[lane] = FiberWaitKind::none;
        }
        waiting_count_ = 0;
    }

#if defined(TERNLOG_FFN_BATCH4_HOOK)
    void execute_ffn_batch() {
        const TernlogFfnBatchRequest& first = ffn_requests_[0];
        for (int lane = 1; lane < kCodecLanes; ++lane) {
            if (ffn_requests_[lane].weights != first.weights
                || ffn_requests_[lane].decoder != first.decoder
                || ffn_requests_[lane].bias != first.bias
                || ffn_requests_[lane].correction_scales
                    != first.correction_scales) {
                throw std::runtime_error("FFN lanes reached different layers");
            }
        }
        std::int32_t* outputs[kCodecLanes];
        const std::int8_t* inputs[kCodecLanes];
        float* decoded_outputs[kCodecLanes];
        const std::uint8_t* final_states[kCodecLanes];
        const std::int32_t* raw_inputs[kCodecLanes];
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            outputs[lane] = ffn_requests_[lane].raw_sums;
            inputs[lane] = ffn_requests_[lane].pooled;
            decoded_outputs[lane] = ffn_requests_[lane].output;
            final_states[lane] = ffn_requests_[lane].final_state;
            raw_inputs[lane] = ffn_requests_[lane].raw_sums;
        }
#if defined(TERNLOG_BATCH4_PROFILE)
        const auto batch_started = std::chrono::steady_clock::now();
#endif
        ternlog_vocab_ffn_sdot_batch4_raw_ptrs_2row(
            outputs, inputs, first.weights
        );
        ternlog_vocab_ffn_batch4_postprocess_ptrs(
            decoded_outputs, final_states, raw_inputs,
            first.decoder, first.bias, first.correction_scales
        );
#if defined(TERNLOG_BATCH4_PROFILE)
        batch_ffn_nanoseconds_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - batch_started
            ).count()
        );
        ++batch_ffn_calls_;
#endif
        if (ffn_batches_executed_ < TERNLOG_FFN_AUDIT_BATCHES) {
            alignas(64) std::int32_t expected[layout::kDimension];
            alignas(64) float expected_output[layout::kDimension];
            for (int lane = 0; lane < kCodecLanes; ++lane) {
                ternlog_vocab_ffn_sdot_1row(
                    expected, inputs[lane], first.weights
                );
                if (!std::equal(
                        expected, expected + layout::kDimension,
                        outputs[lane])) {
                    throw std::runtime_error("FFN batch raw-sum audit failed");
                }
                ternlog_vocab_ffn_scalar_postprocess(
                    expected_output, final_states[lane], expected,
                    first.decoder, first.bias, first.correction_scales
                );
                if (std::memcmp(
                        expected_output, decoded_outputs[lane],
                        sizeof(expected_output)) != 0) {
                    for (std::size_t row = 0; row < layout::kDimension; ++row) {
                        if (std::memcmp(
                                &expected_output[row], &decoded_outputs[lane][row],
                                sizeof(float)) != 0) {
                            std::uint32_t expected_bits = 0;
                            std::uint32_t actual_bits = 0;
                            std::memcpy(
                                &expected_bits, &expected_output[row], sizeof(float)
                            );
                            std::memcpy(
                                &actual_bits, &decoded_outputs[lane][row], sizeof(float)
                            );
                            std::fprintf(
                                stderr,
                                "FFN batch FP audit mismatch lane=%d row=%zu "
                                "expected=%a actual=%a expected_bits=%08x "
                                "actual_bits=%08x\n",
                                lane, row, expected_output[row],
                                decoded_outputs[lane][row], expected_bits, actual_bits
                            );
                            break;
                        }
                    }
                    throw std::runtime_error("FFN batch FP audit failed");
                }
            }
        }
        ++ffn_batches_executed_;
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            waiting_[lane] = false;
            wait_kinds_[lane] = FiberWaitKind::none;
        }
        waiting_count_ = 0;
    }
#endif

    void audit_first_batch() {
        QkvBatchSubmit saved_submit = ternlog_qkv_batch_submit;
        ternlog_qkv_batch_submit = nullptr;
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            const QkvBatchRequest& request = requests_[lane];
            for (int channel = 0; channel < kHeadDim; ++channel) {
                audit_cache_before_[lane][channel] = (*request.value_cache)[
                    shared_value_cache_int8_offset(
                        request.layer_index, 0, request.value_slot, channel
                    )
                ];
            }
            matrix_vector_int8_qkv_tickets_prequantized(
                request.packed_weights,
                request.weight_scales,
                request.ticket_offsets,
                request.ticket_columns,
                request.activation,
                request.activation_scale,
                audit_output_[lane].data(),
                *request.value_cache,
                request.layer_index,
                request.value_slot,
                request.value_scale
            );
            for (int channel = 0; channel < kHeadDim; ++channel) {
                const std::size_t cache_offset = shared_value_cache_int8_offset(
                    request.layer_index, 0, request.value_slot, channel
                );
                audit_cache_expected_[lane][channel] =
                    (*request.value_cache)[cache_offset];
                (*request.value_cache)[cache_offset] =
                    audit_cache_before_[lane][channel];
            }
        }
        ternlog_qkv_batch_submit = saved_submit;
    }

    void verify_first_batch_audit() const {
        constexpr int value_offset =
            (layout::kHeads + layout::kKeyHeads) * kQkHeadDim;
        float maximum_difference = 0.0F;
        float maximum_left_difference = 0.0F;
        float maximum_right_difference = 0.0F;
        std::size_t cache_mismatches = 0;
        int mismatch_lane = -1;
        int mismatch_channel = -1;
        int expected_cache = 0;
        int actual_cache = 0;
        int candidate_left = 0;
        int candidate_combined = 0;
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            const QkvBatchRequest& request = requests_[lane];
            for (int row = 0; row < value_offset; ++row) {
                volatile float left_partial = static_cast<float>(
                    audit_raw_sums_[lane][row]
                ) * request.activation_scale;
                const float left = left_partial * request.weight_scales[row];
                volatile float right_partial = request.activation_scale
                    * request.weight_scales[row];
                const float right = static_cast<float>(
                    audit_raw_sums_[lane][row]
                ) * right_partial;
                maximum_difference = std::max(
                    maximum_difference,
                    std::abs(audit_output_[lane][row] - request.output[row])
                );
                maximum_left_difference = std::max(
                    maximum_left_difference,
                    std::abs(audit_output_[lane][row] - left)
                );
                maximum_right_difference = std::max(
                    maximum_right_difference,
                    std::abs(audit_output_[lane][row] - right)
                );
            }
            for (int index = 0; index < kHeadDim; ++index) {
                const std::size_t cache_offset = shared_value_cache_int8_offset(
                    request.layer_index, 0, request.value_slot, index
                );
                const int expected = audit_cache_expected_[lane][index];
                const int actual = (*request.value_cache)[cache_offset];
                if (expected != actual) {
                    ++cache_mismatches;
                    if (mismatch_lane < 0) {
                        mismatch_lane = lane;
                        mismatch_channel = index;
                        expected_cache = expected;
                        actual_cache = actual;
                        const int row = value_offset + index;
                        const float raw = static_cast<float>(
                            audit_raw_sums_[lane][row]
                        );
                        const float activation = request.activation_scale;
                        const float weight = request.weight_scales[row];
                        const float inverse = 1.0F / request.value_scale;
                        volatile float partial = raw * activation;
                        candidate_left = static_cast<int>(std::nearbyint(
                            (partial * weight) * inverse
                        ));
                        volatile float combined = activation * weight * inverse;
                        candidate_combined = static_cast<int>(std::nearbyint(
                            raw * combined
                        ));
                    }
                }
            }
        }
        if (maximum_difference != 0.0F || cache_mismatches != 0) {
            char message[384];
            std::snprintf(
                message, sizeof(message),
                "QKV drift at batch %zu: batch %.9g, left %.9g, right %.9g, "
                "cache mismatches %zu, lane %d channel %d expected %d actual %d "
                "left %d combined %d",
                batches_executed_,
                maximum_difference, maximum_left_difference,
                maximum_right_difference, cache_mismatches,
                mismatch_lane, mismatch_channel, expected_cache, actual_cache,
                candidate_left, candidate_combined
            );
            throw std::runtime_error(message);
        }
    }

    void execute_scalar_waiters() {
        QkvBatchSubmit saved_submit = ternlog_qkv_batch_submit;
        ternlog_qkv_batch_submit = nullptr;
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            if (!waiting_[lane]) continue;
            if (wait_kinds_[lane] != FiberWaitKind::qkv) {
                throw std::runtime_error("non-QKV scalar fiber waiter");
            }
            const QkvBatchRequest& request = requests_[lane];
            matrix_vector_int8_qkv_tickets_prequantized(
                request.packed_weights,
                request.weight_scales,
                request.ticket_offsets,
                request.ticket_columns,
                request.activation,
                request.activation_scale,
                request.output,
                *request.value_cache,
                request.layer_index,
                request.value_slot,
                request.value_scale
            );
#if defined(TERNLOG_BATCH4_PROFILE)
            ++scalar_tail_qkv_calls_;
#endif
            waiting_[lane] = false;
            wait_kinds_[lane] = FiberWaitKind::none;
        }
        waiting_count_ = 0;
        ternlog_qkv_batch_submit = saved_submit;
    }

    void resume(int lane) {
        current_lane_ = lane;
#if defined(TERNLOG_BATCH4_PROFILE)
        ++fiber_switches_;
#endif
        ternlog_m4_fiber_switch(&scheduler_context_, &contexts_[lane]);
        current_lane_ = -1;
    }

    void yield(int lane) {
#if defined(TERNLOG_BATCH4_PROFILE)
        ++fiber_switches_;
#endif
        ternlog_m4_fiber_switch(&contexts_[lane], &scheduler_context_);
    }

#if defined(TERNLOG_BATCH4_PROFILE)
    void print_batch_profile(
        std::chrono::steady_clock::time_point run_started
    ) const {
        const std::uint64_t total_nanoseconds = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - run_started
            ).count()
        );
        std::fprintf(
            stderr,
            "{\"batch4_profile\":{\"mode\":\"%s\","
            "\"total_nanoseconds\":%llu,\"batch_qkv_nanoseconds\":%llu,"
            "\"batch_qkv_share\":%.6f,\"batch_qkv_calls\":%llu,"
            "\"batch_ffn_nanoseconds\":%llu,\"batch_ffn_share\":%.6f,"
            "\"batch_ffn_calls\":%llu,"
            "\"scalar_tail_qkv_calls\":%llu,\"fiber_switches\":%llu}}\n",
            encoding_ ? "encode" : "decode",
            static_cast<unsigned long long>(total_nanoseconds),
            static_cast<unsigned long long>(batch_qkv_nanoseconds_),
            total_nanoseconds == 0 ? 0.0
                : static_cast<double>(batch_qkv_nanoseconds_)
                    / total_nanoseconds,
            static_cast<unsigned long long>(batch_qkv_calls_),
            static_cast<unsigned long long>(batch_ffn_nanoseconds_),
            total_nanoseconds == 0 ? 0.0
                : static_cast<double>(batch_ffn_nanoseconds_)
                    / total_nanoseconds,
            static_cast<unsigned long long>(batch_ffn_calls_),
            static_cast<unsigned long long>(scalar_tail_qkv_calls_),
            static_cast<unsigned long long>(fiber_switches_)
        );
    }
#endif

    void decode_lane(int lane) {
        const std::size_t lane_bytes = edges_[lane + 1] - edges_[lane];
        while (outputs_[lane].size() < lane_bytes) {
            const std::size_t block_position =
                outputs_[lane].size() % header_.block_size;
            if (block_position == 0) {
                models_[lane]->reset();
                adaptive_mixers_[lane].reset();
                adaptive_integer_mixers_[lane].reset();
            }
            const auto [predicted, match_length] = matchers_[lane]->predict();
            if (block_position == 0) {
                uniform_frequencies(frequencies_[lane]);
            } else {
                models_[lane]->forward(
                    outputs_[lane].back(), logits_[lane].data()
                );
                make_frequencies(
                    frequencies_[lane], logits_[lane].data(), header_.match,
                    predicted, match_length
                );
                if (header_.adaptive_integer) {
                    mix_integer_frequencies_with_uniform(
                        frequencies_[lane],
                        adaptive_integer_mixers_[lane].model_weight()
                    );
                } else if (header_.adaptive_coded) {
                    mix_coded_frequencies_with_uniform(
                        frequencies_[lane], adaptive_mixers_[lane].model_weight()
                    );
                }
            }
            const std::uint32_t value = decoders_[lane]->frequency(
                frequencies_[lane].cumulative[256]
            );
            const auto found = std::upper_bound(
                frequencies_[lane].cumulative.begin() + 1,
                frequencies_[lane].cumulative.end(), value
            );
            const std::uint8_t symbol = static_cast<std::uint8_t>(
                found - frequencies_[lane].cumulative.begin() - 1
            );
            const std::uint32_t start = frequencies_[lane].cumulative[symbol];
            decoders_[lane]->consume(
                start, frequencies_[lane].cumulative[symbol + 1] - start
            );
            outputs_[lane].push_back(symbol);
            if (header_.adaptive_coded && block_position != 0) {
                if (header_.adaptive_integer) {
                    adaptive_integer_mixers_[lane].update(
                        frequencies_[lane].expert_frequencies[symbol],
                        frequencies_[lane].expert_total
                    );
                } else {
                    adaptive_mixers_[lane].update(
                        frequencies_[lane].expert_probabilities[symbol]
                    );
                }
            }
            matchers_[lane]->update(symbol);
        }
    }

    void encode_lane(int lane) {
        const std::size_t lane_begin = edges_[lane];
        const std::size_t lane_bytes = edges_[lane + 1] - lane_begin;
        for (std::size_t position = 0; position < lane_bytes; ++position) {
            const std::size_t block_position = position % header_.block_size;
            if (block_position == 0) {
                models_[lane]->reset();
                adaptive_mixers_[lane].reset();
                adaptive_integer_mixers_[lane].reset();
            }
            const auto [predicted, match_length] = matchers_[lane]->predict();
            if (block_position == 0) {
                uniform_frequencies(frequencies_[lane]);
            } else {
                models_[lane]->forward(
                    (*encoding_input_)[lane_begin + position - 1],
                    logits_[lane].data()
                );
                make_frequencies(
                    frequencies_[lane], logits_[lane].data(), header_.match,
                    predicted, match_length
                );
                if (header_.adaptive_integer) {
                    mix_integer_frequencies_with_uniform(
                        frequencies_[lane],
                        adaptive_integer_mixers_[lane].model_weight()
                    );
                } else if (header_.adaptive_coded) {
                    mix_coded_frequencies_with_uniform(
                        frequencies_[lane], adaptive_mixers_[lane].model_weight()
                    );
                }
            }
            const std::uint8_t symbol = (*encoding_input_)[lane_begin + position];
            const std::uint32_t start = frequencies_[lane].cumulative[symbol];
            encoders_[lane]->encode(
                start,
                frequencies_[lane].cumulative[symbol + 1] - start,
                frequencies_[lane].cumulative[256]
            );
            if (header_.adaptive_coded && block_position != 0) {
                if (header_.adaptive_integer) {
                    adaptive_integer_mixers_[lane].update(
                        frequencies_[lane].expert_frequencies[symbol],
                        frequencies_[lane].expert_total
                    );
                } else {
                    adaptive_mixers_[lane].update(
                        frequencies_[lane].expert_probabilities[symbol]
                    );
                }
            }
            matchers_[lane]->update(symbol);
        }
        encoded_payloads_[lane] = encoders_[lane]->finish();
    }

    std::vector<std::uint8_t> finish_encoding() {
        std::vector<std::uint8_t> result = make_header(
            static_cast<std::size_t>(header_.original_bytes),
            header_.crc,
            header_.match,
            header_.adaptive_integer
        );
        std::memcpy(
            result.data(),
            header_.adaptive_integer
                ? "TLI4" : header_.adaptive_coded ? "TLC4" : "TLZ4",
            4
        );
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            append_scalar(
                result,
                static_cast<std::uint64_t>(encoded_payloads_[lane].size())
            );
        }
        for (const auto& payload : encoded_payloads_) {
            result.insert(result.end(), payload.begin(), payload.end());
        }
        return result;
    }

    ParsedHeader header_{};
    std::array<std::size_t, kCodecLanes + 1> edges_{};
    std::array<std::vector<std::uint8_t>, kCodecLanes> nested_;
    std::array<std::vector<std::uint8_t>, kCodecLanes> outputs_;
    std::array<std::unique_ptr<MatchModelNative>, kCodecLanes> matchers_;
    std::array<std::unique_ptr<Transformer>, kCodecLanes> models_;
    std::array<std::unique_ptr<RangeDecoderNative>, kCodecLanes> decoders_;
    std::array<std::unique_ptr<RangeEncoderNative>, kCodecLanes> encoders_;
    std::array<std::vector<std::uint8_t>, kCodecLanes> encoded_payloads_;
    std::array<Frequencies, kCodecLanes> frequencies_;
    std::array<AdaptiveUniformMixer, kCodecLanes> adaptive_mixers_;
    std::array<AdaptiveIntegerMixer, kCodecLanes> adaptive_integer_mixers_;
    std::array<std::array<float, 256>, kCodecLanes> logits_{};
    std::array<FiberContext, kCodecLanes> contexts_{};
    std::array<std::vector<std::uint8_t>, kCodecLanes> stacks_;
    std::array<QkvFiberStart, kCodecLanes> starts_{};
    std::array<QkvBatchRequest, kCodecLanes> requests_{};
#if defined(TERNLOG_FFN_BATCH4_HOOK)
    std::array<TernlogFfnBatchRequest, kCodecLanes> ffn_requests_{};
#endif
    std::array<std::array<float, layout::kQkvRows>, kCodecLanes> audit_output_{};
    QkvRawSums audit_raw_sums_{};
    std::array<std::array<ValueCacheScalar, kHeadDim>, kCodecLanes>
        audit_cache_before_{};
    std::array<std::array<ValueCacheScalar, kHeadDim>, kCodecLanes>
        audit_cache_expected_{};
    std::array<bool, kCodecLanes> waiting_{};
    std::array<FiberWaitKind, kCodecLanes> wait_kinds_{};
    std::array<bool, kCodecLanes> done_{};
    FiberContext scheduler_context_{};
    std::exception_ptr failure_;
    int current_lane_ = -1;
    int waiting_count_ = 0;
    int active_lanes_ = kCodecLanes;
    std::size_t batches_executed_ = 0;
    std::size_t ffn_batches_executed_ = 0;
    const std::vector<std::uint8_t>* encoding_input_ = nullptr;
    bool encoding_ = false;
#if defined(TERNLOG_BATCH4_PROFILE)
    std::uint64_t batch_qkv_nanoseconds_ = 0;
    std::uint64_t batch_qkv_calls_ = 0;
    std::uint64_t batch_ffn_nanoseconds_ = 0;
    std::uint64_t batch_ffn_calls_ = 0;
    std::uint64_t scalar_tail_qkv_calls_ = 0;
    std::uint64_t fiber_switches_ = 0;
#endif
};

#endif

std::vector<std::uint8_t> decompress_four_lane_lockstep(
    const std::vector<std::uint8_t>& stream
) {
    constexpr std::size_t directory_bytes = kCodecLanes * sizeof(std::uint64_t);
    const bool adaptive_coded = stream.size() >= 4
        && std::memcmp(stream.data(), "TLC4", 4) == 0;
    const bool adaptive_integer = stream.size() >= 4
        && std::memcmp(stream.data(), "TLI4", 4) == 0;
    const bool adaptive = adaptive_coded || adaptive_integer;
    if (stream.size() < kHeaderBytes + directory_bytes
        || (!adaptive && std::memcmp(stream.data(), "TLZ4", 4) != 0)) {
        throw std::runtime_error("not a TLZ4 stream");
    }
    std::vector<std::uint8_t> common(
        stream.begin(), stream.begin() + kHeaderBytes
    );
    std::memcpy(
        common.data(),
        adaptive_integer ? "TLI2" : adaptive_coded ? "TLC2" : "TLZ2", 4
    );
    const ParsedHeader header = parse_header(common);
    const auto edges = lane_boundaries(static_cast<std::size_t>(header.original_bytes));

    std::array<std::size_t, kCodecLanes> payload_bytes{};
    std::size_t offset = kHeaderBytes;
    std::size_t known_payload = 0;
    for (int lane = 0; lane < kCodecLanes; ++lane) {
        payload_bytes[lane] = read_scalar<std::uint64_t>(stream, offset);
        known_payload += payload_bytes[lane];
    }
    if (known_payload > stream.size() - offset
        || offset + known_payload != stream.size()) {
        throw std::runtime_error("four-lane payload length mismatch");
    }

    std::array<std::vector<std::uint8_t>, kCodecLanes> nested;
    std::array<std::vector<std::uint8_t>, kCodecLanes> outputs;
    std::array<std::unique_ptr<MatchModelNative>, kCodecLanes> matchers;
    std::array<std::unique_ptr<Transformer>, kCodecLanes> models;
    std::array<std::unique_ptr<RangeDecoderNative>, kCodecLanes> decoders;
    std::array<Frequencies, kCodecLanes> frequencies;
    std::array<AdaptiveUniformMixer, kCodecLanes> adaptive_mixers;
    std::array<AdaptiveIntegerMixer, kCodecLanes> adaptive_integer_mixers;
    std::array<std::array<float, 256>, kCodecLanes> logits{};
    std::size_t maximum_lane_bytes = 0;

    for (int lane = 0; lane < kCodecLanes; ++lane) {
        const std::size_t lane_bytes = edges[lane + 1] - edges[lane];
        maximum_lane_bytes = std::max(maximum_lane_bytes, lane_bytes);
        nested[lane] = make_header(
            lane_bytes, 0, header.match, header.adaptive_integer
        );
        if (offset + payload_bytes[lane] > stream.size()) {
            throw std::runtime_error("truncated TLZ4 payload");
        }
        nested[lane].insert(
            nested[lane].end(), stream.begin() + offset,
            stream.begin() + offset + payload_bytes[lane]
        );
        offset += payload_bytes[lane];
        outputs[lane].reserve(lane_bytes);
        matchers[lane] = std::make_unique<MatchModelNative>(header.match);
        models[lane] = std::make_unique<Transformer>();
        decoders[lane] = std::make_unique<RangeDecoderNative>(
            nested[lane], kHeaderBytes
        );
    }

    // Advance one symbol per lane. This deliberately uses the scalar predictor:
    // it is the exact scheduling oracle for later batch-4 layer kernels.
    for (std::size_t position = 0; position < maximum_lane_bytes; ++position) {
        for (int lane = 0; lane < kCodecLanes; ++lane) {
            const std::size_t lane_bytes = edges[lane + 1] - edges[lane];
            if (position >= lane_bytes) continue;
            const std::size_t block_position = position % header.block_size;
            if (block_position == 0) {
                models[lane]->reset();
                adaptive_mixers[lane].reset();
                adaptive_integer_mixers[lane].reset();
            }
            const auto [predicted, match_length] = matchers[lane]->predict();
            if (block_position == 0) {
                uniform_frequencies(frequencies[lane]);
            } else {
                models[lane]->forward(outputs[lane].back(), logits[lane].data());
                make_frequencies(
                    frequencies[lane], logits[lane].data(), header.match,
                    predicted, match_length
                );
                if (header.adaptive_integer) {
                    mix_integer_frequencies_with_uniform(
                        frequencies[lane],
                        adaptive_integer_mixers[lane].model_weight()
                    );
                } else if (header.adaptive_coded) {
                    mix_coded_frequencies_with_uniform(
                        frequencies[lane], adaptive_mixers[lane].model_weight()
                    );
                }
            }
            const std::uint32_t value = decoders[lane]->frequency(
                frequencies[lane].cumulative[256]
            );
            const auto found = std::upper_bound(
                frequencies[lane].cumulative.begin() + 1,
                frequencies[lane].cumulative.end(), value
            );
            const std::uint8_t symbol = static_cast<std::uint8_t>(
                found - frequencies[lane].cumulative.begin() - 1
            );
            const std::uint32_t start = frequencies[lane].cumulative[symbol];
            decoders[lane]->consume(
                start, frequencies[lane].cumulative[symbol + 1] - start
            );
            outputs[lane].push_back(symbol);
            if (header.adaptive_coded && block_position != 0) {
                if (header.adaptive_integer) {
                    adaptive_integer_mixers[lane].update(
                        frequencies[lane].expert_frequencies[symbol],
                        frequencies[lane].expert_total
                    );
                } else {
                    adaptive_mixers[lane].update(
                        frequencies[lane].expert_probabilities[symbol]
                    );
                }
            }
            matchers[lane]->update(symbol);
        }
    }

    std::vector<std::uint8_t> result;
    result.reserve(static_cast<std::size_t>(header.original_bytes));
    for (const auto& output : outputs) {
        result.insert(result.end(), output.begin(), output.end());
    }
    const std::uint32_t actual_crc = static_cast<std::uint32_t>(
        crc32(0, result.data(), static_cast<uInt>(result.size()))
    );
    if (actual_crc != header.crc) throw std::runtime_error("checksum mismatch");
    return result;
}

#if defined(TERNLOG_QKV_BATCH4_HOOK)
std::vector<std::uint8_t> decompress_four_lane_batch4(
    const std::vector<std::uint8_t>& stream
) {
    return QkvFiberScheduler(stream).run();
}

std::vector<std::uint8_t> compress_four_lane_batch4(
    const std::vector<std::uint8_t>& input,
    bool use_match,
    AdaptiveCodingMode adaptive_mode = AdaptiveCodingMode::None
) {
    return QkvFiberScheduler(input, use_match, adaptive_mode).run();
}
#endif

constexpr std::size_t kBatch4MinimumInputBytes =
    static_cast<std::size_t>(kCodecLanes) * layout::kContext;

std::vector<std::uint8_t> compress_fast(
    const std::vector<std::uint8_t>& input,
    bool use_match
) {
#if defined(TERNLOG_QKV_BATCH4_HOOK)
    if (input.size() >= kBatch4MinimumInputBytes) {
        return compress_four_lane_batch4(input, use_match);
    }
#endif
    return compress_native(input, use_match);
}

std::vector<std::uint8_t> compress_adaptive_fast(
    const std::vector<std::uint8_t>& input,
    bool use_match
) {
#if defined(TERNLOG_QKV_BATCH4_HOOK)
    if (input.size() >= kBatch4MinimumInputBytes) {
        return compress_four_lane_batch4(
            input, use_match, AdaptiveCodingMode::IntegerReset
        );
    }
#endif
    return compress_native(input, use_match, AdaptiveCodingMode::IntegerReset);
}

std::vector<std::uint8_t> compress_safe(
    const std::vector<std::uint8_t>& input,
    bool use_match
) {
    if (input.size() > std::numeric_limits<std::size_t>::max() - kRawHeaderBytes) {
        throw std::runtime_error("safe input is too large");
    }
    std::vector<std::uint8_t> neural = compress_adaptive_fast(input, use_match);
    if (neural.size() < kRawHeaderBytes + input.size()) return neural;
    return compress_raw(input);
}

std::vector<std::uint8_t> decompress_auto(
    const std::vector<std::uint8_t>& stream
) {
    if (stream.size() >= 4 && std::memcmp(stream.data(), "TLR0", 4) == 0) {
        return decompress_raw(stream);
    }
    if (stream.size() >= 4
        && (std::memcmp(stream.data(), "TLZ4", 4) == 0
            || std::memcmp(stream.data(), "TLC4", 4) == 0
            || std::memcmp(stream.data(), "TLI4", 4) == 0)) {
#if defined(TERNLOG_QKV_BATCH4_HOOK)
        if (stream.size() >= 12) {
            std::size_t offset = 4;
            const std::uint64_t original_bytes =
                read_scalar<std::uint64_t>(stream, offset);
            if (original_bytes >= kBatch4MinimumInputBytes) {
                return decompress_four_lane_batch4(stream);
            }
        }
#endif
        return decompress_four_lane(stream);
    }
    return decompress_native(stream);
}

double median_seconds(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0U) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

enum class PersistentBenchmarkMode {
    Scalar,
    FourLane,
#if defined(TERNLOG_QKV_BATCH4_HOOK)
    FourLaneBatch,
#endif
};

void benchmark_persistent(
    const std::vector<std::uint8_t>& input,
    int iterations,
    PersistentBenchmarkMode mode
) {
    if (iterations < 2) {
        throw std::runtime_error("benchmark iterations must be at least two");
    }
    std::vector<double> encode_seconds;
    std::vector<double> decode_seconds;
    encode_seconds.reserve(static_cast<std::size_t>(iterations));
    decode_seconds.reserve(static_cast<std::size_t>(iterations));
    std::size_t compressed_bytes = 0;
    std::vector<std::uint8_t> reference_stream;

    const auto compress = [&]() {
        switch (mode) {
            case PersistentBenchmarkMode::Scalar:
                return compress_native(input, true);
            case PersistentBenchmarkMode::FourLane:
                return compress_four_lane(input, true);
#if defined(TERNLOG_QKV_BATCH4_HOOK)
            case PersistentBenchmarkMode::FourLaneBatch:
                return compress_four_lane_batch4(input, true);
#endif
        }
        throw std::runtime_error("invalid persistent benchmark mode");
    };
    const auto decompress = [&](const std::vector<std::uint8_t>& stream) {
        switch (mode) {
            case PersistentBenchmarkMode::Scalar:
                return decompress_native(stream);
            case PersistentBenchmarkMode::FourLane:
                return decompress_four_lane(stream);
#if defined(TERNLOG_QKV_BATCH4_HOOK)
            case PersistentBenchmarkMode::FourLaneBatch:
                return decompress_four_lane_batch4(stream);
#endif
        }
        throw std::runtime_error("invalid persistent benchmark mode");
    };
    const char* mode_name = mode == PersistentBenchmarkMode::Scalar
        ? "scalar"
        : mode == PersistentBenchmarkMode::FourLane
            ? "four_lane"
            : "four_lane_batch";

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto encode_started = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> compressed = compress();
        const auto encode_stopped = std::chrono::steady_clock::now();
        const auto decode_started = encode_stopped;
        std::vector<std::uint8_t> restored = decompress(compressed);
        const auto decode_stopped = std::chrono::steady_clock::now();
        if (restored != input) {
            throw std::runtime_error("persistent benchmark round trip failed");
        }
        if (iteration == 0) {
            compressed_bytes = compressed.size();
            reference_stream = compressed;
        } else if (compressed != reference_stream) {
            throw std::runtime_error("persistent benchmark stream changed");
        }
        encode_seconds.push_back(std::chrono::duration<double>(
            encode_stopped - encode_started
        ).count());
        decode_seconds.push_back(std::chrono::duration<double>(
            decode_stopped - decode_started
        ).count());
    }

    const double bytes = static_cast<double>(input.size());
    const std::vector<double> warm_encode(
        encode_seconds.begin() + 1, encode_seconds.end()
    );
    const std::vector<double> warm_decode(
        decode_seconds.begin() + 1, decode_seconds.end()
    );
    std::printf(
        "{\"mode\":\"%s\",\"input_bytes\":%zu,\"compressed_bytes\":%zu,"
        "\"bits_per_byte\":%.9f,\"iterations\":%d,"
        "\"cold_encode_bytes_per_second\":%.3f,"
        "\"cold_decode_bytes_per_second\":%.3f,"
        "\"warm_encode_bytes_per_second_median\":%.3f,"
        "\"warm_decode_bytes_per_second_median\":%.3f,"
        "\"round_trip\":\"exact\"}\n",
        mode_name, input.size(), compressed_bytes,
        input.empty() ? 0.0 : 8.0 * compressed_bytes / bytes,
        iterations,
        input.empty() ? 0.0 : bytes / encode_seconds.front(),
        input.empty() ? 0.0 : bytes / decode_seconds.front(),
        input.empty() ? 0.0 : bytes / median_seconds(warm_encode),
        input.empty() ? 0.0 : bytes / median_seconds(warm_decode)
    );
}

void codec_usage(const char* executable) {
    std::fprintf(stderr,
        "       %s compress-parallel INPUT OUTPUT [--threads 1..4]\n"
        "       %s decompress-parallel INPUT OUTPUT [--threads 1..4]\n",
        executable, executable);
    std::fprintf(
        stderr,
        "Usage: %s compress INPUT OUTPUT [--no-match]\n"
        "       %s compress-fast INPUT OUTPUT [--no-match]\n"
        "       %s compress-adaptive INPUT OUTPUT [--no-match]\n"
        "       %s compress-adaptive-fast INPUT OUTPUT [--no-match]\n"
        "       %s compress-safe INPUT OUTPUT [--no-match]\n"
        "       %s compress-adaptive-legacy INPUT OUTPUT [--no-match]\n"
        "       %s compress-adaptive-coded-legacy INPUT OUTPUT [--no-match]\n"
        "       %s compress-adaptive-carry INPUT OUTPUT [--no-match]\n"
        "       %s decompress INPUT OUTPUT\n"
        "       %s decompress-auto INPUT OUTPUT\n"
        "       %s compress4 INPUT OUTPUT [--no-match]\n"
        "       %s compress4-adaptive INPUT OUTPUT [--no-match]\n"
        "       %s compress4-adaptive-coded-legacy INPUT OUTPUT [--no-match]\n"
        "       %s decompress4 INPUT OUTPUT\n"
        "       %s decompress4-lockstep INPUT OUTPUT\n",
        executable, executable, executable, executable, executable, executable,
        executable, executable, executable, executable, executable, executable,
        executable, executable, executable
    );
#if defined(TERNLOG_QKV_BATCH4_HOOK)
    std::fprintf(
        stderr,
        "       %s compress4-batch INPUT OUTPUT [--no-match]\n"
        "       %s compress4-batch-adaptive INPUT OUTPUT [--no-match]\n"
        "       %s compress4-batch-adaptive-coded-legacy INPUT OUTPUT [--no-match]\n"
        "       %s decompress4-batch INPUT OUTPUT\n",
        executable, executable, executable, executable
    );
#endif
    std::fprintf(
        stderr,
        "       %s benchmark INPUT ITERATIONS\n"
        "       %s benchmark4 INPUT ITERATIONS\n",
        executable, executable
    );
#if defined(TERNLOG_QKV_BATCH4_HOOK)
    std::fprintf(
        stderr,
        "       %s benchmark4-batch INPUT ITERATIONS\n",
        executable
    );
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            codec_usage(argv[0]);
            return 2;
        }
        const std::string_view operation(argv[1]);
        if (operation == "compress-parallel" || operation == "decompress-parallel") {
            int workers = 4;
            if (argc == 6 && std::string_view(argv[4]) == "--threads"
                && std::string_view(argv[5]).size() == 1
                && argv[5][0] >= '1' && argv[5][0] <= '4') {
                workers = argv[5][0] - '0';
            } else if (argc != 4) {
                throw std::runtime_error("usage: compress-parallel/decompress-parallel INPUT OUTPUT [--threads 1..4]");
            }
            const auto input = read_binary(argv[2]);
            if (operation == "compress-parallel") {
                write_binary(argv[3], input.size() < kBatch4MinimumInputBytes
                    ? compress_adaptive_fast(input, true)
                    : compress_four_lane(input, true, AdaptiveCodingMode::IntegerReset, workers));
            } else {
                const bool four_lane = input.size() >= 4
                    && (std::memcmp(input.data(), "TLI4", 4) == 0
                        || std::memcmp(input.data(), "TLC4", 4) == 0
                        || std::memcmp(input.data(), "TLZ4", 4) == 0);
                write_binary(argv[3], four_lane
                    ? decompress_four_lane(input, workers) : decompress_auto(input));
            }
            return 0;
        }
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
        if (operation == "compress-adaptive-layerwise") {
            write_binary(argv[3], compress_native(read_binary(argv[2]), true,
                         AdaptiveCodingMode::IntegerReset, true));
            return 0;
        }
        if (operation == "audit-encoder-layerwise") {
            const auto input = read_binary(argv[2]);
            const int tile = std::stoi(argv[3]);
            if (tile < 1 || tile > 32) throw std::runtime_error("tile must be 1..32");
            Transformer sequential;
            Transformer layerwise;
            alignas(64) float expected[32 * layout::kVocabulary];
            alignas(64) float actual[32 * layout::kVocabulary];
            for (std::size_t base = 0; base < input.size(); base += tile) {
                const auto count = std::min<std::size_t>(tile, input.size() - base);
                for (std::size_t index = 0; index < count; ++index)
                    sequential.forward(input[base + index], expected + index * layout::kVocabulary);
                layerwise.forward_known_tile(input.data() + base, count, actual);
                if (std::memcmp(expected, actual, count * layout::kVocabulary * sizeof(float)) != 0)
                    throw std::runtime_error("layerwise logit mismatch at tile " + std::to_string(base));
                if (sequential.recurrent_state_hashes() != layerwise.recurrent_state_hashes())
                    throw std::runtime_error("layerwise cache mismatch at tile " + std::to_string(base));
            }
            std::printf("{\"gate\":\"layerwise-bit-exact\",\"bytes\":%zu,\"tile\":%d,\"passed\":true}\n",
                        input.size(), tile);
            return 0;
        }
#endif
        if (operation == "compress") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(argv[3], compress_native(read_binary(argv[2]), use_match));
        } else if (operation == "compress-fast") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(argv[3], compress_fast(read_binary(argv[2]), use_match));
        } else if (operation == "compress-adaptive") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3],
                compress_native(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::IntegerReset
                )
            );
        } else if (operation == "compress-adaptive-fast") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3], compress_adaptive_fast(read_binary(argv[2]), use_match)
            );
        } else if (operation == "compress-safe") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(argv[3], compress_safe(read_binary(argv[2]), use_match));
        } else if (operation == "compress-adaptive-legacy") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3],
                compress_native(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::LegacyFloatReset
                )
            );
        } else if (operation == "compress-adaptive-coded-legacy") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3],
                compress_native(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::CodedReset
                )
            );
        } else if (operation == "compress-adaptive-carry") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3],
                compress_native(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::CodedCarry
                )
            );
        } else if (operation == "decompress") {
            if (argc != 4) return 2;
            write_binary(argv[3], decompress_native(read_binary(argv[2])));
        } else if (operation == "decompress-auto") {
            if (argc != 4) return 2;
            write_binary(argv[3], decompress_auto(read_binary(argv[2])));
        } else if (operation == "compress4") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(argv[3], compress_four_lane(read_binary(argv[2]), use_match));
        } else if (operation == "compress4-adaptive") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3], compress_four_lane(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::IntegerReset
                )
            );
        } else if (operation == "compress4-adaptive-coded-legacy") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3], compress_four_lane(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::CodedReset
                )
            );
#if defined(TERNLOG_QKV_BATCH4_HOOK)
        } else if (operation == "compress4-batch") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3], compress_four_lane_batch4(read_binary(argv[2]), use_match)
            );
        } else if (operation == "compress4-batch-adaptive") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3], compress_four_lane_batch4(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::IntegerReset
                )
            );
        } else if (operation == "compress4-batch-adaptive-coded-legacy") {
            const bool use_match = !(argc == 5 && std::string_view(argv[4]) == "--no-match");
            if (argc != 4 && argc != 5) return 2;
            write_binary(
                argv[3], compress_four_lane_batch4(
                    read_binary(argv[2]), use_match,
                    AdaptiveCodingMode::CodedReset
                )
            );
#endif
        } else if (operation == "decompress4") {
            if (argc != 4) return 2;
            write_binary(argv[3], decompress_four_lane(read_binary(argv[2])));
        } else if (operation == "decompress4-lockstep") {
            if (argc != 4) return 2;
            write_binary(
                argv[3], decompress_four_lane_lockstep(read_binary(argv[2]))
            );
#if defined(TERNLOG_QKV_BATCH4_HOOK)
        } else if (operation == "decompress4-batch") {
            if (argc != 4) return 2;
            write_binary(
                argv[3], decompress_four_lane_batch4(read_binary(argv[2]))
            );
#endif
        } else if (operation == "benchmark") {
            if (argc != 4) return 2;
            benchmark_persistent(
                read_binary(argv[2]), std::stoi(argv[3]),
                PersistentBenchmarkMode::Scalar
            );
        } else if (operation == "benchmark4") {
            if (argc != 4) return 2;
            benchmark_persistent(
                read_binary(argv[2]), std::stoi(argv[3]),
                PersistentBenchmarkMode::FourLane
            );
#if defined(TERNLOG_QKV_BATCH4_HOOK)
        } else if (operation == "benchmark4-batch") {
            if (argc != 4) return 2;
            benchmark_persistent(
                read_binary(argv[2]), std::stoi(argv[3]),
                PersistentBenchmarkMode::FourLaneBatch
            );
#endif
        } else {
            codec_usage(argv[0]);
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

#if defined(TERNLOG_QKV_BATCH4_HOOK)
extern "C" void ternlog_m4_fiber_entry(void* opaque) {
    const auto* start = static_cast<const QkvFiberStart*>(opaque);
    start->scheduler->fiber_entry(start->lane);
}
#endif
