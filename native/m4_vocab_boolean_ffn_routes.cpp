#define TERNLOG_BOOLEAN_FFN_NO_MAIN
#include "m4_vocab_boolean_ffn_core.cpp"

#include <bit>
#include <cstring>

#ifndef TERNLOG_BOOLEAN_LAYOUT_HEADER
#define TERNLOG_BOOLEAN_LAYOUT_HEADER "ternlog_vocab_boolean_story_layout.h"
#endif
#include TERNLOG_BOOLEAN_LAYOUT_HEADER

namespace {

namespace layout = ternlog_vocab_layout;
constexpr std::size_t route_bits = layout::kDimension * 8;
constexpr std::size_t route_bytes = route_bits / 8;
constexpr std::size_t route_features = layout::kBooleanFfnCorrectionFeatures;
constexpr std::size_t route_states = 4;
constexpr int pool_schedule = layout::kBooleanFfnPoolSchedule;
static_assert(route_bytes % 16 == 0);
static_assert(route_features % route_states == 0);
static_assert(pool_schedule >= 0 && pool_schedule <= 2);

struct ScatteredRoute {
    std::uint8_t snapshot;
    std::uint8_t byte;
    std::uint8_t shift;
};

constexpr auto scattered_routes = [] {
    std::array<ScatteredRoute, route_features - route_bits / 16> routes{};
    for (std::size_t index = 0; index < routes.size(); ++index) {
        const std::size_t feature = index + route_bits / 16;
        const std::size_t start = (feature * 37) % route_bits;
        routes[index] = {
            static_cast<std::uint8_t>(
                feature / (route_features / route_states)
            ),
            static_cast<std::uint8_t>(start / 8),
            static_cast<std::uint8_t>(start % 8),
        };
    }
    return routes;
}();

void global_byte_rotate(
    const std::uint8_t* input, std::uint8_t* output, unsigned shift
) {
    const std::size_t byte_shift = shift % route_bytes;
    for (std::size_t index = 0; index < route_bytes; ++index) {
        output[index] = input[(index + route_bytes - byte_shift) % route_bytes];
    }
}

void register_rotate(
    const std::uint8_t* input, std::uint8_t* output, unsigned shift
) {
    constexpr std::size_t registers = route_bytes / 16;
    const std::size_t register_shift = shift % registers;
    for (std::size_t index = 0; index < registers; ++index) {
        const std::size_t source = (index + registers - register_shift) % registers;
        vst1q_u8(output + index * 16, vld1q_u8(input + source * 16));
    }
}

template <int Shift>
void full_bit_rotate(const std::uint8_t* input, std::uint8_t* output) {
    static_assert(Shift > 0 && Shift < 8);
    for (std::size_t offset = 0; offset < route_bytes; offset += 16) {
        const uint8x16_t current = vld1q_u8(input + offset);
        const uint8x16_t previous_register = vld1q_u8(
            input + (offset == 0 ? route_bytes - 16 : offset - 16)
        );
        const uint8x16_t previous_bytes = vextq_u8(previous_register, current, 15);
        vst1q_u8(
            output + offset,
            vorrq_u8(
                vshlq_n_u8(current, Shift),
                vshrq_n_u8(previous_bytes, 8 - Shift)
            )
        );
    }
}

int signed_window16_routes(const std::uint8_t* packed, std::size_t start) {
    int ones = 0;
    if (start + 16 <= route_bits) {
        const std::size_t byte = start / 8;
        const unsigned shift = static_cast<unsigned>(start % 8);
        std::uint32_t word = packed[byte];
        if (byte + 1 < route_bytes) word |= static_cast<std::uint32_t>(packed[byte + 1]) << 8;
        if (byte + 2 < route_bytes) word |= static_cast<std::uint32_t>(packed[byte + 2]) << 16;
        ones = std::popcount((word >> shift) & 0xffffU);
    } else {
        for (std::size_t lane = 0; lane < 16; ++lane) {
            const std::size_t bit = (start + lane) % route_bits;
            ones += (packed[bit / 8] >> (bit % 8)) & 1U;
        }
    }
    return 2 * ones - 16;
}

int signed_aligned_window16(
    const std::uint8_t* packed, std::size_t block
) {
    std::uint16_t word;
    std::memcpy(&word, packed + block * 2, sizeof(word));
    return 2 * std::popcount(word) - 16;
}

void pool_aligned_route_features(
    std::int8_t* pooled,
    const std::uint8_t* packed,
    std::size_t first_feature,
    std::size_t feature_count
) {
    for (std::size_t feature = 0; feature < feature_count; feature += 8) {
        const uint8x16_t counts = vcntq_u8(vld1q_u8(
            packed + (first_feature + feature) * 2
        ));
        const int16x8_t signed_counts = vsubq_s16(
            vreinterpretq_s16_u16(vshlq_n_u16(vpaddlq_u8(counts), 1)),
            vdupq_n_s16(16)
        );
        vst1_s8(pooled + first_feature + feature, vqmovn_s16(signed_counts));
    }
}

int signed_window16_padded(
    const std::uint8_t* packed, std::uint8_t byte, std::uint8_t shift
) {
    const std::uint32_t word = static_cast<std::uint32_t>(packed[byte])
        | static_cast<std::uint32_t>(packed[byte + 1]) << 8
        | static_cast<std::uint32_t>(packed[byte + 2]) << 16;
    return 2 * std::popcount((word >> shift) & 0xffffU) - 16;
}

template <int Mode>
void route_operands(
    const std::uint8_t* state,
    std::uint8_t* second,
    std::uint8_t* third,
    int round
) {
    const unsigned shift = 1U << round;
    if constexpr (Mode == 0) {
        route_round(state, second, third, route_bytes, shift);
    } else {
        global_byte_rotate(state, second, shift);
        if constexpr (Mode == 1) {
            register_rotate(state, third, shift);
        } else {
            if (shift == 1) full_bit_rotate<1>(state, third);
            else if (shift == 2) full_bit_rotate<2>(state, third);
            else full_bit_rotate<4>(state, third);
        }
    }
}

template <int Mode>
__attribute__((noinline)) void transform_and_pool_routes(
    std::uint8_t* final_state,
    std::int8_t* pooled,
    const std::uint8_t* input_state,
    const std::uint8_t* program
) {
#if defined(TERNLOG_BOOLEAN_FFN_ROUTE_POOL_PADDED)
    constexpr std::size_t snapshot_bytes = route_bytes + 2;
#else
    constexpr std::size_t snapshot_bytes = route_bytes;
#endif
    alignas(16) std::array<std::uint8_t, route_bytes> state{};
    alignas(16) std::array<std::uint8_t, route_bytes> previous{};
    alignas(16) std::array<std::uint8_t, route_bytes> second{};
    alignas(16) std::array<std::uint8_t, route_bytes> third{};
    alignas(16) std::array<
        std::array<std::uint8_t, snapshot_bytes>, route_states
    > snapshots{};
    const auto save_snapshot = [&](std::size_t index) {
        std::memcpy(snapshots[index].data(), state.data(), route_bytes);
#if defined(TERNLOG_BOOLEAN_FFN_ROUTE_POOL_PADDED)
        snapshots[index][route_bytes] = state[0];
        snapshots[index][route_bytes + 1] = state[1];
#endif
    };
    std::memcpy(state.data(), input_state, route_bytes);
    save_snapshot(0);
    for (int round = 0; round < 3; ++round) {
        previous = state;
        route_operands<Mode>(previous.data(), second.data(), third.data(), round);
        boolean_round(
            previous.data(), second.data(), third.data(),
            program + static_cast<std::size_t>(round) * 8 * route_bytes,
            nullptr, state.data(), route_bytes, "lut3"
        );
        save_snapshot(round + 1);
    }
    constexpr std::size_t features_per_state = route_features / route_states;
    for (std::size_t feature = 0; feature < route_features; ++feature) {
        const std::size_t snapshot = feature / features_per_state;
        if constexpr (pool_schedule == 0) {
            const std::size_t start = feature < route_bits / 16
                ? feature * 16
                : (feature * 37) % route_bits;
            pooled[feature] = static_cast<std::int8_t>(signed_window16_routes(
                snapshots[snapshot].data(), start
            ));
        } else {
            constexpr std::size_t blocks = route_bits / 16;
            const std::size_t local = feature % features_per_state;
            const std::size_t offset = pool_schedule == 1
                ? 0
                : snapshot * (blocks / route_states);
            pooled[feature] = static_cast<std::int8_t>(signed_aligned_window16(
                snapshots[snapshot].data(), (local + offset) % blocks
            ));
        }
    }
    std::memcpy(final_state, state.data(), route_bytes);
}

void sdot_route_correction(
    float* output, const std::int8_t* pooled,
    const std::int8_t* weights, const float* scales
) {
    for (std::size_t row = 0; row < layout::kDimension; ++row) {
        int32x4_t accumulator = vdupq_n_s32(0);
        for (std::size_t feature = 0; feature < route_features; feature += 16) {
            accumulator = vdotq_s32(
                accumulator, vld1q_s8(pooled + feature),
                vld1q_s8(weights + row * route_features + feature)
            );
        }
        output[row] = static_cast<float>(vaddvq_s32(accumulator)) * scales[row];
    }
}

}  // namespace

extern "C" void ternlog_vocab_boolean_ffn_routes_byte_lane(
    std::uint8_t* final_state, std::int8_t* pooled,
    const std::uint8_t* input_state, const std::uint8_t* program
) { transform_and_pool_routes<0>(final_state, pooled, input_state, program); }

extern "C" void ternlog_vocab_boolean_ffn_routes_register_mix(
    std::uint8_t* final_state, std::int8_t* pooled,
    const std::uint8_t* input_state, const std::uint8_t* program
) { transform_and_pool_routes<1>(final_state, pooled, input_state, program); }

extern "C" void ternlog_vocab_boolean_ffn_routes_bitmix(
    std::uint8_t* final_state, std::int8_t* pooled,
    const std::uint8_t* input_state, const std::uint8_t* program
) { transform_and_pool_routes<2>(final_state, pooled, input_state, program); }

#ifndef TERNLOG_BOOLEAN_FFN_ROUTES_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 6) {
            std::cerr << "usage: m4_vocab_boolean_ffn_routes byte_lane|register_mix|bitmix STATE PROGRAM [WEIGHTS SCALES]\n";
            return 2;
        }
        const std::string mode = argv[1];
        const auto state = read_file(argv[2]);
        const auto program = read_file(argv[3]);
        if (state.size() != route_bytes || program.size() != 3 * 8 * route_bytes) {
            throw std::runtime_error("input size mismatch");
        }
        std::array<std::uint8_t, route_bytes> final_state{};
        std::array<std::int8_t, route_features> pooled{};
        auto function = mode == "byte_lane" ? ternlog_vocab_boolean_ffn_routes_byte_lane
            : mode == "register_mix" ? ternlog_vocab_boolean_ffn_routes_register_mix
            : mode == "bitmix" ? ternlog_vocab_boolean_ffn_routes_bitmix : nullptr;
        if (function == nullptr) throw std::runtime_error("invalid route mode");
        function(final_state.data(), pooled.data(), state.data(), program.data());
        std::cout.write(reinterpret_cast<const char*>(final_state.data()), final_state.size());
        std::cout.write(reinterpret_cast<const char*>(pooled.data()), pooled.size());
        if (argc == 6) {
            const auto weights = read_file(argv[4]);
            const auto scales = read_file(argv[5]);
            if (
                weights.size() != layout::kDimension * route_features
                || scales.size() != layout::kDimension * sizeof(float)
            ) {
                throw std::runtime_error("correction size mismatch");
            }
            std::array<float, layout::kDimension> correction{};
            sdot_route_correction(
                correction.data(), pooled.data(),
                reinterpret_cast<const std::int8_t*>(weights.data()),
                reinterpret_cast<const float*>(scales.data())
            );
            std::cout.write(reinterpret_cast<const char*>(correction.data()), sizeof(correction));
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
