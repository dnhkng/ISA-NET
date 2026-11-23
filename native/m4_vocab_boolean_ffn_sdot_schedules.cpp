#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#ifndef TERNLOG_BOOLEAN_LAYOUT_HEADER
#define TERNLOG_BOOLEAN_LAYOUT_HEADER "ternlog_vocab_boolean_story_layout.h"
#endif
#include TERNLOG_BOOLEAN_LAYOUT_HEADER

namespace {

namespace layout = ternlog_vocab_layout;
constexpr std::size_t dimension = layout::kDimension;
constexpr std::size_t correction_features =
    layout::kBooleanFfnCorrectionFeatures;

inline float decode_bits_exact(
    std::uint8_t bits,
    const float* decoder,
    float bias
) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    float decoded = bias;
    for (std::size_t plane = 0; plane < 8; ++plane) {
        const float sign = (bits >> plane) & 1U ? 1.0f : -1.0f;
        decoded = decoded + sign * decoder[plane];
    }
    return decoded;
}

template <std::size_t Rows>
inline void correction_rows(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    int32x4_t accumulators[Rows];
#pragma clang loop unroll(full)
    for (std::size_t row = 0; row < Rows; ++row) {
        accumulators[row] = vdupq_n_s32(0);
    }
    for (
        std::size_t feature = 0;
        feature < correction_features;
        feature += 16
    ) {
        const int8x16_t activation = vld1q_s8(pooled + feature);
#pragma clang loop unroll(full)
        for (std::size_t row = 0; row < Rows; ++row) {
            accumulators[row] = vdotq_s32(
                accumulators[row],
                activation,
                vld1q_s8(weights + row * correction_features + feature)
            );
        }
    }
#pragma clang loop unroll(full)
    for (std::size_t row = 0; row < Rows; ++row) {
        output[row] = vaddvq_s32(accumulators[row]);
    }
}

template <std::size_t Rows>
void correction(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    static_assert(dimension % Rows == 0);
    for (std::size_t row = 0; row < dimension; row += Rows) {
        correction_rows<Rows>(
            output + row, pooled, weights + row * correction_features
        );
    }
}

template <std::size_t Rows>
void decode_correction(
    float* output,
    const std::uint8_t* final_state,
    const std::int8_t* pooled,
    const float* decoder,
    const float* bias,
    const std::int8_t* correction_weights,
    const float* correction_scales
) {
    static_assert(dimension % Rows == 0);
    for (std::size_t row_base = 0; row_base < dimension; row_base += Rows) {
        int32x4_t accumulators[Rows];
#pragma clang loop unroll(full)
        for (std::size_t row = 0; row < Rows; ++row) {
            accumulators[row] = vdupq_n_s32(0);
        }
        for (
            std::size_t feature = 0;
            feature < correction_features;
            feature += 16
        ) {
            const int8x16_t activation = vld1q_s8(pooled + feature);
#pragma clang loop unroll(full)
            for (std::size_t row = 0; row < Rows; ++row) {
                accumulators[row] = vdotq_s32(
                    accumulators[row],
                    activation,
                    vld1q_s8(
                        correction_weights
                        + (row_base + row) * correction_features + feature
                    )
                );
            }
        }
#pragma clang loop unroll(full)
        for (std::size_t row = 0; row < Rows; ++row) {
            const std::size_t output_row = row_base + row;
            const std::uint8_t bits = final_state[output_row];
            const float decoded = decode_bits_exact(
                bits, decoder + output_row * 8, bias[output_row]
            );
            output[output_row] = decoded
                + static_cast<float>(vaddvq_s32(accumulators[row]))
                    * correction_scales[output_row];
        }
    }
}

}  // namespace

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_sdot_1row(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    correction<1>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_sdot_8row(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    correction<8>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_sdot_16row(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    correction<16>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_decode_sdot_1row(
    float* output,
    const std::uint8_t* final_state,
    const std::int8_t* pooled,
    const float* decoder,
    const float* bias,
    const std::int8_t* correction_weights,
    const float* correction_scales
) {
    decode_correction<1>(
        output, final_state, pooled, decoder, bias,
        correction_weights, correction_scales
    );
}

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_decode_sdot_8row(
    float* output,
    const std::uint8_t* final_state,
    const std::int8_t* pooled,
    const float* decoder,
    const float* bias,
    const std::int8_t* correction_weights,
    const float* correction_scales
) {
    decode_correction<8>(
        output, final_state, pooled, decoder, bias,
        correction_weights, correction_scales
    );
}

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_decode_sdot_16row(
    float* output,
    const std::uint8_t* final_state,
    const std::int8_t* pooled,
    const float* decoder,
    const float* bias,
    const std::int8_t* correction_weights,
    const float* correction_scales
) {
    decode_correction<16>(
        output, final_state, pooled, decoder, bias,
        correction_weights, correction_scales
    );
}
