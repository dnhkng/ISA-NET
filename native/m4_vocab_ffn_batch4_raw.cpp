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
constexpr std::size_t features = layout::kBooleanFfnCorrectionFeatures;
#ifndef TERNLOG_FFN_BATCH_LANES
#define TERNLOG_FFN_BATCH_LANES 4
#endif
constexpr std::size_t lanes = TERNLOG_FFN_BATCH_LANES;
static_assert(dimension % 4 == 0);
static_assert(features % 16 == 0);

template <std::size_t Rows>
void batch4_raw_ptrs(
    std::int32_t* const output[lanes],
    const std::int8_t* const pooled[lanes],
    const std::int8_t* weights
) {
    static_assert(dimension % Rows == 0);
    for (std::size_t row_base = 0; row_base < dimension; row_base += Rows) {
        int32x4_t sums[lanes][Rows];
#pragma clang loop unroll(full)
        for (std::size_t lane = 0; lane < lanes; ++lane) {
#pragma clang loop unroll(full)
            for (std::size_t row = 0; row < Rows; ++row) {
                sums[lane][row] = vdupq_n_s32(0);
            }
        }
        for (std::size_t feature = 0; feature < features; feature += 16) {
            int8x16_t activations[lanes];
#pragma clang loop unroll(full)
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                activations[lane] = vld1q_s8(
                    pooled[lane] + feature
                );
            }
#pragma clang loop unroll(full)
            for (std::size_t row = 0; row < Rows; ++row) {
                const int8x16_t weight = vld1q_s8(
                    weights + (row_base + row) * features + feature
                );
#pragma clang loop unroll(full)
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    sums[lane][row] = vdotq_s32(
                        sums[lane][row], weight, activations[lane]
                    );
                }
            }
        }
#pragma clang loop unroll(full)
        for (std::size_t lane = 0; lane < lanes; ++lane) {
#pragma clang loop unroll(full)
            for (std::size_t row = 0; row < Rows; ++row) {
                output[lane][row_base + row] =
                    vaddvq_s32(sums[lane][row]);
            }
        }
    }
}

template <std::size_t Rows>
void batch4_raw_contiguous(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    std::int32_t* outputs[lanes];
    const std::int8_t* inputs[lanes];
#pragma clang loop unroll(full)
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        outputs[lane] = output + lane * dimension;
        inputs[lane] = pooled + lane * features;
    }
    batch4_raw_ptrs<Rows>(outputs, inputs, weights);
}

void scalar_postprocess(
    float* output,
    const std::uint8_t* final_state,
    const std::int32_t* raw_sums,
    const float* decoder,
    const float* bias,
    const float* correction_scales
) {
#pragma clang loop vectorize(disable) interleave(disable)
    for (std::size_t row = 0; row < dimension; ++row) {
        float decoded = bias[row];
        const std::uint8_t bits = final_state[row];
        for (std::size_t plane = 0; plane < 8; ++plane) {
            decoded += ((bits >> plane) & 1U ? 1.0f : -1.0f)
                * decoder[row * 8 + plane];
        }
        output[row] = decoded
            + static_cast<float>(raw_sums[row]) * correction_scales[row];
    }
}

}  // namespace

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_sdot_batch4_raw_1row(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    batch4_raw_contiguous<1>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_sdot_batch4_raw_2row(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    batch4_raw_contiguous<2>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_sdot_batch4_raw_4row(
    std::int32_t* output,
    const std::int8_t* pooled,
    const std::int8_t* weights
) {
    batch4_raw_contiguous<4>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void
ternlog_vocab_ffn_sdot_batch4_raw_ptrs_2row(
    std::int32_t* const output[lanes],
    const std::int8_t* const pooled[lanes],
    const std::int8_t* weights
) {
    batch4_raw_ptrs<2>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void
ternlog_vocab_ffn_sdot_batch4_raw_ptrs_4row(
    std::int32_t* const output[lanes],
    const std::int8_t* const pooled[lanes],
    const std::int8_t* weights
) {
    batch4_raw_ptrs<4>(output, pooled, weights);
}

extern "C" __attribute__((noinline)) void
ternlog_vocab_ffn_batch4_postprocess_ptrs(
    float* const output[lanes],
    const std::uint8_t* const final_state[lanes],
    const std::int32_t* const raw_sums[lanes],
    const float* decoder,
    const float* bias,
    const float* correction_scales
) {
    const uint32x4_t one = vdupq_n_u32(1U);
    const uint32x4_t sign_bit = vdupq_n_u32(0x80000000U);
    for (std::size_t row = 0; row < dimension; ++row) {
        const uint32x4_t bits = {
            final_state[0][row], final_state[1][row],
            final_state[2][row], final_state[3][row],
        };
        float32x4_t coefficients[8];
#pragma clang loop unroll(full)
        for (std::size_t plane = 0; plane < 8; ++plane) {
            const uint32x4_t bit = vandq_u32(
                vshlq_u32(bits, vdupq_n_s32(-static_cast<std::int32_t>(plane))),
                one
            );
            const uint32x4_t negate = vandq_u32(vceqq_u32(bit, vdupq_n_u32(0)), sign_bit);
            const uint32x4_t coefficient = veorq_u32(
                vreinterpretq_u32_f32(vdupq_n_f32(decoder[row * 8 + plane])),
                negate
            );
            coefficients[plane] = vreinterpretq_f32_u32(coefficient);
        }
        const int32x4_t raw = {
            raw_sums[0][row], raw_sums[1][row],
            raw_sums[2][row], raw_sums[3][row],
        };
        const float32x4_t raw_f = vcvtq_f32_s32(raw);
        const float32x4_t scale = vdupq_n_f32(correction_scales[row]);
        const float32x4_t row_bias = vdupq_n_f32(bias[row]);
        float32x4_t sum01;
        float32x4_t sum23;
        float32x4_t sum03;
        float32x4_t sum56;
        float32x4_t sum46;
        float32x4_t sum06;
        float32x4_t correction7;
        float32x4_t base;
        float32x4_t decoded;
        asm volatile(
            "fadd %[sum01].4s, %[c0].4s, %[c1].4s\n"
            "fadd %[sum23].4s, %[c2].4s, %[c3].4s\n"
            "fadd %[sum03].4s, %[sum01].4s, %[sum23].4s\n"
            "fadd %[sum56].4s, %[c5].4s, %[c6].4s\n"
            "fadd %[sum46].4s, %[c4].4s, %[sum56].4s\n"
            "fadd %[sum06].4s, %[sum46].4s, %[sum03].4s\n"
            "mov %[correction7].16b, %[c7].16b\n"
            "fmla %[correction7].4s, %[raw].4s, %[scale].4s\n"
            "fadd %[base].4s, %[bias].4s, %[correction7].4s\n"
            "fadd %[decoded].4s, %[sum06].4s, %[base].4s\n"
            : [sum01] "=&w" (sum01), [sum23] "=&w" (sum23),
              [sum03] "=&w" (sum03), [sum56] "=&w" (sum56),
              [sum46] "=&w" (sum46), [sum06] "=&w" (sum06),
              [correction7] "=&w" (correction7), [base] "=&w" (base),
              [decoded] "=&w" (decoded)
            : [c0] "w" (coefficients[0]), [c1] "w" (coefficients[1]),
              [c2] "w" (coefficients[2]), [c3] "w" (coefficients[3]),
              [c4] "w" (coefficients[4]), [c5] "w" (coefficients[5]),
              [c6] "w" (coefficients[6]), [c7] "w" (coefficients[7]),
              [raw] "w" (raw_f), [scale] "w" (scale), [bias] "w" (row_bias)
        );
        output[0][row] = vgetq_lane_f32(decoded, 0);
        output[1][row] = vgetq_lane_f32(decoded, 1);
        output[2][row] = vgetq_lane_f32(decoded, 2);
        output[3][row] = vgetq_lane_f32(decoded, 3);
    }
}

extern "C" __attribute__((noinline)) void
ternlog_vocab_ffn_scalar_postprocess(
    float* output,
    const std::uint8_t* final_state,
    const std::int32_t* raw_sums,
    const float* decoder,
    const float* bias,
    const float* correction_scales
) {
    scalar_postprocess(
        output, final_state, raw_sums, decoder, bias, correction_scales
    );
}
