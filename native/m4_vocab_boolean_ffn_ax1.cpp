#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#ifndef TERNLOG_BOOLEAN_LAYOUT_HEADER
#define TERNLOG_BOOLEAN_LAYOUT_HEADER "ternlog_vocab_boolean_story_layout.h"
#endif
#include TERNLOG_BOOLEAN_LAYOUT_HEADER

#if defined(TERNLOG_FFN_BATCH4_HOOK)
#include "m4_vocab_ffn_batch4_hook.h"

extern "C" thread_local TernlogFfnBatchSubmit ternlog_ffn_batch_submit = nullptr;
#endif

extern "C" void ternlog_vocab_ffn_encode_linear_fp16(
    std::uint8_t*, const float*, const float*
);
extern "C" void ternlog_vocab_ffn_encode_linear_fp16_neon(
    std::uint8_t*, const float*, const float*
);
extern "C" void ternlog_vocab_ffn_encode_linear_fp16_fused_neon(
    std::uint8_t*, const float*, const float*
);
extern "C" void ternlog_vocab_ffn_encode_linear_rms_fp16_fused_neon(
    std::uint8_t*, const float*, const __fp16*, const float*, float
);
extern "C" void ternlog_vocab_boolean_ffn_routes_register_mix(
    std::uint8_t*, std::int8_t*, const std::uint8_t*, const std::uint8_t*
);
extern "C" void ternlog_vocab_ffn_decode_sdot_8row(
    float*, const std::uint8_t*, const std::int8_t*, const float*, const float*,
    const std::int8_t*, const float*
);
extern "C" void ternlog_vocab_ffn_decode_sdot_16row(
    float*, const std::uint8_t*, const std::int8_t*, const float*, const float*,
    const std::int8_t*, const float*
);
extern "C" void ternlog_matvec_f16input_fmlal_m4(
    const __fp16*, const __fp16*, const __fp16*, float*, int, int
);

namespace {

namespace layout = ternlog_vocab_layout;
constexpr std::size_t dimension = layout::kDimension;
constexpr std::size_t correction_features =
    layout::kBooleanFfnCorrectionFeatures;
static_assert(dimension % 16 == 0);
static_assert(correction_features % 16 == 0);

inline float decode_bits_exact(
    std::uint8_t bits, const float* decoder, float bias
) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    float decoded = bias;
    for (std::size_t plane = 0; plane < 8; ++plane) {
        const float sign = (bits >> plane) & 1U ? 1.0F : -1.0F;
        decoded = decoded + sign * decoder[plane];
    }
    return decoded;
}

inline void ticket16_decode(
    float* output,
    const std::uint8_t* final_state,
    const std::int8_t* pooled,
    const float* decoder,
    const float* bias,
    const std::int8_t* weights,
    const float* scales,
    const std::uint16_t* offsets,
    const std::uint8_t* columns
) {
    for (std::size_t block_row = 0; block_row < dimension / 16; ++block_row) {
        int32x4_t sums[16] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (
            std::size_t ticket = offsets[block_row];
            ticket < offsets[block_row + 1];
            ++ticket
        ) {
            const int8x16_t value = vld1q_s8(
                pooled + static_cast<std::size_t>(columns[ticket]) * 16
            );
            const std::int8_t* ticket_weights = weights + ticket * 16 * 16;
#pragma clang loop unroll(full)
            for (std::size_t row = 0; row < 16; ++row) {
                sums[row] = vdotq_s32(
                    sums[row], value,
                    vld1q_s8(ticket_weights + row * 16)
                );
            }
        }
#pragma clang loop unroll(full)
        for (std::size_t item = 0; item < 16; ++item) {
            const std::size_t row = block_row * 16 + item;
            output[row] = decode_bits_exact(
                final_state[row], decoder + row * 8, bias[row]
            ) + static_cast<float>(vaddvq_s32(sums[item])) * scales[row];
        }
    }
}

#if defined(TERNLOG_BOOLEAN_FFN_FACTOR32)
inline void factor32_decode(
    float* output,
    const std::uint8_t* final_state,
    const std::int8_t* pooled,
    const float* decoder,
    const float* bias,
    const std::int8_t* input_weights,
    const float* input_scales,
    const __fp16* output_weights
) {
    alignas(64) float hidden_float[32];
    alignas(64) __fp16 hidden[32];
    for (std::size_t row = 0; row < 32; row += 8) {
        int32x4_t sums[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (
            std::size_t feature = 0;
            feature < correction_features;
            feature += 16
        ) {
            const int8x16_t value = vld1q_s8(pooled + feature);
#pragma clang loop unroll(full)
            for (std::size_t item = 0; item < 8; ++item) {
                sums[item] = vdotq_s32(
                    sums[item], value,
                    vld1q_s8(
                        input_weights
                            + (row + item) * correction_features
                            + feature
                    )
                );
            }
        }
#pragma clang loop unroll(full)
        for (std::size_t item = 0; item < 8; ++item) {
            hidden_float[row + item] = static_cast<float>(vaddvq_s32(sums[item]))
                * input_scales[row + item];
        }
    }
    for (std::size_t index = 0; index < 32; index += 8) {
        const float16x4_t low = vcvt_f16_f32(vld1q_f32(hidden_float + index));
        const float16x4_t high = vcvt_f16_f32(vld1q_f32(hidden_float + index + 4));
        vst1q_f16(hidden + index, vcombine_f16(low, high));
    }
    ternlog_matvec_f16input_fmlal_m4(
        output_weights, nullptr, hidden, output, dimension, 32
    );
    for (std::size_t row = 0; row < dimension; ++row) {
        output[row] += decode_bits_exact(
            final_state[row], decoder + row * 8, bias[row]
        );
    }
}
#endif

}  // namespace

extern "C" __attribute__((noinline)) void ternlog_vocab_ffn_decode_sdot_tickets16(
    float* output,
    const std::uint8_t* final_state,
    const std::int8_t* pooled,
    const float* decoder,
    const float* bias,
    const std::int8_t* weights,
    const float* scales,
    const std::uint16_t* offsets,
    const std::uint8_t* columns
) {
    ticket16_decode(
        output, final_state, pooled, decoder, bias, weights, scales,
        offsets, columns
    );
}

extern "C" __attribute__((noinline)) void ternlog_vocab_boolean_ffn_ax1(
    float* output,
    const float* input,
    const float* offsets,
    const std::uint8_t* program,
    const float* decoder,
    const std::int8_t* decoder_int8,
    const float* decoder_scales,
    const float* bias,
    const std::int8_t* correction,
    const float* correction_scales,
    const std::uint16_t* correction_ticket_offsets,
    const std::uint8_t* correction_ticket_columns,
    const __fp16* correction_output,
    const __fp16* norm_weights,
    float inverse_rms
) {
    alignas(64) std::uint8_t encoded[dimension];
    alignas(64) std::uint8_t final_state[dimension];
    alignas(64) std::int8_t pooled[correction_features];
#if defined(TERNLOG_FFN_RMS_ENCODE_FUSED)
    ternlog_vocab_ffn_encode_linear_rms_fp16_fused_neon(
        encoded, input, norm_weights, offsets, inverse_rms
    );
#elif defined(TERNLOG_BOOLEAN_FFN_ENCODE_FUSED_NEON)
    ternlog_vocab_ffn_encode_linear_fp16_fused_neon(encoded, input, offsets);
#elif defined(TERNLOG_BOOLEAN_FFN_ENCODE_NEON)
    ternlog_vocab_ffn_encode_linear_fp16_neon(encoded, input, offsets);
#else
    ternlog_vocab_ffn_encode_linear_fp16(encoded, input, offsets);
#endif
    ternlog_vocab_boolean_ffn_routes_register_mix(
        final_state, pooled, encoded, program
    );
#if defined(TERNLOG_FFN_BATCH4_HOOK)
    alignas(64) std::int32_t raw_sums[dimension];
    const TernlogFfnBatchRequest batch_request{
        output, raw_sums, final_state, pooled, correction,
        decoder, bias, correction_scales,
    };
    if (ternlog_ffn_batch_submit != nullptr
        && ternlog_ffn_batch_submit(batch_request)) {
        return;
    }
#endif
#if defined(TERNLOG_BOOLEAN_FFN_CORRECTION_TICKETS16)
    ternlog_vocab_ffn_decode_sdot_tickets16(
        output, final_state, pooled, decoder, bias, correction,
        correction_scales, correction_ticket_offsets, correction_ticket_columns
    );
#elif defined(TERNLOG_BOOLEAN_FFN_FACTOR32)
    factor32_decode(
        output, final_state, pooled, decoder, bias,
        correction, correction_scales, correction_output
    );
#elif defined(TERNLOG_BOOLEAN_FFN_SDOT8) || defined(TERNLOG_BOOLEAN_FFN_SDOT16)
#if defined(TERNLOG_BOOLEAN_FFN_SDOT8)
    ternlog_vocab_ffn_decode_sdot_8row(
        output, final_state, pooled, decoder, bias, correction, correction_scales
    );
#else
    ternlog_vocab_ffn_decode_sdot_16row(
        output, final_state, pooled, decoder, bias, correction, correction_scales
    );
#endif
#else
    for (std::size_t row = 0; row < dimension; ++row) {
#if defined(TERNLOG_BOOLEAN_FFN_DECODER_W8A8) \
    || defined(TERNLOG_BOOLEAN_FFN_DECODER_W8A8_GROUP4)
        alignas(16) std::int8_t signs[16]{};
        const std::uint8_t bits = final_state[row];
#pragma clang loop unroll(full)
        for (std::size_t plane = 0; plane < 8; ++plane) {
            signs[plane] = (bits >> plane) & 1U ? 1 : -1;
        }
        const int32x4_t decoder_dot = vdotq_s32(
            vdupq_n_s32(0), vld1q_s8(signs),
            vld1q_s8(decoder_int8 + row * 16)
        );
#if defined(TERNLOG_BOOLEAN_FFN_DECODER_W8A8_GROUP4)
        float decoded = bias[row]
            + static_cast<float>(vgetq_lane_s32(decoder_dot, 0))
                * decoder_scales[row * 2]
            + static_cast<float>(vgetq_lane_s32(decoder_dot, 1))
                * decoder_scales[row * 2 + 1];
#else
        float decoded = bias[row]
            + static_cast<float>(vaddvq_s32(decoder_dot)) * decoder_scales[row];
#endif
#else
        float decoded = bias[row];
        const std::uint8_t bits = final_state[row];
        for (std::size_t plane = 0; plane < 8; ++plane) {
            decoded += ((bits >> plane) & 1U ? 1.0f : -1.0f)
                * decoder[row * 8 + plane];
        }
#endif
        int32x4_t dot = vdupq_n_s32(0);
        for (
            std::size_t feature = 0;
            feature < correction_features;
            feature += 16
        ) {
            dot = vdotq_s32(
                dot,
                vld1q_s8(pooled + feature),
                vld1q_s8(
                    correction + row * correction_features + feature
                )
            );
        }
        output[row] = decoded
            + static_cast<float>(vaddvq_s32(dot)) * correction_scales[row];
    }
#endif
}
