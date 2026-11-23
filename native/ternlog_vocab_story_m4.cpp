#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(TERNLOG_BOOLEAN_AX1)
#ifndef TERNLOG_BOOLEAN_LAYOUT_HEADER
#define TERNLOG_BOOLEAN_LAYOUT_HEADER "ternlog_vocab_boolean_story_layout.h"
#endif
#include TERNLOG_BOOLEAN_LAYOUT_HEADER
#else
#include "ternlog_vocab_story_layout.h"
#endif

extern "C" const unsigned char ternlog_vocab_story_weights[];
extern "C" const unsigned char ternlog_vocab_story_weights_end[];
extern "C" void ternlog_matvec_f16_fmlal_m4(
    const __fp16* weights,
    const __fp16* bias,
    const float* input,
    float* output,
    int rows,
    int columns
);
extern "C" void ternlog_matvec_f16input_fmlal_m4(
    const __fp16* weights,
    const __fp16* bias,
    const __fp16* input,
    float* output,
    int rows,
    int columns
);
extern "C" void ternlog_matvec_f16input_sve_m4(
    const __fp16* weights,
    const __fp16* bias,
    const __fp16* input,
    float* output,
    int rows,
    int columns
);
#if defined(TERNLOG_BOOLEAN_QKV_ASM8)
extern "C" void ternlog_vocab_qkv_sdot_480x160_m4_8row(
    const std::int8_t* weights,
    const std::int8_t* activation,
    std::int32_t* sums
);
#elif defined(TERNLOG_BOOLEAN_QKV_ASM16)
extern "C" void ternlog_vocab_qkv_sdot_480x160_m4_16row(
    const std::int8_t* weights,
    const std::int8_t* activation,
    std::int32_t* sums
);
#endif
extern "C" void ternlog_vocab_qkv_packed_sdot_192x160_m4_8row(
    const std::int8_t* weights,
    const std::int8_t* activation,
    std::int32_t* sums
);
extern "C" void ternlog_vocab_qkv_packed_sdot_192x160_m4_16row(
    const std::int8_t* weights,
    const std::int8_t* activation,
    std::int32_t* sums
);
#if defined(TERNLOG_BOOLEAN_AX1)
extern "C" void ternlog_vocab_boolean_embedding160(
    float*, const std::uint8_t*, std::size_t
);
extern "C" void ternlog_vocab_boolean_hybrid64_thirdscale_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const float*
);
extern "C" void ternlog_vocab_boolean_hybrid64_factor32_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const __fp16*, const float*
);
extern "C" void ternlog_vocab_boolean_hybrid64_factor64_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const __fp16*, const float*
);
extern "C" void ternlog_vocab_boolean_hybrid64_factor_native_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const __fp16*, const float*
);
extern "C" void ternlog_vocab_boolean_hierarchy64_factor32_prepare(
    float*, std::uint8_t*, __fp16*, const float*, const std::uint8_t*,
    const std::int8_t*, const float*, const __fp16*, const float*
);
extern "C" void ternlog_vocab_boolean_hierarchy64_factor32_leaf(
    float*, const std::uint8_t*, const __fp16*, const std::uint8_t*,
    const std::uint16_t*, const __fp16*, const float*
);
extern "C" void ternlog_vocab_boolean_hybrid64_factor32_w8a8_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const std::int8_t*, const float*, const float*
);
extern "C" void ternlog_vocab_boolean_hybrid64_factor32_w8a8_group16_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const std::int8_t*, const float*, const float*
);
extern "C" void ternlog_vocab_boolean_hybrid64_factor48_w8a8_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const std::int8_t*, const float*, const float*
);
extern "C" void ternlog_vocab_boolean_hybrid64_factor48_w8a8_group16_2048(
    float*, const float*, const std::uint8_t*, const std::int8_t*,
    const float*, const std::int8_t*, const float*, const float*
);
extern "C" void ternlog_vocab_boolean_ffn_ax1(
    float*, const float*, const float*, const std::uint8_t*, const float*,
    const std::int8_t*, const float*, const float*, const std::int8_t*,
    const float*, const std::uint16_t*, const std::uint8_t*, const __fp16*,
    const __fp16*, float
);
#if defined(TERNLOG_BOOLEAN_QKV_SIGNED_SQUEEZE)
extern "C" void ternlog_vocab_qkv_signed_counts40_m4(
    std::int8_t*, const float*, const float*, const float*, const std::uint32_t*
);
extern "C" void ternlog_vocab_qkv_signed_sdot_480x48_m4(
    std::int32_t*, const std::int8_t*, const std::int8_t*
);
extern "C" void ternlog_vocab_qkv_round160_m4(float*, const float*, int);
#endif
#endif

namespace {

namespace layout = ternlog_vocab_layout;
using Half = __fp16;
#if defined(TERNLOG_INT8_QK_CACHE_FIXED4) \
    || defined(TERNLOG_INT8_QK_CACHE_FIXED4_FP16) \
    || defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
using KeyCacheScalar = std::int8_t;
#elif defined(TERNLOG_FP16_KEY_CACHE)
using KeyCacheScalar = Half;
#else
using KeyCacheScalar = float;
#endif
#if defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
using ValueCacheScalar = std::int8_t;
#elif defined(TERNLOG_FP16_VALUE_CACHE)
using ValueCacheScalar = Half;
#else
using ValueCacheScalar = float;
#endif

constexpr int kDim = layout::kDimension;
constexpr int kHeads = layout::kHeads;
constexpr int kHeadDim = kDim / kHeads;
constexpr int kQkHeadDim = layout::kQkHeadDimension;
constexpr int kLocalQkHeadDim = layout::kLocalQkHeadDimension;
static_assert(kDim > 0 && kDim % 16 == 0);
static_assert(kHeads > 0 && kDim % kHeads == 0);
static_assert(kHeadDim % 4 == 0);
static_assert(layout::kLayers > 0 && layout::kLayers <= 64);
static_assert(layout::kLocalHeads >= 0 && layout::kLocalHeads <= kHeads);
static_assert(kQkHeadDim > 0 && kQkHeadDim % 4 == 0);
static_assert(kLocalQkHeadDim > 0 && kLocalQkHeadDim % 4 == 0);
static_assert(layout::kBooleanFfnCorrectionFeatures > 0);
static_assert(layout::kBooleanFfnCorrectionFeatures % 16 == 0);
static_assert(layout::kBooleanFfnCorrectionFeatures % 4 == 0);
static_assert(layout::kBooleanVocabCorrectionFeatures > 0);
static_assert(layout::kBooleanVocabCorrectionRank > 0);
static_assert(layout::kBooleanVocabCorrectionRank % 16 == 0);
#if defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR) \
    || defined(TERNLOG_QKV_VALUE_CACHE_FUSED_NEON)
static_assert(std::is_same_v<ValueCacheScalar, std::int8_t>);
std::size_t shared_value_cache_int8_offset(
    int layer, int value_head, int slot, int channel
);
#endif
constexpr int kRopePairs =
#if defined(TERNLOG_BOOLEAN_AX1)
    layout::kRopePairs;
#else
    kQkHeadDim / 2;
#endif
constexpr float kAttentionScale = layout::kAttentionScale;
constexpr float kRmsEpsilon = std::numeric_limits<float>::epsilon();
constexpr float kQkRmsEpsilon = 1.0e-6F;
constexpr std::string_view kEndOfText = "<|endoftext|>";
constexpr std::string_view kDefaultPrompt =
    "\n<|endoftext|>\nOnce upon a time, there was a little girl named Lily.";

template <typename T>
const T* at(std::size_t offset) {
    return reinterpret_cast<const T*>(ternlog_vocab_story_weights + offset);
}

#if defined(TERNLOG_BOOLEAN_AX1)
struct LayerWeights {
    const float* residual_scales;
    const Half* attention_norm;
    const float* attention_slopes;
    const float* value_cache_scale;
    const Half* qkv;
    const Half* attention_output;
    const std::int8_t* qkv_int8;
    const float* qkv_scales;
    const Half* qkv_query_transform;
    const Half* qkv_key_transform;
    const std::int8_t* qkv_local_int8;
    const float* qkv_local_scales;
    const std::int8_t* qkv_global_int8;
    const float* qkv_global_scales;
    const Half* qkv_global_transform;
    const Half* qkv_route_weight;
    const std::uint8_t* qkv_route_offsets;
    const std::int8_t* qkv_value_int8;
    const float* qkv_value_scales;
    const std::uint16_t* qkv_ticket_offsets;
    const std::uint8_t* qkv_ticket_columns;
    const std::uint8_t* qkv_value_block_mask;
    const float* qkv_squeeze_offsets;
    const float* qkv_squeeze_thresholds;
    const std::uint32_t* qkv_squeeze_polarities;
    const std::int8_t* qkv_squeeze_int8;
    const float* qkv_squeeze_scales;
    const std::int8_t* qkv_bypass_int8;
    const float* qkv_bypass_scales;
    const std::uint32_t* qkv_bypass_start;
    const std::int8_t* attention_output_int8;
    const float* attention_output_scales;
    const std::uint8_t* attention_output_block_mask;
    const std::uint16_t* attention_output_ticket_offsets;
    const std::uint8_t* attention_output_ticket_columns;
    const Half* feedforward_norm;
    const float* encoder_offsets;
    const std::uint8_t* boolean_program;
    const float* decoder_weight;
    const std::int8_t* decoder_int8;
    const float* decoder_scales;
    const float* decoder_bias;
    const std::int8_t* correction_int8;
    const float* correction_scales;
    const std::uint16_t* correction_ticket_offsets;
    const std::uint8_t* correction_ticket_columns;
    const Half* correction_output_f16;
};

struct Weights {
    const std::uint8_t* token_codes = at<std::uint8_t>(layout::kBooleanTokenCodes);
    const std::int8_t* ngram_table =
        at<std::int8_t>(layout::kNgramTableInt8);
    const float* ngram_scales = at<float>(layout::kNgramTableScales);
    const Half* input_norm = at<Half>(layout::kInputNormWeight);
    const Half* final_norm = at<Half>(layout::kFinalNormWeight);
    const std::int8_t* readout_correction =
        at<std::int8_t>(layout::kReadoutCorrection);
    const float* readout_correction_scales =
        at<float>(layout::kReadoutCorrectionScales);
    const Half* readout_correction_output =
        at<Half>(layout::kReadoutCorrectionOutput);
    const std::int8_t* readout_correction_output_int8 =
        at<std::int8_t>(layout::kReadoutCorrectionOutputInt8);
    const float* readout_correction_output_scales =
        at<float>(layout::kReadoutCorrectionOutputScales);
    const float* readout_bias = at<float>(layout::kReadoutBias);
    const std::uint8_t* hierarchy_leaf_codes =
        at<std::uint8_t>(layout::kReadoutHierarchyLeafCodes);
    const std::uint8_t* hierarchy_cluster_codes =
        at<std::uint8_t>(layout::kReadoutHierarchyClusterCodes);
    const Half* hierarchy_cluster_output =
        at<Half>(layout::kReadoutHierarchyClusterOutput);
    const float* hierarchy_cluster_bias =
        at<float>(layout::kReadoutHierarchyClusterBias);
    const std::uint16_t* hierarchy_token_ids =
        at<std::uint16_t>(layout::kReadoutHierarchyTokenIds);
    const std::uint16_t* hierarchy_token_index =
        at<std::uint16_t>(layout::kReadoutHierarchyTokenIndex);
    std::array<LayerWeights, layout::kLayers> layers{};

    Weights() {
        const std::size_t embedded_bytes = static_cast<std::size_t>(
            ternlog_vocab_story_weights_end - ternlog_vocab_story_weights
        );
        if (embedded_bytes != layout::kBinaryBytes) {
            std::fprintf(
                stderr, "embedded model size mismatch: %zu != %zu\n",
                embedded_bytes, layout::kBinaryBytes
            );
            std::exit(2);
        }
        for (int index = 0; index < layout::kLayers; ++index) {
            const auto& source = layout::kLayerOffsets[index];
            layers[index] = {
                at<float>(source.residual_scales),
                at<Half>(source.attention_norm),
                at<float>(source.attention_slopes),
                at<float>(source.value_cache_scale),
                at<Half>(source.qkv),
                at<Half>(source.attention_output),
                at<std::int8_t>(source.qkv_int8),
                at<float>(source.qkv_scales),
                at<Half>(source.qkv_query_transform),
                at<Half>(source.qkv_key_transform),
                at<std::int8_t>(source.qkv_local_int8),
                at<float>(source.qkv_local_scales),
                at<std::int8_t>(source.qkv_global_int8),
                at<float>(source.qkv_global_scales),
                at<Half>(source.qkv_global_transform),
                at<Half>(source.qkv_route_weight),
                at<std::uint8_t>(source.qkv_route_offsets),
                at<std::int8_t>(source.qkv_value_int8),
                at<float>(source.qkv_value_scales),
                at<std::uint16_t>(source.qkv_ticket_offsets),
                at<std::uint8_t>(source.qkv_ticket_columns),
                at<std::uint8_t>(source.qkv_value_block_mask),
                at<float>(source.qkv_squeeze_offsets),
                at<float>(source.qkv_squeeze_thresholds),
                at<std::uint32_t>(source.qkv_squeeze_polarities),
                at<std::int8_t>(source.qkv_squeeze_int8),
                at<float>(source.qkv_squeeze_scales),
                at<std::int8_t>(source.qkv_bypass_int8),
                at<float>(source.qkv_bypass_scales),
                at<std::uint32_t>(source.qkv_bypass_start),
                at<std::int8_t>(source.attention_output_int8),
                at<float>(source.attention_output_scales),
                at<std::uint8_t>(source.attention_output_block_mask),
                at<std::uint16_t>(source.attention_output_ticket_offsets),
                at<std::uint8_t>(source.attention_output_ticket_columns),
                at<Half>(source.feedforward_norm),
                at<float>(source.encoder_offsets),
                at<std::uint8_t>(source.boolean_program),
                at<float>(source.decoder_weight),
                at<std::int8_t>(source.decoder_int8),
                at<float>(source.decoder_scales),
                at<float>(source.decoder_bias),
                at<std::int8_t>(source.correction_int8),
                at<float>(source.correction_scales),
                at<std::uint16_t>(source.correction_ticket_offsets),
                at<std::uint8_t>(source.correction_ticket_columns),
                at<Half>(source.correction_output_f16),
            };
        }
    }
};
#else
struct LayerWeights {
    const float* residual_scales;
    const Half* attention_norm;
    const float* attention_slopes;
    const Half* qkv;
    const Half* attention_output;
    const Half* feedforward_norm;
    const Half* ff_up;
    const Half* ff_down;
    const std::int8_t* qkv_int8;
    const float* qkv_scale;
    const std::int8_t* attention_output_int8;
    const float* attention_output_scale;
    const std::int8_t* ff_up_int8;
    const float* ff_up_scale;
    const std::int8_t* ff_down_int8;
    const float* ff_down_scale;
    int ff_dimension;
};

struct Weights {
    const float* bigram_scale = at<float>(layout::kBigramScale);
    const float* route_strength = at<float>(layout::kDynamicRouteStrength);
    const float* route_scale = at<float>(layout::kDynamicRouteScale);
    const float* route_bias = at<float>(layout::kDynamicRouteBias);
    const Half* token_embedding = at<Half>(layout::kTokenEmbeddingWeight);
    const std::int8_t* token_embedding_int8 =
        at<std::int8_t>(layout::kInt8TokenEmbeddingWeight);
    const float* token_embedding_scale =
        at<float>(layout::kInt8TokenEmbeddingScale);
    const Half* bigram_embedding = at<Half>(layout::kBigramEmbeddingWeight);
    const Half* input_norm = at<Half>(layout::kInputNormWeight);
    const Half* final_norm = at<Half>(layout::kFinalNormWeight);
    std::array<LayerWeights, layout::kLayers> layers{};

    Weights() {
        const std::size_t embedded_bytes = static_cast<std::size_t>(
            ternlog_vocab_story_weights_end - ternlog_vocab_story_weights
        );
        if (embedded_bytes != layout::kBinaryBytes) {
            std::fprintf(
                stderr,
                "embedded model size mismatch: %zu != %zu\n",
                embedded_bytes,
                layout::kBinaryBytes
            );
            std::exit(2);
        }
        for (int index = 0; index < layout::kLayers; ++index) {
            const auto& source = layout::kLayerOffsets[index];
            layers[index] = {
                at<float>(source.residual_scales),
                at<Half>(source.attention_norm),
                at<float>(source.attention_slopes),
                at<Half>(source.qkv),
                at<Half>(source.attention_output),
                at<Half>(source.feedforward_norm),
                at<Half>(source.ff_up),
                at<Half>(source.ff_down),
                at<std::int8_t>(source.qkv_int8),
                at<float>(source.qkv_scale),
                at<std::int8_t>(source.attention_output_int8),
                at<float>(source.attention_output_scale),
                at<std::int8_t>(source.ff_up_int8),
                at<float>(source.ff_up_scale),
                at<std::int8_t>(source.ff_down_int8),
                at<float>(source.ff_down_scale),
                source.ff_dimension,
            };
        }
    }
};
#endif

inline float horizontal_sum(float32x4_t value) { return vaddvq_f32(value); }
inline float inverse_square_root(float value);

float quantize_activation_int8(
    const float* input, std::int8_t* output, int columns
) {
    float32x4_t maximum = vdupq_n_f32(0.0F);
    for (int column = 0; column < columns; column += 4) {
        maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(input + column)));
    }
    const float scale = std::max(vmaxvq_f32(maximum) / 127.0F, 1.0e-12F);
    const float inverse = 1.0F / scale;
    for (int column = 0; column < columns; column += 16) {
        const int32x4_t q0 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column), inverse)
        );
        const int32x4_t q1 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column + 4), inverse)
        );
        const int32x4_t q2 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column + 8), inverse)
        );
        const int32x4_t q3 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column + 12), inverse)
        );
        const int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(
            output + column,
            vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1))
        );
    }
    return scale;
}

void quantize_fixed_int8(
    const float* input, std::int8_t* output, int columns, float scale
) {
    const float inverse = 1.0F / scale;
    for (int column = 0; column < columns; column += 16) {
        const int32x4_t q0 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column), inverse)
        );
        const int32x4_t q1 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column + 4), inverse)
        );
        const int32x4_t q2 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column + 8), inverse)
        );
        const int32x4_t q3 = vcvtnq_s32_f32(
            vmulq_n_f32(vld1q_f32(input + column + 12), inverse)
        );
        const int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(
            output + column,
            vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1))
        );
    }
}

void quantize_fixed_int8_fp16(
    const float* input, std::int8_t* output, int columns, float scale
) {
    alignas(32) float rounded[kQkHeadDim];
    for (int column = 0; column < columns; column += 8) {
        const float16x8_t value = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(input + column)),
            vcvt_f16_f32(vld1q_f32(input + column + 4))
        );
        vst1q_f32(rounded + column, vcvt_f32_f16(vget_low_f16(value)));
        vst1q_f32(rounded + column + 4, vcvt_f32_f16(vget_high_f16(value)));
    }
    quantize_fixed_int8(rounded, output, columns, scale);
}

float rms_norm_quantize_activation_int8(
    const float* input,
    const Half* weights,
    std::int8_t* output,
    int columns,
    float epsilon
) {
    float32x4_t square_sum = vdupq_n_f32(0.0F);
    float32x4_t maximum = vdupq_n_f32(0.0F);
    for (int column = 0; column < columns; column += 8) {
        const float32x4_t low = vld1q_f32(input + column);
        const float32x4_t high = vld1q_f32(input + column + 4);
        const float16x8_t weight = vld1q_f16(weights + column);
        square_sum = vfmaq_f32(square_sum, low, low);
        square_sum = vfmaq_f32(square_sum, high, high);
        maximum = vmaxq_f32(
            maximum,
            vabsq_f32(vmulq_f32(low, vcvt_f32_f16(vget_low_f16(weight))))
        );
        maximum = vmaxq_f32(
            maximum,
            vabsq_f32(vmulq_f32(high, vcvt_f32_f16(vget_high_f16(weight))))
        );
    }
    const float inverse_rms = inverse_square_root(
        horizontal_sum(square_sum) / columns + epsilon
    );
    const float scale = std::max(
        vmaxvq_f32(maximum) * inverse_rms / 127.0F, 1.0e-12F
    );
    const float multiplier = inverse_rms / scale;
    for (int column = 0; column < columns; column += 16) {
        const float16x8_t weight0 = vld1q_f16(weights + column);
        const float16x8_t weight1 = vld1q_f16(weights + column + 8);
        const int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(
            vmulq_f32(
                vld1q_f32(input + column),
                vcvt_f32_f16(vget_low_f16(weight0))
            ), multiplier
        ));
        const int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(
            vmulq_f32(
                vld1q_f32(input + column + 4),
                vcvt_f32_f16(vget_high_f16(weight0))
            ), multiplier
        ));
        const int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(
            vmulq_f32(
                vld1q_f32(input + column + 8),
                vcvt_f32_f16(vget_low_f16(weight1))
            ), multiplier
        ));
        const int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(
            vmulq_f32(
                vld1q_f32(input + column + 12),
                vcvt_f32_f16(vget_high_f16(weight1))
            ), multiplier
        ));
        const int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(
            output + column,
            vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1))
        );
    }
    return scale;
}

void matrix_vector_int8_qkv_prequantized(
    const std::int8_t* weights,
    const float* weight_scales,
    const std::int8_t* activation,
    float activation_scale,
    float* output
) {
    static_assert(layout::kQkvRows % 8 == 0);
    static_assert(kDim % 16 == 0);
    for (int row = 0; row < layout::kQkvRows; row += 8) {
        int32x4_t sum[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int column = 0; column < kDim; column += 16) {
            const int8x16_t value = vld1q_s8(activation + column);
            for (int item = 0; item < 8; ++item) {
                sum[item] = vdotq_s32(
                    sum[item],
                    vld1q_s8(
                        weights
                            + static_cast<std::size_t>(row + item) * kDim
                            + column
                    ),
                    value
                );
            }
        }
        for (int item = 0; item < 8; ++item) {
            output[row + item] = static_cast<float>(vaddvq_s32(sum[item]))
                * activation_scale * weight_scales[row + item];
        }
    }
}

#if defined(TERNLOG_BOOLEAN_QKV_SHARED_BASIS) \
    || defined(TERNLOG_BOOLEAN_QKV_LOCAL_GLOBAL) \
    || defined(TERNLOG_BOOLEAN_QKV_CYCLIC_ROUTES) \
    || defined(TERNLOG_BOOLEAN_QKV_TIED) \
    || defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE) \
    || defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE_GLOBAL)
template <int Rows>
void matrix_vector_int8_dense_prequantized(
    const std::int8_t* weights,
    const float* weight_scales,
    const std::int8_t* activation,
    float activation_scale,
    float* output
) {
    static_assert(Rows % 8 == 0);
    for (int row = 0; row < Rows; row += 8) {
        int32x4_t sums[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int column = 0; column < kDim; column += 16) {
            const int8x16_t value = vld1q_s8(activation + column);
#pragma clang loop unroll(full)
            for (int item = 0; item < 8; ++item) {
                sums[item] = vdotq_s32(
                    sums[item],
                    vld1q_s8(
                        weights + static_cast<std::size_t>(row + item) * kDim
                        + column
                    ),
                    value
                );
            }
        }
#pragma clang loop unroll(full)
        for (int item = 0; item < 8; ++item) {
            output[row + item] = static_cast<float>(vaddvq_s32(sums[item]))
                * activation_scale * weight_scales[row + item];
        }
    }
}

#if defined(TERNLOG_BOOLEAN_QKV_SHARED_BASIS)
void matrix_vector_int8_qkv_shared_basis_prequantized(
    const LayerWeights& layer,
    const std::int8_t* activation,
    float activation_scale,
    float* output
) {
    static_assert(layout::kQkvSharedBases == 1 || layout::kQkvSharedBases == 2);
    static_assert(layout::kQkvBaseRows % 8 == 0);
    static_assert(kQkHeadDim == 16);
    alignas(64) float bases[layout::kQkvBaseRows];
    alignas(64) Half rounded[2 * layout::kQkvSharedBases * kQkHeadDim];
    matrix_vector_int8_dense_prequantized<layout::kQkvBaseRows>(
        layer.qkv_int8, layer.qkv_scales, activation, activation_scale, bases
    );
    constexpr int qk_base_values = 2 * layout::kQkvSharedBases * kQkHeadDim;
    for (int index = 0; index < qk_base_values; index += 8) {
        const float16x4_t low = vcvt_f16_f32(vld1q_f32(bases + index));
        const float16x4_t high = vcvt_f16_f32(vld1q_f32(bases + index + 4));
        vst1q_f16(rounded + index, vcombine_f16(low, high));
    }
    for (int head = 0; head < kHeads; ++head) {
        const int base = layout::kQkvSharedBases == 1
            ? 0
            : (head < layout::kLocalHeads ? 0 : 1);
        ternlog_matvec_f16input_fmlal_m4(
            layer.qkv_query_transform
                + static_cast<std::size_t>(head) * kQkHeadDim * kQkHeadDim,
            nullptr,
            rounded + base * kQkHeadDim,
            output + head * kQkHeadDim,
            kQkHeadDim,
            kQkHeadDim
        );
        ternlog_matvec_f16input_fmlal_m4(
            layer.qkv_key_transform
                + static_cast<std::size_t>(head) * kQkHeadDim * kQkHeadDim,
            nullptr,
            rounded + layout::kQkvSharedBases * kQkHeadDim
                + base * kQkHeadDim,
            output + kHeads * kQkHeadDim + head * kQkHeadDim,
            kQkHeadDim,
            kQkHeadDim
        );
    }
    constexpr int value_offset = (kHeads + layout::kKeyHeads) * kQkHeadDim;
    constexpr int base_value_offset =
        2 * layout::kQkvSharedBases * kQkHeadDim;
    std::memcpy(
        output + value_offset,
        bases + base_value_offset,
        layout::kValueHeads * kHeadDim * sizeof(float)
    );
}
#endif

#if defined(TERNLOG_BOOLEAN_QKV_TIED) \
    || defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE) \
    || defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE_GLOBAL)
void matrix_vector_int8_qkv_head_subspace_prequantized(
    const LayerWeights& layer,
    const std::int8_t* activation,
    float activation_scale,
    float* output
) {
    static_assert(kQkHeadDim == 16);
    constexpr int qk_rows = kHeads * kQkHeadDim;
#if defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE_GLOBAL)
    static_assert(layout::kQkvGlobalRank == 8);
    constexpr int global_rank = layout::kQkvGlobalRank;
#else
    constexpr int global_rank = 0;
#endif
    constexpr int base_rows =
        qk_rows + global_rank + layout::kValueHeads * kHeadDim;
    alignas(64) float bases[base_rows];
    matrix_vector_int8_dense_prequantized<base_rows>(
        layer.qkv_int8, layer.qkv_scales,
        activation, activation_scale, bases
    );
#if defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE) \
    || defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE_GLOBAL)
    constexpr int transform_columns = kQkHeadDim + global_rank;
    alignas(64) Half rounded[qk_rows + global_rank];
    alignas(64) Half combined[transform_columns];
    for (int index = 0; index < qk_rows + global_rank; index += 8) {
        vst1q_f16(
            rounded + index,
            vcombine_f16(
                vcvt_f16_f32(vld1q_f32(bases + index)),
                vcvt_f16_f32(vld1q_f32(bases + index + 4))
            )
        );
    }
    for (int head = 0; head < kHeads; ++head) {
        const Half* transform_input = rounded + head * kQkHeadDim;
#if defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE_GLOBAL)
        std::memcpy(
            combined,
            rounded + head * kQkHeadDim,
            kQkHeadDim * sizeof(Half)
        );
        std::memcpy(
            combined + kQkHeadDim,
            rounded + qk_rows,
            global_rank * sizeof(Half)
        );
        transform_input = combined;
#endif
        ternlog_matvec_f16input_fmlal_m4(
            layer.qkv_query_transform
                + static_cast<std::size_t>(head)
                    * kQkHeadDim * transform_columns,
            nullptr,
            transform_input,
            output + head * kQkHeadDim,
            kQkHeadDim,
            transform_columns
        );
        ternlog_matvec_f16input_fmlal_m4(
            layer.qkv_key_transform
                + static_cast<std::size_t>(head)
                    * kQkHeadDim * transform_columns,
            nullptr,
            transform_input,
            output + qk_rows + head * kQkHeadDim,
            kQkHeadDim,
            transform_columns
        );
    }
#else
    std::memcpy(output, bases, qk_rows * sizeof(float));
    std::memcpy(output + qk_rows, bases, qk_rows * sizeof(float));
#endif
    std::memcpy(
        output + 2 * qk_rows,
        bases + qk_rows + global_rank,
        layout::kValueHeads * kHeadDim * sizeof(float)
    );
}
#endif

#if defined(TERNLOG_BOOLEAN_QKV_LOCAL_GLOBAL)
void matrix_vector_int8_qkv_local_global_prequantized(
    const LayerWeights& layer,
    const std::int8_t* activation,
    float activation_scale,
    float* output
) {
    static_assert(layout::kQkvGlobalRank == 16);
    static_assert(kQkHeadDim == 16);
    static_assert(layout::kQkvLocalWidth % kHeadDim == 0);
    constexpr int global_value_rows =
        layout::kQkvGlobalRank + layout::kValueHeads * kHeadDim;
    alignas(64) float global_value[global_value_rows];
    alignas(64) Half rounded_global[layout::kQkvGlobalRank];
    matrix_vector_int8_dense_prequantized<global_value_rows>(
        layer.qkv_global_int8,
        layer.qkv_global_scales,
        activation,
        activation_scale,
        global_value
    );
    for (int index = 0; index < layout::kQkvGlobalRank; index += 8) {
        const float16x4_t low = vcvt_f16_f32(vld1q_f32(global_value + index));
        const float16x4_t high = vcvt_f16_f32(
            vld1q_f32(global_value + index + 4)
        );
        vst1q_f16(rounded_global + index, vcombine_f16(low, high));
    }
    for (int family = 0; family < 2; ++family) {
        for (int head = 0; head < kHeads; ++head) {
            const int qk_row = (family * kHeads + head) * kQkHeadDim;
            ternlog_matvec_f16input_fmlal_m4(
                layer.qkv_global_transform
                    + static_cast<std::size_t>(qk_row) * layout::kQkvGlobalRank,
                nullptr,
                rounded_global,
                output + qk_row,
                kQkHeadDim,
                layout::kQkvGlobalRank
            );
            for (int row = 0; row < kQkHeadDim; ++row) {
                int32x4_t sum = vdupq_n_s32(0);
                const std::int8_t* weights = layer.qkv_local_int8
                    + static_cast<std::size_t>(qk_row + row)
                        * layout::kQkvLocalWidth;
                for (
                    int column = 0;
                    column < layout::kQkvLocalWidth;
                    column += 16
                ) {
                    const int local_group = column / kHeadDim;
                    const int within_group = column % kHeadDim;
                    const int source_group = (head + local_group) % kHeads;
                    sum = vdotq_s32(
                        sum,
                        vld1q_s8(weights + column),
                        vld1q_s8(
                            activation + source_group * kHeadDim + within_group
                        )
                    );
                }
                output[qk_row + row] += static_cast<float>(vaddvq_s32(sum))
                    * activation_scale * layer.qkv_local_scales[qk_row + row];
            }
        }
    }
    constexpr int value_offset = 2 * kHeads * kQkHeadDim;
    std::memcpy(
        output + value_offset,
        global_value + layout::kQkvGlobalRank,
        layout::kValueHeads * kHeadDim * sizeof(float)
    );
}
#endif

#if defined(TERNLOG_BOOLEAN_QKV_CYCLIC_ROUTES)
void matrix_vector_qkv_cyclic_routes(
    const LayerWeights& layer,
    const float* normalized,
    const std::int8_t* activation,
    float activation_scale,
    float* output
) {
    static_assert(layout::kQkvRouteCount == 16);
    static_assert(2 * kHeads * kQkHeadDim == kDim);
    alignas(64) Half duplicated[2 * kDim];
    for (int index = 0; index < kDim; index += 8) {
        const float16x8_t value = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(normalized + index)),
            vcvt_f16_f32(vld1q_f32(normalized + index + 4))
        );
        vst1q_f16(duplicated + index, value);
        vst1q_f16(duplicated + kDim + index, value);
    }
    for (int row = 0; row < kDim; row += 8) {
        float32x4_t low = vdupq_n_f32(0.0F);
        float32x4_t high = vdupq_n_f32(0.0F);
#pragma clang loop unroll(full)
        for (int route = 0; route < layout::kQkvRouteCount; ++route) {
            const float16x8_t value = vld1q_f16(
                duplicated + layer.qkv_route_offsets[route] + row
            );
            const float16x8_t weight = vld1q_f16(
                layer.qkv_route_weight
                    + static_cast<std::size_t>(route) * kDim + row
            );
            low = vfmlalq_low_f16(low, value, weight);
            high = vfmlalq_high_f16(high, value, weight);
        }
        vst1q_f32(output + row, low);
        vst1q_f32(output + row + 4, high);
    }
    matrix_vector_int8_dense_prequantized<layout::kValueHeads * kHeadDim>(
        layer.qkv_value_int8,
        layer.qkv_value_scales,
        activation,
        activation_scale,
        output + kDim
    );
}
#endif
#endif

#if defined(TERNLOG_QKV_BATCH4_HOOK)
#if !defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR)
#error "batch-4 QKV hook currently requires scalar-fused value-cache export"
#endif
struct QkvBatchRequest {
    const std::int8_t* packed_weights;
    const float* weight_scales;
    const std::uint16_t* ticket_offsets;
    const std::uint8_t* ticket_columns;
    const std::int8_t* activation;
    float activation_scale;
    float* output;
    std::vector<ValueCacheScalar>* value_cache;
    int layer_index;
    int value_slot;
    float value_scale;
};

using QkvBatchSubmit = bool (*)(const QkvBatchRequest&);
thread_local QkvBatchSubmit ternlog_qkv_batch_submit = nullptr;
#endif

#if defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR)
__attribute__((noinline)) void store_qkv_value_cache_scalar_group(
    const std::int32_t* raw_sums,
    int count,
    float activation_scale,
    const float* weight_scales,
    float inverse_value_scale,
    std::vector<ValueCacheScalar>& value_cache,
    int layer_index,
    int value_slot,
    int first_channel
) {
    for (int item = 0; item < count; ++item) {
        volatile float scaled = static_cast<float>(raw_sums[item])
            * activation_scale;
        volatile float weighted = scaled * weight_scales[item];
        volatile float quantized = weighted * inverse_value_scale;
        const int rounded = std::clamp(
            static_cast<int>(std::nearbyint(quantized)), -127, 127
        );
        value_cache[shared_value_cache_int8_offset(
            layer_index, 0, value_slot, first_channel + item
        )] = static_cast<std::int8_t>(rounded);
    }
}
#endif

void matrix_vector_int8_qkv_tickets_prequantized(
    const std::int8_t* packed_weights,
    const float* weight_scales,
    const std::uint16_t* ticket_offsets,
    const std::uint8_t* ticket_columns,
    const std::int8_t* activation,
    float activation_scale,
    float* output
#if defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR) \
    || defined(TERNLOG_QKV_VALUE_CACHE_FUSED_NEON)
    , std::vector<ValueCacheScalar>& value_cache,
    int layer_index,
    int value_slot,
    float value_scale
#endif
) {
#if defined(TERNLOG_QKV_BATCH4_HOOK)
    if (ternlog_qkv_batch_submit != nullptr) {
        const QkvBatchRequest request{
            packed_weights, weight_scales, ticket_offsets, ticket_columns,
            activation, activation_scale, output, &value_cache,
            layer_index, value_slot, value_scale,
        };
        if (ternlog_qkv_batch_submit(request)) return;
    }
#endif
#if defined(TERNLOG_QKV_PACKED_FIXED8) || defined(TERNLOG_QKV_PACKED_FIXED16)
    static_assert(layout::kQkvRows == 192);
    static_assert(kDim == 160);
    alignas(64) std::int32_t raw_sums[layout::kQkvRows];
#if defined(TERNLOG_QKV_PACKED_FIXED16)
    ternlog_vocab_qkv_packed_sdot_192x160_m4_16row(
        packed_weights, activation, raw_sums
    );
#else
    ternlog_vocab_qkv_packed_sdot_192x160_m4_8row(
        packed_weights, activation, raw_sums
    );
#endif
    for (int row = 0; row < layout::kQkvRows; ++row) {
        output[row] = static_cast<float>(raw_sums[row])
            * activation_scale * weight_scales[row];
    }
#else
    constexpr int row_block = 8;
    constexpr int row_blocks = layout::kQkvRows / row_block;
#if defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR) \
    || defined(TERNLOG_QKV_VALUE_CACHE_FUSED_NEON)
    constexpr int value_offset =
        (layout::kHeads + layout::kKeyHeads) * kQkHeadDim;
    static_assert(layout::kQkvRows - value_offset == kHeadDim);
    const float inverse_value_scale = 1.0F / value_scale;
#endif
    for (int block = 0; block < row_blocks; ++block) {
#if defined(TERNLOG_SHARED_QK45)
        static_assert(layout::kHeads * kQkHeadDim == 80);
        static_assert(layout::kKeyHeads * kQkHeadDim == 80);
        if (layer_index == 5 && block < 20) {
            continue;
        }
#endif
        int32x4_t sums[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int ticket = ticket_offsets[block];
             ticket < ticket_offsets[block + 1]; ++ticket) {
#if defined(TERNLOG_QKV_PREFETCH_TICKETS)
            constexpr int prefetch_distance = TERNLOG_QKV_PREFETCH_TICKETS;
            static_assert(prefetch_distance > 0);
            if (ticket + prefetch_distance < ticket_offsets[block + 1]) {
                __builtin_prefetch(
                    packed_weights
                        + static_cast<std::size_t>(ticket + prefetch_distance)
                            * row_block * 16,
                    0,
                    0
                );
            }
#endif
            const int8x16_t value = vld1q_s8(
                activation + 16 * ticket_columns[ticket]
            );
            const std::int8_t* weights = packed_weights
                + static_cast<std::size_t>(ticket) * row_block * 16;
            for (int row = 0; row < row_block; ++row) {
                sums[row] = vdotq_s32(
                    sums[row], vld1q_s8(weights + row * 16), value
                );
            }
        }
        const int first_output_row = block * row_block;
#if defined(TERNLOG_QKV_VALUE_CACHE_FUSED_NEON)
        if (first_output_row >= value_offset) {
            alignas(16) std::int32_t raw_sums[row_block];
            alignas(8) std::int8_t quantized[row_block];
#pragma clang loop unroll(full)
            for (int row = 0; row < row_block; ++row) {
                raw_sums[row] = vaddvq_s32(sums[row]);
            }
            float32x4_t low = vcvtq_f32_s32(vld1q_s32(raw_sums));
            float32x4_t high = vcvtq_f32_s32(vld1q_s32(raw_sums + 4));
            low = vmulq_n_f32(low, activation_scale);
            high = vmulq_n_f32(high, activation_scale);
            low = vmulq_f32(low, vld1q_f32(weight_scales + first_output_row));
            high = vmulq_f32(
                high, vld1q_f32(weight_scales + first_output_row + 4)
            );
            low = vmulq_n_f32(low, inverse_value_scale);
            high = vmulq_n_f32(high, inverse_value_scale);
            const float32x4_t minimum = vdupq_n_f32(-127.0F);
            const float32x4_t maximum = vdupq_n_f32(127.0F);
            low = vminq_f32(vmaxq_f32(low, minimum), maximum);
            high = vminq_f32(vmaxq_f32(high, minimum), maximum);
            const int16x8_t narrowed = vcombine_s16(
                vqmovn_s32(vcvtnq_s32_f32(low)),
                vqmovn_s32(vcvtnq_s32_f32(high))
            );
            vst1_s8(quantized, vqmovn_s16(narrowed));
#pragma clang loop unroll(full)
            for (int row = 0; row < row_block; ++row) {
                const int channel = first_output_row + row - value_offset;
                value_cache[shared_value_cache_int8_offset(
                    layer_index, 0, value_slot, channel
                )] = quantized[row];
            }
            continue;
        }
#endif
#if defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR)
        if (first_output_row >= value_offset) {
            alignas(16) std::int32_t raw_sums[row_block];
            for (int row = 0; row < row_block; ++row) {
                raw_sums[row] = vaddvq_s32(sums[row]);
            }
            store_qkv_value_cache_scalar_group(
                raw_sums, row_block, activation_scale,
                weight_scales + first_output_row, inverse_value_scale,
                value_cache, layer_index, value_slot,
                first_output_row - value_offset
            );
            continue;
        }
#endif
        for (int row = 0; row < row_block; ++row) {
            const int output_row = first_output_row + row;
            volatile float scaled = static_cast<float>(
                vaddvq_s32(sums[row])
            ) * activation_scale;
            output[output_row] = scaled * weight_scales[output_row];
        }
    }
#endif
}

template <int Rows, int RowBlock>
void matrix_vector_int8_tickets_prequantized(
    const std::int8_t* packed_weights,
    const float* weight_scales,
    const std::uint16_t* ticket_offsets,
    const std::uint8_t* ticket_columns,
    const std::int8_t* activation,
    float activation_scale,
    float* output
) {
    static_assert(Rows % RowBlock == 0);
    constexpr int row_blocks = Rows / RowBlock;
    for (int block = 0; block < row_blocks; ++block) {
        int32x4_t sums[RowBlock];
#pragma clang loop unroll(full)
        for (int row = 0; row < RowBlock; ++row) {
            sums[row] = vdupq_n_s32(0);
        }
        for (int ticket = ticket_offsets[block];
             ticket < ticket_offsets[block + 1]; ++ticket) {
            const int8x16_t value = vld1q_s8(
                activation + 16 * ticket_columns[ticket]
            );
            const std::int8_t* weights = packed_weights
                + static_cast<std::size_t>(ticket) * RowBlock * 16;
#pragma clang loop unroll(full)
            for (int row = 0; row < RowBlock; ++row) {
                sums[row] = vdotq_s32(
                    sums[row], vld1q_s8(weights + row * 16), value
                );
            }
        }
#pragma clang loop unroll(full)
        for (int row = 0; row < RowBlock; ++row) {
            const int output_row = block * RowBlock + row;
            output[output_row] = static_cast<float>(vaddvq_s32(sums[row]))
                * activation_scale * weight_scales[output_row];
        }
    }
}

template <int RowBlock>
void matrix_vector_int8_output_tickets(
    const std::int8_t* packed_weights,
    const float* weight_scales,
    const std::uint16_t* ticket_offsets,
    const std::uint8_t* ticket_columns,
    const float* input,
    float* output
) {
    static_assert(kDim % 16 == 0);
    alignas(64) std::int8_t activation[kDim];
    alignas(64) float rounded_input[kDim];
    if constexpr (layout::kAttentionOutputInputFP16) {
        for (int index = 0; index < kDim; index += 8) {
            const float16x4_t low = vcvt_f16_f32(vld1q_f32(input + index));
            const float16x4_t high = vcvt_f16_f32(vld1q_f32(input + index + 4));
            vst1q_f32(rounded_input + index, vcvt_f32_f16(low));
            vst1q_f32(rounded_input + index + 4, vcvt_f32_f16(high));
        }
        input = rounded_input;
    }
    const float activation_scale = quantize_activation_int8(
        input, activation, kDim
    );
    matrix_vector_int8_tickets_prequantized<kDim, RowBlock>(
        packed_weights, weight_scales, ticket_offsets, ticket_columns,
        activation, activation_scale, output
    );
}

#if defined(TERNLOG_OUTPUT_CODEBOOK32)
constexpr int kOutputCodewords = 32;
constexpr int kOutputCodebookBlocks = kDim / 16;
constexpr int kOutputCodebookResponseShift = 10;

inline int32x4_t output_codebook_dot4(
    const std::int8_t* packed,
    int8x16_t activation
) {
    int32x4_t sums = vdupq_n_s32(0);
    sums = vdotq_laneq_s32(sums, vld1q_s8(packed), activation, 0);
    sums = vdotq_laneq_s32(sums, vld1q_s8(packed + 16), activation, 1);
    sums = vdotq_laneq_s32(sums, vld1q_s8(packed + 32), activation, 2);
    return vdotq_laneq_s32(
        sums, vld1q_s8(packed + 48), activation, 3
    );
}

inline int8x16_t output_codebook_table16(
    const std::int8_t* packed,
    int8x16_t activation
) {
    int32x4_t sums[4];
#pragma clang loop unroll(full)
    for (int group = 0; group < 4; ++group) {
        sums[group] = output_codebook_dot4(packed + group * 64, activation);
    }
    const int16x8_t low = vcombine_s16(
        vqrshrn_n_s32(sums[0], kOutputCodebookResponseShift),
        vqrshrn_n_s32(sums[1], kOutputCodebookResponseShift)
    );
    const int16x8_t high = vcombine_s16(
        vqrshrn_n_s32(sums[2], kOutputCodebookResponseShift),
        vqrshrn_n_s32(sums[3], kOutputCodebookResponseShift)
    );
    return vcombine_s8(vqmovn_s16(low), vqmovn_s16(high));
}

void matrix_vector_int8_output_codebook32(
    const std::int8_t* packed_codebooks,
    const float* row_scales,
    const std::uint8_t* indices,
    const float* input,
    float* output
) {
    alignas(64) std::int8_t activation[kDim];
    alignas(64) float rounded_input[kDim];
    if constexpr (layout::kAttentionOutputInputFP16) {
        for (int index = 0; index < kDim; index += 8) {
            const float16x4_t low = vcvt_f16_f32(vld1q_f32(input + index));
            const float16x4_t high = vcvt_f16_f32(vld1q_f32(input + index + 4));
            vst1q_f32(rounded_input + index, vcvt_f32_f16(low));
            vst1q_f32(rounded_input + index + 4, vcvt_f32_f16(high));
        }
        input = rounded_input;
    }
    const float activation_scale = quantize_activation_int8(
        input, activation, kDim
    );
    std::array<int16x8_t, 2 * kOutputCodebookBlocks> sums{};
    for (int block_column = 0;
         block_column < kOutputCodebookBlocks;
         ++block_column) {
        const int8x16_t value = vld1q_s8(activation + block_column * 16);
        const std::int8_t* codebook = packed_codebooks
            + static_cast<std::size_t>(block_column) * kOutputCodewords * 16;
        const int8x16x2_t tables = {{
            output_codebook_table16(codebook, value),
            output_codebook_table16(codebook + 4 * 64, value),
        }};
        for (int block_row = 0;
             block_row < kOutputCodebookBlocks;
             ++block_row) {
            const uint8x16_t selectors = vld1q_u8(
                indices
                    + static_cast<std::size_t>(block_column) * kDim
                    + block_row * 16
            );
            const int8x16_t selected = vqtbl2q_s8(tables, selectors);
            sums[2 * block_row] = vaddw_s8(
                sums[2 * block_row], vget_low_s8(selected)
            );
            sums[2 * block_row + 1] = vaddw_high_s8(
                sums[2 * block_row + 1], selected
            );
        }
    }
    const float response_scale = static_cast<float>(
        1 << kOutputCodebookResponseShift
    ) * activation_scale;
    for (int block_row = 0;
         block_row < kOutputCodebookBlocks;
         ++block_row) {
        for (int half = 0; half < 2; ++half) {
            const int row = block_row * 16 + half * 8;
            const int16x8_t packed = sums[2 * block_row + half];
            const float32x4_t low = vcvtq_f32_s32(
                vmovl_s16(vget_low_s16(packed))
            );
            const float32x4_t high = vcvtq_f32_s32(vmovl_high_s16(packed));
            vst1q_f32(
                output + row,
                vmulq_f32(
                    vmulq_n_f32(low, response_scale),
                    vld1q_f32(row_scales + row)
                )
            );
            vst1q_f32(
                output + row + 4,
                vmulq_f32(
                    vmulq_n_f32(high, response_scale),
                    vld1q_f32(row_scales + row + 4)
                )
            );
        }
    }
}
#endif

#if defined(TERNLOG_OUTPUT_HEADLOCAL_RANK8)
void matrix_vector_int8_output_headlocal_rank8(
    const std::int8_t* packed_weights,
    const float* weight_scales,
    const std::uint8_t* rank_output_bytes,
    const float* input,
    float* output
) {
    constexpr int kLocalWeightCount = kHeads * kHeadDim * kHeadDim;
    constexpr int kRank = 8;
    alignas(64) std::int8_t activation[kDim];
    alignas(64) float rounded_input[kDim];
    for (int index = 0; index < kDim; index += 8) {
        const float16x4_t low = vcvt_f16_f32(vld1q_f32(input + index));
        const float16x4_t high = vcvt_f16_f32(vld1q_f32(input + index + 4));
        vst1q_f32(rounded_input + index, vcvt_f32_f16(low));
        vst1q_f32(rounded_input + index + 4, vcvt_f32_f16(high));
    }
    const float activation_scale = quantize_activation_int8(
        rounded_input, activation, kDim
    );
    for (int head = 0; head < kHeads; ++head) {
        for (int row_block = 0; row_block < kHeadDim; row_block += 8) {
            int32x4_t sums[8] = {
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0),
            };
            for (int column = 0; column < kHeadDim; column += 16) {
                const int8x16_t value = vld1q_s8(
                    activation + head * kHeadDim + column
                );
                for (int item = 0; item < 8; ++item) {
                    const std::size_t offset =
                        static_cast<std::size_t>(head) * kHeadDim * kHeadDim
                        + static_cast<std::size_t>(row_block + item) * kHeadDim
                        + column;
                    sums[item] = vdotq_s32(
                        sums[item], vld1q_s8(packed_weights + offset), value
                    );
                }
            }
            for (int item = 0; item < 8; ++item) {
                const int row = head * kHeadDim + row_block + item;
                output[row] = static_cast<float>(vaddvq_s32(sums[item]))
                    * activation_scale * weight_scales[row];
            }
        }
    }

    const std::int8_t* rank_input = packed_weights + kLocalWeightCount;
    const float* rank_scales = weight_scales + kDim;
    alignas(64) float hidden[kRank];
    int32x4_t rank_sums[kRank] = {
        vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
    };
    for (int column = 0; column < kDim; column += 16) {
        const int8x16_t value = vld1q_s8(activation + column);
        for (int row = 0; row < kRank; ++row) {
            rank_sums[row] = vdotq_s32(
                rank_sums[row],
                vld1q_s8(rank_input + row * kDim + column), value
            );
        }
    }
    for (int row = 0; row < kRank; ++row) {
        hidden[row] = static_cast<float>(vaddvq_s32(rank_sums[row]))
            * activation_scale * rank_scales[row];
    }
    const float16x8_t hidden_half = vcombine_f16(
        vcvt_f16_f32(vld1q_f32(hidden)),
        vcvt_f16_f32(vld1q_f32(hidden + 4))
    );
    const Half* rank_output = reinterpret_cast<const Half*>(rank_output_bytes);
    for (int row = 0; row < kDim; ++row) {
        const float16x8_t weight = vld1q_f16(rank_output + row * kRank);
        float32x4_t sum = vdupq_n_f32(0.0F);
        sum = vfmlalq_low_f16(sum, weight, hidden_half);
        sum = vfmlalq_high_f16(sum, weight, hidden_half);
        output[row] += vaddvq_f32(sum);
    }
}
#endif

template <int RowBlock, bool AccumulateSquareSum>
float matrix_vector_int8_output_tickets_residual(
    const std::int8_t* packed_weights,
    const float* weight_scales,
    const std::uint16_t* ticket_offsets,
    const std::uint8_t* ticket_columns,
    const float* input,
    float* state,
    float state_scale,
    float branch_scale
) {
    static_assert(kDim % RowBlock == 0);
    static_assert(RowBlock % 4 == 0);
    alignas(64) std::int8_t activation[kDim];
    alignas(64) float rounded_input[kDim];
    if constexpr (layout::kAttentionOutputInputFP16) {
        for (int index = 0; index < kDim; index += 8) {
            const float16x4_t low = vcvt_f16_f32(vld1q_f32(input + index));
            const float16x4_t high = vcvt_f16_f32(vld1q_f32(input + index + 4));
            vst1q_f32(rounded_input + index, vcvt_f32_f16(low));
            vst1q_f32(rounded_input + index + 4, vcvt_f32_f16(high));
        }
        input = rounded_input;
    }
    const float activation_scale = quantize_activation_int8(
        input, activation, kDim
    );
    float32x4_t square_sum = vdupq_n_f32(0.0F);
    constexpr int row_blocks = kDim / RowBlock;
    for (int block = 0; block < row_blocks; ++block) {
        int32x4_t sums[RowBlock];
#pragma clang loop unroll(full)
        for (int row = 0; row < RowBlock; ++row) sums[row] = vdupq_n_s32(0);
        for (
            int ticket = ticket_offsets[block];
            ticket < ticket_offsets[block + 1];
            ++ticket
        ) {
            const int8x16_t value = vld1q_s8(
                activation + 16 * ticket_columns[ticket]
            );
            const std::int8_t* weights = packed_weights
                + static_cast<std::size_t>(ticket) * RowBlock * 16;
#pragma clang loop unroll(full)
            for (int row = 0; row < RowBlock; ++row) {
                sums[row] = vdotq_s32(
                    sums[row], vld1q_s8(weights + row * 16), value
                );
            }
        }
        for (int row = 0; row < RowBlock; row += 4) {
            const int output_row = block * RowBlock + row;
            alignas(16) float branch_values[4];
#pragma clang loop unroll(full)
            for (int item = 0; item < 4; ++item) {
                branch_values[item] = static_cast<float>(
                    vaddvq_s32(sums[row + item])
                ) * activation_scale * weight_scales[output_row + item];
            }
            const float32x4_t updated = vfmaq_n_f32(
                vmulq_n_f32(vld1q_f32(state + output_row), state_scale),
                vld1q_f32(branch_values),
                branch_scale
            );
            vst1q_f32(state + output_row, updated);
            if constexpr (AccumulateSquareSum) {
                square_sum = vfmaq_f32(square_sum, updated, updated);
            }
        }
    }
    if constexpr (AccumulateSquareSum) return horizontal_sum(square_sum);
    return 0.0F;
}

void matrix_vector_int8(
    const std::int8_t* weights,
    const float* weight_scales,
    const float* input,
    float* output,
    int rows,
    int columns
) {
    alignas(64) std::int8_t activation[1040];
    const float activation_scale = quantize_activation_int8(
        input, activation, columns
    );
#if defined(TERNLOG_BOOLEAN_QKV_ASM8) || defined(TERNLOG_BOOLEAN_QKV_ASM16)
    if (rows == 480 && columns == 160) {
        alignas(64) std::int32_t sums[480];
#if defined(TERNLOG_BOOLEAN_QKV_ASM8)
        ternlog_vocab_qkv_sdot_480x160_m4_8row(weights, activation, sums);
#else
        ternlog_vocab_qkv_sdot_480x160_m4_16row(weights, activation, sums);
#endif
        for (int row = 0; row < rows; ++row) {
            output[row] = static_cast<float>(sums[row])
                * activation_scale * weight_scales[row];
        }
        return;
    }
#endif
    for (int row = 0; row < rows; row += 8) {
        const std::int8_t* weight[8];
        for (int item = 0; item < 8; ++item) {
            weight[item] = weights
                + static_cast<std::size_t>(row + item) * columns;
        }
        int32x4_t sum[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int column = 0; column < columns; column += 16) {
            const int8x16_t value = vld1q_s8(activation + column);
            for (int item = 0; item < 8; ++item) {
                sum[item] = vdotq_s32(
                    sum[item], vld1q_s8(weight[item] + column), value
                );
            }
        }
        for (int item = 0; item < 8; ++item) {
            output[row + item] = vaddvq_s32(sum[item])
                * activation_scale * weight_scales[row + item];
        }
    }
}

#if defined(TERNLOG_BOOLEAN_QKV_SIGNED_SQUEEZE)
void matrix_vector_qkv_signed_squeeze(
    const LayerWeights& layer,
    const float* input,
    float* output
) {
    static_assert(layout::kQkvSqueezeFeatures == 40);
    static_assert(layout::kQkvSqueezePolarities == 1);
    static_assert(layout::kQkvSqueezeBypass == 64);
    alignas(64) float rounded[kDim];
    ternlog_vocab_qkv_round160_m4(
        rounded, input, layout::kQkvSqueezeRoundingScale
    );
    alignas(64) std::int8_t counts[48];
    ternlog_vocab_qkv_signed_counts40_m4(
        counts, rounded, layer.qkv_squeeze_offsets,
        layer.qkv_squeeze_thresholds, layer.qkv_squeeze_polarities
    );
    const int bypass_start = static_cast<int>(*layer.qkv_bypass_start);
    alignas(64) std::int8_t bypass_activation[layout::kQkvSqueezeBypass];
    const float bypass_activation_scale = quantize_activation_int8(
        rounded + bypass_start, bypass_activation, layout::kQkvSqueezeBypass
    );
    constexpr int kSqueezeStride = 48;
    constexpr int kBypassStride = 64;
    for (int row = 0; row < 3 * kDim; row += 8) {
        int32x4_t squeeze_sum[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        int32x4_t bypass_sum[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int column = 0; column < kSqueezeStride; column += 16) {
            const int8x16_t activation = vld1q_s8(counts + column);
            for (int item = 0; item < 8; ++item) {
                squeeze_sum[item] = vdotq_s32(
                    squeeze_sum[item],
                    vld1q_s8(
                        layer.qkv_squeeze_int8
                        + static_cast<std::size_t>(row + item) * kSqueezeStride
                        + column
                    ),
                    activation
                );
            }
        }
        for (int column = 0; column < kBypassStride; column += 16) {
            const int8x16_t activation = vld1q_s8(bypass_activation + column);
            for (int item = 0; item < 8; ++item) {
                bypass_sum[item] = vdotq_s32(
                    bypass_sum[item],
                    vld1q_s8(
                        layer.qkv_bypass_int8
                        + static_cast<std::size_t>(row + item) * kBypassStride
                        + column
                    ),
                    activation
                );
            }
        }
        for (int item = 0; item < 8; ++item) {
            output[row + item] =
                static_cast<float>(vaddvq_s32(squeeze_sum[item]))
                    * layer.qkv_squeeze_scales[row + item]
                + static_cast<float>(vaddvq_s32(bypass_sum[item]))
                    * bypass_activation_scale * layer.qkv_bypass_scales[row + item];
        }
    }
}
#endif

void matrix_vector_int8_block16(
    const std::int8_t* weights,
    const float* weight_scales,
    const std::uint8_t* block_mask,
    const float* input,
    float* output
) {
    static_assert(kDim % 16 == 0);
    alignas(64) std::int8_t activation[kDim];
    alignas(64) float rounded_input[kDim];
#if defined(TERNLOG_BOOLEAN_AX1)
    if constexpr (layout::kAttentionOutputInputFP16) {
        for (int index = 0; index < kDim; index += 8) {
            const float16x4_t low = vcvt_f16_f32(vld1q_f32(input + index));
            const float16x4_t high = vcvt_f16_f32(vld1q_f32(input + index + 4));
            vst1q_f32(rounded_input + index, vcvt_f32_f16(low));
            vst1q_f32(rounded_input + index + 4, vcvt_f32_f16(high));
        }
        input = rounded_input;
    }
#endif
    const float activation_scale = quantize_activation_int8(
        input, activation, kDim
    );
    constexpr int blocks = kDim / 16;
    for (int block_row = 0; block_row < blocks; ++block_row) {
        for (int half = 0; half < 2; ++half) {
            const int row = block_row * 16 + half * 8;
            int32x4_t sum[8] = {
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            };
            for (int block_column = 0; block_column < blocks; ++block_column) {
                if (!block_mask[block_row * blocks + block_column]) continue;
                const int column = block_column * 16;
                const int8x16_t value = vld1q_s8(activation + column);
                for (int item = 0; item < 8; ++item) {
                    sum[item] = vdotq_s32(
                        sum[item],
                        vld1q_s8(
                            weights
                            + static_cast<std::size_t>(row + item) * kDim
                            + column
                        ),
                        value
                    );
                }
            }
            for (int item = 0; item < 8; ++item) {
                output[row + item] = static_cast<float>(vaddvq_s32(sum[item]))
                    * activation_scale * weight_scales[row + item];
            }
        }
    }
}

void matrix_vector_int8_qk_value_block16(
    const std::int8_t* weights,
    const float* weight_scales,
    const std::uint8_t* value_block_mask,
    const float* input,
    float* output
) {
    static_assert(kDim % 16 == 0);
    constexpr int qk_rows =
        (layout::kHeads + layout::kKeyHeads) * kQkHeadDim;
    constexpr int value_rows = layout::kValueHeads * kHeadDim;
    static_assert(qk_rows % 8 == 0);
    static_assert(value_rows % 16 == 0);
    alignas(64) std::int8_t activation[kDim];
    const float activation_scale = quantize_activation_int8(
        input, activation, kDim
    );
    for (int row = 0; row < qk_rows; row += 8) {
        int32x4_t sum[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int column = 0; column < kDim; column += 16) {
            const int8x16_t value = vld1q_s8(activation + column);
            for (int item = 0; item < 8; ++item) {
                sum[item] = vdotq_s32(
                    sum[item],
                    vld1q_s8(
                        weights
                            + static_cast<std::size_t>(row + item) * kDim
                            + column
                    ),
                    value
                );
            }
        }
        for (int item = 0; item < 8; ++item) {
            output[row + item] = static_cast<float>(vaddvq_s32(sum[item]))
                * activation_scale * weight_scales[row + item];
        }
    }
    constexpr int input_blocks = kDim / 16;
    constexpr int value_row_blocks = value_rows / 16;
    for (int block_row = 0; block_row < value_row_blocks; ++block_row) {
        for (int half = 0; half < 2; ++half) {
            const int row = qk_rows + block_row * 16 + half * 8;
            int32x4_t sum[8] = {
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            };
            for (
                int block_column = 0;
                block_column < input_blocks;
                ++block_column
            ) {
                if (!value_block_mask[
                    block_row * input_blocks + block_column
                ]) continue;
                const int column = block_column * 16;
                const int8x16_t value = vld1q_s8(activation + column);
                for (int item = 0; item < 8; ++item) {
                    sum[item] = vdotq_s32(
                        sum[item],
                        vld1q_s8(
                            weights
                            + static_cast<std::size_t>(row + item) * kDim
                            + column
                        ),
                        value
                    );
                }
            }
            for (int item = 0; item < 8; ++item) {
                output[row + item] = static_cast<float>(vaddvq_s32(sum[item]))
                    * activation_scale * weight_scales[row + item];
            }
        }
    }
}

void matrix_vector(
    const Half* weights,
    const std::int8_t* int8_weights,
    const float* int8_scales,
    const float* input,
    float* output,
    int rows,
    int columns
) {
#if defined(TERNLOG_INT8_MATVEC)
    matrix_vector_int8(
        int8_weights, int8_scales, input, output, rows, columns
    );
#elif defined(TERNLOG_PREROUNDED_MATVEC)
    alignas(64) Half rounded[1040];
    for (int index = 0; index < columns; index += 8) {
        const float16x4_t low = vcvt_f16_f32(vld1q_f32(input + index));
        const float16x4_t high = vcvt_f16_f32(vld1q_f32(input + index + 4));
        vst1q_f16(rounded + index, vcombine_f16(low, high));
    }
#if defined(TERNLOG_SVE_MATVEC)
    ternlog_matvec_f16input_sve_m4(
        weights, nullptr, rounded, output, rows, columns
    );
#else
    ternlog_matvec_f16input_fmlal_m4(
        weights, nullptr, rounded, output, rows, columns
    );
#endif
#else
    ternlog_matvec_f16_fmlal_m4(weights, nullptr, input, output, rows, columns);
#endif
}

void load_half(const Half* input, float* output, int count) {
    int index = 0;
    for (; index + 8 <= count; index += 8) {
        const float16x8_t value = vld1q_f16(input + index);
        vst1q_f32(output + index, vcvt_f32_f16(vget_low_f16(value)));
        vst1q_f32(output + index + 4, vcvt_f32_f16(vget_high_f16(value)));
    }
    for (; index < count; ++index) {
        output[index] = static_cast<float>(input[index]);
    }
}

inline float inverse_square_root(float value) {
#if defined(TERNLOG_RSQRT_ONE_NEWTON) || defined(TERNLOG_RSQRT_TWO_NEWTON)
    const float32x2_t argument = vdup_n_f32(value);
    float32x2_t estimate = vrsqrte_f32(argument);
    estimate = vmul_f32(
        estimate,
        vrsqrts_f32(vmul_f32(argument, estimate), estimate)
    );
#if defined(TERNLOG_RSQRT_TWO_NEWTON)
    estimate = vmul_f32(
        estimate,
        vrsqrts_f32(vmul_f32(argument, estimate), estimate)
    );
#endif
    return vget_lane_f32(estimate, 0);
#else
    return 1.0F / std::sqrt(value);
#endif
}

void rms_norm(
    const float* input,
    const Half* weights,
    float* output,
    int count,
    float epsilon
) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    int index = 0;
    for (; index + 4 <= count; index += 4) {
        const float32x4_t value = vld1q_f32(input + index);
        sum = vfmaq_f32(sum, value, value);
    }
    float square_sum = horizontal_sum(sum);
    for (; index < count; ++index) {
        square_sum += input[index] * input[index];
    }
    const float inverse_rms = inverse_square_root(square_sum / count + epsilon);
    index = 0;
    for (; index + 8 <= count; index += 8) {
        const float16x8_t weight = vld1q_f16(weights + index);
        vst1q_f32(
            output + index,
            vmulq_f32(
                vmulq_n_f32(vld1q_f32(input + index), inverse_rms),
                vcvt_f32_f16(vget_low_f16(weight))
            )
        );
        vst1q_f32(
            output + index + 4,
            vmulq_f32(
                vmulq_n_f32(vld1q_f32(input + index + 4), inverse_rms),
                vcvt_f32_f16(vget_high_f16(weight))
            )
        );
    }
    for (; index < count; ++index) {
        output[index] = input[index] * inverse_rms * static_cast<float>(weights[index]);
    }
}

float rms_inverse(
    const float* input,
    int count,
    float epsilon
) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    int index = 0;
    for (; index + 4 <= count; index += 4) {
        const float32x4_t value = vld1q_f32(input + index);
        sum = vfmaq_f32(sum, value, value);
    }
    float square_sum = horizontal_sum(sum);
    for (; index < count; ++index) {
        square_sum += input[index] * input[index];
    }
    return inverse_square_root(square_sum / count + epsilon);
}

void rms_norm_from_square_sum(
    const float* input,
    const Half* weights,
    float* output,
    int count,
    float epsilon,
    float square_sum
) {
    const float inverse_rms = inverse_square_root(square_sum / count + epsilon);
    int index = 0;
    for (; index + 8 <= count; index += 8) {
        const float16x8_t weight = vld1q_f16(weights + index);
        vst1q_f32(
            output + index,
            vmulq_f32(
                vmulq_n_f32(vld1q_f32(input + index), inverse_rms),
                vcvt_f32_f16(vget_low_f16(weight))
            )
        );
        vst1q_f32(
            output + index + 4,
            vmulq_f32(
                vmulq_n_f32(vld1q_f32(input + index + 4), inverse_rms),
                vcvt_f32_f16(vget_high_f16(weight))
            )
        );
    }
    for (; index < count; ++index) {
        output[index] = input[index] * inverse_rms * static_cast<float>(weights[index]);
    }
}

void qk_rms_norm(float* state, int count) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    for (int index = 0; index < count; index += 4) {
        const float32x4_t value = vld1q_f32(state + index);
        sum = vfmaq_f32(sum, value, value);
    }
    const float inverse_rms = inverse_square_root(
        horizontal_sum(sum) / count + kQkRmsEpsilon
    );
    for (int index = 0; index < count; index += 4) {
        vst1q_f32(
            state + index,
            vmulq_n_f32(vld1q_f32(state + index), inverse_rms)
        );
    }
}

float dot(const float* left, const float* right, int count) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    for (int index = 0; index < count; index += 4) {
        sum = vfmaq_f32(sum, vld1q_f32(left + index), vld1q_f32(right + index));
    }
    return horizontal_sum(sum);
}

float dot(const Half* left, const Half* right, int count) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    for (int index = 0; index < count; index += 8) {
        const float16x8_t left_value = vld1q_f16(left + index);
        const float16x8_t right_value = vld1q_f16(right + index);
        sum = vfmlalq_low_f16(sum, left_value, right_value);
        sum = vfmlalq_high_f16(sum, left_value, right_value);
    }
    return horizontal_sum(sum);
}

std::int32_t dot(const std::int8_t* left, const std::int8_t* right, int count) {
    int32x4_t sum = vdupq_n_s32(0);
    for (int index = 0; index < count; index += 16) {
        sum = vdotq_s32(sum, vld1q_s8(left + index), vld1q_s8(right + index));
    }
    return vaddvq_s32(sum);
}

void store_cache(float* destination, const float* source, int count) {
    std::memcpy(destination, source, sizeof(float) * count);
}

void store_cache(Half* destination, const float* source, int count) {
    for (int index = 0; index < count; index += 8) {
        const float16x4_t low = vcvt_f16_f32(vld1q_f32(source + index));
        const float16x4_t high = vcvt_f16_f32(vld1q_f32(source + index + 4));
        vst1q_f16(destination + index, vcombine_f16(low, high));
    }
}

#if defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
static_assert(layout::kContext % 4 == 0);
static_assert(kHeadDim % 4 == 0);

std::size_t shared_value_cache_int8_offset(
    int layer, int value_head, int slot, int channel
) {
    const std::size_t block = static_cast<std::size_t>(slot / 4);
    return (
        (
            (static_cast<std::size_t>(layer) * layout::kValueHeads + value_head)
            * (layout::kContext / 4) + block
        ) * kHeadDim + channel
    ) * 4 + static_cast<std::size_t>(slot & 3);
}

void store_shared_value_cache_int8(
    std::vector<ValueCacheScalar>& cache,
    int layer,
    int value_head,
    int slot,
    const float* source,
    float scale
) {
    const float inverse_scale = 1.0F / scale;
    for (int channel = 0; channel < kHeadDim; ++channel) {
        const int rounded = std::clamp(
            static_cast<int>(std::nearbyint(source[channel] * inverse_scale)),
            -127,
            127
        );
        cache[shared_value_cache_int8_offset(
            layer, value_head, slot, channel
        )] = static_cast<std::int8_t>(rounded);
    }
}

void accumulate_shared_value_cache_int8(
    float* output,
    const std::vector<ValueCacheScalar>& cache,
    int layer,
    int value_head,
    int first_slot,
    const std::int8_t* weights,
    int count,
    float value_scale
) {
    alignas(16) std::int32_t sums[kHeadDim]{};
    int denominator = 0;
    int item = 0;

    const auto accumulate_scalar = [&](int offset) {
        const int slot = (first_slot + offset) % layout::kContext;
        const int weight = weights[offset];
        denominator += weight;
        for (int channel = 0; channel < kHeadDim; ++channel) {
            sums[channel] += weight * static_cast<int>(cache[
                shared_value_cache_int8_offset(
                    layer, value_head, slot, channel
                )
            ]);
        }
    };

    while (item < count && ((first_slot + item) & 3) != 0) {
        accumulate_scalar(item++);
    }
    while (item + 4 <= count) {
        int slot = first_slot + item;
#if defined(TERNLOG_CONTIGUOUS_VALUE_RING)
        if (slot >= layout::kContext) slot -= layout::kContext;
#else
        slot %= layout::kContext;
        if (slot + 4 > layout::kContext) {
            break;
        }
#endif
        std::int32_t packed_weights;
        std::memcpy(&packed_weights, weights + item, sizeof(packed_weights));
        const int8x16_t repeated_weights = vreinterpretq_s8_s32(
            vdupq_n_s32(packed_weights)
        );
        const std::size_t block_base = shared_value_cache_int8_offset(
            layer, value_head, slot, 0
        );
        for (int channel = 0; channel < kHeadDim; channel += 4) {
            int32x4_t accumulator = vld1q_s32(sums + channel);
            accumulator = vdotq_s32(
                accumulator,
                vld1q_s8(cache.data() + block_base + channel * 4),
                repeated_weights
            );
            vst1q_s32(sums + channel, accumulator);
        }
        denominator += static_cast<int>(weights[item])
            + static_cast<int>(weights[item + 1])
            + static_cast<int>(weights[item + 2])
            + static_cast<int>(weights[item + 3]);
        item += 4;
    }
    while (item < count) {
        accumulate_scalar(item++);
    }

    const float scale = value_scale / static_cast<float>(std::max(denominator, 1));
    for (int channel = 0; channel < kHeadDim; channel += 4) {
        vst1q_f32(
            output + channel,
            vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(sums + channel)), scale)
        );
    }
}

void accumulate_shared_local_value_cache_int8(
    float* output,
    const std::vector<ValueCacheScalar>& cache,
    int layer,
    int first_slot,
    const std::int8_t weights[layout::kLocalHeads][layout::kLocalWindow],
    int count,
    float value_scale
) {
    alignas(16) std::int32_t sums[layout::kLocalHeads][kHeadDim]{};
    int denominators[layout::kLocalHeads]{};
    int item = 0;

    const auto accumulate_scalar = [&](int offset) {
        const int slot = (first_slot + offset) % layout::kContext;
        for (int head = 0; head < layout::kLocalHeads; ++head) {
            denominators[head] += weights[head][offset];
        }
        for (int channel = 0; channel < kHeadDim; ++channel) {
            const int value = cache[shared_value_cache_int8_offset(
                layer, 0, slot, channel
            )];
            for (int head = 0; head < layout::kLocalHeads; ++head) {
                sums[head][channel] += weights[head][offset] * value;
            }
        }
    };

#if !defined(TERNLOG_LOCAL_MASKED_SDOT)
    while (item < count && ((first_slot + item) & 3) != 0) {
        accumulate_scalar(item++);
    }
    while (item + 4 <= count) {
        const int slot = (first_slot + item) % layout::kContext;
        if (slot + 4 > layout::kContext) {
            break;
        }
#else
    while (item < count) {
        const int current_slot = (first_slot + item) % layout::kContext;
        const int subslot = current_slot & 3;
        const int slot = current_slot - subslot;
        const int take = std::min(4 - subslot, count - item);
#endif
        int8x16_t repeated_weights[layout::kLocalHeads];
        for (int head = 0; head < layout::kLocalHeads; ++head) {
#if defined(TERNLOG_LOCAL_MASKED_SDOT)
            // Padded coefficients let SDOT read the aligned physical cache
            // block without admitting stale/future slots to the prediction.
            std::int32_t packed_weights = 0;
            auto* bytes = reinterpret_cast<std::int8_t*>(&packed_weights);
            for (int j = 0; j < take; ++j) {
                bytes[subslot + j] = weights[head][item + j];
                denominators[head] += weights[head][item + j];
            }
            repeated_weights[head] = vreinterpretq_s8_s32(vdupq_n_s32(packed_weights));
#else
            std::int32_t packed_weights;
            std::memcpy(
                &packed_weights, weights[head] + item, sizeof(packed_weights)
            );
            repeated_weights[head] = vreinterpretq_s8_s32(
                vdupq_n_s32(packed_weights)
            );
            denominators[head] += static_cast<int>(weights[head][item])
                + static_cast<int>(weights[head][item + 1])
                + static_cast<int>(weights[head][item + 2])
                + static_cast<int>(weights[head][item + 3]);
#endif
        }
        const std::size_t block_base = shared_value_cache_int8_offset(
            layer, 0, slot, 0
        );
        for (int channel = 0; channel < kHeadDim; channel += 4) {
            const int8x16_t values = vld1q_s8(
                cache.data() + block_base + channel * 4
            );
            for (int head = 0; head < layout::kLocalHeads; ++head) {
                int32x4_t accumulator = vld1q_s32(sums[head] + channel);
                accumulator = vdotq_s32(
                    accumulator, values, repeated_weights[head]
                );
                vst1q_s32(sums[head] + channel, accumulator);
            }
        }
        item +=
#if defined(TERNLOG_LOCAL_MASKED_SDOT)
            take;
#else
            4;
#endif
    }
    while (item < count) {
        accumulate_scalar(item++);
    }

    for (int head = 0; head < layout::kLocalHeads; ++head) {
        const float scale = value_scale
            / static_cast<float>(std::max(denominators[head], 1));
        for (int channel = 0; channel < kHeadDim; channel += 4) {
            vst1q_f32(
                output + head * kHeadDim + channel,
                vmulq_n_f32(
                    vcvtq_f32_s32(vld1q_s32(sums[head] + channel)), scale
                )
            );
        }
    }
}
#endif

void accumulate_weighted(
    float* output, const float* values, float weight, int count
) {
    for (int index = 0; index < count; index += 4) {
        vst1q_f32(
            output + index,
            vfmaq_n_f32(
                vld1q_f32(output + index),
                vld1q_f32(values + index),
                weight
            )
        );
    }
}

void accumulate_weighted(
    float* output, const Half* values, float weight, int count
) {
#if defined(TERNLOG_FMLAL_VALUE_ATTENTION)
    const float16x8_t multiplier = vdupq_n_f16(static_cast<Half>(weight));
    for (int index = 0; index < count; index += 8) {
        const float16x8_t value = vld1q_f16(values + index);
        vst1q_f32(
            output + index,
            vfmlalq_low_f16(vld1q_f32(output + index), value, multiplier)
        );
        vst1q_f32(
            output + index + 4,
            vfmlalq_high_f16(vld1q_f32(output + index + 4), value, multiplier)
        );
    }
#else
    for (int index = 0; index < count; index += 8) {
        const float16x8_t value = vld1q_f16(values + index);
        vst1q_f32(
            output + index,
            vfmaq_n_f32(
                vld1q_f32(output + index),
                vcvt_f32_f16(vget_low_f16(value)),
                weight
            )
        );
        vst1q_f32(
            output + index + 4,
            vfmaq_n_f32(
                vld1q_f32(output + index + 4),
                vcvt_f32_f16(vget_high_f16(value)),
                weight
            )
        );
    }
#endif
}

#if defined(TERNLOG_SHARED_LOCAL_VALUE4)
void accumulate_shared_local_values4(
    float* output,
    const Half* const* values,
    const float weights[4][layout::kLocalWindow],
    int count
) {
    static_assert(layout::kLocalHeads == 4);
    static_assert(kHeadDim == 32);
#if defined(TERNLOG_SHARED_LOCAL_VALUE4_REGISTER)
    for (int index = 0; index < kHeadDim; index += 8) {
        float32x4_t low[4] = {
            vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
            vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
        };
        float32x4_t high[4] = {
            vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
            vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
        };
        for (int item = 0; item < count; ++item) {
            const float16x8_t value = vld1q_f16(values[item] + index);
#pragma clang loop unroll(full)
            for (int head = 0; head < 4; ++head) {
                const float16x8_t multiplier = vdupq_n_f16(
                    static_cast<Half>(weights[head][item])
                );
                low[head] = vfmlalq_low_f16(low[head], value, multiplier);
                high[head] = vfmlalq_high_f16(high[head], value, multiplier);
            }
        }
#pragma clang loop unroll(full)
        for (int head = 0; head < 4; ++head) {
            vst1q_f32(output + head * kHeadDim + index, low[head]);
            vst1q_f32(output + head * kHeadDim + index + 4, high[head]);
        }
    }
#else
    std::fill(output, output + 4 * kHeadDim, 0.0F);
    for (int item = 0; item < count; ++item) {
        for (int index = 0; index < kHeadDim; index += 8) {
            const float16x8_t value = vld1q_f16(values[item] + index);
#pragma clang loop unroll(full)
            for (int head = 0; head < 4; ++head) {
                const float16x8_t multiplier = vdupq_n_f16(
                    static_cast<Half>(weights[head][item])
                );
                float* destination = output + head * kHeadDim + index;
                vst1q_f32(
                    destination,
                    vfmlalq_low_f16(vld1q_f32(destination), value, multiplier)
                );
                vst1q_f32(
                    destination + 4,
                    vfmlalq_high_f16(
                        vld1q_f32(destination + 4), value, multiplier
                    )
                );
            }
        }
    }
#endif
}
#endif

float reciprocal_refined(float value) {
    float32x2_t denominator = vdup_n_f32(value);
    float32x2_t estimate = vrecpe_f32(denominator);
    estimate = vmul_f32(
        estimate, vrecps_f32(denominator, estimate)
    );
    estimate = vmul_f32(
        estimate, vrecps_f32(denominator, estimate)
    );
    return vget_lane_f32(estimate, 0);
}

inline float clipped_linear_weight(float score, float maximum, float slope) {
    const float32x2_t delta = vdup_n_f32(score - maximum);
    const float32x2_t weight = vfma_n_f32(vdup_n_f32(1.0F), delta, slope);
    return vget_lane_f32(vmax_f32(vdup_n_f32(0.0F), weight), 0);
}

#if defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
void clipped_linear_weights_int8(
    const float* scores,
    float maximum,
    float slope,
    std::int8_t* output,
    int count
) {
    const float32x4_t one = vdupq_n_f32(1.0F);
    const float32x4_t zero = vdupq_n_f32(0.0F);
    const float32x4_t maximum_vector = vdupq_n_f32(maximum);
    const int32x4_t integer_zero = vdupq_n_s32(0);
    const int32x4_t integer_maximum = vdupq_n_s32(127);
    int item = 0;
    for (; item + 4 <= count; item += 4) {
        const float32x4_t delta = vsubq_f32(
            vld1q_f32(scores + item), maximum_vector
        );
        const float32x4_t weights = vmaxq_f32(
            zero, vfmaq_n_f32(one, delta, slope)
        );
        const int32x4_t integers = vminq_s32(
            integer_maximum,
            vmaxq_s32(
                integer_zero,
                vcvtnq_s32_f32(vmulq_n_f32(weights, 127.0F))
            )
        );
        const int16x4_t narrowed16 = vqmovn_s32(integers);
        const int8x8_t narrowed8 = vqmovn_s16(
            vcombine_s16(narrowed16, vdup_n_s16(0))
        );
        vst1_lane_s32(
            reinterpret_cast<std::int32_t*>(output + item),
            vreinterpret_s32_s8(narrowed8),
            0
        );
    }
    for (; item < count; ++item) {
        const float weight = clipped_linear_weight(
            scores[item], maximum, slope
        );
        output[item] = static_cast<std::int8_t>(std::clamp(
            static_cast<int>(std::nearbyint(weight * 127.0F)), 0, 127
        ));
    }
}
#endif

class Tokenizer {
  public:
    Tokenizer()
        : byte_rows_(at<std::uint16_t>(layout::kTokenizerByteTokenIds)),
          offsets_(at<std::uint32_t>(layout::kTokenizerTokenOffsets)),
          bytes_(at<std::uint8_t>(layout::kTokenizerTokenBytes)),
          triples_(at<std::uint16_t>(layout::kTokenizerMergeTriples)) {
        for (int rank = 0; rank < layout::kMergeCount; ++rank) {
            const std::uint16_t* triple = triples_ + rank * 3;
            merges_.emplace(
                (static_cast<std::uint32_t>(triple[0]) << 16) | triple[1],
                Merge{rank, triple[2]}
            );
        }
    }

    std::vector<std::uint16_t> encode(std::string_view text) const {
        std::vector<std::uint16_t> tokens;
        for (std::size_t index = 0; index < text.size();) {
            if (text.substr(index).starts_with(kEndOfText)) {
                tokens.push_back(0);
                index += kEndOfText.size();
            } else {
                tokens.push_back(byte_rows_[static_cast<unsigned char>(text[index])]);
                ++index;
            }
        }
        while (tokens.size() >= 2) {
            int best_rank = std::numeric_limits<int>::max();
            std::uint32_t best_pair = 0;
            std::uint16_t replacement = 0;
            for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
                const std::uint32_t key =
                    (static_cast<std::uint32_t>(tokens[index]) << 16)
                    | tokens[index + 1];
                const auto found = merges_.find(key);
                if (found != merges_.end() && found->second.rank < best_rank) {
                    best_rank = found->second.rank;
                    best_pair = key;
                    replacement = found->second.output;
                }
            }
            if (best_rank == std::numeric_limits<int>::max()) {
                break;
            }
            std::vector<std::uint16_t> merged;
            merged.reserve(tokens.size());
            for (std::size_t index = 0; index < tokens.size();) {
                if (
                    index + 1 < tokens.size()
                    && ((static_cast<std::uint32_t>(tokens[index]) << 16)
                        | tokens[index + 1]) == best_pair
                ) {
                    merged.push_back(replacement);
                    index += 2;
                } else {
                    merged.push_back(tokens[index++]);
                }
            }
            tokens.swap(merged);
        }
        return tokens;
    }

    std::string_view decode(std::uint16_t token) const {
        const std::uint32_t begin = offsets_[token];
        const std::uint32_t end = offsets_[token + 1];
        return std::string_view(
            reinterpret_cast<const char*>(bytes_ + begin), end - begin
        );
    }

  private:
    struct Merge {
        int rank;
        std::uint16_t output;
    };
    const std::uint16_t* byte_rows_;
    const std::uint32_t* offsets_;
    const std::uint8_t* bytes_;
    const std::uint16_t* triples_;
    std::unordered_map<std::uint32_t, Merge> merges_;
};

class Transformer {
  public:
    Transformer()
        : key_cache_(
#if defined(TERNLOG_SHARED_QK45)
              (layout::kLayers - 1) * key_cache_values_per_layer()
#else
              layout::kLayers * key_cache_values_per_layer()
#endif
          ),
#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
          key_scale_cache_(layout::kLayers * cache_slots_per_layer()),
#endif
          value_cache_(
#if defined(TERNLOG_SHARED_VALUE_HEAD)
              layout::kLayers * layout::kValueHeads * layout::kContext * kHeadDim
#else
              layout::kLayers * value_cache_values_per_layer()
#endif
          ) {
#if defined(TERNLOG_SHARED_ROPE)
        initialize_rope();
#endif
    }

    void reset() {
        tokens_seen_ = 0;
        previous_token_ = 0;
#if defined(TERNLOG_SHARED_ROPE)
        set_rope_position(0);
#endif
    }

    std::array<std::uint64_t, 3> recurrent_state_hashes() const {
        const auto hash_bytes = [](const void* pointer, std::size_t bytes) {
            const auto* data = static_cast<const std::uint8_t*>(pointer);
            std::uint64_t hash = 1469598103934665603ULL;
            for (std::size_t index = 0; index < bytes; ++index) {
                hash ^= data[index];
                hash *= 1099511628211ULL;
            }
            return hash;
        };
        const std::uint64_t key_hash = hash_bytes(
            key_cache_.data(), key_cache_.size() * sizeof(KeyCacheScalar)
        );
        const std::uint64_t value_hash = hash_bytes(
            value_cache_.data(), value_cache_.size() * sizeof(ValueCacheScalar)
        );
        std::uint64_t position_hash = 1469598103934665603ULL;
#if defined(TERNLOG_SHARED_ROPE)
        const auto mix_position = [&](const auto& values) {
            const std::uint64_t component = hash_bytes(
                values.data(), values.size() * sizeof(values[0])
            );
            position_hash ^= component;
            position_hash *= 1099511628211ULL;
        };
        mix_position(rope_cosine_);
        mix_position(rope_sine_);
        mix_position(local_rope_cosine_);
        mix_position(local_rope_sine_);
#endif
        return {key_hash, value_hash, position_hash};
    }

    void add_ngram_features(std::uint16_t token, float* state) const {
        if constexpr (layout::kNgramOrderCount == 0) return;
        constexpr int width =
            layout::kNgramOrderCount * layout::kNgramDimension;
        static_assert(width <= kDim);
        static_assert(layout::kNgramDimension % 16 == 0);
        static_assert(
            layout::kNgramOrderCount == 0
            || (layout::kNgramHeads > 0 && layout::kNgramHeads <= 4)
        );
        constexpr std::uint32_t multipliers[] = {
            0x5BD1E995U, 0x27D4EB2FU, 0x165667B1U, 0x61C88647U
        };
        for (int order_index = 0;
             order_index < layout::kNgramOrderCount; ++order_index) {
            const int order = layout::kNgramOrders[order_index];
            float* destination = state + kDim - width
                + order_index * layout::kNgramDimension;
            for (int head = 0; head < layout::kNgramHeads; ++head) {
                std::uint32_t hash = static_cast<std::uint32_t>(order);
                for (int position = 0; position < order; ++position) {
                    const std::int64_t stream_position =
                        static_cast<std::int64_t>(tokens_seen_)
                        - (order - 1 - position);
                    const std::uint32_t byte = stream_position < 0
                        ? 256U
                        : stream_position == static_cast<std::int64_t>(tokens_seen_)
                        ? static_cast<std::uint32_t>(token)
                        : static_cast<std::uint32_t>(
                            byte_history_[static_cast<std::size_t>(stream_position) & 15]
                        );
                    hash = ((hash * multipliers[head]) ^ (byte + 1U))
                        & 0x7FFFFFFFU;
                }
                hash ^= hash >> 15;
                const int region = order_index * layout::kNgramHeads + head;
                const std::size_t row = static_cast<std::size_t>(region)
                    * layout::kNgramTableRows
                    + (hash & (layout::kNgramTableRows - 1));
                const std::int8_t* source = weights_.ngram_table
                    + row * layout::kNgramDimension;
                const float scale = weights_.ngram_scales[region];
                for (int channel = 0;
                     channel < layout::kNgramDimension; channel += 16) {
                    const int8x16_t packed = vld1q_s8(source + channel);
                    const int16x8_t low16 = vmovl_s8(vget_low_s8(packed));
                    const int16x8_t high16 = vmovl_high_s8(packed);
                    vst1q_f32(
                        destination + channel,
                        vfmaq_n_f32(
                            vld1q_f32(destination + channel),
                            vcvtq_f32_s32(vmovl_s16(vget_low_s16(low16))),
                            scale
                        )
                    );
                    vst1q_f32(
                        destination + channel + 4,
                        vfmaq_n_f32(
                            vld1q_f32(destination + channel + 4),
                            vcvtq_f32_s32(vmovl_high_s16(low16)),
                            scale
                        )
                    );
                    vst1q_f32(
                        destination + channel + 8,
                        vfmaq_n_f32(
                            vld1q_f32(destination + channel + 8),
                            vcvtq_f32_s32(vmovl_s16(vget_low_s16(high16))),
                            scale
                        )
                    );
                    vst1q_f32(
                        destination + channel + 12,
                        vfmaq_n_f32(
                            vld1q_f32(destination + channel + 12),
                            vcvtq_f32_s32(vmovl_high_s16(high16)),
                            scale
                        )
                    );
                }
            }
        }
    }

#if defined(TERNLOG_PROFILE_PHASES)
    void reset_profile() {
        profile_nanoseconds_.fill(0);
        profile_calls_.fill(0);
    }

    void print_profile() const {
        static constexpr const char* names[] = {
            "embedding_norm", "qkv", "attention_kv", "attention_output",
            "boolean_ffn", "final_readout",
        };
        std::uint64_t total = 0;
        for (std::uint64_t value : profile_nanoseconds_) total += value;
        std::fputs("{\"profile_phases\":{", stdout);
        for (std::size_t index = 0; index < profile_nanoseconds_.size(); ++index) {
            if (index != 0) std::fputc(',', stdout);
            const double share = total == 0 ? 0.0
                : static_cast<double>(profile_nanoseconds_[index]) / total;
            const double nanoseconds_per_call = profile_calls_[index] == 0
                ? 0.0
                : static_cast<double>(profile_nanoseconds_[index])
                    / profile_calls_[index];
            std::fprintf(
                stdout,
                "\"%s\":{\"nanoseconds\":%llu,\"calls\":%llu,"
                "\"nanoseconds_per_call\":%.3f,\"share\":%.6f}",
                names[index],
                static_cast<unsigned long long>(profile_nanoseconds_[index]),
                static_cast<unsigned long long>(profile_calls_[index]),
                nanoseconds_per_call,
                share
            );
        }
        std::fprintf(
            stdout, "},\"profiled_nanoseconds\":%llu}\n",
            static_cast<unsigned long long>(total)
        );
    }
#endif

#if defined(TERNLOG_BOOLEAN_VOCAB_HIERARCHY32X64)
    void forward_hierarchy_fast(std::uint16_t token) {
        forward(token, nullptr);
    }

    const float* hierarchy_cluster_logits() const {
        return hierarchy_cluster_logits_.data();
    }

    void hierarchy_leaf_logits(int cluster, float* leaf_logits) const {
        const std::size_t leaf_base = static_cast<std::size_t>(cluster) * 64;
        ternlog_vocab_boolean_hierarchy64_factor32_leaf(
            leaf_logits,
            hierarchy_query_.data(),
            hierarchy_hidden_.data(),
            weights_.token_codes,
            weights_.hierarchy_token_ids + leaf_base,
            weights_.readout_correction_output + leaf_base * 32,
            weights_.readout_bias + leaf_base
        );
    }

    std::uint16_t hierarchy_token(int cluster, int leaf) const {
        return weights_.hierarchy_token_ids[
            static_cast<std::size_t>(cluster) * 64 + leaf
        ];
    }

    void materialize_hierarchy_logits(float* logits) const {
        const float cluster_maximum = *std::max_element(
            hierarchy_cluster_logits_.begin(), hierarchy_cluster_logits_.end()
        );
        double cluster_denominator = 0.0;
        for (float value : hierarchy_cluster_logits_) {
            cluster_denominator += std::exp(
                static_cast<double>(value - cluster_maximum)
            );
        }
        const float cluster_log_normalizer = cluster_maximum
            + static_cast<float>(std::log(cluster_denominator));
        alignas(64) float leaf_logits[64];
        for (int cluster = 0; cluster < 32; ++cluster) {
            hierarchy_leaf_logits(cluster, leaf_logits);
            float leaf_maximum = leaf_logits[0];
            for (int leaf = 1; leaf < 64; ++leaf) {
                leaf_maximum = std::max(leaf_maximum, leaf_logits[leaf]);
            }
            double leaf_denominator = 0.0;
            for (float value : leaf_logits) {
                leaf_denominator += std::exp(
                    static_cast<double>(value - leaf_maximum)
                );
            }
            const float leaf_log_normalizer = leaf_maximum
                + static_cast<float>(std::log(leaf_denominator));
            const float cluster_log_probability =
                hierarchy_cluster_logits_[cluster] - cluster_log_normalizer;
            for (int leaf = 0; leaf < 64; ++leaf) {
                logits[hierarchy_token(cluster, leaf)] = cluster_log_probability
                    + leaf_logits[leaf] - leaf_log_normalizer;
            }
        }
    }
#endif

#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
#if defined(TERNLOG_RECURRENT_ROPE) || defined(TERNLOG_SHARED_QK45) || !defined(TERNLOG_BOOLEAN_AX1)
#error "Layerwise probe requires independent Boolean layers and direct RoPE"
#endif
    void forward_known_tile(const std::uint8_t* tokens, std::size_t count, float* logits) {
        static_assert(layout::kAttentionSharedInputMask == 0,
                      "Cross-layer shared sources require explicit tile storage");
        if (count > 32) throw std::runtime_error("encoder probe tile exceeds 32 tokens");
        alignas(64) float states[32 * kDim];
        const auto start = tokens_seen_;
        for (int layer = 0; layer < layout::kLayers; ++layer) {
            tokens_seen_ = start;
            for (std::size_t index = 0; index < count; ++index) {
                forward(tokens[index], logits + index * layout::kVocabulary,
                        layer, states + index * kDim);
            }
        }
    }
#endif

    void forward(std::uint16_t token, float* logits
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
                 , int selected_layer = -1, float* tile_state = nullptr
#endif
    ) {
        alignas(64) float state[kDim];
        alignas(64) float input_state[kDim];
        alignas(64) float normalized[kDim];
        alignas(64) float qkv[layout::kQkvRows];
        alignas(64) float attention[kDim];
        alignas(64) float projected[kDim];
        alignas(64) float hidden[1040];
        alignas(64) float scores[layout::kContext];

#if defined(TERNLOG_PROFILE_PHASES)
        profile_restart();
#endif

#if defined(TERNLOG_SHARED_ROPE) && !defined(TERNLOG_RECURRENT_ROPE)
        set_rope_position(tokens_seen_);
#endif

#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
        if (selected_layer <= 0) {
#endif
#if defined(TERNLOG_BOOLEAN_AX1)
        ternlog_vocab_boolean_embedding160(
            state, weights_.token_codes + static_cast<std::size_t>(token) * 8, 64
        );
        add_ngram_features(token, state);
#else
        load_half(weights_.token_embedding + static_cast<std::size_t>(token) * kDim, state, kDim);
        const std::uint32_t hash =
            ((36313U * token) ^ (27191U * previous_token_))
            & (layout::kBigramRows - 1);
        const Half* bigram = weights_.bigram_embedding
            + static_cast<std::size_t>(hash) * layout::kBigramDimension;
        for (int index = 0; index < layout::kBigramDimension; index += 8) {
            const float16x8_t value = vld1q_f16(bigram + index);
            vst1q_f32(
                state + index,
                vfmaq_n_f32(
                    vld1q_f32(state + index),
                    vcvt_f32_f16(vget_low_f16(value)),
                    *weights_.bigram_scale
                )
            );
            vst1q_f32(
                state + index + 4,
                vfmaq_n_f32(
                    vld1q_f32(state + index + 4),
                    vcvt_f32_f16(vget_high_f16(value)),
                    *weights_.bigram_scale
                )
            );
        }
#endif
        rms_norm(state, weights_.input_norm, input_state, kDim, kRmsEpsilon);
        std::memcpy(state, input_state, sizeof(state));
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
        } else {
            std::memcpy(state, tile_state, sizeof(state));
        }
#endif
#if defined(TERNLOG_PROFILE_PHASES)
        profile_mark(ProfilePhase::embedding_norm);
#endif
        alignas(64) float shared_attention_source[kDim];
#if defined(TERNLOG_SHARED_QK45)
        alignas(64) float shared_attention_qk[
            (layout::kHeads + layout::kKeyHeads) * kQkHeadDim
        ];
#endif

        for (int layer_index =
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
                 selected_layer < 0 ? 0 : selected_layer;
             layer_index < (selected_layer < 0 ? layout::kLayers : selected_layer + 1);
#else
                 0; layer_index < layout::kLayers;
#endif
             ++layer_index) {
#if defined(TERNLOG_PROFILE_PHASES)
            profile_restart();
#endif
#if !defined(TERNLOG_BOOLEAN_AX1)
            apply_dynamic_route(layer_index, state, input_state);
#endif
            if (layer_index == layout::kAttentionSharedInputFirst) {
                std::memcpy(shared_attention_source, state, sizeof(state));
            }
            const bool uses_shared_attention_source =
                (layout::kAttentionSharedInputMask & (1ULL << layer_index)) != 0;
            const float* attention_source = uses_shared_attention_source
                ? shared_attention_source
                : state;
            const LayerWeights& layer = weights_.layers[layer_index];
#if defined(TERNLOG_OUTPUT_RESIDUAL_RMS_FUSED)
            float fused_attention_square_sum = 0.0F;
#endif
            if ((layout::kAttentionSkipMask & (1ULL << layer_index)) == 0ULL) {
#if defined(TERNLOG_BOOLEAN_QKV_CYCLIC_ROUTES)
            rms_norm(
                attention_source, layer.attention_norm, normalized, kDim,
                kRmsEpsilon
            );
            alignas(64) std::int8_t qkv_activation[kDim];
            const float qkv_activation_scale = quantize_activation_int8(
                normalized, qkv_activation, kDim
            );
            matrix_vector_qkv_cyclic_routes(
                layer, normalized, qkv_activation, qkv_activation_scale, qkv
            );
#elif defined(TERNLOG_FUSED_QKV_RMS_QUANT)
            alignas(64) std::int8_t qkv_activation[kDim];
            const float qkv_activation_scale = rms_norm_quantize_activation_int8(
                attention_source, layer.attention_norm, qkv_activation, kDim,
                kRmsEpsilon
            );
            #if defined(TERNLOG_BOOLEAN_QKV_LOCAL_GLOBAL)
            matrix_vector_int8_qkv_local_global_prequantized(
                layer, qkv_activation, qkv_activation_scale, qkv
            );
            #elif defined(TERNLOG_BOOLEAN_QKV_TIED) \
                || defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE) \
                || defined(TERNLOG_BOOLEAN_QKV_HEAD_SUBSPACE_GLOBAL)
            matrix_vector_int8_qkv_head_subspace_prequantized(
                layer, qkv_activation, qkv_activation_scale, qkv
            );
            #elif defined(TERNLOG_BOOLEAN_QKV_SHARED_BASIS)
            matrix_vector_int8_qkv_shared_basis_prequantized(
                layer, qkv_activation, qkv_activation_scale, qkv
            );
            #elif defined(TERNLOG_BOOLEAN_QKV_TICKETS)
            matrix_vector_int8_qkv_tickets_prequantized(
                layer.qkv_int8, layer.qkv_scales,
                layer.qkv_ticket_offsets, layer.qkv_ticket_columns,
                qkv_activation, qkv_activation_scale, qkv
#if defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR) \
    || defined(TERNLOG_QKV_VALUE_CACHE_FUSED_NEON)
                , value_cache_, layer_index,
                static_cast<int>(tokens_seen_ % layout::kContext),
                *layer.value_cache_scale
#endif
            );
            #else
            matrix_vector_int8_qkv_prequantized(
                layer.qkv_int8, layer.qkv_scales, qkv_activation,
                qkv_activation_scale, qkv
            );
            #endif
#else
            rms_norm(
                attention_source, layer.attention_norm, normalized, kDim,
                kRmsEpsilon
            );
#if defined(TERNLOG_BOOLEAN_QKV_VALUE_BLOCK_INT8)
            matrix_vector_int8_qk_value_block16(
                layer.qkv_int8, layer.qkv_scales, layer.qkv_value_block_mask,
                normalized, qkv
            );
#elif defined(TERNLOG_BOOLEAN_QKV_SIGNED_SQUEEZE)
            matrix_vector_qkv_signed_squeeze(layer, normalized, qkv);
#elif defined(TERNLOG_BOOLEAN_QKV_INT8)
            matrix_vector_int8(
                layer.qkv_int8, layer.qkv_scales,
                normalized, qkv, 3 * kDim, kDim
            );
#else
            matrix_vector(
                layer.qkv,
#if defined(TERNLOG_BOOLEAN_AX1)
                nullptr, nullptr,
#else
                layer.qkv_int8, layer.qkv_scale,
#endif
                normalized, qkv, 3 * kDim, kDim
            );
#endif
#endif
#if defined(TERNLOG_SHARED_QK45)
            if (layer_index == 5) {
                std::memcpy(qkv, shared_attention_qk, sizeof(shared_attention_qk));
            } else
#endif
            for (int head = 0; head < kHeads; ++head) {
                const int qk_dimension = qk_head_dimension(head);
                qk_rms_norm(qkv + head * kQkHeadDim, qk_dimension);
                qk_rms_norm(
                    qkv + layout::kHeads * kQkHeadDim + head * kQkHeadDim,
                    qk_dimension
                );
                apply_rope(qkv + head * kQkHeadDim, tokens_seen_, qk_dimension);
                apply_rope(
                    qkv + layout::kHeads * kQkHeadDim + head * kQkHeadDim,
                    tokens_seen_, qk_dimension
                );
            }
#if defined(TERNLOG_SHARED_QK45)
            if (layer_index == 4) {
                std::memcpy(shared_attention_qk, qkv, sizeof(shared_attention_qk));
            }
#endif
#if defined(TERNLOG_PROFILE_PHASES)
            profile_mark(ProfilePhase::qkv);
#endif
#if defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
            alignas(16) std::int8_t
                local_value_weights[layout::kLocalHeads][layout::kLocalWindow]{};
            int local_value_count = 0;
#elif defined(TERNLOG_SHARED_LOCAL_VALUE4) \
    && !defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
            static_assert(layout::kLocalHeads == 4);
            alignas(64) float local_attention_weights[4][layout::kLocalWindow]{};
            alignas(16) float local_attention_denominators[4]{};
            int local_attention_count = 0;
#endif
            for (int head = 0; head < kHeads; ++head) {
                const int capacity = head_capacity(head);
                const int slot = static_cast<int>(tokens_seen_ % capacity);
                KeyCacheScalar* key = key_cache(layer_index, head, slot);
                const int qk_dimension = qk_head_dimension(head);
#if defined(TERNLOG_INT8_QK_CACHE_FIXED4) \
    || defined(TERNLOG_INT8_QK_CACHE_FIXED4_FP16)
                constexpr float qk_cache_scale = 4.0F / 127.0F;
#endif
#if defined(TERNLOG_SHARED_QK45)
                if (layer_index != 5) {
#endif
#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
                *key_scale_cache(layer_index, head, slot) = quantize_activation_int8(
                    qkv + layout::kHeads * kQkHeadDim + head * kQkHeadDim,
                    key, qk_dimension
                );
#elif defined(TERNLOG_INT8_QK_CACHE_FIXED4) \
    || defined(TERNLOG_INT8_QK_CACHE_FIXED4_FP16)
#if defined(TERNLOG_INT8_QK_CACHE_FIXED4_FP16)
                quantize_fixed_int8_fp16(
#else
                quantize_fixed_int8(
#endif
                    qkv + layout::kHeads * kQkHeadDim + head * kQkHeadDim,
                    key, qk_dimension, qk_cache_scale
                );
#else
                store_cache(
                    key,
                    qkv + layout::kHeads * kQkHeadDim + head * kQkHeadDim,
                    qk_dimension
                );
#endif
#if defined(TERNLOG_SHARED_QK45)
                }
#endif
#if defined(TERNLOG_SHARED_VALUE_HEAD)
                if (head == 0) {
#if defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
#if !defined(TERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR) \
    && !defined(TERNLOG_QKV_VALUE_CACHE_FUSED_NEON)
                    store_shared_value_cache_int8(
                        value_cache_, layer_index, 0,
                        static_cast<int>(tokens_seen_ % layout::kContext),
                        qkv + (layout::kHeads + layout::kKeyHeads) * kQkHeadDim,
                        *layer.value_cache_scale
                    );
#endif
#else
                    ValueCacheScalar* value = shared_value_cache(
                        value_cache_, layer_index, 0,
                        static_cast<int>(tokens_seen_ % layout::kContext)
                    );
                    store_cache(
                        value,
                        qkv + (layout::kHeads + layout::kKeyHeads) * kQkHeadDim,
                        kHeadDim
                    );
#endif
                }
#else
                ValueCacheScalar* value = value_cache(layer_index, head, slot);
                store_cache(
                    value,
                    qkv + (layout::kHeads + layout::kKeyHeads) * kQkHeadDim
                        + head * kHeadDim,
                    kHeadDim
                );
#endif
                const int count = static_cast<int>(std::min<std::uint64_t>(tokens_seen_ + 1, capacity));
                const int oldest = tokens_seen_ + 1 > static_cast<std::uint64_t>(capacity)
                    ? (slot + 1) % capacity
                    : 0;
                const float* query = qkv + head * kQkHeadDim;
                const float attention_scale = head < layout::kLocalHeads
                    ? layout::kLocalAttentionScale
                    : layout::kAttentionScale;
#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
                alignas(16) std::int8_t rounded_query[kQkHeadDim];
                const float rounded_query_scale = quantize_activation_int8(
                    query, rounded_query, qk_dimension
                );
#elif defined(TERNLOG_INT8_QK_CACHE_FIXED4) \
    || defined(TERNLOG_INT8_QK_CACHE_FIXED4_FP16)
                alignas(16) std::int8_t rounded_query[kQkHeadDim];
#if defined(TERNLOG_INT8_QK_CACHE_FIXED4_FP16)
                quantize_fixed_int8_fp16(
#else
                quantize_fixed_int8(
#endif
                    query, rounded_query, qk_dimension, qk_cache_scale
                );
#elif defined(TERNLOG_FP16_KEY_CACHE)
                alignas(16) Half rounded_query[kQkHeadDim];
                store_cache(rounded_query, query, qk_dimension);
#endif
                float maximum = -std::numeric_limits<float>::max();
#if defined(TERNLOG_INT8_QK_CACHE_FIXED4) \
    || defined(TERNLOG_INT8_QK_CACHE_FIXED4_FP16) \
    || defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
#if defined(TERNLOG_CONTIGUOUS_QK_RING)
                const KeyCacheScalar* head_keys = key_cache(
                    layer_index, head, 0
                );
                const int first_count = std::min(count, capacity - oldest);
                for (int item = 0; item < first_count; ++item) {
                    const int cached_slot = oldest + item;
                    const KeyCacheScalar* cached_key = head_keys
                        + static_cast<std::size_t>(cached_slot) * qk_dimension;
#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
                    const float key_scale = *key_scale_cache(
                        layer_index, head, cached_slot
                    );
                    scores[item] = static_cast<float>(dot(
                        rounded_query, cached_key, qk_dimension
                    )) * (rounded_query_scale * key_scale * attention_scale);
#else
                    scores[item] = static_cast<float>(dot(
                        rounded_query, cached_key, qk_dimension
                    )) * (qk_cache_scale * qk_cache_scale * attention_scale);
#endif
                    maximum = std::max(maximum, scores[item]);
                }
                for (int item = first_count; item < count; ++item) {
                    const int cached_slot = item - first_count;
                    const KeyCacheScalar* cached_key = head_keys
                        + static_cast<std::size_t>(cached_slot) * qk_dimension;
#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
                    const float key_scale = *key_scale_cache(
                        layer_index, head, cached_slot
                    );
                    scores[item] = static_cast<float>(dot(
                        rounded_query, cached_key, qk_dimension
                    )) * (rounded_query_scale * key_scale * attention_scale);
#else
                    scores[item] = static_cast<float>(dot(
                        rounded_query, cached_key, qk_dimension
                    )) * (qk_cache_scale * qk_cache_scale * attention_scale);
#endif
                    maximum = std::max(maximum, scores[item]);
                }
#else
                for (int item = 0; item < count; ++item) {
                    const int cached_slot = (oldest + item) % capacity;
                    const KeyCacheScalar* cached_key = key_cache(
                        layer_index, head, cached_slot
                    );
#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
                    const float key_scale = *key_scale_cache(
                        layer_index, head, cached_slot
                    );
                    scores[item] = static_cast<float>(dot(
                        rounded_query, cached_key, qk_dimension
                    )) * (rounded_query_scale * key_scale * attention_scale);
#else
                    scores[item] = static_cast<float>(dot(
                        rounded_query, cached_key, qk_dimension
                    )) * (qk_cache_scale * qk_cache_scale * attention_scale);
#endif
                    maximum = std::max(maximum, scores[item]);
                }
#endif
#elif defined(TERNLOG_ATTENTION_SCORE4) || defined(TERNLOG_ATTENTION_SCORE8)
#if defined(TERNLOG_ATTENTION_SCORE8)
                constexpr int score_group = 8;
#else
                constexpr int score_group = 4;
#endif
                int item = 0;
                for (; item + score_group <= count; item += score_group) {
                    float32x4_t sums[score_group];
#pragma clang loop unroll(full)
                    for (int key = 0; key < score_group; ++key) {
                        sums[key] = vdupq_n_f32(0.0f);
                    }
                    for (int index = 0; index < qk_dimension; index += 8) {
                        const float16x8_t query_value = vld1q_f16(
                            rounded_query + index
                        );
#pragma clang loop unroll(full)
                        for (int key = 0; key < score_group; ++key) {
                            const KeyCacheScalar* cached_key = key_cache(
                                layer_index, head, (oldest + item + key) % capacity
                            );
                            const float16x8_t key_value = vld1q_f16(
                                cached_key + index
                            );
                            sums[key] = vfmlalq_low_f16(
                                sums[key], query_value, key_value
                            );
                            sums[key] = vfmlalq_high_f16(
                                sums[key], query_value, key_value
                            );
                        }
                    }
#pragma clang loop unroll(full)
                    for (int key = 0; key < score_group; ++key) {
                        scores[item + key] = vaddvq_f32(sums[key])
                            * attention_scale;
                        maximum = std::max(maximum, scores[item + key]);
                    }
                }
                for (; item < count; ++item) {
                    const KeyCacheScalar* cached_key = key_cache(
                        layer_index, head, (oldest + item) % capacity
                    );
                    scores[item] = dot(rounded_query, cached_key, qk_dimension)
                        * attention_scale;
                    maximum = std::max(maximum, scores[item]);
                }
#else
                for (int item = 0; item < count; ++item) {
                    const KeyCacheScalar* cached_key = key_cache(
                        layer_index, head, (oldest + item) % capacity
                    );
#if defined(TERNLOG_FP16_KEY_CACHE)
                    scores[item] = dot(rounded_query, cached_key, qk_dimension) * attention_scale;
#else
                    scores[item] = dot(query, cached_key, qk_dimension) * attention_scale;
#endif
                    maximum = std::max(maximum, scores[item]);
                }
#endif
                float denominator = 0.0F;
#if defined(TERNLOG_FUSED_CLIPPED_ATTENTION)
#if defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
                alignas(16) std::int8_t value_weights[layout::kContext];
                clipped_linear_weights_int8(
                    scores, maximum, layer.attention_slopes[head],
                    value_weights, count
                );
                if (head < layout::kLocalHeads) {
                    std::memcpy(
                        local_value_weights[head], value_weights,
                        static_cast<std::size_t>(count)
                    );
                    local_value_count = count;
                } else {
                    accumulate_shared_value_cache_int8(
                        attention + head * kHeadDim,
                        value_cache_,
                        layer_index,
                        0,
                        static_cast<int>(
                            (tokens_seen_ + 1 - count) % layout::kContext
                        ),
                        value_weights,
                        count,
                        *layer.value_cache_scale
                    );
                }
#else
#if defined(TERNLOG_SHARED_LOCAL_VALUE4)
                if (head < layout::kLocalHeads) {
                    for (int item = 0; item < count; ++item) {
                        const float weight = clipped_linear_weight(
                            scores[item], maximum,
                            layer.attention_slopes[head]
                        );
                        denominator += weight;
                        local_attention_weights[head][item] = weight;
                    }
                    local_attention_denominators[head] = denominator;
                    local_attention_count = count;
                } else {
#endif
                float* output = attention + head * kHeadDim;
                std::fill(output, output + kHeadDim, 0.0F);
                for (int item = 0; item < count; ++item) {
                    const float weight = clipped_linear_weight(
                        scores[item], maximum, layer.attention_slopes[head]
                    );
                    denominator += weight;
                    const ValueCacheScalar* cached_value =
#if defined(TERNLOG_SHARED_VALUE_HEAD)
                        shared_value_cache(
                            value_cache_, layer_index, 0,
                            static_cast<int>((tokens_seen_ + 1 - count + item) % layout::kContext)
                        );
#else
                        value_cache(layer_index, head, (oldest + item) % capacity);
#endif
                    accumulate_weighted(output, cached_value, weight, kHeadDim);
                }
#if defined(TERNLOG_RECIPROCAL_ATTENTION)
                const float inverse = reciprocal_refined(
                    std::max(denominator, 1.0e-12F)
                );
#else
                const float inverse = 1.0F / std::max(denominator, 1.0e-12F);
#endif
                for (int index = 0; index < kHeadDim; index += 4) {
                    vst1q_f32(
                        output + index,
                        vmulq_n_f32(vld1q_f32(output + index), inverse)
                    );
                }
#if defined(TERNLOG_SHARED_LOCAL_VALUE4)
                }
#endif
#endif
#else
                for (int item = 0; item < count; ++item) {
                    scores[item] = std::max(
                        0.0F,
                        1.0F + layer.attention_slopes[head] * (scores[item] - maximum)
                    );
                    denominator += scores[item];
                }
                float* output = attention + head * kHeadDim;
                std::fill(output, output + kHeadDim, 0.0F);
                const float inverse = 1.0F / std::max(denominator, 1.0e-12F);
                for (int item = 0; item < count; ++item) {
                    const float probability = scores[item] * inverse;
                    const ValueCacheScalar* cached_value =
#if defined(TERNLOG_SHARED_VALUE_HEAD)
                        shared_value_cache(
                            value_cache_, layer_index, 0,
                            static_cast<int>((tokens_seen_ + 1 - count + item) % layout::kContext)
                        );
#else
                        value_cache(layer_index, head, (oldest + item) % capacity);
#endif
                    accumulate_weighted(output, cached_value, probability, kHeadDim);
                }
#endif
            }
#if defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
            accumulate_shared_local_value_cache_int8(
                attention,
                value_cache_,
                layer_index,
                static_cast<int>(
                    (tokens_seen_ + 1 - local_value_count) % layout::kContext
                ),
                local_value_weights,
                local_value_count,
                *layer.value_cache_scale
            );
#endif
#if defined(TERNLOG_SHARED_LOCAL_VALUE4) \
    && !defined(TERNLOG_INT8_VALUE_CACHE_SCALAR)
            alignas(64) const Half* local_values[layout::kLocalWindow];
            for (int item = 0; item < local_attention_count; ++item) {
                local_values[item] = shared_value_cache(
                    value_cache_, layer_index, 0,
                    static_cast<int>(
                        (tokens_seen_ + 1 - local_attention_count + item)
                        % layout::kContext
                    )
                );
            }
            accumulate_shared_local_values4(
                attention,
                local_values,
                local_attention_weights,
                local_attention_count
            );
            for (int head = 0; head < layout::kLocalHeads; ++head) {
                float* output = attention + head * kHeadDim;
#if defined(TERNLOG_RECIPROCAL_ATTENTION)
                const float inverse = reciprocal_refined(std::max(
                    local_attention_denominators[head], 1.0e-12F
                ));
#else
                const float inverse = 1.0F / std::max(
                    local_attention_denominators[head], 1.0e-12F
                );
#endif
                for (int index = 0; index < kHeadDim; index += 4) {
                    vst1q_f32(
                        output + index,
                        vmulq_n_f32(vld1q_f32(output + index), inverse)
                    );
                }
            }
#endif
#if defined(TERNLOG_PROFILE_PHASES)
            profile_mark(ProfilePhase::attention_kv);
#endif
#if defined(TERNLOG_OUTPUT_HEADLOCAL_RANK8)
            matrix_vector_int8_output_headlocal_rank8(
                layer.attention_output_int8,
                layer.attention_output_scales,
                layer.attention_output_ticket_columns,
                attention,
                projected
            );
#elif defined(TERNLOG_OUTPUT_CODEBOOK32)
            matrix_vector_int8_output_codebook32(
                layer.attention_output_int8,
                layer.attention_output_scales,
                layer.attention_output_ticket_columns,
                attention,
                projected
            );
#elif defined(TERNLOG_BOOLEAN_OUTPUT_TICKETS8)
            matrix_vector_int8_output_tickets<8>(
                layer.attention_output_int8,
                layer.attention_output_scales,
                layer.attention_output_ticket_offsets,
                layer.attention_output_ticket_columns,
                attention,
                projected
            );
#elif defined(TERNLOG_BOOLEAN_OUTPUT_TICKETS16)
#if defined(TERNLOG_OUTPUT_RESIDUAL_RMS_FUSED)
            fused_attention_square_sum =
                matrix_vector_int8_output_tickets_residual<16, true>(
                    layer.attention_output_int8,
                    layer.attention_output_scales,
                    layer.attention_output_ticket_offsets,
                    layer.attention_output_ticket_columns,
                    attention,
                    state,
                    layer.residual_scales[0],
                    layer.residual_scales[1]
                );
#elif defined(TERNLOG_OUTPUT_RESIDUAL_FUSED)
            matrix_vector_int8_output_tickets_residual<16, false>(
                layer.attention_output_int8,
                layer.attention_output_scales,
                layer.attention_output_ticket_offsets,
                layer.attention_output_ticket_columns,
                attention,
                state,
                layer.residual_scales[0],
                layer.residual_scales[1]
            );
#else
            matrix_vector_int8_output_tickets<16>(
                layer.attention_output_int8,
                layer.attention_output_scales,
                layer.attention_output_ticket_offsets,
                layer.attention_output_ticket_columns,
                attention,
                projected
            );
#endif
#elif defined(TERNLOG_BOOLEAN_OUTPUT_BLOCK_INT8)
            matrix_vector_int8_block16(
                layer.attention_output_int8,
                layer.attention_output_scales,
                layer.attention_output_block_mask,
                attention,
                projected
            );
#elif defined(TERNLOG_BOOLEAN_OUTPUT_INT8)
            matrix_vector_int8(
                layer.attention_output_int8,
                layer.attention_output_scales,
                attention,
                projected,
                kDim,
                kDim
            );
#else
            matrix_vector(
                layer.attention_output,
#if defined(TERNLOG_BOOLEAN_AX1)
                nullptr,
                nullptr,
#else
                layer.attention_output_int8,
                layer.attention_output_scale,
#endif
                attention,
                projected,
                kDim,
                kDim
            );
#endif
#if !defined(TERNLOG_OUTPUT_RESIDUAL_FUSED) \
    && !defined(TERNLOG_OUTPUT_RESIDUAL_RMS_FUSED)
            residual_update(
                state,
                projected,
                layer.residual_scales[0],
                layer.residual_scales[1]
            );
#endif
            } else {
                for (int index = 0; index < kDim; index += 4) {
                    vst1q_f32(
                        state + index,
                        vmulq_n_f32(
                            vld1q_f32(state + index),
                            layer.residual_scales[0]
                        )
                    );
                }
            }
#if defined(TERNLOG_PROFILE_PHASES)
            profile_mark(ProfilePhase::attention_output);
#endif
#if defined(TERNLOG_FFN_RMS_ENCODE_FUSED)
            const float ffn_inverse_rms =
#if defined(TERNLOG_OUTPUT_RESIDUAL_RMS_FUSED)
                (layout::kAttentionSkipMask & (1ULL << layer_index)) == 0ULL
                ? inverse_square_root(
                    fused_attention_square_sum / kDim + kRmsEpsilon
                )
                : rms_inverse(state, kDim, kRmsEpsilon);
#else
                rms_inverse(state, kDim, kRmsEpsilon);
#endif
#elif defined(TERNLOG_OUTPUT_RESIDUAL_RMS_FUSED)
            if ((layout::kAttentionSkipMask & (1ULL << layer_index)) == 0ULL) {
                rms_norm_from_square_sum(
                    state, layer.feedforward_norm, normalized, kDim,
                    kRmsEpsilon, fused_attention_square_sum
                );
            } else {
                rms_norm(
                    state, layer.feedforward_norm, normalized, kDim, kRmsEpsilon
                );
            }
#else
            rms_norm(state, layer.feedforward_norm, normalized, kDim, kRmsEpsilon);
#endif
#if defined(TERNLOG_BOOLEAN_AX1)
            ternlog_vocab_boolean_ffn_ax1(
                projected,
#if defined(TERNLOG_FFN_RMS_ENCODE_FUSED)
                state,
#else
                normalized,
#endif
                layer.encoder_offsets,
                layer.boolean_program,
                layer.decoder_weight,
                layer.decoder_int8,
                layer.decoder_scales,
                layer.decoder_bias,
                layer.correction_int8,
                layer.correction_scales,
                layer.correction_ticket_offsets,
                layer.correction_ticket_columns,
                layer.correction_output_f16,
#if defined(TERNLOG_FFN_RMS_ENCODE_FUSED)
                layer.feedforward_norm,
                ffn_inverse_rms
#else
                nullptr,
                1.0F
#endif
            );
#else
            matrix_vector(
                layer.ff_up, layer.ff_up_int8, layer.ff_up_scale,
                normalized, hidden, layer.ff_dimension, kDim
            );
            for (int index = 0; index < layer.ff_dimension; index += 4) {
                const float32x4_t value = vmaxq_f32(
                    vdupq_n_f32(0.0F), vld1q_f32(hidden + index)
                );
                vst1q_f32(hidden + index, vmulq_f32(value, value));
            }
            matrix_vector(
                layer.ff_down, layer.ff_down_int8, layer.ff_down_scale,
                hidden, projected, kDim, layer.ff_dimension
            );
#endif
            residual_update(
                state,
                projected,
                layer.residual_scales[2],
                layer.residual_scales[3]
            );
#if defined(TERNLOG_PROFILE_PHASES)
            profile_mark(ProfilePhase::boolean_ffn);
#endif
        }
#if defined(TERNLOG_PROFILE_PHASES)
        profile_restart();
#endif
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
        if (selected_layer >= 0) std::memcpy(tile_state, state, sizeof(state));
        if (selected_layer < 0 || selected_layer == layout::kLayers - 1) {
#endif
        rms_norm(state, weights_.final_norm, normalized, kDim, kRmsEpsilon);
#if defined(TERNLOG_BOOLEAN_AX1)
#if defined(TERNLOG_BOOLEAN_VOCAB_HIERARCHY32X64)
        ternlog_vocab_boolean_hierarchy64_factor32_prepare(
            hierarchy_cluster_logits_.data(),
            hierarchy_query_.data(),
            hierarchy_hidden_.data(),
            normalized,
            weights_.hierarchy_cluster_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.hierarchy_cluster_output,
            weights_.hierarchy_cluster_bias
        );
        if (logits != nullptr) materialize_hierarchy_logits(logits);
#elif defined(TERNLOG_BOOLEAN_READOUT_FACTOR48_W8A8_GROUP16)
        ternlog_vocab_boolean_hybrid64_factor48_w8a8_group16_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_correction_output_int8,
            weights_.readout_correction_output_scales,
            weights_.readout_bias
        );
#elif defined(TERNLOG_BOOLEAN_READOUT_FACTOR48_W8A8)
        ternlog_vocab_boolean_hybrid64_factor48_w8a8_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_correction_output_int8,
            weights_.readout_correction_output_scales,
            weights_.readout_bias
        );
#elif defined(TERNLOG_BOOLEAN_READOUT_FACTOR32_W8A8_GROUP16)
        ternlog_vocab_boolean_hybrid64_factor32_w8a8_group16_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_correction_output_int8,
            weights_.readout_correction_output_scales,
            weights_.readout_bias
        );
#elif defined(TERNLOG_BOOLEAN_READOUT_FACTOR32_W8A8)
        ternlog_vocab_boolean_hybrid64_factor32_w8a8_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_correction_output_int8,
            weights_.readout_correction_output_scales,
            weights_.readout_bias
        );
#elif defined(TERNLOG_BOOLEAN_READOUT_FACTORIZED_NATIVE)
        ternlog_vocab_boolean_hybrid64_factor_native_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_correction_output,
            weights_.readout_bias
        );
#elif defined(TERNLOG_BOOLEAN_READOUT_FACTOR64)
        ternlog_vocab_boolean_hybrid64_factor64_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_correction_output,
            weights_.readout_bias
        );
#elif defined(TERNLOG_BOOLEAN_READOUT_FACTOR32)
        ternlog_vocab_boolean_hybrid64_factor32_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_correction_output,
            weights_.readout_bias
        );
#else
        ternlog_vocab_boolean_hybrid64_thirdscale_2048(
            logits,
            normalized,
            weights_.token_codes,
            weights_.readout_correction,
            weights_.readout_correction_scales,
            weights_.readout_bias
        );
#endif
#else
        matrix_vector(
            weights_.token_embedding,
            weights_.token_embedding_int8,
            weights_.token_embedding_scale,
            normalized,
            logits,
            layout::kVocabulary,
            kDim
        );
#endif
#if defined(TERNLOG_PROFILE_PHASES)
        profile_mark(ProfilePhase::final_readout);
#endif
#if defined(TERNLOG_ENCODER_LAYERWISE_PROBE)
        }
#endif
        previous_token_ = token;
        byte_history_[static_cast<std::size_t>(tokens_seen_) & 15] = token;
        ++tokens_seen_;
#if defined(TERNLOG_RECURRENT_ROPE)
        advance_rope();
#endif
    }

  private:
#if defined(TERNLOG_PROFILE_PHASES)
    enum class ProfilePhase : std::size_t {
        embedding_norm,
        qkv,
        attention_kv,
        attention_output,
        boolean_ffn,
        final_readout,
        count,
    };

    void profile_restart() {
        profile_started_ = std::chrono::steady_clock::now();
    }

    void profile_mark(ProfilePhase phase) {
        const auto stopped = std::chrono::steady_clock::now();
        const std::size_t index = static_cast<std::size_t>(phase);
        profile_nanoseconds_[index] += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stopped - profile_started_
            ).count()
        );
        ++profile_calls_[index];
        profile_started_ = stopped;
    }
#endif

    static int head_capacity(int head) {
        return head < layout::kLocalHeads ? layout::kLocalWindow : layout::kContext;
    }

    static int qk_head_dimension(int head) {
        return head < layout::kLocalHeads ? kLocalQkHeadDim : kQkHeadDim;
    }

    static std::size_t head_slot_offset(int head) {
        if (head < layout::kLocalHeads) {
            return static_cast<std::size_t>(head) * layout::kLocalWindow;
        }
        return static_cast<std::size_t>(layout::kLocalHeads) * layout::kLocalWindow
            + static_cast<std::size_t>(head - layout::kLocalHeads) * layout::kContext;
    }

    static std::size_t cache_slots_per_layer() {
        return (
            static_cast<std::size_t>(layout::kLocalHeads) * layout::kLocalWindow
            + static_cast<std::size_t>(kHeads - layout::kLocalHeads) * layout::kContext
        );
    }

    static std::size_t key_cache_values_per_layer() {
        return static_cast<std::size_t>(layout::kLocalHeads)
                * layout::kLocalWindow * kLocalQkHeadDim
            + static_cast<std::size_t>(kHeads - layout::kLocalHeads)
                * layout::kContext * kQkHeadDim;
    }

    static std::size_t value_cache_values_per_layer() {
        return cache_slots_per_layer() * kHeadDim;
    }

    KeyCacheScalar* key_cache(int layer, int head, int slot) {
#if defined(TERNLOG_SHARED_QK45)
        if (layer == 5) {
            layer = 4;
        }
#endif
        const std::size_t head_offset = head < layout::kLocalHeads
            ? static_cast<std::size_t>(head) * layout::kLocalWindow
                * kLocalQkHeadDim
            : static_cast<std::size_t>(layout::kLocalHeads)
                    * layout::kLocalWindow * kLocalQkHeadDim
                + static_cast<std::size_t>(head - layout::kLocalHeads)
                    * layout::kContext * kQkHeadDim;
        return key_cache_.data()
            + static_cast<std::size_t>(layer) * key_cache_values_per_layer()
            + head_offset + static_cast<std::size_t>(slot) * qk_head_dimension(head);
    }

#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
    float* key_scale_cache(int layer, int head, int slot) {
        return key_scale_cache_.data()
            + static_cast<std::size_t>(layer) * cache_slots_per_layer()
            + head_slot_offset(head) + slot;
    }
#endif

    ValueCacheScalar* value_cache(int layer, int head, int slot) {
        return value_cache_.data()
            + static_cast<std::size_t>(layer) * value_cache_values_per_layer()
            + (head_slot_offset(head) + slot) * kHeadDim;
    }

    template <typename Scalar>
    static Scalar* shared_value_cache(
        std::vector<Scalar>& values,
        int layer,
        int value_head,
        int slot
    ) {
        return values.data()
            + (
                (static_cast<std::size_t>(layer) * layout::kValueHeads + value_head)
                * layout::kContext + slot
            ) * kHeadDim;
    }

    void apply_rope(
        float* state, std::uint64_t position, int qk_dimension
    ) const {
        const bool local = qk_dimension == kLocalQkHeadDim
            && kLocalQkHeadDim != kQkHeadDim;
        const int pairs = qk_dimension * kRopePairs / kQkHeadDim;
        for (int index = 0; index < pairs; ++index) {
            const float left = state[index];
#if defined(TERNLOG_SHARED_ROPE)
            const float cosine = local ? local_rope_cosine_[index] : rope_cosine_[index];
            const float sine = local ? local_rope_sine_[index] : rope_sine_[index];
#else
            const float frequency = std::exp(
                -std::log(10000.0F) * index / pairs
            );
            const float angle = static_cast<float>(position) * frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
#endif
            const float right = state[index + qk_dimension / 2];
            state[index] = left * cosine - right * sine;
            state[index + qk_dimension / 2] = right * cosine + left * sine;
        }
    }

#if defined(TERNLOG_SHARED_ROPE)
    void set_rope_position(std::uint64_t position) {
        for (int index = 0; index < kRopePairs; ++index) {
            const float angle = static_cast<float>(position) * rope_frequency_[index];
            rope_cosine_[index] = std::cos(angle);
            rope_sine_[index] = std::sin(angle);
        }
        constexpr int local_pairs = kLocalQkHeadDim * kRopePairs / kQkHeadDim;
        for (int index = 0; index < local_pairs; ++index) {
            const float angle = static_cast<float>(position)
                * local_rope_frequency_[index];
            local_rope_cosine_[index] = std::cos(angle);
            local_rope_sine_[index] = std::sin(angle);
        }
    }

    void initialize_rope() {
        for (int index = 0; index < kRopePairs; ++index) {
            rope_frequency_[index] = std::exp(
                -std::log(10000.0F) * index / kRopePairs
            );
            rope_step_cosine_[index] = std::cos(rope_frequency_[index]);
            rope_step_sine_[index] = std::sin(rope_frequency_[index]);
        }
        constexpr int local_pairs = kLocalQkHeadDim * kRopePairs / kQkHeadDim;
        for (int index = 0; index < local_pairs; ++index) {
            local_rope_frequency_[index] = std::exp(
                -std::log(10000.0F) * index / local_pairs
            );
            local_rope_step_cosine_[index] = std::cos(local_rope_frequency_[index]);
            local_rope_step_sine_[index] = std::sin(local_rope_frequency_[index]);
        }
        set_rope_position(0);
    }

#if defined(TERNLOG_RECURRENT_ROPE)
    void advance_rope() {
        if ((tokens_seen_ & 4095U) == 0) {
            set_rope_position(tokens_seen_);
            return;
        }
        for (int index = 0; index < kRopePairs; ++index) {
            const float cosine = rope_cosine_[index];
            const float sine = rope_sine_[index];
            rope_cosine_[index] =
                cosine * rope_step_cosine_[index] - sine * rope_step_sine_[index];
            rope_sine_[index] =
                sine * rope_step_cosine_[index] + cosine * rope_step_sine_[index];
        }
        constexpr int local_pairs = kLocalQkHeadDim * kRopePairs / kQkHeadDim;
        for (int index = 0; index < local_pairs; ++index) {
            const float cosine = local_rope_cosine_[index];
            const float sine = local_rope_sine_[index];
            local_rope_cosine_[index] = cosine * local_rope_step_cosine_[index]
                - sine * local_rope_step_sine_[index];
            local_rope_sine_[index] = sine * local_rope_step_cosine_[index]
                + cosine * local_rope_step_sine_[index];
        }
    }
#endif
#endif

    static void residual_update(
        float* state,
        const float* branch,
        float state_scale,
        float branch_scale
    ) {
        for (int index = 0; index < kDim; index += 4) {
            vst1q_f32(
                state + index,
                vfmaq_n_f32(
                    vmulq_n_f32(vld1q_f32(state + index), state_scale),
                    vld1q_f32(branch + index),
                    branch_scale
                )
            );
        }
    }

#if !defined(TERNLOG_BOOLEAN_AX1)
    void apply_dynamic_route(int layer, float* state, const float* input_state) const {
        int route = -1;
        if (layer == 4) route = 0;
        if (layer == 7) route = 1;
        if (layer == 9) route = 2;
        if (route < 0) return;
        float32x4_t sum = vdupq_n_f32(0.0F);
        for (int index = 0; index < 16; index += 4) {
            sum = vaddq_f32(sum, vld1q_f32(state + index));
        }
        const float signal = horizontal_sum(sum) * (1.0F / 16.0F);
        const float gate = std::clamp(
            0.5F + weights_.route_scale[route] * signal + weights_.route_bias[route],
            0.0F,
            1.0F
        );
        const float strength = weights_.route_strength[route] * gate;
        for (int index = 0; index < kDim; index += 4) {
            vst1q_f32(
                state + index,
                vfmaq_n_f32(
                    vld1q_f32(state + index),
                    vld1q_f32(input_state + index),
                    strength
                )
            );
        }
    }
#endif

    Weights weights_;
    std::vector<KeyCacheScalar> key_cache_;
#if defined(TERNLOG_INT8_QK_CACHE_DYNAMIC)
    std::vector<float> key_scale_cache_;
#endif
    std::vector<ValueCacheScalar> value_cache_;
#if defined(TERNLOG_PROFILE_PHASES)
    std::array<std::uint64_t, static_cast<std::size_t>(ProfilePhase::count)>
        profile_nanoseconds_{};
    std::array<std::uint64_t, static_cast<std::size_t>(ProfilePhase::count)>
        profile_calls_{};
    std::chrono::steady_clock::time_point profile_started_{};
#endif
#if defined(TERNLOG_SHARED_ROPE)
    std::array<float, kQkHeadDim / 2> rope_frequency_{};
    std::array<float, kQkHeadDim / 2> rope_step_cosine_{};
    std::array<float, kQkHeadDim / 2> rope_step_sine_{};
    std::array<float, kQkHeadDim / 2> rope_cosine_{};
    std::array<float, kQkHeadDim / 2> rope_sine_{};
    std::array<float, kQkHeadDim / 2> local_rope_frequency_{};
    std::array<float, kQkHeadDim / 2> local_rope_step_cosine_{};
    std::array<float, kQkHeadDim / 2> local_rope_step_sine_{};
    std::array<float, kQkHeadDim / 2> local_rope_cosine_{};
    std::array<float, kQkHeadDim / 2> local_rope_sine_{};
#endif
    std::uint64_t tokens_seen_ = 0;
    std::uint16_t previous_token_ = 0;
    std::array<std::uint16_t, 16> byte_history_{};
#if defined(TERNLOG_BOOLEAN_VOCAB_HIERARCHY32X64)
    alignas(64) std::array<float, 32> hierarchy_cluster_logits_{};
    alignas(64) std::array<std::uint8_t, 8> hierarchy_query_{};
    alignas(64) std::array<Half, 32> hierarchy_hidden_{};
#endif
};

class SplitMix64 {
  public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        std::uint64_t value = (state_ += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }
    double uniform() {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }
  private:
    std::uint64_t state_;
};

int sample_categorical_logits(
    const float* logits,
    int count,
    SplitMix64& random
) {
    float maximum = logits[0];
    for (int index = 1; index < count; ++index) {
        maximum = std::max(maximum, logits[index]);
    }
    std::array<double, layout::kVocabulary> probabilities{};
    double total = 0.0;
    for (int index = 0; index < count; ++index) {
        probabilities[index] = std::exp(
            static_cast<double>(logits[index] - maximum)
        );
        total += probabilities[index];
    }
    double target = random.uniform() * total;
    for (int index = 0; index < count; ++index) {
        target -= probabilities[index];
        if (target <= 0.0) return index;
    }
    return count - 1;
}

std::uint16_t sample_ancestral(float* logits, SplitMix64& random) {
    return static_cast<std::uint16_t>(sample_categorical_logits(
        logits, layout::kVocabulary, random
    ));
}

#if defined(TERNLOG_BOOLEAN_VOCAB_HIERARCHY32X64)
std::uint16_t sample_hierarchy_ancestral(
    Transformer& model,
    SplitMix64& random
) {
    const int cluster = sample_categorical_logits(
        model.hierarchy_cluster_logits(), 32, random
    );
    alignas(64) float leaf_logits[64];
    model.hierarchy_leaf_logits(cluster, leaf_logits);
    const int leaf = sample_categorical_logits(leaf_logits, 64, random);
    return model.hierarchy_token(cluster, leaf);
}
#endif

std::uint16_t sample(float* logits, SplitMix64& random, bool greedy) {
#if defined(TERNLOG_GREEDY_SCALAR_ARGMAX)
    if (greedy) {
        float maximum = logits[0];
        std::uint16_t best_token = 0;
        for (int token = 1; token < layout::kVocabulary; ++token) {
            if (logits[token] > maximum) {
                maximum = logits[token];
                best_token = static_cast<std::uint16_t>(token);
            }
        }
        return best_token;
    }
#elif defined(TERNLOG_GREEDY_NEON_ARGMAX)
    if (greedy) {
        float32x4_t maximum = vld1q_f32(logits);
        for (int token = 4; token < layout::kVocabulary; token += 4) {
            maximum = vmaxq_f32(maximum, vld1q_f32(logits + token));
        }
        const float scalar_maximum = vmaxvq_f32(maximum);
        for (int token = 0; token < layout::kVocabulary; ++token) {
            if (logits[token] == scalar_maximum) {
                return static_cast<std::uint16_t>(token);
            }
        }
    }
#endif
    struct Candidate { float logit; std::uint16_t token; };
    std::array<Candidate, layout::kVocabulary> candidates{};
    for (int token = 0; token < layout::kVocabulary; ++token) {
        candidates[token] = {logits[token], static_cast<std::uint16_t>(token)};
    }
    constexpr int kTopK = 40;
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + kTopK,
        candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            return left.logit > right.logit;
        }
    );
    if (greedy) return candidates[0].token;
    const float maximum = candidates[0].logit;
    std::array<float, kTopK> probabilities{};
    double total = 0.0;
#if defined(TERNLOG_SAMPLE_EXP2_CUBIC) || defined(TERNLOG_SAMPLE_EXP2_QUARTIC)
    for (int index = 0; index < kTopK; ++index) {
        probabilities[index] = (candidates[index].logit - maximum) / 0.7F;
    }
    float32x4_t vector_total = vdupq_n_f32(0.0F);
    for (int index = 0; index < kTopK; index += 4) {
        const float32x4_t exponent = vmaxq_f32(
            vdupq_n_f32(-16.0F),
            vminq_f32(vdupq_n_f32(0.0F), vld1q_f32(probabilities.data() + index))
        );
        const float32x4_t base2 = vmulq_n_f32(exponent, 1.4426950408889634F);
        const int32x4_t integer = vcvtmq_s32_f32(base2);
        const float32x4_t fraction = vsubq_f32(base2, vcvtq_f32_s32(integer));
        const float32x4_t power = vreinterpretq_f32_s32(
            vshlq_n_s32(vaddq_s32(integer, vdupq_n_s32(127)), 23)
        );
#if defined(TERNLOG_SAMPLE_EXP2_QUARTIC)
        float32x4_t polynomial = vfmaq_n_f32(
            vdupq_n_f32(0.05550410866482158F), fraction, 0.009618129107628477F
        );
#else
        float32x4_t polynomial = vdupq_n_f32(0.05550410866482158F);
#endif
        polynomial = vfmaq_f32(
            vdupq_n_f32(0.2402265069591007F), fraction, polynomial
        );
        polynomial = vfmaq_f32(
            vdupq_n_f32(0.6931471805599453F), fraction, polynomial
        );
        polynomial = vfmaq_f32(vdupq_n_f32(1.0F), fraction, polynomial);
        const float32x4_t values = vmulq_f32(power, polynomial);
        vst1q_f32(probabilities.data() + index, values);
        vector_total = vaddq_f32(vector_total, values);
    }
    total = horizontal_sum(vector_total);
#else
    for (int index = 0; index < kTopK; ++index) {
        probabilities[index] = std::exp((candidates[index].logit - maximum) / 0.7F);
        total += probabilities[index];
    }
#endif
    double target = random.uniform() * total;
    for (int index = 0; index < kTopK; ++index) {
        target -= probabilities[index];
        if (target <= 0.0) return candidates[index].token;
    }
    return candidates[kTopK - 1].token;
}

bool parse_u64(std::string_view text, std::uint64_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

std::vector<std::uint16_t> parse_tokens(std::string_view text) {
    std::vector<std::uint16_t> result;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        const std::string_view item = text.substr(0, comma);
        std::uint64_t value = 0;
        if (!parse_u64(item, value) || value >= layout::kVocabulary) {
            std::fputs("--verify-tokens requires comma-separated in-range IDs\n", stderr);
            std::exit(2);
        }
        result.push_back(static_cast<std::uint16_t>(value));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return result;
}

void usage(const char* executable) {
    std::fprintf(
        stderr,
        "Usage: %s [--prompt TEXT] [--max-bytes N] [--seed N] [--greedy]\n"
        "       %s [--prompt TEXT] --benchmark-tokens N [--benchmark-sampling|--benchmark-ancestral]\n"
        "       %s --tokenize TEXT\n"
        "       %s --decode-tokens ID,ID,...\n"
        "       %s --evaluate-token-file PATH [--evaluation-windows N]\n"
        "       %s --verify-tokens ID,ID,...\n",
        executable,
        executable,
        executable,
        executable,
        executable,
        executable
    );
}

}  // namespace

#if !defined(TERNLOG_STORY_NO_MAIN)
int main(int argc, char** argv) {
    std::string prompt(kDefaultPrompt);
    std::uint64_t maximum_bytes = 256;
    std::uint64_t benchmark_tokens = 0;
    bool benchmark_sampling = false;
    bool benchmark_ancestral = false;
    std::uint64_t seed = 20260824;
    bool greedy = false;
    std::string tokenize_text;
    bool tokenize_only = false;
    std::string decode_tokens;
    std::string evaluation_token_file;
    std::uint64_t evaluation_windows = 512;
    std::string verify_tokens;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            return 0;
        } else if (argument == "--prompt" && index + 1 < argc) {
            prompt = argv[++index];
        } else if (argument == "--max-bytes" && index + 1 < argc) {
            if (!parse_u64(argv[++index], maximum_bytes) || maximum_bytes == 0) return 2;
        } else if (argument == "--benchmark-tokens" && index + 1 < argc) {
            if (!parse_u64(argv[++index], benchmark_tokens) || benchmark_tokens == 0) return 2;
        } else if (argument == "--benchmark-sampling") {
            benchmark_sampling = true;
        } else if (argument == "--benchmark-ancestral") {
            benchmark_ancestral = true;
        } else if (argument == "--seed" && index + 1 < argc) {
            if (!parse_u64(argv[++index], seed)) return 2;
        } else if (argument == "--greedy") {
            greedy = true;
        } else if (argument == "--tokenize" && index + 1 < argc) {
            tokenize_text = argv[++index];
            tokenize_only = true;
        } else if (argument == "--decode-tokens" && index + 1 < argc) {
            decode_tokens = argv[++index];
        } else if (argument == "--evaluate-token-file" && index + 1 < argc) {
            evaluation_token_file = argv[++index];
        } else if (argument == "--evaluation-windows" && index + 1 < argc) {
            if (!parse_u64(argv[++index], evaluation_windows) || evaluation_windows == 0) return 2;
        } else if (argument == "--verify-tokens" && index + 1 < argc) {
            verify_tokens = argv[++index];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (benchmark_sampling && benchmark_ancestral) {
        std::fprintf(
            stderr,
            "--benchmark-sampling and --benchmark-ancestral are mutually exclusive\n"
        );
        return 2;
    }

    Tokenizer tokenizer;
    if (tokenize_only) {
        const std::vector<std::uint16_t> tokens = tokenizer.encode(tokenize_text);
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (index != 0) std::fputc(',', stdout);
            std::fprintf(stdout, "%u", static_cast<unsigned>(tokens[index]));
        }
        std::fputc('\n', stdout);
        return 0;
    }
    if (!decode_tokens.empty()) {
        for (std::uint16_t token : parse_tokens(decode_tokens)) {
            const std::string_view bytes = tokenizer.decode(token);
            if (std::fwrite(bytes.data(), 1, bytes.size(), stdout) != bytes.size()) {
                return 1;
            }
        }
        return 0;
    }
    Transformer model;
    alignas(64) float logits[layout::kVocabulary];
    if (!evaluation_token_file.empty()) {
        std::FILE* handle = std::fopen(evaluation_token_file.c_str(), "rb");
        if (handle == nullptr) {
            std::perror("unable to open evaluation token file");
            return 2;
        }
        if (std::fseek(handle, 0, SEEK_END) != 0) return 2;
        const long file_bytes = std::ftell(handle);
        if (file_bytes < 0 || std::fseek(handle, 0, SEEK_SET) != 0) return 2;
        std::vector<std::uint16_t> values(
            static_cast<std::size_t>(file_bytes) / sizeof(std::uint16_t)
        );
        if (std::fread(values.data(), sizeof(std::uint16_t), values.size(), handle)
            != values.size()) {
            std::fclose(handle);
            return 2;
        }
        std::fclose(handle);
        const std::uint64_t available_windows = values.size() > layout::kContext
            ? (values.size() - layout::kContext - 1) / layout::kContext + 1
            : 0;
        evaluation_windows = std::min(evaluation_windows, available_windows);
        double negative_log_likelihood = 0.0;
        std::uint64_t decoded_bytes = 0;
        std::uint64_t evaluated_tokens = 0;
        const auto evaluation_started = std::chrono::steady_clock::now();
        for (std::uint64_t window = 0; window < evaluation_windows; ++window) {
            model.reset();
            const std::size_t start = window * layout::kContext;
            for (int position = 0; position < layout::kContext; ++position) {
                model.forward(values[start + position], logits);
                const std::uint16_t target = values[start + position + 1];
                float maximum = logits[0];
                for (int token = 1; token < layout::kVocabulary; ++token) {
                    maximum = std::max(maximum, logits[token]);
                }
                double denominator = 0.0;
                for (int token = 0; token < layout::kVocabulary; ++token) {
                    denominator += std::exp(
                        static_cast<double>(logits[token] - maximum)
                    );
                }
                negative_log_likelihood += std::log(denominator)
                    + maximum - logits[target];
                decoded_bytes += tokenizer.decode(target).size();
                ++evaluated_tokens;
            }
        }
        const double evaluation_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - evaluation_started
        ).count();
        std::fprintf(
            stdout,
            "{\"windows\":%llu,\"tokens\":%llu,\"decoded_bytes\":%llu,"
            "\"loss_per_token\":%.9f,\"bpb\":%.9f,\"seconds\":%.6f,"
            "\"tokens_per_second\":%.1f}\n",
            static_cast<unsigned long long>(evaluation_windows),
            static_cast<unsigned long long>(evaluated_tokens),
            static_cast<unsigned long long>(decoded_bytes),
            negative_log_likelihood / evaluated_tokens,
            negative_log_likelihood / decoded_bytes / std::log(2.0),
            evaluation_seconds,
            evaluated_tokens / evaluation_seconds
        );
        return 0;
    }
    if (!verify_tokens.empty()) {
        for (std::uint16_t token : parse_tokens(verify_tokens)) {
            model.forward(token, logits);
        }
        return std::fwrite(logits, sizeof(float), layout::kVocabulary, stdout)
                == layout::kVocabulary
            ? 0
            : 1;
    }

    const std::vector<std::uint16_t> prompt_tokens = tokenizer.encode(prompt);
    const auto started = std::chrono::steady_clock::now();
    for (std::uint16_t token : prompt_tokens) {
        model.forward(token, logits);
    }
    SplitMix64 random(seed);
    if (benchmark_tokens != 0) {
        std::uint64_t decoded_bytes = 0;
#if defined(TERNLOG_PROFILE_PHASES)
        model.reset_profile();
#endif
        const auto benchmark_started = std::chrono::steady_clock::now();
        for (std::uint64_t index = 0; index < benchmark_tokens; ++index) {
            const std::uint16_t token =
#if defined(TERNLOG_BOOLEAN_VOCAB_HIERARCHY32X64)
                benchmark_ancestral
                    ? sample_hierarchy_ancestral(model, random)
                    : sample(logits, random, !benchmark_sampling);
#else
                benchmark_ancestral
                    ? sample_ancestral(logits, random)
                    : sample(logits, random, !benchmark_sampling);
#endif
            decoded_bytes += tokenizer.decode(token).size();
#if defined(TERNLOG_BOOLEAN_VOCAB_HIERARCHY32X64)
            if (benchmark_ancestral) {
                model.forward_hierarchy_fast(token);
            } else {
                model.forward(token, logits);
            }
#else
            model.forward(token, logits);
#endif
        }
        const double benchmark_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - benchmark_started
        ).count();
        std::fprintf(
            stdout,
            "tokens=%llu bytes=%llu seconds=%.6f tokens_per_second=%.1f chars_per_second=%.1f\n",
            static_cast<unsigned long long>(benchmark_tokens),
            static_cast<unsigned long long>(decoded_bytes),
            benchmark_elapsed,
            benchmark_tokens / benchmark_elapsed,
            decoded_bytes / benchmark_elapsed
        );
#if defined(TERNLOG_PROFILE_PHASES)
        model.print_profile();
#endif
        return 0;
    }
    std::string generated;
    std::uint64_t generated_tokens = 0;
    while (generated.size() < maximum_bytes) {
        const std::uint16_t token = sample(logits, random, greedy);
        if (token == 0) break;
        generated.append(tokenizer.decode(token));
        ++generated_tokens;
        model.forward(token, logits);
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    if (std::fwrite(generated.data(), 1, generated.size(), stdout) != generated.size()) {
        return 1;
    }
    std::fprintf(
        stderr,
        "\ntokens=%llu bytes=%zu seconds=%.6f tokens_per_second=%.1f chars_per_second=%.1f\n",
        static_cast<unsigned long long>(generated_tokens),
        generated.size(),
        elapsed,
        generated_tokens / elapsed,
        generated.size() / elapsed
    );
    return 0;
}
#endif
