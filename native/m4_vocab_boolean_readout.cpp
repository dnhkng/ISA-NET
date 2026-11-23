#include <arm_neon.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TERNLOG_BOOLEAN_LAYOUT_HEADER
#define TERNLOG_BOOLEAN_LAYOUT_HEADER "ternlog_vocab_boolean_story_layout.h"
#endif
#include TERNLOG_BOOLEAN_LAYOUT_HEADER

extern "C" void ternlog_matvec_f16input_fmlal_m4(
    const __fp16*, const __fp16*, const __fp16*, float*, int, int
);

namespace {

namespace layout = ternlog_vocab_layout;

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("unable to open " + path);
    const std::streamsize size = stream.tellg();
    stream.seekg(0);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("unable to read " + path);
    }
    return data;
}

template <class T>
const T* typed(const std::vector<std::uint8_t>& values) {
    return reinterpret_cast<const T*>(values.data());
}

void pack_query(
    std::uint8_t* packed,
    const float* features,
    std::size_t bits
) {
    std::memset(packed, 0, (bits + 7) / 8);
    for (std::size_t bit = 0; bit < bits; ++bit) {
        packed[bit / 8] |= static_cast<std::uint8_t>(features[bit] >= 0.0f)
            << (bit % 8);
    }
}

std::int8_t signed_group16(const float* features) {
    const float32x4_t values = vld1q_f32(features);
    uint32x4_t count = vshrq_n_u32(
        vcgeq_f32(values, vdupq_n_f32(-1.5f)), 31
    );
    count = vaddq_u32(count, vshrq_n_u32(
        vcgeq_f32(values, vdupq_n_f32(-0.5f)), 31
    ));
    count = vaddq_u32(count, vshrq_n_u32(
        vcgeq_f32(values, vdupq_n_f32(0.5f)), 31
    ));
    count = vaddq_u32(count, vshrq_n_u32(
        vcgeq_f32(values, vdupq_n_f32(1.5f)), 31
    ));
    return static_cast<std::int8_t>(2 * vaddvq_u32(count) - 16);
}

std::int8_t signed_routed_group16(
    const float* features,
    std::size_t start
) {
    constexpr std::array<float, 4> thresholds{-1.5f, -0.5f, 0.5f, 1.5f};
    int count = 0;
    for (std::size_t lane = 0; lane < 16; ++lane) {
        const std::size_t encoded =
            (start + lane) % (layout::kDimension * 4);
        count += features[encoded / 4] >= thresholds[encoded % 4];
    }
    return static_cast<std::int8_t>(2 * count - 16);
}

template <std::size_t Groups, std::size_t PaddedGroups>
void pool_correction_features(
    std::array<std::int8_t, PaddedGroups>& pooled,
    const float* features
) {
    static_assert(Groups <= PaddedGroups);
    for (std::size_t group = 0; group < Groups; ++group) {
        pooled[group] = group < static_cast<std::size_t>(layout::kDimension / 4)
            ? signed_group16(features + 4 * group)
            : signed_routed_group16(
                features, (group * 37) % (layout::kDimension * 4)
            );
    }
}

template <std::size_t Bits>
int centered_hamming(
    const std::uint8_t* query,
    const std::uint8_t* code
) {
    static_assert(Bits % 8 == 0);
    constexpr std::size_t bytes = Bits / 8;
    std::size_t offset = 0;
    int mismatches = 0;
    for (; offset + sizeof(std::uint64_t) <= bytes; offset += sizeof(std::uint64_t)) {
        std::uint64_t query_word;
        std::uint64_t code_word;
        std::memcpy(&query_word, query + offset, sizeof(query_word));
        std::memcpy(&code_word, code + offset, sizeof(code_word));
        mismatches += std::popcount(query_word ^ code_word);
    }
    if constexpr (bytes % sizeof(std::uint64_t) != 0) {
        std::uint32_t query_word;
        std::uint32_t code_word;
        std::memcpy(&query_word, query + offset, sizeof(query_word));
        std::memcpy(&code_word, code + offset, sizeof(code_word));
        mismatches += std::popcount(query_word ^ code_word);
    }
    return static_cast<int>(Bits) - 2 * mismatches;
}

template <std::size_t CodeBits, std::size_t CorrectionFeatures, std::size_t PaddedFeatures>
void boolean_hybrid_readout(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    static_assert(CodeBits % 8 == 0);
    static_assert(PaddedFeatures % 16 == 0);
    alignas(16) std::array<std::uint8_t, CodeBits / 8> query{};
    alignas(16) std::array<std::int8_t, PaddedFeatures> pooled{};
    pack_query(query.data(), features, CodeBits);
    pool_correction_features<CorrectionFeatures>(pooled, features);
    for (
        std::size_t token = 0;
        token < static_cast<std::size_t>(layout::kVocabulary);
        ++token
    ) {
        int32x4_t accumulators = vdupq_n_s32(0);
        for (std::size_t group = 0; group < PaddedFeatures; group += 16) {
            accumulators = vdotq_s32(
                accumulators,
                vld1q_s8(pooled.data() + group),
                vld1q_s8(correction + token * PaddedFeatures + group)
            );
        }
        const int correction_dot = vaddvq_s32(accumulators);
        const float main = static_cast<float>(centered_hamming<CodeBits>(
            query.data(), codes + token * (CodeBits / 8)
        )) * (0.25f / std::sqrt(static_cast<float>(CodeBits)));
        logits[token] = main
            + static_cast<float>(correction_dot) * correction_scales[token]
            + bias[token];
    }
}

int centered_hamming128(
    const std::uint8_t* query,
    const std::uint8_t* code
) {
    const uint8x16_t mismatch = veorq_u8(
        vld1q_u8(query), vld1q_u8(code)
    );
    return 128 - 2 * static_cast<int>(vaddlvq_u8(vcntq_u8(mismatch)));
}

int centered_hamming159(
    const std::uint8_t* query,
    const std::uint8_t* code
) {
    int mismatches = static_cast<int>(vaddlvq_u8(vcntq_u8(veorq_u8(
        vld1q_u8(query), vld1q_u8(code)
    ))));
    for (std::size_t index = 16; index < 19; ++index) {
        mismatches += std::popcount(
            static_cast<unsigned>(query[index] ^ code[index])
        );
    }
    mismatches += std::popcount(
        static_cast<unsigned>((query[19] ^ code[19]) & 0x7fU)
    );
    return 159 - 2 * mismatches;
}

template <std::size_t Rows>
void boolean_hybrid64_thirdscale_rows(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    static_assert(2048 % Rows == 0);
    constexpr std::size_t correction_features =
        layout::kBooleanVocabCorrectionFeatures;
    constexpr std::size_t padded_features =
        (correction_features + 31) / 32 * 32;
    constexpr std::size_t scale_groups = padded_features / 32;
    alignas(16) std::array<std::uint8_t, 8> query{};
    alignas(16) std::array<std::int8_t, padded_features> pooled{};
    pack_query(query.data(), features, 64);
    pool_correction_features<correction_features, padded_features>(
        pooled, features
    );
    for (std::size_t token_base = 0; token_base < 2048; token_base += Rows) {
        int32x4_t accumulators[Rows][scale_groups];
#pragma clang loop unroll(full)
        for (std::size_t row = 0; row < Rows; ++row) {
#pragma clang loop unroll(full)
            for (std::size_t group = 0; group < scale_groups; ++group) {
                accumulators[row][group] = vdupq_n_s32(0);
            }
        }
        for (std::size_t group = 0; group < scale_groups; ++group) {
            for (std::size_t block = 0; block < 2; ++block) {
                const std::size_t feature = group * 32 + block * 16;
                const int8x16_t activation = vld1q_s8(pooled.data() + feature);
#pragma clang loop unroll(full)
                for (std::size_t row = 0; row < Rows; ++row) {
                    accumulators[row][group] = vdotq_s32(
                        accumulators[row][group], activation,
                        vld1q_s8(
                            correction
                                + (token_base + row) * padded_features
                                + feature
                        )
                    );
                }
            }
        }
#pragma clang loop unroll(full)
        for (std::size_t row = 0; row < Rows; ++row) {
            const std::size_t token = token_base + row;
            float correction_sum = 0.0f;
            for (std::size_t group = 0; group < scale_groups; ++group) {
                correction_sum = correction_sum
                    + static_cast<float>(vaddvq_s32(accumulators[row][group]))
                        * correction_scales[token * scale_groups + group];
            }
            const float main = static_cast<float>(centered_hamming<64>(
                query.data(), codes + token * 8
            )) * (0.25f / std::sqrt(64.0f));
            logits[token] = main + correction_sum + bias[token];
        }
    }
}

template <std::size_t Rank>
void factor_hidden(
    std::array<std::uint8_t, 8>& query,
    std::array<__fp16, Rank>& hidden,
    const float* features,
    const std::int8_t* correction_input,
    const float* correction_input_scales
) {
    constexpr std::size_t correction_features =
        layout::kBooleanVocabCorrectionFeatures;
    constexpr std::size_t padded_features =
        (correction_features + 15) / 16 * 16;
    alignas(64) std::array<std::int8_t, padded_features> pooled{};
    static_assert(Rank % 16 == 0);
    alignas(64) std::array<float, Rank> hidden_float{};
    pack_query(query.data(), features, 64);
    pool_correction_features<correction_features, padded_features>(
        pooled, features
    );
    for (std::size_t row = 0; row < Rank; row += 8) {
        int32x4_t sums[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (
            std::size_t feature = 0;
            feature < padded_features;
            feature += 16
        ) {
            const int8x16_t value = vld1q_s8(pooled.data() + feature);
#pragma clang loop unroll(full)
            for (std::size_t item = 0; item < 8; ++item) {
                sums[item] = vdotq_s32(
                    sums[item], value,
                    vld1q_s8(
                        correction_input
                            + (row + item) * padded_features
                            + feature
                    )
                );
            }
        }
#pragma clang loop unroll(full)
        for (std::size_t item = 0; item < 8; ++item) {
            hidden_float[row + item] = static_cast<float>(vaddvq_s32(sums[item]))
                * correction_input_scales[row + item];
        }
    }
    for (std::size_t index = 0; index < Rank; index += 8) {
        vst1q_f16(hidden.data() + index, vcombine_f16(
            vcvt_f16_f32(vld1q_f32(hidden_float.data() + index)),
            vcvt_f16_f32(vld1q_f32(hidden_float.data() + index + 4))
        ));
    }
}

template <std::size_t Rank>
float quantize_factor_hidden(
    std::array<std::int8_t, Rank>& quantized,
    const std::array<__fp16, Rank>& hidden
) {
    static_assert(Rank % 16 == 0);
    float32x4_t maximum = vdupq_n_f32(1.0e-12F);
    for (std::size_t index = 0; index < Rank; index += 8) {
        const float16x8_t values = vld1q_f16(hidden.data() + index);
        maximum = vmaxq_f32(maximum, vabsq_f32(vcvt_f32_f16(vget_low_f16(values))));
        maximum = vmaxq_f32(maximum, vabsq_f32(vcvt_f32_f16(vget_high_f16(values))));
    }
    const float max_value = vmaxvq_f32(maximum);
    const float multiplier = 127.0F / max_value;
    for (std::size_t index = 0; index < Rank; index += 16) {
        const float16x8_t first = vld1q_f16(hidden.data() + index);
        const float16x8_t second = vld1q_f16(hidden.data() + index + 8);
        const int16x8_t first16 = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(
                vcvt_f32_f16(vget_low_f16(first)), multiplier
            ))),
            vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(
                vcvt_f32_f16(vget_high_f16(first)), multiplier
            )))
        );
        const int16x8_t second16 = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(
                vcvt_f32_f16(vget_low_f16(second)), multiplier
            ))),
            vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(
                vcvt_f32_f16(vget_high_f16(second)), multiplier
            )))
        );
        vst1q_s8(quantized.data() + index, vcombine_s8(
            vqmovn_s16(first16), vqmovn_s16(second16)
        ));
    }
    return max_value / 127.0F;
}

void add_factor_main(
    float* logits,
    const std::array<std::uint8_t, 8>& query,
    const std::uint8_t* codes,
    const float* bias
) {
    for (
        std::size_t token = 0;
        token < static_cast<std::size_t>(layout::kVocabulary);
        ++token
    ) {
        const float main = static_cast<float>(centered_hamming<64>(
            query.data(), codes + token * 8
        )) * (0.25F / std::sqrt(64.0F));
        logits[token] += main + bias[token];
    }
}

template <std::size_t Rank>
void factor_readout(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const __fp16* correction_output,
    const float* bias
) {
    alignas(64) std::array<std::uint8_t, 8> query{};
    alignas(64) std::array<__fp16, Rank> hidden{};
    factor_hidden<Rank>(
        query, hidden, features, correction_input, correction_input_scales
    );
    ternlog_matvec_f16input_fmlal_m4(
        correction_output, nullptr, hidden.data(), logits,
        layout::kVocabulary, Rank
    );
    add_factor_main(logits, query, codes, bias);
}

template <std::size_t Rank, bool Group16>
void factor_w8a8_readout(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const std::int8_t* correction_output,
    const float* correction_output_scales,
    const float* bias
) {
    static_assert(Rank % 16 == 0);
    constexpr std::size_t groups = Rank / 16;
    alignas(64) std::array<std::uint8_t, 8> query{};
    alignas(64) std::array<__fp16, Rank> hidden{};
    alignas(64) std::array<std::int8_t, Rank> quantized{};
    factor_hidden<Rank>(
        query, hidden, features, correction_input, correction_input_scales
    );
    const float activation_scale = quantize_factor_hidden<Rank>(quantized, hidden);
    for (std::size_t token_base = 0; token_base < 2048; token_base += 8) {
        if constexpr (Group16) {
#pragma clang loop unroll(full)
            for (std::size_t row = 0; row < 8; ++row) {
                const std::size_t token = token_base + row;
                const std::int8_t* weight = correction_output + token * Rank;
                float correction = 0.0F;
#pragma clang loop unroll(full)
                for (std::size_t group = 0; group < groups; ++group) {
                    const int dot = vaddvq_s32(vdotq_s32(
                        vdupq_n_s32(0),
                        vld1q_s8(quantized.data() + group * 16),
                        vld1q_s8(weight + group * 16)
                    ));
                    correction += static_cast<float>(dot)
                        * correction_output_scales[token * groups + group];
                }
                logits[token] = activation_scale * correction;
            }
        } else {
            int32x4_t sums[8] = {
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            };
#pragma clang loop unroll(full)
            for (std::size_t group = 0; group < groups; ++group) {
                const int8x16_t activation = vld1q_s8(
                    quantized.data() + group * 16
                );
#pragma clang loop unroll(full)
                for (std::size_t row = 0; row < 8; ++row) {
                    sums[row] = vdotq_s32(
                        sums[row], activation,
                        vld1q_s8(
                            correction_output
                            + (token_base + row) * Rank + group * 16
                        )
                    );
                }
            }
#pragma clang loop unroll(full)
            for (std::size_t row = 0; row < 8; ++row) {
                logits[token_base + row] = static_cast<float>(vaddvq_s32(sums[row]))
                    * activation_scale * correction_output_scales[token_base + row];
            }
        }
    }
    add_factor_main(logits, query, codes, bias);
}

}  // namespace

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_thirdscale_2048_rows1(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    boolean_hybrid64_thirdscale_rows<1>(
        logits, features, codes, correction, correction_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_thirdscale_2048_rows4(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    boolean_hybrid64_thirdscale_rows<4>(
        logits, features, codes, correction, correction_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_thirdscale_2048_rows8(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    boolean_hybrid64_thirdscale_rows<8>(
        logits, features, codes, correction, correction_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    boolean_hybrid_readout<128, 31, 32>(
        logits, features, codes, correction, correction_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid96_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    boolean_hybrid_readout<96, 63, 64>(
        logits, features, codes, correction, correction_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    boolean_hybrid_readout<64, 95, 96>(
        logits, features, codes, correction, correction_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_blockscale_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
    alignas(16) std::array<std::uint8_t, 8> query{};
    alignas(16) std::array<std::int8_t, 96> pooled{};
    pack_query(query.data(), features, 64);
    pool_correction_features<95, 96>(pooled, features);
    for (std::size_t token = 0; token < 2048; ++token) {
        float correction_sum = 0.0f;
        for (std::size_t block = 0; block < 6; ++block) {
            const int32x4_t dot = vdotq_s32(
                vdupq_n_s32(0),
                vld1q_s8(pooled.data() + 16 * block),
                vld1q_s8(correction + token * 96 + 16 * block)
            );
            correction_sum += static_cast<float>(vaddvq_s32(dot))
                * correction_scales[token * 6 + block];
        }
        const float main = static_cast<float>(centered_hamming<64>(
            query.data(), codes + token * 8
        )) * (0.25f / std::sqrt(64.0f));
        logits[token] = main + correction_sum + bias[token];
    }
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_thirdscale_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction,
    const float* correction_scales,
    const float* bias
) {
#if defined(TERNLOG_BOOLEAN_READOUT_SDOT4)
    ternlog_vocab_boolean_hybrid64_thirdscale_2048_rows4(
        logits, features, codes, correction, correction_scales, bias
    );
#elif defined(TERNLOG_BOOLEAN_READOUT_SDOT8)
    ternlog_vocab_boolean_hybrid64_thirdscale_2048_rows8(
        logits, features, codes, correction, correction_scales, bias
    );
#else
    ternlog_vocab_boolean_hybrid64_thirdscale_2048_rows1(
        logits, features, codes, correction, correction_scales, bias
    );
#endif
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_factor32_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const __fp16* correction_output,
    const float* bias
) {
    factor_readout<32>(
        logits, features, codes, correction_input, correction_input_scales,
        correction_output, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_factor64_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const __fp16* correction_output,
    const float* bias
) {
    factor_readout<64>(
        logits, features, codes, correction_input, correction_input_scales,
        correction_output, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_factor_native_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const __fp16* correction_output,
    const float* bias
) {
    constexpr std::size_t rank = layout::kBooleanVocabCorrectionRank;
    static_assert(rank % 16 == 0);
    if constexpr (rank > 0) {
        factor_readout<rank>(
            logits, features, codes, correction_input, correction_input_scales,
            correction_output, bias
        );
    }
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hierarchy64_factor32_prepare(
    float* cluster_logits,
    std::uint8_t* packed_query,
    __fp16* hidden_output,
    const float* features,
    const std::uint8_t* cluster_codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const __fp16* cluster_output,
    const float* cluster_bias
) {
    alignas(64) std::array<std::uint8_t, 8> query{};
    alignas(64) std::array<__fp16, 32> hidden{};
    factor_hidden<32>(
        query, hidden, features, correction_input, correction_input_scales
    );
    ternlog_matvec_f16input_fmlal_m4(
        cluster_output, nullptr, hidden.data(), cluster_logits, 32, 32
    );
    for (std::size_t cluster = 0; cluster < 32; ++cluster) {
        const float main = static_cast<float>(centered_hamming<64>(
            query.data(), cluster_codes + cluster * 8
        )) * (0.25F / 8.0F);
        cluster_logits[cluster] += main + cluster_bias[cluster];
    }
    std::memcpy(packed_query, query.data(), query.size());
    std::memcpy(hidden_output, hidden.data(), hidden.size() * sizeof(__fp16));
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hierarchy64_factor32_leaf(
    float* leaf_logits,
    const std::uint8_t* packed_query,
    const __fp16* hidden,
    const std::uint8_t* token_codes,
    const std::uint16_t* leaf_token_ids,
    const __fp16* leaf_output,
    const float* leaf_bias
) {
    ternlog_matvec_f16input_fmlal_m4(
        leaf_output, nullptr, hidden, leaf_logits, 64, 32
    );
    for (std::size_t leaf = 0; leaf < 64; ++leaf) {
        const float main = static_cast<float>(centered_hamming<64>(
            packed_query,
            token_codes + static_cast<std::size_t>(leaf_token_ids[leaf]) * 8
        )) * (0.25F / 8.0F);
        leaf_logits[leaf] += main + leaf_bias[leaf];
    }
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_factor32_w8a8_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const std::int8_t* correction_output,
    const float* correction_output_scales,
    const float* bias
) {
    factor_w8a8_readout<32, false>(
        logits, features, codes, correction_input, correction_input_scales,
        correction_output, correction_output_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_factor32_w8a8_group16_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const std::int8_t* correction_output,
    const float* correction_output_scales,
    const float* bias
) {
    factor_w8a8_readout<32, true>(
        logits, features, codes, correction_input, correction_input_scales,
        correction_output, correction_output_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_factor48_w8a8_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const std::int8_t* correction_output,
    const float* correction_output_scales,
    const float* bias
) {
    factor_w8a8_readout<48, false>(
        logits, features, codes, correction_input, correction_input_scales,
        correction_output, correction_output_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_hybrid64_factor48_w8a8_group16_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const std::int8_t* correction_input,
    const float* correction_input_scales,
    const std::int8_t* correction_output,
    const float* correction_output_scales,
    const float* bias
) {
    factor_w8a8_readout<48, true>(
        logits, features, codes, correction_input, correction_input_scales,
        correction_output, correction_output_scales, bias
    );
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_tied_2048(
    float* logits,
    const float* features,
    const std::uint8_t* codes,
    const float* bias
) {
    alignas(16) std::array<std::uint8_t, 20> query{};
    pack_query(query.data(), features, 159);
    for (std::size_t token = 0; token < 2048; ++token) {
        logits[token] = static_cast<float>(centered_hamming159(
            query.data(), codes + token * 20
        )) * (0.25f / std::sqrt(159.0f)) + bias[token];
    }
}

extern "C" __attribute__((noinline))
void ternlog_vocab_boolean_embedding160(
    float* output,
    const std::uint8_t* code,
    std::size_t code_bits
) {
    for (
        std::size_t dimension = 0;
        dimension < static_cast<std::size_t>(layout::kDimension);
        ++dimension
    ) {
        const std::size_t source = dimension % code_bits;
        const int bit = (code[source / 8] >> (source % 8)) & 1;
        output[dimension] = static_cast<float>(2 * bit - 1) * 0.02f;
    }
}

#ifndef TERNLOG_VOCAB_BOOLEAN_READOUT_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: m4_vocab_boolean_readout embed|hybrid|hybrid96|hybrid64|tied ...\n";
            return 2;
        }
        const std::string mode = argv[1];
        if (mode == "embed") {
            if (argc != 4) throw std::runtime_error("embed CODE CODE_BITS");
            const auto code = read_file(argv[2]);
            const std::size_t bits = static_cast<std::size_t>(std::stoul(argv[3]));
            if (code.size() != (bits + 7) / 8) throw std::runtime_error("bad code size");
            std::array<float, 160> output{};
            ternlog_vocab_boolean_embedding160(output.data(), code.data(), bits);
            std::cout.write(reinterpret_cast<const char*>(output.data()), sizeof(output));
            return 0;
        }
        if (mode == "hybrid" || mode == "hybrid96" || mode == "hybrid64"
            || mode == "hybrid64block" || mode == "hybrid64third") {
            if (argc != 7) {
                throw std::runtime_error("hybrid FEATURES CODES CORRECTION SCALES BIAS");
            }
            const auto features = read_file(argv[2]);
            const auto codes = read_file(argv[3]);
            const auto correction = read_file(argv[4]);
            const auto scales = read_file(argv[5]);
            const auto bias = read_file(argv[6]);
            const std::size_t code_bytes = mode == "hybrid"
                ? 16 : mode == "hybrid96" ? 12 : 8;
            const std::size_t padded_correction = mode == "hybrid"
                ? 32 : mode == "hybrid96" ? 64 : 96;
            if (features.size() != 160 * sizeof(float)
                || codes.size() != 2048 * code_bytes
                || correction.size() != 2048 * padded_correction
                || scales.size() != 2048 * sizeof(float)
                    * (mode == "hybrid64block" ? 6 : mode == "hybrid64third" ? 3 : 1)
                || bias.size() != 2048 * sizeof(float)) {
                throw std::runtime_error("hybrid input size mismatch");
            }
            std::array<float, 2048> logits{};
            const auto* feature_values = typed<float>(features);
            const auto* correction_values = typed<std::int8_t>(correction);
            if (mode == "hybrid") {
                ternlog_vocab_boolean_hybrid_2048(
                    logits.data(), feature_values, codes.data(), correction_values,
                    typed<float>(scales), typed<float>(bias)
                );
            } else if (mode == "hybrid96") {
                ternlog_vocab_boolean_hybrid96_2048(
                    logits.data(), feature_values, codes.data(), correction_values,
                    typed<float>(scales), typed<float>(bias)
                );
            } else if (mode == "hybrid64") {
                ternlog_vocab_boolean_hybrid64_2048(
                    logits.data(), feature_values, codes.data(), correction_values,
                    typed<float>(scales), typed<float>(bias)
                );
            } else if (mode == "hybrid64block") {
                ternlog_vocab_boolean_hybrid64_blockscale_2048(
                    logits.data(), feature_values, codes.data(), correction_values,
                    typed<float>(scales), typed<float>(bias)
                );
            } else {
                ternlog_vocab_boolean_hybrid64_thirdscale_2048(
                    logits.data(), feature_values, codes.data(), correction_values,
                    typed<float>(scales), typed<float>(bias)
                );
            }
            std::cout.write(reinterpret_cast<const char*>(logits.data()), sizeof(logits));
            return 0;
        }
        if (mode == "tied") {
            if (argc != 5) throw std::runtime_error("tied FEATURES CODES BIAS");
            const auto features = read_file(argv[2]);
            const auto codes = read_file(argv[3]);
            const auto bias = read_file(argv[4]);
            if (features.size() != 160 * sizeof(float)
                || codes.size() != 2048 * 20
                || bias.size() != 2048 * sizeof(float)) {
                throw std::runtime_error("tied input size mismatch");
            }
            std::array<float, 2048> logits{};
            ternlog_vocab_boolean_tied_2048(
                logits.data(), typed<float>(features), codes.data(), typed<float>(bias)
            );
            std::cout.write(reinterpret_cast<const char*>(logits.data()), sizeof(logits));
            return 0;
        }
        throw std::runtime_error("invalid Boolean readout mode");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
