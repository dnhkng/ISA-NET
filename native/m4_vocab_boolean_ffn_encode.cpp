#include <arm_neon.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TERNLOG_BOOLEAN_LAYOUT_HEADER
#define TERNLOG_BOOLEAN_LAYOUT_HEADER "ternlog_vocab_boolean_story_layout.h"
#endif
#include TERNLOG_BOOLEAN_LAYOUT_HEADER

namespace {

namespace layout = ternlog_vocab_layout;
constexpr std::size_t dimension = layout::kDimension;
static_assert(dimension % 16 == 0);

std::vector<std::uint8_t> read_encode_file(const std::string& path) {
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

template <int Mode>
const std::array<float, 8>& thresholds() {
    static const std::array<float, 8> linear{
        -1.75f, -1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f, 1.75f
    };
    static const std::array<float, 8> gaussian{
        -1.2206403017044067f, -0.7647097110748291f,
        -0.4307272434234619f, -0.13971029222011566f,
        0.13971036672592163f, 0.4307273328304291f,
        0.7647097110748291f, 1.2206403017044067f
    };
    static const std::array<float, 8> power2{
        -2.0f, -1.0f, -0.5f, -0.25f, 0.25f, 0.5f, 1.0f, 2.0f
    };
    if constexpr (Mode == 0) return linear;
    if constexpr (Mode == 1) return gaussian;
    return power2;
}

template <int Mode>
__attribute__((noinline)) void encode_ffn(
    std::uint8_t* packed, const float* input, const float* offsets
) {
    const auto& values = thresholds<Mode>();
    for (std::size_t row = 0; row < dimension; ++row) {
        std::uint8_t byte = 0;
        const float centered = input[row] - offsets[row];
        for (unsigned plane = 0; plane < 8; ++plane) {
            byte |= static_cast<std::uint8_t>(centered >= values[plane]) << plane;
        }
        packed[row] = byte;
    }
}

template <int Mode>
inline uint8x16_t encode_centered_neon(
    float32x4_t centered0,
    float32x4_t centered1,
    float32x4_t centered2,
    float32x4_t centered3
) {
    const auto& values = thresholds<Mode>();
    uint32x4_t bits0 = vdupq_n_u32(0);
    uint32x4_t bits1 = vdupq_n_u32(0);
    uint32x4_t bits2 = vdupq_n_u32(0);
    uint32x4_t bits3 = vdupq_n_u32(0);
#pragma clang loop unroll(full)
    for (unsigned plane = 0; plane < 8; ++plane) {
        const float32x4_t threshold = vdupq_n_f32(values[plane]);
        const uint32x4_t bit = vdupq_n_u32(1U << plane);
        bits0 = vorrq_u32(
            bits0, vandq_u32(vcgeq_f32(centered0, threshold), bit)
        );
        bits1 = vorrq_u32(
            bits1, vandq_u32(vcgeq_f32(centered1, threshold), bit)
        );
        bits2 = vorrq_u32(
            bits2, vandq_u32(vcgeq_f32(centered2, threshold), bit)
        );
        bits3 = vorrq_u32(
            bits3, vandq_u32(vcgeq_f32(centered3, threshold), bit)
        );
    }
    const uint16x8_t half0 = vcombine_u16(
        vmovn_u32(bits0), vmovn_u32(bits1)
    );
    const uint16x8_t half1 = vcombine_u16(
        vmovn_u32(bits2), vmovn_u32(bits3)
    );
    return vcombine_u8(vmovn_u16(half0), vmovn_u16(half1));
}

template <int Mode>
__attribute__((noinline)) void encode_ffn_neon(
    std::uint8_t* packed, const float* input, const float* offsets
) {
    for (std::size_t row = 0; row < dimension; row += 16) {
        const float32x4_t centered0 = vsubq_f32(
            vld1q_f32(input + row), vld1q_f32(offsets + row)
        );
        const float32x4_t centered1 = vsubq_f32(
            vld1q_f32(input + row + 4), vld1q_f32(offsets + row + 4)
        );
        const float32x4_t centered2 = vsubq_f32(
            vld1q_f32(input + row + 8), vld1q_f32(offsets + row + 8)
        );
        const float32x4_t centered3 = vsubq_f32(
            vld1q_f32(input + row + 12), vld1q_f32(offsets + row + 12)
        );
        vst1q_u8(
            packed + row,
            encode_centered_neon<Mode>(
                centered0, centered1, centered2, centered3
            )
        );
    }
}

template <int Rounding>
inline void round_input(float* output, const float* input) {
    if constexpr (Rounding == 1) {
        for (std::size_t index = 0; index < dimension; index += 8) {
            const float32x4_t low = vld1q_f32(input + index);
            const float32x4_t high = vld1q_f32(input + index + 4);
            const float16x8_t half = vcombine_f16(
                vcvt_f16_f32(low), vcvt_f16_f32(high)
            );
            vst1q_f32(output + index, vcvt_f32_f16(vget_low_f16(half)));
            vst1q_f32(output + index + 4, vcvt_f32_f16(vget_high_f16(half)));
        }
    } else {
        static_assert(Rounding == 2);
        for (std::size_t index = 0; index < dimension; index += 4) {
            const float32x4_t scaled = vmulq_n_f32(
                vld1q_f32(input + index), 256.0f
            );
            vst1q_f32(
                output + index,
                vmulq_n_f32(vrndnq_f32(scaled), 1.0f / 256.0f)
            );
        }
    }
}

template <int Mode, int Rounding>
__attribute__((noinline)) void encode_ffn_rounded(
    std::uint8_t* packed, const float* input, const float* offsets
) {
    alignas(64) std::array<float, dimension> rounded{};
    round_input<Rounding>(rounded.data(), input);
    encode_ffn<Mode>(packed, rounded.data(), offsets);
}

template <int Mode>
__attribute__((noinline)) void encode_ffn_rounded_neon(
    std::uint8_t* packed, const float* input, const float* offsets
) {
    alignas(64) std::array<float, dimension> rounded{};
    round_input<1>(rounded.data(), input);
    encode_ffn_neon<Mode>(packed, rounded.data(), offsets);
}

template <int Mode>
__attribute__((noinline)) void encode_ffn_fp16_fused_neon(
    std::uint8_t* packed, const float* input, const float* offsets
) {
    for (std::size_t row = 0; row < dimension; row += 16) {
        const float32x4_t rounded0 = vcvt_f32_f16(
            vcvt_f16_f32(vld1q_f32(input + row))
        );
        const float32x4_t rounded1 = vcvt_f32_f16(
            vcvt_f16_f32(vld1q_f32(input + row + 4))
        );
        const float32x4_t rounded2 = vcvt_f32_f16(
            vcvt_f16_f32(vld1q_f32(input + row + 8))
        );
        const float32x4_t rounded3 = vcvt_f32_f16(
            vcvt_f16_f32(vld1q_f32(input + row + 12))
        );
        vst1q_u8(
            packed + row,
            encode_centered_neon<Mode>(
                vsubq_f32(rounded0, vld1q_f32(offsets + row)),
                vsubq_f32(rounded1, vld1q_f32(offsets + row + 4)),
                vsubq_f32(rounded2, vld1q_f32(offsets + row + 8)),
                vsubq_f32(rounded3, vld1q_f32(offsets + row + 12))
            )
        );
    }
}

template <int Mode>
__attribute__((noinline)) void encode_ffn_rms_fp16_fused_neon(
    std::uint8_t* packed,
    const float* input,
    const __fp16* norm_weights,
    const float* offsets,
    float inverse_rms
) {
    for (std::size_t row = 0; row < dimension; row += 16) {
        const float16x8_t norm0 = vld1q_f16(norm_weights + row);
        const float16x8_t norm1 = vld1q_f16(norm_weights + row + 8);
        const float32x4_t normalized0 = vmulq_f32(
            vmulq_n_f32(vld1q_f32(input + row), inverse_rms),
            vcvt_f32_f16(vget_low_f16(norm0))
        );
        const float32x4_t normalized1 = vmulq_f32(
            vmulq_n_f32(vld1q_f32(input + row + 4), inverse_rms),
            vcvt_f32_f16(vget_high_f16(norm0))
        );
        const float32x4_t normalized2 = vmulq_f32(
            vmulq_n_f32(vld1q_f32(input + row + 8), inverse_rms),
            vcvt_f32_f16(vget_low_f16(norm1))
        );
        const float32x4_t normalized3 = vmulq_f32(
            vmulq_n_f32(vld1q_f32(input + row + 12), inverse_rms),
            vcvt_f32_f16(vget_high_f16(norm1))
        );
        const float32x4_t rounded0 = vcvt_f32_f16(vcvt_f16_f32(normalized0));
        const float32x4_t rounded1 = vcvt_f32_f16(vcvt_f16_f32(normalized1));
        const float32x4_t rounded2 = vcvt_f32_f16(vcvt_f16_f32(normalized2));
        const float32x4_t rounded3 = vcvt_f32_f16(vcvt_f16_f32(normalized3));
        vst1q_u8(
            packed + row,
            encode_centered_neon<Mode>(
                vsubq_f32(rounded0, vld1q_f32(offsets + row)),
                vsubq_f32(rounded1, vld1q_f32(offsets + row + 4)),
                vsubq_f32(rounded2, vld1q_f32(offsets + row + 8)),
                vsubq_f32(rounded3, vld1q_f32(offsets + row + 12))
            )
        );
    }
}

}  // namespace

extern "C" void ternlog_vocab_ffn_encode_linear(
    std::uint8_t* output, const float* input, const float* offsets
) { encode_ffn<0>(output, input, offsets); }

extern "C" void ternlog_vocab_ffn_encode_gaussian(
    std::uint8_t* output, const float* input, const float* offsets
) { encode_ffn<1>(output, input, offsets); }

extern "C" void ternlog_vocab_ffn_encode_power2(
    std::uint8_t* output, const float* input, const float* offsets
) { encode_ffn<2>(output, input, offsets); }

extern "C" void ternlog_vocab_ffn_encode_linear_fp16(
    std::uint8_t* output, const float* input, const float* offsets
) { encode_ffn_rounded<0, 1>(output, input, offsets); }

extern "C" void ternlog_vocab_ffn_encode_linear_fp16_neon(
    std::uint8_t* output, const float* input, const float* offsets
) { encode_ffn_rounded_neon<0>(output, input, offsets); }

extern "C" void ternlog_vocab_ffn_encode_linear_fp16_fused_neon(
    std::uint8_t* output, const float* input, const float* offsets
) { encode_ffn_fp16_fused_neon<0>(output, input, offsets); }

extern "C" void ternlog_vocab_ffn_encode_linear_rms_fp16_fused_neon(
    std::uint8_t* output,
    const float* input,
    const __fp16* norm_weights,
    const float* offsets,
    float inverse_rms
) {
    encode_ffn_rms_fp16_fused_neon<0>(
        output, input, norm_weights, offsets, inverse_rms
    );
}

extern "C" void ternlog_vocab_ffn_encode_linear_fixed8(
    std::uint8_t* output, const float* input, const float* offsets
) { encode_ffn_rounded<0, 2>(output, input, offsets); }

#ifndef TERNLOG_BOOLEAN_FFN_ENCODE_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "usage: m4_vocab_boolean_ffn_encode linear|gaussian|power2 INPUT OFFSETS [none|fp16|fixed8]\n";
            return 2;
        }
        const auto input = read_encode_file(argv[2]);
        const auto offsets = read_encode_file(argv[3]);
        if (input.size() != dimension * sizeof(float)
            || offsets.size() != dimension * sizeof(float)) {
            throw std::runtime_error("input size mismatch");
        }
        std::array<std::uint8_t, dimension> output{};
        const std::string rounding = argc == 5 ? argv[4] : "none";
        auto function = rounding == "fp16" && std::string(argv[1]) == "linear"
            ? ternlog_vocab_ffn_encode_linear_fp16
            : rounding == "fixed8" && std::string(argv[1]) == "linear"
            ? ternlog_vocab_ffn_encode_linear_fixed8
            : rounding != "none" ? nullptr
            : std::string(argv[1]) == "linear" ? ternlog_vocab_ffn_encode_linear
            : std::string(argv[1]) == "gaussian" ? ternlog_vocab_ffn_encode_gaussian
            : std::string(argv[1]) == "power2" ? ternlog_vocab_ffn_encode_power2 : nullptr;
        if (function == nullptr) throw std::runtime_error("invalid threshold mode");
        function(
            output.data(), reinterpret_cast<const float*>(input.data()),
            reinterpret_cast<const float*>(offsets.data())
        );
        std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
