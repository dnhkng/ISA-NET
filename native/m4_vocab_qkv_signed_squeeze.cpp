#include <arm_neon.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kDimension = 160;
constexpr int kPlanes = 8;
constexpr int kGroups = 40;
constexpr int kPolarities = 2;
constexpr int kFeatures = kGroups * kPolarities;
constexpr int kRows = 480;
constexpr std::array<float, kPlanes> kThresholds{
    -1.75F, -1.25F, -0.75F, -0.25F,
     0.25F,  0.75F,  1.25F,  1.75F,
};

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

inline int popcount32(std::uint32_t value) {
    const uint8x8_t bytes = vreinterpret_u8_u64(vcreate_u64(value));
    return static_cast<int>(vaddlv_u8(vcnt_u8(bytes)));
}

}  // namespace

extern "C" void ternlog_vocab_qkv_round160_m4(
    float* output, const float* input, int fixed_scale
) {
    if (fixed_scale == 0) {
        for (int index = 0; index < kDimension; index += 8) {
            const float16x8_t half = vcombine_f16(
                vcvt_f16_f32(vld1q_f32(input + index)),
                vcvt_f16_f32(vld1q_f32(input + index + 4))
            );
            vst1q_f32(output + index, vcvt_f32_f16(vget_low_f16(half)));
            vst1q_f32(
                output + index + 4,
                vcvt_f32_f16(vget_high_f16(half))
            );
        }
        return;
    }
    const float scale = static_cast<float>(fixed_scale);
    const float inverse = 1.0F / scale;
    for (int index = 0; index < kDimension; index += 4) {
        vst1q_f32(
            output + index,
            vmulq_n_f32(
                vrndnq_f32(vmulq_n_f32(vld1q_f32(input + index), scale)),
                inverse
            )
        );
    }
}

extern "C" void ternlog_vocab_qkv_signed_counts_m4(
    std::int8_t* output,
    const float* input,
    const float* offsets,
    const std::uint32_t* polarity_masks
) {
    alignas(64) std::array<float, kDimension> rounded{};
    for (int index = 0; index < kDimension; index += 8) {
        const float16x8_t half = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(input + index)),
            vcvt_f16_f32(vld1q_f32(input + index + 4))
        );
        vst1q_f32(rounded.data() + index, vcvt_f32_f16(vget_low_f16(half)));
        vst1q_f32(
            rounded.data() + index + 4,
            vcvt_f32_f16(vget_high_f16(half))
        );
    }
    for (int plane = 0; plane < kPlanes; ++plane) {
        for (int group = 0; group < 5; ++group) {
            std::uint32_t packed = 0;
            const int base = group * 32;
            const uint32x4_t lane_bits = {1U, 2U, 4U, 8U};
            for (int lane = 0; lane < 32; lane += 4) {
                const float32x4_t boundary = vaddq_f32(
                    vld1q_f32(offsets + base + lane),
                    vdupq_n_f32(kThresholds[plane])
                );
                const uint32x4_t compared = vcgeq_f32(
                    vld1q_f32(rounded.data() + base + lane), boundary
                );
                packed |= vaddvq_u32(vandq_u32(compared, lane_bits)) << lane;
            }
            const int feature_group = plane * 5 + group;
            for (int polarity = 0; polarity < kPolarities; ++polarity) {
                const int mismatch = popcount32(
                    packed ^ polarity_masks[polarity * kGroups + feature_group]
                );
                output[polarity * kGroups + feature_group] =
                    static_cast<std::int8_t>(32 - 2 * mismatch);
            }
        }
    }
}

extern "C" void ternlog_vocab_qkv_signed_counts40_m4(
    std::int8_t* output,
    const float* input,
    const float* offsets,
    const float* thresholds,
    const std::uint32_t* polarity_masks
) {
    alignas(64) std::array<float, kDimension> rounded{};
    for (int index = 0; index < kDimension; index += 8) {
        const float16x8_t half = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(input + index)),
            vcvt_f16_f32(vld1q_f32(input + index + 4))
        );
        vst1q_f32(rounded.data() + index, vcvt_f32_f16(vget_low_f16(half)));
        vst1q_f32(
            rounded.data() + index + 4,
            vcvt_f32_f16(vget_high_f16(half))
        );
    }
    for (int plane = 0; plane < kPlanes; ++plane) {
        for (int group = 0; group < 5; ++group) {
            std::uint32_t packed = 0;
            const int base = group * 32;
            const uint32x4_t lane_bits = {1U, 2U, 4U, 8U};
            for (int lane = 0; lane < 32; lane += 4) {
                const float32x4_t boundary = vaddq_f32(
                    vld1q_f32(offsets + base + lane),
                    vdupq_n_f32(thresholds[plane])
                );
                const uint32x4_t compared = vcgeq_f32(
                    vld1q_f32(rounded.data() + base + lane), boundary
                );
                packed |= vaddvq_u32(vandq_u32(compared, lane_bits)) << lane;
            }
            const int feature_group = plane * 5 + group;
            const int mismatch = popcount32(
                packed ^ polarity_masks[feature_group]
            );
            output[feature_group] = static_cast<std::int8_t>(32 - 2 * mismatch);
        }
    }
    std::memset(output + 40, 0, 8);
}

extern "C" void ternlog_vocab_qkv_signed_sdot_480x80_m4(
    std::int32_t* output,
    const std::int8_t* weights,
    const std::int8_t* features
) {
    for (int row = 0; row < kRows; row += 8) {
        int32x4_t sums[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int column = 0; column < kFeatures; column += 16) {
            const int8x16_t activation = vld1q_s8(features + column);
            for (int item = 0; item < 8; ++item) {
                sums[item] = vdotq_s32(
                    sums[item],
                    vld1q_s8(
                        weights + static_cast<std::size_t>(row + item) * kFeatures
                        + column
                    ),
                    activation
                );
            }
        }
        for (int item = 0; item < 8; ++item) {
            output[row + item] = vaddvq_s32(sums[item]);
        }
    }
}

extern "C" void ternlog_vocab_qkv_signed_sdot_480x48_m4(
    std::int32_t* output,
    const std::int8_t* weights,
    const std::int8_t* features
) {
    constexpr int kPaddedFeatures = 48;
    for (int row = 0; row < kRows; row += 8) {
        int32x4_t sums[8] = {
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
            vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int column = 0; column < kPaddedFeatures; column += 16) {
            const int8x16_t activation = vld1q_s8(features + column);
            for (int item = 0; item < 8; ++item) {
                sums[item] = vdotq_s32(
                    sums[item],
                    vld1q_s8(
                        weights
                        + static_cast<std::size_t>(row + item) * kPaddedFeatures
                        + column
                    ),
                    activation
                );
            }
        }
        for (int item = 0; item < 8; ++item) {
            output[row + item] = vaddvq_s32(sums[item]);
        }
    }
}

#ifndef TERNLOG_QKV_SIGNED_SQUEEZE_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--round") {
            const auto input = read_file(argv[2]);
            if (input.size() != kDimension * sizeof(float)) {
                throw std::runtime_error("rounding input size mismatch");
            }
            std::array<float, kDimension> rounded{};
            ternlog_vocab_qkv_round160_m4(
                rounded.data(), reinterpret_cast<const float*>(input.data()),
                std::atoi(argv[3])
            );
            std::cout.write(
                reinterpret_cast<const char*>(rounded.data()),
                rounded.size() * sizeof(float)
            );
            return 0;
        }
        if (argc != 5) {
            std::cerr << "usage: m4_vocab_qkv_signed_squeeze INPUT OFFSETS POLARITIES WEIGHTS\n";
            return 2;
        }
        const auto input = read_file(argv[1]);
        const auto offsets = read_file(argv[2]);
        const auto polarities = read_file(argv[3]);
        const auto weights = read_file(argv[4]);
        if (input.size() != kDimension * sizeof(float)
            || offsets.size() != kDimension * sizeof(float)
            || polarities.size() != kPolarities * kGroups * sizeof(std::uint32_t)
            || weights.size() != kRows * kFeatures) {
            throw std::runtime_error("signed-squeeze input size mismatch");
        }
        std::array<std::int8_t, kFeatures> features{};
        std::array<std::int32_t, kRows> sums{};
        ternlog_vocab_qkv_signed_counts_m4(
            features.data(), reinterpret_cast<const float*>(input.data()),
            reinterpret_cast<const float*>(offsets.data()),
            reinterpret_cast<const std::uint32_t*>(polarities.data())
        );
        ternlog_vocab_qkv_signed_sdot_480x80_m4(
            sums.data(), reinterpret_cast<const std::int8_t*>(weights.data()),
            features.data()
        );
        std::cout.write(
            reinterpret_cast<const char*>(features.data()), features.size()
        );
        std::cout.write(
            reinterpret_cast<const char*>(sums.data()),
            sums.size() * sizeof(std::int32_t)
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
