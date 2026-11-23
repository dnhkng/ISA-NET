#include <arm_neon.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

std::uint64_t rotate_left(std::uint64_t value, unsigned shift) {
    shift &= 63U;
    return shift == 0 ? value : (value << shift) | (value >> (64U - shift));
}

void route_round(
    const std::uint8_t* state,
    std::uint8_t* byte_rotated,
    std::uint8_t* lane_rotated,
    std::size_t bytes,
    unsigned shift
) {
    const std::size_t byte_shift = shift % bytes;
    for (std::size_t output = 0; output < bytes; ++output) {
        byte_rotated[output] = state[(output + bytes - byte_shift) % bytes];
    }
    for (std::size_t lane = 0; lane < bytes; lane += 8) {
        std::uint64_t value = 0;
        for (unsigned byte = 0; byte < 8; ++byte) {
            value |= static_cast<std::uint64_t>(state[lane + byte]) << (8U * byte);
        }
        value = rotate_left(value, shift);
        for (unsigned byte = 0; byte < 8; ++byte) {
            lane_rotated[lane + byte] = static_cast<std::uint8_t>(
                value >> (8U * byte)
            );
        }
    }
}

void boolean_round(
    const std::uint8_t* state,
    const std::uint8_t* second,
    const std::uint8_t* third,
    const std::uint8_t* program,
    const std::uint8_t* gates,
    std::uint8_t* output,
    std::size_t bytes,
    const std::string& cell
) {
    for (std::size_t offset = 0; offset < bytes; offset += 16) {
        if (gates != nullptr && gates[offset / 16] == 0) {
            vst1q_u8(output + offset, vld1q_u8(state + offset));
            continue;
        }
        const uint8x16_t a = vld1q_u8(state + offset);
        const uint8x16_t b = vld1q_u8(second + offset);
        const uint8x16_t c = vld1q_u8(third + offset);
        uint8x16_t branch;
        if (cell == "lut3") {
            const uint8x16_t p0 = vld1q_u8(program + 0 * bytes + offset);
            const uint8x16_t p1 = vld1q_u8(program + 1 * bytes + offset);
            const uint8x16_t p2 = vld1q_u8(program + 2 * bytes + offset);
            const uint8x16_t p3 = vld1q_u8(program + 3 * bytes + offset);
            const uint8x16_t p4 = vld1q_u8(program + 4 * bytes + offset);
            const uint8x16_t p5 = vld1q_u8(program + 5 * bytes + offset);
            const uint8x16_t p6 = vld1q_u8(program + 6 * bytes + offset);
            const uint8x16_t p7 = vld1q_u8(program + 7 * bytes + offset);
            const uint8x16_t low00 = vbslq_u8(c, p1, p0);
            const uint8x16_t low01 = vbslq_u8(c, p3, p2);
            const uint8x16_t high10 = vbslq_u8(c, p5, p4);
            const uint8x16_t high11 = vbslq_u8(c, p7, p6);
            const uint8x16_t low = vbslq_u8(b, low01, low00);
            const uint8x16_t high = vbslq_u8(b, high11, high10);
            branch = vbslq_u8(a, high, low);
        } else if (cell == "factorized_minterm") {
            const uint8x16_t enable = vld1q_u8(program + 0 * bytes + offset);
            const uint8x16_t polarity_a = vld1q_u8(program + 1 * bytes + offset);
            const uint8x16_t polarity_b = vld1q_u8(program + 2 * bytes + offset);
            const uint8x16_t polarity_c = vld1q_u8(program + 3 * bytes + offset);
            const uint8x16_t ones = vdupq_n_u8(0xff);
            const uint8x16_t match_a = veor3q_u8(a, polarity_a, ones);
            const uint8x16_t match_b = veor3q_u8(b, polarity_b, ones);
            const uint8x16_t match_c = veor3q_u8(c, polarity_c, ones);
            branch = vandq_u8(enable, vandq_u8(match_a, vandq_u8(match_b, match_c)));
        } else if (cell == "dual3") {
            const uint8x16_t selector = vld1q_u8(program + 0 * bytes + offset);
            const uint8x16_t bias = vld1q_u8(program + 1 * bytes + offset);
            const uint8x16_t nonlinear_a = vbcaxq_u8(a, b, c);
            const uint8x16_t nonlinear_b = vbslq_u8(a, b, c);
            const uint8x16_t selected = vbslq_u8(selector, nonlinear_a, nonlinear_b);
            vst1q_u8(output + offset, veor3q_u8(a, selected, bias));
            continue;
        } else {
            throw std::runtime_error("unknown Boolean cell " + cell);
        }
        vst1q_u8(output + offset, veorq_u8(a, branch));
    }
}

}  // namespace

#ifndef TERNLOG_BOOLEAN_FFN_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
            std::cerr << "usage: m4_vocab_boolean_ffn_core STATE PROGRAM GATES|- ROUNDS [CELL]\n";
            return 2;
        }
        std::vector<std::uint8_t> state = read_file(argv[1]);
        const std::vector<std::uint8_t> program = read_file(argv[2]);
        const int rounds = std::stoi(argv[4]);
        const std::string cell = argc == 6 ? argv[5] : "lut3";
        if (state.empty() || state.size() % 16 != 0 || rounds <= 0) {
            throw std::runtime_error("state must contain whole 128-bit blocks");
        }
        const std::size_t bytes = state.size();
        const std::size_t planes = cell == "lut3" ? 8 : cell == "factorized_minterm" ? 4 : cell == "dual3" ? 2 : 0;
        if (planes == 0) throw std::runtime_error("unknown Boolean cell " + cell);
        if (program.size() != static_cast<std::size_t>(rounds) * planes * bytes) {
            throw std::runtime_error("program size does not match state and rounds");
        }
        std::vector<std::uint8_t> gates;
        if (std::string(argv[3]) != "-") {
            gates = read_file(argv[3]);
            if (gates.size() != static_cast<std::size_t>(rounds) * bytes / 16) {
                throw std::runtime_error("gate size does not match hardware blocks");
            }
        }
        std::vector<std::uint8_t> previous(bytes);
        std::vector<std::uint8_t> second(bytes);
        std::vector<std::uint8_t> third(bytes);
        for (int round = 0; round < rounds; ++round) {
            previous = state;
            const unsigned shift = 1U << (round % 5);
            route_round(previous.data(), second.data(), third.data(), bytes, shift);
            boolean_round(
                previous.data(), second.data(), third.data(),
                program.data() + static_cast<std::size_t>(round) * planes * bytes,
                gates.empty()
                    ? nullptr
                    : gates.data() + static_cast<std::size_t>(round) * bytes / 16,
                state.data(), bytes, cell
            );
        }
        std::cout.write(
            reinterpret_cast<const char*>(state.data()),
            static_cast<std::streamsize>(state.size())
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
