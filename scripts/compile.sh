#!/bin/sh
# Compile the ISA-NET codec for Apple M4 from a layout header + weight image.
# usage: scripts/compile.sh [HEADER WEIGHTS OUTPUT]
# Defaults build the published Jolt model in model/ into build/isa-net-m4.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
header="${1:-$root/model/ternlog_vocab_boolean_story_layout.h}"
weights="${2:-$root/model/ternlog_vocab_boolean_story_weights.bin}"
output="${3:-$root/build/isa-net-m4}"
src="$root/native"
mkdir -p "$(dirname -- "$output")"

# -ffast-math and -mcpu=apple-m4 are part of the codec's numeric contract:
# encoder and decoder must be built identically. Don't change them casually.
clang++ -O3 -std=c++20 -mcpu=apple-m4 -ffast-math \
  -I"$src" \
  -DTERNLOG_BOOLEAN_LAYOUT_HEADER="\"$header\"" \
  -DTERNLOG_BOOLEAN_WEIGHTS_PATH="\"$weights\"" \
  -DTERNLOG_BOOLEAN_QKV_INT8 -DTERNLOG_BOOLEAN_QKV_TICKETS \
  -DTERNLOG_SHARED_VALUE_HEAD -DTERNLOG_BOOLEAN_OUTPUT_TICKETS16 \
  -DTERNLOG_FUSED_QKV_RMS_QUANT -DTERNLOG_BOOLEAN_READOUT_FACTORIZED_NATIVE \
  -DTERNLOG_BOOLEAN_FFN_ROUTE_POOL_NEON -DTERNLOG_BOOLEAN_FFN_ROUTE_POOL_PADDED \
  -DTERNLOG_INT8_QK_CACHE_FIXED4 -DTERNLOG_INT8_VALUE_CACHE_SCALAR \
  -DTERNLOG_QKV_VALUE_CACHE_FUSED_SCALAR -DTERNLOG_BOOLEAN_AX1 \
  -DTERNLOG_BOOLEAN_FFN_ENCODE_NO_MAIN -DTERNLOG_BOOLEAN_FFN_ROUTES_NO_MAIN \
  -DTERNLOG_VOCAB_BOOLEAN_READOUT_NO_MAIN -DTERNLOG_FP16_KEY_CACHE \
  -DTERNLOG_FP16_VALUE_CACHE -DTERNLOG_SHARED_ROPE -DTERNLOG_PREROUNDED_MATVEC \
  -DTERNLOG_FMLAL_VALUE_ATTENTION -DTERNLOG_FUSED_CLIPPED_ATTENTION \
  -DTERNLOG_OUTPUT_RESIDUAL_RMS_FUSED -DTERNLOG_BOOLEAN_FFN_ENCODE_NEON \
  -DTERNLOG_QKV_BATCH4_HOOK -DTERNLOG_FFN_BATCH4_HOOK \
  "$src/ternlog_byte_codec_m4.cpp" \
  "$src/m4_vocab_boolean_ffn_ax1.cpp" \
  "$src/m4_vocab_boolean_ffn_sdot_schedules.cpp" \
  "$src/m4_vocab_ffn_batch4_raw.cpp" \
  "$src/m4_vocab_boolean_ffn_encode.cpp" \
  "$src/m4_vocab_boolean_ffn_routes.cpp" \
  "$src/m4_vocab_boolean_readout.cpp" \
  "$src/m4_vocab_qkv_sdot.S" \
  "$src/m4_vocab_qkv_packed_sdot.S" \
  "$src/m4_vocab_qkv_signed_squeeze.cpp" \
  "$src/m4_story_matvec_fmlal.S" \
  "$src/m4_story_matvec_fmlal_half.S" \
  "$src/m4_fiber_switch.S" \
  "$src/ternlog_vocab_boolean_story_weights.S" \
  -DTERNLOG_QKV_SIGNED_SQUEEZE_NO_MAIN -lz -o "$output"
echo "built $output"
