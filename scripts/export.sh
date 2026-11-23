#!/bin/sh
# Export a trained checkpoint to the native layout header + weight image.
# usage: scripts/export.sh CHECKPOINT.pt MATCH_MIXING.json OUTPUT_DIR
# Needs Python 3.11+, NumPy and PyTorch (CPU is fine).
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
checkpoint="${1:?usage: scripts/export.sh CHECKPOINT.pt MATCH_MIXING.json OUTPUT_DIR}"
mixing="${2:?usage: scripts/export.sh CHECKPOINT.pt MATCH_MIXING.json OUTPUT_DIR}"
out="${3:?usage: scripts/export.sh CHECKPOINT.pt MATCH_MIXING.json OUTPUT_DIR}"
mkdir -p "$out"
PYTHONPATH="$root/python/src:$root/python/tools" "${PYTHON:-python3}" \
  "$root/python/tools/export_vocab_boolean_story_m4.py" "$checkpoint" \
  --output "$out/ternlog_vocab_boolean_story_weights.bin" \
  --header "$out/ternlog_vocab_boolean_story_layout.h" \
  --corpus-manifest "$root/model/corpus_manifest.json" \
  --match-mixing "$mixing" \
  --dense-mode qkv_tickets_prepacked_output16 \
  --qk-cache-precision int8_fixed4 --qkv-active-fraction 1.0
echo "exported to $out"
