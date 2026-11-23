#!/bin/sh
# Lossless round trip with the single-worker codec: compress, decompress, compare.
set -eu
input="${1:?usage: bench/roundtrip.sh INPUT}"
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin="${ISA_NET_BIN:-$root/build/isa-net-m4}"
tmp=$(mktemp -d /tmp/isa-net-rt.XXXXXX)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
"$bin" compress-adaptive-fast "$input" "$tmp/packed.tli"
"$bin" decompress-auto "$tmp/packed.tli" "$tmp/restored"
cmp "$input" "$tmp/restored"
printf 'exact round trip: %s -> %s bytes\n' "$input" "$(wc -c < "$tmp/packed.tli" | tr -d ' ')"
