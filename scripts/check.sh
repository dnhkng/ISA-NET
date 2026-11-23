#!/bin/sh
# Verify a build reproduces the reference Jolt streams exactly on held-out enwik9 slices.
# The reference sizes were produced by the original binary on an Apple M4.
set -eu
enwik9="${1:?usage: scripts/check.sh path/to/enwik9}"
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin="${ISA_NET_BIN:-$root/build/isa-net-m4}"
tmp=$(mktemp -d /tmp/isa-net-check.XXXXXX)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
status=0
for spec in "909000000 169645 validation" "959000000 180600 test"; do
  set -- $spec
  dd if="$enwik9" of="$tmp/in" bs=1000000 skip=$(( $1 / 1000000 )) count=1 2>/dev/null
  "$bin" compress-adaptive-fast "$tmp/in" "$tmp/p.tli"
  "$bin" decompress-auto "$tmp/p.tli" "$tmp/out"
  cmp "$tmp/in" "$tmp/out"
  got=$(wc -c < "$tmp/p.tli" | tr -d ' ')
  if [ "$got" = "$2" ]; then echo "ok   enwik9[$1:+1MB] ($3): $got bytes, exact round trip"
  else echo "DIFF enwik9[$1:+1MB] ($3): $got bytes, expected $2 (round trip still exact)"; status=1; fi
done
exit $status
