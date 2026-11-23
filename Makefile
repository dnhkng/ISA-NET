.PHONY: help build check roundtrip export clean

help:
	@echo "make build                     # build build/isa-net-m4 from the published Jolt model (Apple M4, macOS)"
	@echo "make check ENWIK9=path/enwik9  # build, then verify exact sizes on two held-out 1 MB slices"
	@echo "make roundtrip INPUT=file      # lossless round trip of any file"
	@echo "make export CHECKPOINT=best.pt # re-export a checkpoint into model-export/ (needs PyTorch)"

build:
	scripts/compile.sh

check: build
	@test -n "$(ENWIK9)" || (echo "ENWIK9=path/to/enwik9 is required" >&2; exit 2)
	scripts/check.sh "$(ENWIK9)"

roundtrip:
	@test -n "$(INPUT)" || (echo "INPUT is required" >&2; exit 2)
	bench/roundtrip.sh "$(INPUT)"

export:
	@test -n "$(CHECKPOINT)" || (echo "CHECKPOINT is required" >&2; exit 2)
	scripts/export.sh "$(CHECKPOINT)" model/match_mixing.json model-export

clean:
	rm -rf build model-export
