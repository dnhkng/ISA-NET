#!/usr/bin/env python3
"""Export the selected vocabulary-native story model and tokenizer for M4."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ternlog_vocab.config import VocabLMConfig


DEFAULT_CHECKPOINT = Path(
    "runs/vocab_lm/g1_official_vocab2k_continue_15mb/last.pt"
)
DEFAULT_OUTPUT = Path("native/ternlog_vocab_story_weights.bin")
DEFAULT_HEADER = Path("native/ternlog_vocab_story_layout.h")


def _constant(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    return "k" + "".join(part[:1].upper() + part[1:] for part in parts if part)


class BinaryWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.entries: dict[str, dict[str, Any]] = {}

    def add(self, name: str, array: np.ndarray, *, layout: str = "row-major") -> None:
        padding = (-len(self.data)) % 64
        self.data.extend(b"\0" * padding)
        contiguous = np.ascontiguousarray(array)
        raw = contiguous.tobytes(order="C")
        self.entries[name] = {
            "offset": len(self.data),
            "bytes": len(raw),
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "layout": layout,
        }
        self.data.extend(raw)


def _tensor(state: dict[str, torch.Tensor], name: str, dtype: str) -> np.ndarray:
    tensor = state[name].detach().cpu()
    if dtype == "f16":
        return tensor.numpy().astype("<f2", copy=False)
    if dtype == "f32":
        return tensor.numpy().astype("<f4", copy=False)
    raise ValueError(dtype)


def _quantize_rows(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = tensor.detach().float().cpu().numpy()
    maximum = np.maximum(np.abs(values).max(axis=1), 1.0e-12)
    scales = maximum / 127.0
    quantized = np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8)
    return quantized, scales.astype("<f4")


def _tokenizer_entries(corpus_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    token_bytes = [
        bytes.fromhex(value)
        for value in json.loads((corpus_dir / "token_bytes.json").read_text())
    ]
    byte_rows = np.full(256, np.iinfo(np.uint16).max, dtype="<u2")
    offsets = [0]
    data = bytearray()
    for token_id, value in enumerate(token_bytes):
        data.extend(value)
        offsets.append(len(data))
        if len(value) == 1:
            byte_rows[value[0]] = token_id
    if np.any(byte_rows == np.iinfo(np.uint16).max):
        missing = np.flatnonzero(byte_rows == np.iinfo(np.uint16).max)
        raise ValueError(f"tokenizer lacks single-byte fallback rows: {missing[:8]}")

    specification = json.loads((corpus_dir / "tokenizer.json").read_text())
    vocabulary = specification["model"]["vocab"]
    triples: list[tuple[int, int, int]] = []
    for left, right in specification["model"]["merges"]:
        merged = left + right
        if left not in vocabulary or right not in vocabulary or merged not in vocabulary:
            raise ValueError(f"invalid tokenizer merge: {(left, right)!r}")
        triples.append((vocabulary[left], vocabulary[right], vocabulary[merged]))
    return (
        {
            "tokenizer.byte_token_ids": byte_rows,
            "tokenizer.token_offsets": np.asarray(offsets, dtype="<u4"),
            "tokenizer.token_bytes": np.frombuffer(bytes(data), dtype=np.uint8),
            "tokenizer.merge_triples": np.asarray(triples, dtype="<u2"),
        },
        len(triples),
    )


def _expected_state_names(config: VocabLMConfig) -> set[str]:
    names = {
        "bigram_scale",
        "dynamic_route_strength",
        "dynamic_route_scale",
        "dynamic_route_bias",
        "token_embedding.weight",
        "bigram_embedding.weight",
        "input_norm.weight",
        "final_norm.weight",
        "readout.weight",
    }
    for layer in range(config.layers):
        prefix = f"layers.{layer}."
        names.update(
            {
                prefix + "residual_scales",
                prefix + "attention_norm.weight",
                prefix + "attention.clipped_linear_log_slope",
                prefix + "attention.qkv.weight",
                prefix + "attention.output.weight",
                prefix + "feedforward_norm.weight",
                prefix + "ff_up.weight",
                prefix + "ff_down.weight",
            }
        )
    return names


def _write_header(
    path: Path,
    config: VocabLMConfig,
    entries: dict[str, dict[str, Any]],
    *,
    vocabulary: int,
    merge_count: int,
    binary_bytes: int,
) -> None:
    def offset(name: str) -> int:
        return int(entries[name]["offset"]) if name in entries else 0

    scalar_entries = (
        "tokenizer.byte_token_ids",
        "tokenizer.token_offsets",
        "tokenizer.token_bytes",
        "tokenizer.merge_triples",
        "bigram_scale",
        "dynamic_route_strength",
        "dynamic_route_scale",
        "dynamic_route_bias",
        "token_embedding.weight",
        "int8.token_embedding.weight",
        "int8.token_embedding.scale",
        "bigram_embedding.weight",
        "input_norm.weight",
        "final_norm.weight",
    )
    lines = [
        "// Generated by export_vocab_story_m4.py; do not edit.",
        "#pragma once",
        "#include <cstddef>",
        "namespace ternlog_vocab_layout {",
        f"inline constexpr int kVocabulary = {vocabulary};",
        f"inline constexpr int kContext = {config.context_tokens};",
        f"inline constexpr int kDimension = {config.dimension};",
        f"inline constexpr int kLayers = {config.layers};",
        f"inline constexpr int kHeads = {config.heads};",
        f"inline constexpr int kKvHeads = {config.kv_heads};",
        f"inline constexpr int kBigramRows = {config.bigram_hash_size};",
        f"inline constexpr int kBigramDimension = {config.bigram_dimension};",
        f"inline constexpr int kLocalWindow = {config.local_attention_window};",
        f"inline constexpr int kLocalHeads = {config.local_attention_heads};",
        f"inline constexpr int kMergeCount = {merge_count};",
        f"inline constexpr std::size_t kBinaryBytes = {binary_bytes};",
    ]
    for name in scalar_entries:
        entry = entries[name]
        lines.append(
            f"inline constexpr std::size_t {_constant(name)} = {entry['offset']};"
        )
        lines.append(
            f"inline constexpr std::size_t {_constant(name)}Bytes = {entry['bytes']};"
        )
    lines.extend(
        [
            "struct LayerOffsets {",
            "  std::size_t residual_scales;",
            "  std::size_t attention_norm;",
            "  std::size_t attention_slopes;",
            "  std::size_t qkv;",
            "  std::size_t attention_output;",
            "  std::size_t feedforward_norm;",
            "  std::size_t ff_up;",
            "  std::size_t ff_down;",
            "  std::size_t qkv_int8;",
            "  std::size_t qkv_scale;",
            "  std::size_t attention_output_int8;",
            "  std::size_t attention_output_scale;",
            "  std::size_t ff_up_int8;",
            "  std::size_t ff_up_scale;",
            "  std::size_t ff_down_int8;",
            "  std::size_t ff_down_scale;",
            "  int ff_dimension;",
            "};",
            "inline constexpr LayerOffsets kLayerOffsets[] = {",
        ]
    )
    for layer in range(config.layers):
        prefix = f"layers.{layer}."
        ff_dimension = config.ff_dimension + (
            config.ff_extra_dimension
            if layer < config.ff_extra_dimension_layers
            else 0
        )
        offsets = [
            offset(prefix + "residual_scales"),
            offset(prefix + "attention_norm.weight"),
            offset(prefix + "attention.clipped_linear_slope"),
            offset(prefix + "attention.qkv.weight"),
            offset(prefix + "attention.output.weight"),
            offset(prefix + "feedforward_norm.weight"),
            offset(prefix + "ff_up.weight"),
            offset(prefix + "ff_down.weight"),
            offset("int8." + prefix + "attention.qkv.weight"),
            offset("int8." + prefix + "attention.qkv.scale"),
            offset("int8." + prefix + "attention.output.weight"),
            offset("int8." + prefix + "attention.output.scale"),
            offset("int8." + prefix + "ff_up.weight"),
            offset("int8." + prefix + "ff_up.scale"),
            offset("int8." + prefix + "ff_down.weight"),
            offset("int8." + prefix + "ff_down.scale"),
        ]
        lines.append(
            "  {" + ", ".join(str(value) for value in offsets) + f", {ff_dimension}}},"
        )
    lines.extend(["};", "}  // namespace ternlog_vocab_layout", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument(
        "--int8-only",
        action="store_true",
        help="omit FP16 dense matrices that the W8A8 path does not execute",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = VocabLMConfig.from_dict(checkpoint["config"])
    expected = {
        "dimension": 224,
        "layers": 10,
        "heads": 7,
        "kv_heads": 7,
        "token_feature": "hashed_bigram",
        "attention_normalizer": "clipped_linear",
        "attention_window_policy": "head_local",
        "residual_routing": "dynamic_input",
        "feedforward_activation": "relu_squared",
        "tie_embeddings": True,
        "qk_norm": True,
    }
    mismatches = {
        name: (getattr(config, name), value)
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint does not match vocabulary-native ABI: {mismatches}")
    corpus_dir = Path(config.corpus_dir)
    corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text())
    vocabulary = int(corpus_manifest["logical_vocabulary_size"])
    if vocabulary != 2048:
        raise ValueError("the selected native ABI currently requires the 2K vocabulary")

    state = checkpoint["model_state_dict"]
    expected_names = _expected_state_names(config)
    missing = expected_names - set(state)
    extra = set(state) - expected_names
    if missing or extra:
        raise ValueError(f"unexpected state tensors; missing={missing}, extra={extra}")
    if not torch.equal(state["readout.weight"], state["token_embedding.weight"]):
        raise ValueError("native ABI requires the tied readout to equal token embedding")

    writer = BinaryWriter()
    tokenizer_arrays, merge_count = _tokenizer_entries(corpus_dir)
    for name, array in tokenizer_arrays.items():
        writer.add(name, array, layout="tokenizer-table")
    for name in (
        "bigram_scale",
        "dynamic_route_strength",
        "dynamic_route_scale",
        "dynamic_route_bias",
    ):
        writer.add(name, _tensor(state, name, "f32"), layout="scalar")
    for name in (
        "token_embedding.weight",
        "bigram_embedding.weight",
        "input_norm.weight",
    ):
        writer.add(name, _tensor(state, name, "f16"))
    for layer in range(config.layers):
        prefix = f"layers.{layer}."
        writer.add(prefix + "residual_scales", _tensor(state, prefix + "residual_scales", "f32"), layout="scalar")
        writer.add(prefix + "attention_norm.weight", _tensor(state, prefix + "attention_norm.weight", "f16"))
        slope = state[prefix + "attention.clipped_linear_log_slope"].detach().float().exp().cpu().numpy().astype("<f4")
        writer.add(prefix + "attention.clipped_linear_slope", slope, layout="per-head")
        for suffix in (
            "attention.qkv.weight",
            "attention.output.weight",
            "feedforward_norm.weight",
            "ff_up.weight",
            "ff_down.weight",
        ):
            if args.int8_only and suffix != "feedforward_norm.weight":
                continue
            writer.add(prefix + suffix, _tensor(state, prefix + suffix, "f16"))
    writer.add("final_norm.weight", _tensor(state, "final_norm.weight", "f16"))

    dense_names = ["token_embedding.weight"]
    for layer in range(config.layers):
        prefix = f"layers.{layer}."
        dense_names.extend(
            prefix + suffix
            for suffix in (
                "attention.qkv.weight",
                "attention.output.weight",
                "ff_up.weight",
                "ff_down.weight",
            )
        )
    for name in dense_names:
        quantized, scales = _quantize_rows(state[name])
        writer.add("int8." + name, quantized, layout="row-major-int8")
        writer.add(
            "int8." + name.removesuffix(".weight") + ".scale",
            scales,
            layout="per-row",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(writer.data)
    _write_header(
        args.header,
        config,
        writer.entries,
        vocabulary=vocabulary,
        merge_count=merge_count,
        binary_bytes=len(writer.data),
    )
    manifest = {
        "format": (
            "ternlog-vocab-story-m4-int8-production-v2"
            if args.int8_only
            else "ternlog-vocab-story-m4-fp16-int8-experimental-v2"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": checkpoint.get("update"),
        "decoded_bytes_processed": checkpoint.get("decoded_bytes_processed"),
        "validation": checkpoint.get("validation"),
        "logical_vocabulary_size": vocabulary,
        "tokenizer_sha256": corpus_manifest["tokenizer_sha256"],
        "source_train_sha256": corpus_manifest["source_train_sha256"],
        "source_validation_sha256": corpus_manifest["source_validation_sha256"],
        "parameter_count": sum(parameter.numel() for parameter in state.values())
        - state["readout.weight"].numel(),
        "binary_bytes": len(writer.data),
        "matrix_layout": "row-major",
        "dense_kernel": (
            "eight-row dynamic-W8A8 FP32-scaled SDOT"
            if args.int8_only
            else "eight-row FP16-activation FP32-FMLAL plus experimental W8A8"
        ),
        "native_operations": {
            "tokenizer": ["integer merge-table lookup", "direct byte-table decode"],
            "embedding": ["integer address", "FP16 load", "FCVTL"],
            "hashed_bigram": ["MUL", "EOR", "AND", "FP16 load", "FMLA"],
            "rms_and_qk_norm": ["FMLA reduction", "FSQRT", "FDIV", "FMUL"],
            "rope": ["shared scalar SINCOS", "FMUL", "FMLA/FMLS"],
            "dense": (
                ["FMAX reduction", "FDIV", "FCVTNS", "SQXTN", "SDOT", "SCVTF", "FMUL"]
                if args.int8_only
                else ["FCVTN", "FMLAL", "FMLAL2", "FADDP"]
            ),
            "attention": [
                "FMLAL key dot", "FMAX", "FMLA", "FP16 probability rounding", "FMLAL value accumulation", "reduction", "FDIV"
            ],
            "feedforward": ["FMAX", "FMUL"],
            "residual_routing": ["FMLA reduction", "FMAX", "FMIN", "FMUL", "FMLA"],
            "kv_cache": ["FP16 store/load", "ring-buffer integer addressing"],
            "greedy_selection": ["NEON FMAX reduction", "first-index recovery"],
            "sampling": [
                "top-k partial sort", "FMAX/FMIN", "FCVTMS", "SHL/ADD exponent bits", "quartic FMLA"
            ],
        },
        "config": config.to_dict(),
        "tensors": writer.entries,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
