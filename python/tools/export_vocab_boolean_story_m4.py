#!/usr/bin/env python3
"""Export the selected AX1 Boolean vocabulary Transformer for Apple M4."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from export_vocab_story_m4 import BinaryWriter, _tensor, _tokenizer_entries
from ternlog_vocab.config import VocabLMConfig


DEFAULT_CHECKPOINT = Path(
    "runs/vocab_lm/ax1_local32_fp16_seed20260825_3mb/last.pt"
)
DEFAULT_OUTPUT = Path("native/ternlog_vocab_boolean_story_weights.bin")
DEFAULT_HEADER = Path("native/ternlog_vocab_boolean_story_layout.h")


def inference_config_from_checkpoint(values: dict) -> VocabLMConfig:
    """Accept newer training-only metadata without weakening model validation."""
    export_config = dict(values)
    # The bundled inference config predates matched-region training. Neither
    # region size nor sampler changes inference; preserve the source dictionary.
    for field in ("training_region_tokens", "validation_domain_bytes",
                  "training_decoded_bytes_per_update", "training_micro_batch_size",
                  "run_source_decoded_bytes", "save_gate_checkpoints"):
        export_config.pop(field, None)
    if export_config.get("training_sampling") in {"shared_regions", "byte_proportional"}:
        export_config["training_sampling"] = "blocks"
    if export_config.get("validation_sampling") == "domain_bytes":
        export_config["validation_sampling"] = "spread"
    return VocabLMConfig.from_dict(export_config)


def scalable_native_abi_errors(config: VocabLMConfig) -> list[str]:
    """Return shape-contract failures for the compile-time AX1 lowering."""
    qk_head_dimension = (
        config.qk_head_dimension or config.dimension // config.heads
    )
    local_qk_head_dimension = (
        config.local_qk_head_dimension or qk_head_dimension
    )
    key_heads = (
        config.kv_heads
        if config.kv_head_sharing in {"both", "key"}
        else config.heads
    )
    value_heads = (
        config.kv_heads
        if config.kv_head_sharing in {"both", "value"}
        else config.heads
    )
    qkv_rows = (
        (config.heads + key_heads) * qk_head_dimension
        + value_heads * (config.dimension // config.heads)
    )
    requirements = {
        "dimension_multiple_16": config.dimension % 16 == 0,
        "eight_boolean_planes": config.boolean_width == config.dimension * 8,
        "correction_multiple_16": (
            config.boolean_ffn_correction_features > 0
            and config.boolean_ffn_correction_features % 16 == 0
        ),
        "equal_trajectory_partition": (
            config.boolean_ffn_correction_features
            % (config.boolean_rounds + 1)
            == 0
        ),
        "value_head_dimension_multiple_4": (
            config.dimension // config.heads
        ) % 4 == 0,
        "qk_head_dimension_multiple_4": (
            qk_head_dimension > 0 and qk_head_dimension % 4 == 0
        ),
        "local_qk_head_dimension_multiple_4": (
            local_qk_head_dimension > 0
            and local_qk_head_dimension % 4 == 0
        ),
        "qkv_rows_multiple_8": qkv_rows % 8 == 0,
        "layer_mask_capacity": 0 < config.layers <= 64,
        "local_heads_in_range": (
            0 <= config.local_attention_heads <= config.heads
        ),
        "vocab_rank_positive_multiple_16": (
            config.boolean_vocab_correction_rank > 0
            and config.boolean_vocab_correction_rank % 16 == 0
        ),
        "vocab_correction_features_positive": (
            config.dimension - config.boolean_vocab_code_width - 1 > 0
        ),
    }
    return [name for name, valid in requirements.items() if not valid]


def quantize_rows(weight: torch.Tensor, *, divisor: float = 1.0):
    values = weight.detach().float().cpu().numpy()
    maximum = np.maximum(np.abs(values).max(axis=1), 1.0e-12)
    scales = maximum / 127.0
    quantized = np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8)
    return quantized, (scales / divisor).astype("<f4")


def quantize_decoder_group4(weight: torch.Tensor):
    """Quantize each four-plane decoder group and pad rows to one SDOT vector."""
    values = weight.detach().float().cpu().reshape(weight.shape[0], 2, 4)
    scales = values.abs().amax(dim=2).clamp_min(1.0e-12) / 127.0
    quantized = torch.round(values / scales[:, :, None]).clamp(-127, 127)
    padded = torch.zeros(weight.shape[0], 16, dtype=torch.int8)
    padded[:, :8] = quantized.to(torch.int8).reshape(weight.shape[0], 8)
    return padded.numpy(), scales.numpy().astype("<f4")


def quantize_decoder_rows(weight: torch.Tensor):
    quantized, scales = quantize_rows(weight)
    padded = np.zeros((weight.shape[0], 16), dtype=np.int8)
    padded[:, :8] = quantized
    return padded, scales


def quantize_readout(weight: torch.Tensor):
    padded_features = (weight.shape[1] + 31) // 32 * 32
    scale_groups = padded_features // 32
    padded = torch.zeros(
        weight.shape[0], padded_features, dtype=weight.dtype
    )
    padded[:, : weight.shape[1]] = weight
    groups = padded.reshape(weight.shape[0], scale_groups, 32)
    scales = groups.abs().amax(dim=2).clamp_min(1.0e-8) / 127.0
    quantized = torch.round(groups / scales[:, :, None]).clamp(-127, 127)
    return (
        quantized.to(torch.int8).reshape(
            weight.shape[0], padded_features
        ).numpy(),
        (scales / 4.0).numpy().astype("<f4"),
    )


def pack_qkv_tickets(weight: torch.Tensor, mask: torch.Tensor, block_rows: int):
    expanded = mask.repeat_interleave(block_rows, 0).repeat_interleave(16, 1)
    quantized, scales = quantize_rows(weight * expanded)
    mask_array = mask.detach().cpu().numpy().astype(np.uint8)
    offsets = [0]
    columns: list[int] = []
    tickets = []
    for block_row in range(mask_array.shape[0]):
        for block_column in np.flatnonzero(mask_array[block_row]):
            columns.append(int(block_column))
            row = block_row * block_rows
            column = int(block_column) * 16
            tickets.append(quantized[row : row + block_rows, column : column + 16])
        offsets.append(len(columns))
    return (
        np.asarray(offsets, dtype="<u2"),
        np.asarray(columns, dtype=np.uint8),
        np.stack(tickets).astype(np.int8),
        scales,
    )


def pack_product_codebook(
    codebook: torch.Tensor,
    code_index_logits: torch.Tensor,
):
    blocks, codewords, width = codebook.shape
    if width != 16 or codewords not in {32, 64}:
        raise ValueError("invalid product-codebook export geometry")
    quantized = codebook.detach().round().clamp(-127, 127).to(torch.int8)
    packed = (
        quantized.reshape(blocks, codewords // 4, 4, 4, 4)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .numpy()
    )
    indices = code_index_logits.detach().argmax(dim=-1).to(torch.uint8).numpy()
    return packed, indices


def expanded_block_mask(mask: torch.Tensor, block_rows: int) -> torch.Tensor:
    return mask.repeat_interleave(block_rows, 0).repeat_interleave(16, 1)


def magnitude_block_mask(
    weight: torch.Tensor, *, block_rows: int, active_fraction: float
) -> torch.Tensor:
    row_blocks = weight.shape[0] // block_rows
    column_blocks = weight.shape[1] // 16
    scores = weight.detach().reshape(
        row_blocks, block_rows, column_blocks, 16
    ).square().sum(dim=(1, 3))
    active = max(1, round(scores.numel() * active_fraction))
    mask = torch.zeros_like(scores)
    mask.flatten()[scores.flatten().topk(active).indices] = 1
    return mask


def write_header(
    path: Path,
    config: VocabLMConfig,
    entries: dict[str, dict],
    *,
    vocabulary: int,
    merge_count: int,
    binary_bytes: int,
    model_fingerprint: bytes,
    match_mixing: dict | None,
) -> None:
    def offset(name: str) -> int:
        return int(entries[name]["offset"]) if name in entries else 0

    ffn_pool_schedule = {
        "legacy_mixed": 0,
        "state_aligned": 1,
        "state_rotated": 2,
    }[config.boolean_ffn_pool_schedule]
    lines = [
        "// Generated by export_vocab_boolean_story_m4.py; do not edit.",
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "namespace ternlog_vocab_layout {",
        f"inline constexpr int kVocabulary = {vocabulary};",
        f"inline constexpr int kContext = {config.context_tokens};",
        f"inline constexpr int kDimension = {config.dimension};",
        f"inline constexpr int kLayers = {config.layers};",
        "inline constexpr std::uint64_t kAttentionSkipMask = "
        f"{sum(1 << layer for layer in config.attention_skip_layers)}ULL;",
        "inline constexpr std::uint64_t kAttentionSharedInputMask = "
        f"{sum(1 << layer for layer in config.attention_shared_input_layers)}ULL;",
        "inline constexpr int kAttentionSharedInputFirst = "
        f"{min(config.attention_shared_input_layers) if config.attention_shared_input_layers else -1};",
        f"inline constexpr int kHeads = {config.heads};",
        f"inline constexpr int kKvHeads = {config.kv_heads};",
        f"inline constexpr int kKeyHeads = {config.kv_heads if config.kv_head_sharing in {'both', 'key'} else config.heads};",
        f"inline constexpr int kValueHeads = {config.kv_heads if config.kv_head_sharing in {'both', 'value'} else config.heads};",
        f"inline constexpr int kQkHeadDimension = {config.qk_head_dimension or config.dimension // config.heads};",
        f"inline constexpr int kLocalQkHeadDimension = {config.local_qk_head_dimension or config.qk_head_dimension or config.dimension // config.heads};",
        "inline constexpr int kValueHeadDimension = kDimension / kHeads;",
        "inline constexpr int kQkvRows = (kHeads + kKeyHeads) * kQkHeadDimension + kValueHeads * kValueHeadDimension;",
        f"inline constexpr int kQkvSharedBases = {config.qkv_shared_bases if config.attention_qkv_projection_mode == 'shared_qk_basis' else 0};",
        "inline constexpr int kQkvBaseRows = 2 * kQkvSharedBases * kQkHeadDimension + kValueHeads * kValueHeadDimension;",
        f"inline constexpr int kQkvLocalWidth = {config.qkv_local_width};",
        f"inline constexpr int kQkvGlobalRank = {config.qkv_global_rank};",
        f"inline constexpr int kQkvRouteCount = {config.qkv_route_count};",
        f"inline constexpr int kBooleanFfnCorrectionFeatures = {config.boolean_ffn_correction_features};",
        f"inline constexpr int kBooleanFfnCorrectionRank = {config.boolean_ffn_correction_rank};",
        f"inline constexpr int kBooleanFfnPoolSchedule = {ffn_pool_schedule};",
        f"inline constexpr int kBooleanVocabCorrectionRank = {config.boolean_vocab_correction_rank};",
        f"inline constexpr int kBooleanVocabCorrectionFeatures = {config.dimension - config.boolean_vocab_code_width - 1 if config.vocabulary_path == 'boolean_hybrid' else 0};",
        "inline constexpr bool kVocabularyHierarchy32x64 = "
        + (
            "true;"
            if config.vocabulary_hierarchy == "balanced_32x64"
            else "false;"
        ),
        f"inline constexpr int kRopePairs = {int((config.qk_head_dimension or config.dimension / config.heads) / 2 * config.rope_fraction)};",
        f"inline constexpr float kAttentionScale = {1.0 / np.sqrt(config.qk_head_dimension or config.dimension / config.heads):.17g}F;",
        f"inline constexpr float kLocalAttentionScale = {1.0 / np.sqrt(config.local_qk_head_dimension or config.qk_head_dimension or config.dimension / config.heads):.17g}F;",
        "inline constexpr int kBigramRows = 1;",
        "inline constexpr int kBigramDimension = 0;",
        f"inline constexpr int kLocalWindow = {config.local_attention_window};",
        f"inline constexpr int kLocalHeads = {config.local_attention_heads};",
        "inline constexpr bool kQkvSignedSqueeze = "
        + (
            "true;"
            if config.attention_qkv_projection_mode == "signed_squeeze_hybrid"
            else "false;"
        ),
        f"inline constexpr int kQkvSqueezeBitplanes = {config.qkv_squeeze_bitplanes};",
        f"inline constexpr int kQkvSqueezePolarities = {config.qkv_squeeze_polarities};",
        "inline constexpr int kQkvSqueezeFeatures = "
        f"{config.dimension * config.qkv_squeeze_bitplanes // config.qkv_squeeze_group_bits * config.qkv_squeeze_polarities};",
        f"inline constexpr int kQkvSqueezeBypass = {config.qkv_squeeze_bypass};",
        "inline constexpr int kQkvSqueezeRoundingScale = "
        + str(
            {"fp16": 0, "fixed8": 8, "fixed4": 4}.get(
                config.qkv_squeeze_input_rounding, 0
            )
        )
        + ";",
        "inline constexpr bool kAttentionOutputInputFP16 = "
        + ("true;" if config.attention_output_input_rounding == "fp16" else "false;"),
        f"inline constexpr int kMergeCount = {merge_count};",
        f"inline constexpr int kNgramOrderCount = {len(config.ngram_orders)};",
        f"inline constexpr int kNgramHeads = {config.ngram_heads if config.ngram_orders else 0};",
        f"inline constexpr int kNgramTableRows = {config.ngram_table_rows if config.ngram_orders else 0};",
        f"inline constexpr int kNgramDimension = {config.ngram_dimension if config.ngram_orders else 0};",
        "inline constexpr int kNgramOrders[] = {"
        + (", ".join(str(value) for value in config.ngram_orders) or "0")
        + "};",
        "inline constexpr std::uint8_t kModelFingerprint[16] = {"
        + ", ".join(str(value) for value in model_fingerprint)
        + "};",
        f"inline constexpr std::size_t kBinaryBytes = {binary_bytes};",
    ]
    if match_mixing is not None:
        lines.extend([
            f"inline constexpr std::uint32_t kMatchHistory = {int(match_mixing['history'])}U;",
            f"inline constexpr int kMatchTableBits = {int(match_mixing['table_bits'])};",
            f"inline constexpr int kMatchContext = {int(match_mixing['context'])};",
            "inline constexpr double kMatchWeights[4] = {"
            + ", ".join(f"{float(value):.17g}" for value in match_mixing["weights"])
            + "};",
            "inline constexpr double kMatchConfidences[4] = {"
            + ", ".join(f"{float(value):.17g}" for value in match_mixing["confidences"])
            + "};",
        ])
    else:
        lines.extend([
            "inline constexpr std::uint32_t kMatchHistory = 0U;",
            "inline constexpr int kMatchTableBits = 0;",
            "inline constexpr int kMatchContext = 0;",
            "inline constexpr double kMatchWeights[4] = {0, 0, 0, 0};",
            "inline constexpr double kMatchConfidences[4] = {0, 0, 0, 0};",
        ])
    globals_ = {
        "kTokenizerByteTokenIds": "tokenizer.byte_token_ids",
        "kTokenizerTokenOffsets": "tokenizer.token_offsets",
        "kTokenizerTokenBytes": "tokenizer.token_bytes",
        "kTokenizerMergeTriples": "tokenizer.merge_triples",
        "kBooleanTokenCodes": "boolean_token_codes",
        "kNgramTableInt8": "ngram_table.int8",
        "kNgramTableScales": "ngram_table.scales",
        "kInputNormWeight": "input_norm.weight",
        "kFinalNormWeight": "final_norm.weight",
        "kReadoutCorrection": "readout.correction_int8",
        "kReadoutCorrectionScales": "readout.correction_scales",
        "kReadoutCorrectionOutput": "readout.correction_output_f16",
        "kReadoutCorrectionOutputInt8": "readout.correction_output_int8",
        "kReadoutCorrectionOutputScales": "readout.correction_output_scales",
        "kReadoutBias": "readout.bias",
        "kReadoutHierarchyLeafCodes": "readout.hierarchy_leaf_codes",
        "kReadoutHierarchyClusterCodes": "readout.hierarchy_cluster_codes",
        "kReadoutHierarchyClusterOutput": "readout.hierarchy_cluster_output_f16",
        "kReadoutHierarchyClusterBias": "readout.hierarchy_cluster_bias",
        "kReadoutHierarchyTokenIds": "readout.hierarchy_token_ids",
        "kReadoutHierarchyTokenIndex": "readout.hierarchy_token_index",
    }
    for constant, name in globals_.items():
        lines.append(f"inline constexpr std::size_t {constant} = {offset(name)};")
        lines.append(
            f"inline constexpr std::size_t {constant}Bytes = {entries[name]['bytes']};"
        )
    lines.extend(
        [
            "struct LayerOffsets {",
            "  std::size_t residual_scales;",
            "  std::size_t attention_norm;",
            "  std::size_t attention_slopes;",
            "  std::size_t value_cache_scale;",
            "  std::size_t qkv;",
            "  std::size_t attention_output;",
            "  std::size_t qkv_int8;",
            "  std::size_t qkv_scales;",
            "  std::size_t qkv_query_transform;",
            "  std::size_t qkv_key_transform;",
            "  std::size_t qkv_local_int8;",
            "  std::size_t qkv_local_scales;",
            "  std::size_t qkv_global_int8;",
            "  std::size_t qkv_global_scales;",
            "  std::size_t qkv_global_transform;",
            "  std::size_t qkv_route_weight;",
            "  std::size_t qkv_route_offsets;",
            "  std::size_t qkv_value_int8;",
            "  std::size_t qkv_value_scales;",
            "  std::size_t qkv_ticket_offsets;",
            "  std::size_t qkv_ticket_columns;",
            "  std::size_t qkv_value_block_mask;",
            "  std::size_t qkv_squeeze_offsets;",
            "  std::size_t qkv_squeeze_thresholds;",
            "  std::size_t qkv_squeeze_polarities;",
            "  std::size_t qkv_squeeze_int8;",
            "  std::size_t qkv_squeeze_scales;",
            "  std::size_t qkv_bypass_int8;",
            "  std::size_t qkv_bypass_scales;",
            "  std::size_t qkv_bypass_start;",
            "  std::size_t attention_output_int8;",
            "  std::size_t attention_output_scales;",
            "  std::size_t attention_output_block_mask;",
            "  std::size_t attention_output_ticket_offsets;",
            "  std::size_t attention_output_ticket_columns;",
            "  std::size_t feedforward_norm;",
            "  std::size_t encoder_offsets;",
            "  std::size_t boolean_program;",
            "  std::size_t decoder_weight;",
            "  std::size_t decoder_int8;",
            "  std::size_t decoder_scales;",
            "  std::size_t decoder_bias;",
            "  std::size_t correction_int8;",
            "  std::size_t correction_scales;",
            "  std::size_t correction_ticket_offsets;",
            "  std::size_t correction_ticket_columns;",
            "  std::size_t correction_output_f16;",
            "};",
            "inline constexpr LayerOffsets kLayerOffsets[] = {",
        ]
    )
    for layer in range(config.layers):
        prefix = f"layers.{layer}."
        names = (
            "residual_scales",
            "attention_norm.weight",
            "attention.clipped_linear_slope",
            "attention.value_cache_scale",
            "attention.qkv.weight",
            "attention.output.weight",
            "attention.qkv_int8",
            "attention.qkv_scales",
            "attention.qkv_query_transform",
            "attention.qkv_key_transform",
            "attention.qkv_local_int8",
            "attention.qkv_local_scales",
            "attention.qkv_global_int8",
            "attention.qkv_global_scales",
            "attention.qkv_global_transform",
            "attention.qkv_route_weight",
            "attention.qkv_route_offsets",
            "attention.qkv_value_int8",
            "attention.qkv_value_scales",
            "attention.qkv_ticket_offsets",
            "attention.qkv_ticket_columns",
            "attention.qkv_value_block_mask",
            "attention.qkv_squeeze_offsets",
            "attention.qkv_squeeze_thresholds",
            "attention.qkv_squeeze_polarities",
            "attention.qkv_squeeze_int8",
            "attention.qkv_squeeze_scales",
            "attention.qkv_bypass_int8",
            "attention.qkv_bypass_scales",
            "attention.qkv_bypass_start",
            "attention.output_int8",
            "attention.output_scales",
            "attention.output_block_mask",
            "attention.output_ticket_offsets",
            "attention.output_ticket_columns",
            "feedforward_norm.weight",
            "boolean_ffn.encoder_offsets",
            "boolean_ffn.program",
            "boolean_ffn.decoder_weight",
            "boolean_ffn.decoder_int8",
            "boolean_ffn.decoder_scales",
            "boolean_ffn.decoder_bias",
            "boolean_ffn.correction_int8",
            "boolean_ffn.correction_scales",
            "boolean_ffn.correction_ticket_offsets",
            "boolean_ffn.correction_ticket_columns",
            "boolean_ffn.correction_output_f16",
        )
        lines.append(
            "  {" + ", ".join(str(offset(prefix + name)) for name in names) + "},"
        )
    lines.extend(["};", "}  // namespace ternlog_vocab_layout", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument(
        "--corpus-manifest", type=Path, default=None,
        help="manifest override for portable byte-model bundles",
    )
    parser.add_argument(
        "--match-mixing", type=Path, default=None,
        help="optional match_mixing.json to embed in the native codec target",
    )
    parser.add_argument(
        "--attention-skip-layers",
        default=None,
        help="comma-separated static attention layers to omit from the native program",
    )
    parser.add_argument("--qkv-active-fraction", type=float, default=None)
    parser.add_argument("--attention-output-active-fraction", type=float, default=None)
    parser.add_argument("--ffn-correction-active-fraction", type=float, default=None)
    parser.add_argument(
        "--ffn-decoder-output",
        choices=("fp32", "w8a8_row", "w8a8_group4"),
        default="fp32",
        help="deployed precision and scale grouping for Boolean FFN plane decoders",
    )
    parser.add_argument(
        "--qk-cache-precision",
        choices=("fp32", "int8_fixed4", "int8_fixed4_fp16", "int8_dynamic"),
        default=None,
        help="override the deployed Q/K cache representation recorded in the ABI",
    )
    parser.add_argument(
        "--readout-factor-output",
        choices=("fp16", "w8a8_row", "w8a8_group16"),
        default="fp16",
        help="deployed precision and scale grouping for the rank-32 readout output",
    )
    parser.add_argument(
        "--dense-mode",
        choices=(
            "fp16", "qkv_int8", "all_int8", "block_output_int8",
            "block_output_value_int8", "signed_squeeze_block_output",
            "qkv_tickets_block_output",
            "qkv_tickets_prepacked_output8",
            "qkv_tickets_prepacked_output16",
            "shared_qk_basis_output16",
            "local_global_qk_output16",
            "cyclic_routes_qk_output16",
            "tied_qk_output16",
            "head_subspace_qk_output16",
            "head_subspace_global_qk_output16",
            "product_codebook_output16",
            "head_local_output_rank8",
        ),
        default="fp16",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = inference_config_from_checkpoint(checkpoint["config"])
    if args.qk_cache_precision is not None:
        config = dataclasses.replace(
            config, qk_cache_precision=args.qk_cache_precision
        )
    if args.attention_skip_layers is not None:
        skipped = tuple(
            int(value)
            for value in args.attention_skip_layers.split(",")
            if value.strip()
        )
        config = dataclasses.replace(config, attention_skip_layers=skipped)
    if args.qkv_active_fraction is not None:
        config = dataclasses.replace(
            config, attention_value_block_active_fraction=args.qkv_active_fraction
        )
    if args.attention_output_active_fraction is not None:
        config = dataclasses.replace(
            config,
            attention_output_block_active_fraction=(
                args.attention_output_active_fraction
            ),
        )
    if args.ffn_correction_active_fraction is not None:
        config = dataclasses.replace(
            config,
            boolean_ffn_correction_block_active_fraction=(
                args.ffn_correction_active_fraction
            ),
        )
    config.validate()
    required = {
        "feedforward_activation": "boolean_lut3",
        "boolean_rounds": 3,
        "boolean_ffn_correction_taps": "equal_trajectory",
        "boolean_ffn_routes": "register_mix",
        "boolean_ffn_thresholds": "linear",
        "boolean_ffn_input_rounding": "fp16",
        "attention_normalizer": "clipped_linear",
        "attention_window_policy": "head_local",
        "vocabulary_path": "boolean_hybrid",
        "boolean_vocab_code_width": 64,
        "boolean_vocab_pool_bits": 16,
        "boolean_vocab_pool_schedule": "uniform",
        "boolean_token_expansion": "repeat",
        "residual_routing": "ordinary",
        "tie_embeddings": True,
        "qk_norm": True,
    }
    mismatches = {
        key: (getattr(config, key), value)
        for key, value in required.items()
        if getattr(config, key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint does not match AX1 native ABI: {mismatches}")
    invalid_scalable = scalable_native_abi_errors(config)
    if invalid_scalable:
        raise ValueError(
            "checkpoint violates scalable AX1 native ABI: "
            f"{invalid_scalable}"
        )
    corpus_dir = Path(config.corpus_dir)
    manifest_path = args.corpus_manifest or corpus_dir / "manifest.json"
    corpus_manifest = json.loads(manifest_path.read_text())
    vocabulary = int(corpus_manifest["logical_vocabulary_size"])
    if vocabulary <= 0 or vocabulary % 16:
        raise ValueError("AX1 native ABI requires a positive 16-row-aligned vocabulary")

    state = checkpoint["model_state_dict"]
    writer = BinaryWriter()
    if vocabulary == 256:
        tokenizer = {
            "tokenizer.byte_token_ids": np.arange(256, dtype="<u2"),
            "tokenizer.token_offsets": np.arange(257, dtype="<u4"),
            "tokenizer.token_bytes": np.arange(256, dtype=np.uint8),
            "tokenizer.merge_triples": np.empty((0, 3), dtype="<u2"),
        }
        merge_count = 0
    else:
        tokenizer, merge_count = _tokenizer_entries(corpus_dir)
    for name, array in tokenizer.items():
        writer.add(name, array, layout="tokenizer-table")
    codes = np.packbits(
        (state["token_embedding.weight"] >= 0).numpy().astype(np.uint8),
        axis=1,
        bitorder="little",
    )
    if codes.shape[0] != vocabulary:
        raise ValueError(f"checkpoint vocabulary {codes.shape[0]} != manifest {vocabulary}")
    writer.add("boolean_token_codes", codes, layout=f"{vocabulary}x64-packed")
    if "ngram_table.weight" in state:
        regions = len(config.ngram_orders) * config.ngram_heads
        values = state["ngram_table.weight"].detach().float().cpu().reshape(
            regions, config.ngram_table_rows, config.ngram_dimension
        )
        maximum = values.abs().amax(dim=(1, 2)).clamp_min(1.0e-12)
        scales = maximum / 127.0
        quantized = torch.round(values / scales[:, None, None]).clamp(-127, 127)
        writer.add(
            "ngram_table.int8", quantized.to(torch.int8).numpy(),
            layout="region-major, 64-byte cache-line rows",
        )
        writer.add(
            "ngram_table.scales", scales.numpy().astype("<f4"),
            layout="one scale per order/head region",
        )
    else:
        writer.add("ngram_table.int8", np.empty((0,), dtype=np.int8))
        writer.add("ngram_table.scales", np.empty((0,), dtype="<f4"))
    hierarchy = config.vocabulary_hierarchy == "balanced_32x64"
    if hierarchy:
        if args.readout_factor_output != "fp16":
            raise ValueError("the 32x64 hierarchy currently requires FP16 factor output")
        order_tensor = state["readout.cluster_token_ids"].detach().cpu().flatten()
        order = order_tensor.numpy().astype("<u2")
        inverse = np.asarray([], dtype="<u2")
        hierarchy_leaf_codes = np.asarray([], dtype=np.uint8)
        hierarchy_cluster_codes = np.packbits(
            (state["readout.cluster_code_logits"] >= 0)
            .detach().cpu().numpy().astype(np.uint8),
            axis=1,
            bitorder="little",
        )
        hierarchy_cluster_output = _tensor(
            state, "readout.cluster_correction_output_weight", "f16"
        )
        hierarchy_cluster_bias = _tensor(state, "readout.cluster_bias", "f32")
    else:
        order_tensor = torch.arange(vocabulary)
        hierarchy_leaf_codes = np.asarray([], dtype=np.uint8)
        hierarchy_cluster_codes = np.asarray([], dtype=np.uint8)
        hierarchy_cluster_output = np.asarray([], dtype="<f2")
        hierarchy_cluster_bias = np.asarray([], dtype="<f4")
        order = np.asarray([], dtype="<u2")
        inverse = np.asarray([], dtype="<u2")
    writer.add(
        "readout.hierarchy_leaf_codes",
        hierarchy_leaf_codes,
        layout="unused-token-codes-addressed-through-hierarchy-token-ids",
    )
    writer.add(
        "readout.hierarchy_cluster_codes",
        hierarchy_cluster_codes,
        layout="32x64-packed",
    )
    writer.add(
        "readout.hierarchy_cluster_output_f16",
        hierarchy_cluster_output,
        layout="32xrank-row-major-fp16",
    )
    writer.add(
        "readout.hierarchy_cluster_bias",
        hierarchy_cluster_bias,
        layout="32xf32",
    )
    writer.add(
        "readout.hierarchy_token_ids",
        order,
        layout="cluster-major-32x64-u16-token-ids",
    )
    writer.add(
        "readout.hierarchy_token_index",
        inverse,
        layout=f"{vocabulary}xu16-token-to-cluster-major-index",
    )
    writer.add("input_norm.weight", _tensor(state, "input_norm.weight", "f16"))
    writer.add("final_norm.weight", _tensor(state, "final_norm.weight", "f16"))

    for layer in range(config.layers):
        prefix = f"layers.{layer}."
        export_attention = layer not in config.attention_skip_layers
        writer.add(
            prefix + "residual_scales",
            _tensor(state, prefix + "residual_scales", "f32"),
            layout="scalar",
        )
        writer.add(
            prefix + "attention_norm.weight",
            _tensor(state, prefix + "attention_norm.weight", "f16"),
        )
        slope = (
            state[prefix + "attention.clipped_linear_log_slope"]
            .detach().float().exp().numpy().astype("<f4")
        )
        writer.add(prefix + "attention.clipped_linear_slope", slope, layout="per-head")
        value_heads = (
            config.kv_heads
            if config.kv_head_sharing in {"both", "value"}
            else config.heads
        )
        value_rows = value_heads * (config.dimension // config.heads)
        value_weight = state[prefix + "attention.qkv.weight"][-value_rows:]
        block_mask_name = prefix + "attention.qkv.block_mask"
        if block_mask_name in state:
            value_weight = value_weight * expanded_block_mask(
                state[block_mask_name], config.qkv_block_rows
            )[-value_rows:]
        value_cache_scale = (
            4.0 * value_weight.detach().float().square().sum(dim=-1).sqrt()
            / 127.0
        ).amax().reshape(1).numpy().astype("<f4")
        writer.add(
            prefix + "attention.value_cache_scale",
            value_cache_scale,
            layout="one-fixed-weight-rms-scale-per-layer",
        )
        writer.add(
            prefix + "feedforward_norm.weight",
            _tensor(state, prefix + "feedforward_norm.weight", "f16"),
        )
        if export_attention and args.dense_mode == "fp16":
            writer.add(
                prefix + "attention.qkv.weight",
                _tensor(state, prefix + "attention.qkv.weight", "f16"),
            )
        if export_attention and args.dense_mode not in {
            "all_int8", "block_output_int8", "block_output_value_int8",
            "signed_squeeze_block_output", "qkv_tickets_block_output",
            "qkv_tickets_prepacked_output8", "qkv_tickets_prepacked_output16",
            "shared_qk_basis_output16", "local_global_qk_output16",
            "cyclic_routes_qk_output16",
            "tied_qk_output16", "head_subspace_qk_output16",
            "head_subspace_global_qk_output16",
            "product_codebook_output16",
            "head_local_output_rank8",
        }:
            writer.add(
                prefix + "attention.output.weight",
                _tensor(state, prefix + "attention.output.weight", "f16"),
            )
        int8_matrices = []
        if export_attention and args.dense_mode not in {
            "fp16", "block_output_value_int8", "signed_squeeze_block_output"
            , "qkv_tickets_block_output"
            , "qkv_tickets_prepacked_output8", "qkv_tickets_prepacked_output16"
            , "shared_qk_basis_output16", "local_global_qk_output16"
            , "cyclic_routes_qk_output16"
            , "tied_qk_output16", "head_subspace_qk_output16"
            , "head_subspace_global_qk_output16"
            , "product_codebook_output16"
            , "head_local_output_rank8"
        }:
            int8_matrices.append(("attention.qkv.weight", "attention.qkv"))
        if export_attention and args.dense_mode == "all_int8":
            int8_matrices.append(("attention.output.weight", "attention.output"))
        if export_attention and args.dense_mode == "product_codebook_output16":
            if config.attention_output_mode != "product_codebook_16":
                raise ValueError("product-codebook export requires a codebook checkpoint")
            output = prefix + "attention.output."
            packed, indices = pack_product_codebook(
                state[output + "codebook"], state[output + "code_index_logits"]
            )
            writer.add(
                prefix + "attention.output_int8",
                packed,
                layout=(
                    f"{config.attention_output_codewords}-codeword-"
                    "4row-chunk-major-int8"
                ),
            )
            writer.add(
                prefix + "attention.output_scales",
                _tensor(state, output + "row_scale", "f32"),
                layout="per-row",
            )
            writer.add(
                prefix + "attention.output_ticket_columns",
                indices,
                layout=(
                    f"{config.dimension // 16}x{config.dimension}"
                    "-u8-codeword-indices"
                ),
            )
        elif export_attention and args.dense_mode == "head_local_output_rank8":
            if (
                config.attention_output_mode != "head_local_hybrid"
                or config.attention_output_head_local_rank != 8
            ):
                raise ValueError(
                    "head-local rank-8 export requires a matching checkpoint"
                )
            output = prefix + "attention.output."
            local = state[output + "local_weight"].reshape(config.dimension, 32)
            local_int8, local_scales = quantize_rows(local)
            rank_int8, rank_scales = quantize_rows(
                state[output + "global_input_weight"]
            )
            writer.add(
                prefix + "attention.output_int8",
                np.concatenate((local_int8.reshape(-1), rank_int8.reshape(-1))),
                layout=(
                    f"{config.heads}x32x32-head-local-then-8x"
                    f"{config.dimension}-rank-input-int8"
                ),
            )
            writer.add(
                prefix + "attention.output_scales",
                np.concatenate((local_scales, rank_scales)),
                layout=(
                    f"{config.dimension}-head-local-plus-8-rank-input-row-scales"
                ),
            )
            writer.add(
                prefix + "attention.output_ticket_columns",
                _tensor(state, output + "global_output_weight", "f16"),
                layout=(
                    f"{config.dimension}x8-row-major-fp16-rank-output"
                ),
            )
        elif export_attention and args.dense_mode in {
            "block_output_int8", "block_output_value_int8",
            "signed_squeeze_block_output", "qkv_tickets_block_output",
            "qkv_tickets_prepacked_output8", "qkv_tickets_prepacked_output16",
            "shared_qk_basis_output16", "local_global_qk_output16",
            "cyclic_routes_qk_output16",
            "tied_qk_output16", "head_subspace_qk_output16",
            "head_subspace_global_qk_output16",
        }:
            mask = state[prefix + "attention.output.block_mask"].detach().float()
            output_weight = state[prefix + "attention.output.weight"]
            if args.attention_output_active_fraction is not None:
                output_weight = output_weight * expanded_block_mask(mask, 16)
                mask = magnitude_block_mask(
                    output_weight,
                    block_rows=16,
                    active_fraction=args.attention_output_active_fraction,
                )
            if args.dense_mode in {
                "qkv_tickets_prepacked_output8", "qkv_tickets_prepacked_output16",
                "shared_qk_basis_output16", "local_global_qk_output16",
                "cyclic_routes_qk_output16",
                "tied_qk_output16", "head_subspace_qk_output16",
                "head_subspace_global_qk_output16",
            }:
                output_block_rows = (
                    8 if args.dense_mode == "qkv_tickets_prepacked_output8" else 16
                )
                output_mask = mask.repeat_interleave(2, 0) \
                    if output_block_rows == 8 else mask
                offsets, columns, tickets, scales = pack_qkv_tickets(
                    output_weight,
                    output_mask,
                    output_block_rows,
                )
                writer.add(
                    prefix + "attention.output_int8", tickets,
                    layout=f"prepacked-{output_block_rows}x16-ticket-weights",
                )
                writer.add(
                    prefix + "attention.output_scales", scales, layout="per-row"
                )
                writer.add(
                    prefix + "attention.output_ticket_offsets", offsets,
                    layout=f"{len(offsets)}xu16-ticket-offsets",
                )
                writer.add(
                    prefix + "attention.output_ticket_columns", columns,
                    layout=f"{len(columns)}xu8-column-tickets",
                )
            else:
                expanded = mask.repeat_interleave(16, 0).repeat_interleave(16, 1)
                quantized, scales = quantize_rows(
                    output_weight * expanded
                )
                writer.add(
                    prefix + "attention.output_int8",
                    quantized,
                    layout="row-major-int8-block16",
                )
                writer.add(
                    prefix + "attention.output_scales", scales, layout="per-row"
                )
                writer.add(
                    prefix + "attention.output_block_mask",
                    mask.to(torch.uint8).numpy(),
                    layout="10x10-block-ticket",
                )
        if export_attention and args.dense_mode in {
            "qkv_tickets_block_output", "qkv_tickets_prepacked_output8",
            "qkv_tickets_prepacked_output16",
            "product_codebook_output16",
            "head_local_output_rank8",
        }:
            if config.attention_qkv_projection_mode != "block_sparse_16x16":
                raise ValueError("QKV-ticket export requires a ticket checkpoint")
            if config.qkv_block_rows != 8:
                raise ValueError("complete native ticket path currently requires 8x16")
            qkv = prefix + "attention.qkv."
            qkv_weight = state[qkv + "weight"]
            qkv_mask = state[qkv + "block_mask"].clone()
            if args.qkv_active_fraction is not None:
                qkv_weight = qkv_weight * expanded_block_mask(
                    qkv_mask, config.qkv_block_rows
                )
                qkv_mask = magnitude_block_mask(
                    qkv_weight,
                    block_rows=config.qkv_block_rows,
                    active_fraction=args.qkv_active_fraction,
                )
            if layer in config.attention_shared_qk_layers[1:]:
                query_rows = config.heads * config.qk_head_dimension
                key_heads = (
                    config.kv_heads
                    if config.kv_head_sharing in {"both", "key"}
                    else config.heads
                )
                key_rows = key_heads * config.qk_head_dimension
                omitted_rows = key_rows + (
                    query_rows if config.attention_shared_qk_mode == "qk" else 0
                )
                begin_row = (
                    0
                    if config.attention_shared_qk_mode == "qk"
                    else query_rows
                )
                qkv_mask[
                    begin_row // config.qkv_block_rows :
                    (begin_row + omitted_rows) // config.qkv_block_rows
                ] = 0
            offsets, columns, tickets, scales = pack_qkv_tickets(
                qkv_weight, qkv_mask, 8
            )
            writer.add(
                prefix + "attention.qkv_int8", tickets,
                layout="prepacked-8x16-ticket-weights",
            )
            writer.add(prefix + "attention.qkv_scales", scales, layout="per-row")
            writer.add(
                prefix + "attention.qkv_ticket_offsets", offsets,
                layout=f"{len(offsets)}xu16-ticket-offsets",
            )
            writer.add(
                prefix + "attention.qkv_ticket_columns", columns,
                layout=f"{len(columns)}xu8-column-tickets",
            )
        if export_attention and args.dense_mode == "shared_qk_basis_output16":
            if config.attention_qkv_projection_mode != "shared_qk_basis":
                raise ValueError("shared-basis export requires a shared-basis checkpoint")
            qkv = prefix + "attention.qkv."
            base_int8, base_scales = quantize_rows(state[qkv + "base_weight"])
            writer.add(
                prefix + "attention.qkv_int8", base_int8,
                layout="row-major-int8-shared-qk-bases-and-v",
            )
            writer.add(
                prefix + "attention.qkv_scales", base_scales, layout="per-base-row"
            )
            writer.add(
                prefix + "attention.qkv_query_transform",
                _tensor(state, qkv + "query_transform", "f16"),
                layout="head-major-5x16x16-fp16",
            )
            writer.add(
                prefix + "attention.qkv_key_transform",
                _tensor(state, qkv + "key_transform", "f16"),
                layout="head-major-5x16x16-fp16",
            )
        if export_attention and args.dense_mode in {
            "tied_qk_output16", "head_subspace_qk_output16",
            "head_subspace_global_qk_output16",
        }:
            expected_mode = {
                "tied_qk_output16": "tied_qk",
                "head_subspace_qk_output16": "head_subspace_qk",
                "head_subspace_global_qk_output16": "head_subspace_global_qk",
            }[args.dense_mode]
            if config.attention_qkv_projection_mode != expected_mode:
                raise ValueError("head-subspace export requires a matching checkpoint")
            qkv = prefix + "attention.qkv."
            base_int8, base_scales = quantize_rows(state[qkv + "base_weight"])
            writer.add(
                prefix + "attention.qkv_int8",
                base_int8,
                layout="head-subspaces-then-shared-value-row-major-int8",
            )
            writer.add(
                prefix + "attention.qkv_scales",
                base_scales,
                layout="per-head-subspace-and-value-row",
            )
            if expected_mode in {"head_subspace_qk", "head_subspace_global_qk"}:
                transform_columns = config.qk_head_dimension + (
                    config.qkv_global_rank
                    if expected_mode == "head_subspace_global_qk"
                    else 0
                )
                writer.add(
                    prefix + "attention.qkv_query_transform",
                    _tensor(state, qkv + "query_transform", "f16"),
                    layout=f"head-major-5x16x{transform_columns}-fp16",
                )
                writer.add(
                    prefix + "attention.qkv_key_transform",
                    _tensor(state, qkv + "key_transform", "f16"),
                    layout=f"head-major-5x16x{transform_columns}-fp16",
                )
        if export_attention and args.dense_mode == "local_global_qk_output16":
            if config.attention_qkv_projection_mode != "local_global_qk":
                raise ValueError("local-global export requires a matching checkpoint")
            qkv = prefix + "attention.qkv."
            local_weight = state[qkv + "local_weight"].reshape(
                2 * config.heads * config.qk_head_dimension,
                config.qkv_local_width,
            )
            local_int8, local_scales = quantize_rows(local_weight)
            global_value = torch.cat(
                (state[qkv + "global_weight"], state[qkv + "value_weight"]),
                dim=0,
            )
            global_int8, global_scales = quantize_rows(global_value)
            writer.add(
                prefix + "attention.qkv_local_int8",
                local_int8,
                layout=(
                    f"family-head-row-major-int8-local{config.qkv_local_width}"
                ),
            )
            writer.add(
                prefix + "attention.qkv_local_scales",
                local_scales,
                layout="per-local-qk-row",
            )
            writer.add(
                prefix + "attention.qkv_global_int8",
                global_int8,
                layout="global-basis-then-shared-value-row-major-int8",
            )
            writer.add(
                prefix + "attention.qkv_global_scales",
                global_scales,
                layout="per-global-and-value-row",
            )
            writer.add(
                prefix + "attention.qkv_global_transform",
                _tensor(state, qkv + "global_transform", "f16"),
                layout="family-head-16x16-fp16",
            )
        if export_attention and args.dense_mode == "cyclic_routes_qk_output16":
            if config.attention_qkv_projection_mode != "cyclic_routes_qk":
                raise ValueError("cyclic-route export requires a matching checkpoint")
            qkv = prefix + "attention.qkv."
            writer.add(
                prefix + "attention.qkv_route_weight",
                state[qkv + "route_weight"].detach().T.contiguous()
                .to(torch.float16).numpy(),
                layout=(
                    f"route-major-{config.qkv_route_count}x"
                    f"{config.dimension}-fp16"
                ),
            )
            writer.add(
                prefix + "attention.qkv_route_offsets",
                state[qkv + "route_offsets"].detach().to(torch.uint8).numpy(),
                layout=f"{config.qkv_route_count}xu8-cyclic-offsets",
            )
            value_int8, value_scales = quantize_rows(state[qkv + "value_weight"])
            writer.add(
                prefix + "attention.qkv_value_int8",
                value_int8,
                layout=(
                    f"{config.dimension // config.heads}x{config.dimension}"
                    "-row-major-int8-shared-value"
                ),
            )
            writer.add(
                prefix + "attention.qkv_value_scales",
                value_scales,
                layout=(
                    f"{config.dimension // config.heads}"
                    "xf32-value-row-scales"
                ),
            )
        if export_attention and args.dense_mode == "block_output_value_int8":
            value_mask = state[
                prefix + "attention.qkv.value_block_mask"
            ].detach().float()
            expanded_value = value_mask.repeat_interleave(16, 0).repeat_interleave(16, 1)
            qk_rows = (
                config.heads
                + (
                    config.kv_heads
                    if config.kv_head_sharing in {"both", "key"}
                    else config.heads
                )
            ) * config.qk_head_dimension
            expanded_qkv = torch.cat(
                (torch.ones(qk_rows, config.dimension), expanded_value), dim=0
            )
            quantized, scales = quantize_rows(
                state[prefix + "attention.qkv.weight"] * expanded_qkv
            )
            writer.add(
                prefix + "attention.qkv_int8",
                quantized,
                layout="row-major-int8-dense-qk-block16-v",
            )
            writer.add(prefix + "attention.qkv_scales", scales, layout="per-row")
            writer.add(
                prefix + "attention.qkv_value_block_mask",
                value_mask.to(torch.uint8).numpy(),
                layout="10x10-value-block-ticket",
            )
        if export_attention and args.dense_mode == "signed_squeeze_block_output":
            if config.attention_qkv_projection_mode != "signed_squeeze_hybrid":
                raise ValueError("signed-squeeze export requires a signed-squeeze checkpoint")
            qkv = prefix + "attention.qkv."
            features = (
                config.dimension * config.qkv_squeeze_bitplanes
                // config.qkv_squeeze_group_bits * config.qkv_squeeze_polarities
            )
            padded_features = (features + 15) // 16 * 16
            squeeze_weight = state[qkv + "squeeze_weight"].detach().float()
            padded_weight = torch.zeros(
                squeeze_weight.shape[0], padded_features, dtype=squeeze_weight.dtype
            )
            padded_weight[:, :features] = squeeze_weight
            squeeze_int8, squeeze_scales = quantize_rows(
                padded_weight, divisor=float(config.qkv_squeeze_group_bits) ** 0.5
            )
            polarity_bits = (
                state[qkv + "polarity_logits"].detach() >= 0
            ).to(torch.uint8).numpy()
            polarity_words = np.packbits(
                polarity_bits, axis=-1, bitorder="little"
            ).view("<u4").reshape(config.qkv_squeeze_polarities, -1)
            bypass_indices = state[qkv + "bypass_indices"].detach().cpu().numpy()
            bypass_start = int(bypass_indices[0])
            expected = np.arange(
                bypass_start, bypass_start + config.qkv_squeeze_bypass
            )
            if not np.array_equal(bypass_indices, expected):
                raise ValueError("native signed-squeeze bypass must be contiguous")
            bypass_int8, bypass_scales = quantize_rows(
                state[qkv + "bypass_weight"]
            )
            writer.add(
                prefix + "attention.qkv_squeeze_offsets",
                _tensor(state, qkv + "encoder_offsets", "f32"),
            )
            writer.add(
                prefix + "attention.qkv_squeeze_thresholds",
                _tensor(state, qkv + "encoder_thresholds", "f32"),
            )
            writer.add(
                prefix + "attention.qkv_squeeze_polarities", polarity_words,
                layout="polarity-group-u32",
            )
            writer.add(
                prefix + "attention.qkv_squeeze_int8", squeeze_int8,
                layout=f"row-major-int8-padded{padded_features}",
            )
            writer.add(
                prefix + "attention.qkv_squeeze_scales", squeeze_scales,
                layout="per-row-includes-signed-count-normalization",
            )
            writer.add(
                prefix + "attention.qkv_bypass_int8", bypass_int8,
                layout="row-major-int8-contiguous-bypass",
            )
            writer.add(
                prefix + "attention.qkv_bypass_scales", bypass_scales,
                layout="per-row",
            )
            writer.add(
                prefix + "attention.qkv_bypass_start",
                np.asarray([bypass_start], dtype="<u4"), layout="scalar",
            )
        for source, target in int8_matrices:
            quantized, scales = quantize_rows(state[prefix + source])
            writer.add(prefix + target + "_int8", quantized, layout="row-major-int8")
            writer.add(prefix + target + "_scales", scales, layout="per-row")
        writer.add(
            prefix + "boolean_ffn.encoder_offsets",
            _tensor(state, prefix + "boolean_ffn.encoder_offsets", "f32"),
        )
        program = np.packbits(
            (state[prefix + "boolean_ffn.program_logits"] >= 0)
            .numpy().astype(np.uint8),
            axis=-1,
            bitorder="little",
        )
        writer.add(prefix + "boolean_ffn.program", program, layout="round-plane-packed")
        decoder_weight = state[prefix + "boolean_ffn.decoder_weight"]
        if args.ffn_decoder_output == "fp32":
            decoder_float = _tensor(
                state, prefix + "boolean_ffn.decoder_weight", "f32"
            )
            decoder_int8 = np.asarray([], dtype=np.int8)
            decoder_scales = np.asarray([], dtype="<f4")
            decoder_layout = "unused"
        elif args.ffn_decoder_output == "w8a8_row":
            decoder_float = np.asarray([], dtype="<f4")
            decoder_int8, decoder_scales = quantize_decoder_rows(decoder_weight)
            decoder_layout = "row-major-int8-padded16"
        else:
            decoder_float = np.asarray([], dtype="<f4")
            decoder_int8, decoder_scales = quantize_decoder_group4(decoder_weight)
            decoder_layout = "row-major-int8-padded16"
        writer.add(prefix + "boolean_ffn.decoder_weight", decoder_float)
        writer.add(
            prefix + "boolean_ffn.decoder_int8",
            decoder_int8,
            layout=decoder_layout,
        )
        writer.add(
            prefix + "boolean_ffn.decoder_scales",
            decoder_scales,
            layout=(
                "two-groups-per-row" if args.ffn_decoder_output == "w8a8_group4"
                else "per-row"
            ),
        )
        writer.add(
            prefix + "boolean_ffn.decoder_bias",
            _tensor(state, prefix + "boolean_ffn.decoder_bias", "f32"),
        )
        if config.boolean_ffn_correction_rank:
            if config.boolean_ffn_correction_block_active_fraction < 1:
                raise ValueError("factorized FFN correction cannot use block tickets")
            correction, scales = quantize_rows(
                state[prefix + "boolean_ffn.correction_input_weight"], divisor=4.0
            )
            correction_output = _tensor(
                state, prefix + "boolean_ffn.correction_output_weight", "f16"
            )
            correction_offsets = np.asarray([], dtype="<u2")
            correction_columns = np.asarray([], dtype=np.uint8)
            correction_layout = "row-major-int8"
        else:
            correction_weight = state[prefix + "boolean_ffn.correction_weight"]
            correction_fraction = (
                config.boolean_ffn_correction_block_active_fraction
            )
            if correction_fraction < 1:
                mask_name = prefix + "boolean_ffn.correction_block_mask"
                if args.ffn_correction_active_fraction is None and mask_name in state:
                    correction_mask = state[mask_name].detach().float()
                else:
                    if mask_name in state:
                        correction_weight = correction_weight * expanded_block_mask(
                            state[mask_name].detach().float(), 16
                        )
                    correction_mask = magnitude_block_mask(
                        correction_weight,
                        block_rows=16,
                        active_fraction=correction_fraction,
                    )
                (
                    correction_offsets,
                    correction_columns,
                    correction,
                    scales,
                ) = pack_qkv_tickets(correction_weight, correction_mask, 16)
                scales = scales / 4.0
                correction_layout = "prepacked-16x16-ticket-weights"
            else:
                correction, scales = quantize_rows(
                    correction_weight, divisor=4.0
                )
                correction_offsets = np.asarray([], dtype="<u2")
                correction_columns = np.asarray([], dtype=np.uint8)
                correction_layout = "row-major-int8"
            correction_output = np.asarray([], dtype="<f2")
        writer.add(
            prefix + "boolean_ffn.correction_int8",
            correction,
            layout=correction_layout,
        )
        writer.add(prefix + "boolean_ffn.correction_scales", scales, layout="per-row")
        writer.add(
            prefix + "boolean_ffn.correction_ticket_offsets",
            correction_offsets,
            layout=f"{len(correction_offsets)}xu16-ticket-offsets",
        )
        writer.add(
            prefix + "boolean_ffn.correction_ticket_columns",
            correction_columns,
            layout=f"{len(correction_columns)}xu8-column-tickets",
        )
        writer.add(
            prefix + "boolean_ffn.correction_output_f16",
            correction_output,
            layout="row-major-fp16-factor-output",
        )

    if config.boolean_vocab_correction_rank:
        padded_readout_features = (
            (state["readout.correction_input_weight"].shape[1] + 15) // 16 * 16
        )
        padded_readout_input = torch.zeros(
            config.boolean_vocab_correction_rank, padded_readout_features,
            dtype=state["readout.correction_input_weight"].dtype,
        )
        padded_readout_input[:, : state["readout.correction_input_weight"].shape[1]] = (
            state["readout.correction_input_weight"]
        )
        readout, readout_scales = quantize_rows(
            padded_readout_input, divisor=4.0
        )
        output_weight = state["readout.correction_output_weight"]
        if hierarchy:
            output_weight = output_weight.index_select(
                0, order_tensor.to(output_weight.device)
            )
        if args.readout_factor_output == "fp16":
            readout_output = (
                output_weight.detach().cpu().numpy().astype("<f2", copy=False)
            )
            readout_output_int8 = np.asarray([], dtype=np.int8)
            readout_output_scales = np.asarray([], dtype="<f4")
        elif args.readout_factor_output == "w8a8_row":
            readout_output = np.asarray([], dtype="<f2")
            readout_output_int8, readout_output_scales = quantize_rows(
                output_weight
            )
        else:
            readout_output = np.asarray([], dtype="<f2")
            grouped = output_weight.detach().float().reshape(
                output_weight.shape[0], -1, 16
            )
            grouped_maximum = grouped.abs().amax(dim=2).clamp_min(1.0e-12)
            grouped_scales = grouped_maximum / 127.0
            readout_output_int8 = torch.round(
                grouped / grouped_scales[:, :, None]
            ).clamp(-127, 127).to(torch.int8).reshape(
                output_weight.shape[0], -1
            ).numpy()
            readout_output_scales = grouped_scales.numpy().astype("<f4")
        readout_layout = (
            f"row-major-int8-factor-input-padded{padded_readout_features}"
        )
        readout_scale_layout = "per-factor-row-includes-signed-count-normalization"
    else:
        if args.readout_factor_output != "fp16":
            raise ValueError("W8A8 readout factor output requires a factorized readout")
        readout, readout_scales = quantize_readout(state["readout.correction_weight"])
        readout_output = np.asarray([], dtype="<f2")
        readout_output_int8 = np.asarray([], dtype=np.int8)
        readout_output_scales = np.asarray([], dtype="<f4")
        readout_layout = "row-major-int8-3x32"
        readout_scale_layout = "per-row-3-group"
    writer.add("readout.correction_int8", readout, layout=readout_layout)
    writer.add("readout.correction_scales", readout_scales, layout=readout_scale_layout)
    writer.add(
        "readout.correction_output_f16", readout_output,
        layout="row-major-fp16-factor-output",
    )
    writer.add(
        "readout.correction_output_int8", readout_output_int8,
        layout=f"row-major-int8-factor-output-{args.readout_factor_output}",
    )
    writer.add(
        "readout.correction_output_scales", readout_output_scales,
        layout=f"factor-output-scales-{args.readout_factor_output}",
    )
    readout_bias = state["readout.bias"]
    if hierarchy:
        readout_bias = readout_bias.index_select(
            0, order_tensor.to(readout_bias.device)
        )
    writer.add(
        "readout.bias",
        readout_bias.detach().cpu().numpy().astype("<f4", copy=False),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(writer.data)
    # Identify the representation that the native decoder actually consumes,
    # not the source FP32 checkpoint.  Quantization mode, packing, and layout
    # are therefore part of stream compatibility and a mismatched decoder
    # fails before attempting arithmetic decoding.
    fingerprint = hashlib.sha256(writer.data).digest()[:16]
    match_mixing = (
        None if args.match_mixing is None
        else json.loads(args.match_mixing.read_text())
    )
    write_header(
        args.header,
        config,
        writer.entries,
        vocabulary=vocabulary,
        merge_count=merge_count,
        binary_bytes=len(writer.data),
        model_fingerprint=fingerprint,
        match_mixing=match_mixing,
    )
    value_cache_operations = (
        [
            "FCVTNS/SQXTN fixed weight-RMS INT8 value store",
            "FMLA/FMAX/FCVTNS/SQXTN clipped-weight quantization",
            "time-blocked SDOT weighted-value reduction",
        ]
        if config.value_cache_precision.startswith("int8_")
        else ["FP16 value store/load"]
    )
    manifest = {
        "format": "ternlog-vocab-boolean-ax1-m4-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": checkpoint.get("update"),
        "decoded_bytes_processed": checkpoint.get("decoded_bytes_processed"),
        "validation": checkpoint.get("validation"),
        "binary_bytes": len(writer.data),
        "dense_mode": args.dense_mode,
        "readout_factor_output": args.readout_factor_output,
        "native_operations": {
            "embedding": ["packed address", "EOR", "SCVTF", "FMUL"],
            "dense_attention": (
                (
                    ["FCVTN", "FCMGE", "EOR", "CNT", "SDOT", "SCVTF", "FMLA"]
                    if config.qkv_squeeze_input_rounding == "fp16"
                    else ["FMUL", "FRINTN", "FMUL", "FCMGE", "EOR", "CNT", "SDOT", "SCVTF", "FMLA"]
                )
                if args.dense_mode == "signed_squeeze_block_output"
                else (
                    [
                        "FMAX", "FCVTNS", "SQXTN", "SDOT", "SCVTF", "FMUL",
                        "FCVTN", "FMLAL", "FMLAL2",
                    ]
                    if args.dense_mode == "shared_qk_basis_output16"
                    else (
                        ["FCVTN", "FMLAL", "FMLAL2"]
                        if args.dense_mode == "fp16"
                        else ["FMAX", "FCVTNS", "SQXTN", "SDOT", "SCVTF", "FMUL"]
                    )
                )
            ),
            "attention_output": (
                [
                    "FCVTN/FCVTL FP16 input canonicalization",
                    "FMAX/FCVTNS/SQXTN activation quantization",
                    "SDOT codeword responses",
                    "SQRSHRN/SQXTN response quantization",
                    "TBL2 learned codeword assembly",
                    "SADDW accumulation",
                    "SSHLL/SCVTF/FMUL output scaling",
                ]
                if args.dense_mode == "product_codebook_output16"
                else [
                    "FCVTN/FCVTL FP16 input canonicalization",
                    "FMAX/FCVTNS/SQXTN shared activation quantization",
                    "SDOT head-local 32x32 blocks",
                    "SDOT rank-8 global input",
                    "FCVTN/FMLAL/FMLAL2 rank-8 global output",
                    "FADD local/global composition",
                ]
                if args.dense_mode == "head_local_output_rank8"
                else ["included in dense_attention"]
            ),
            "boolean_boundary": ["FCVTN", "FCVTL", "FCMGE"],
            "boolean_program": ["EXT", "BSL", "EOR"],
            "boolean_decode": ["CNT", "SDOT", "FMLA"],
            "readout": (
                [
                    "FCMGE", "pack", "EOR", "CNT", "SDOT", "FCVTN",
                    "FMLAL", "FMLAL2", "FMAX", "exp/log normalization",
                    "two-level categorical sampling",
                ]
                if hierarchy
                else
                [
                    "EOR", "CNT", "SDOT", "FCVTN", "FMAX", "FCVTNS",
                    "SQXTN", "SDOT", "SCVTF", "FMUL", "FADD",
                ]
                if config.boolean_vocab_correction_rank
                and args.readout_factor_output != "fp16"
                else ["EOR", "CNT", "SDOT", "FCVTN", "FMLAL", "FMLAL2", "FADD"]
                if config.boolean_vocab_correction_rank
                else ["EOR", "CNT", "SDOT", "FMLA"]
            ),
            "kv_cache": (
                [
                    "FP16 canonicalization",
                    "FCVTNS/SQXTN fixed-scale INT8 key store",
                    "SDOT query-key score",
                    *value_cache_operations,
                    "ring-buffer integer addressing",
                ]
                if config.qk_cache_precision == "int8_fixed4_fp16"
                else [
                    "FCVTNS/SQXTN fixed-scale INT8 key store",
                    "SDOT query-key score",
                    *value_cache_operations,
                    "ring-buffer integer addressing",
                ]
                if config.qk_cache_precision == "int8_fixed4"
                else [
                    "FMAX/FDIV/FCVTNS/SQXTN dynamic-scale INT8 key store",
                    "SDOT query-key score",
                    *value_cache_operations,
                    "ring-buffer integer addressing",
                ]
                if config.qk_cache_precision == "int8_dynamic"
                else [*value_cache_operations, "ring-buffer integer addressing"]
            ),
            "attention_skip": (
                [
                    "static layer mask",
                    "FMUL retained residual",
                    "omit QKV/RoPE/KV/score/output projection",
                ]
                if config.attention_skip_layers
                else ["none"]
            ),
            "attention_shared_input": (
                [
                    f"{config.dimension}-lane residual snapshot",
                    "reuse saved source for late RMSNorm/QKV",
                    "retain sequential residual/FFN updates",
                ]
                if config.attention_shared_input_layers
                else ["none"]
            ),
            "attention_shared_qk": (
                [
                    "reuse post-RoPE fixed-boundary Q/K",
                    "omit complete consumer 8x16 SDOT row blocks",
                    "share consumer key-cache storage",
                ]
                if config.attention_shared_qk_layers
                else ["none"]
            ),
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
