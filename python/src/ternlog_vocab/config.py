from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(slots=True)
class VocabLMConfig:
    corpus_dir: str = "data/tinystories_vocab_4k"
    output_dir: str = "runs/vocab_lm/v0_8m"
    context_tokens: int = 256
    stride_tokens: int = 256
    validation_stride_tokens: int | None = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    maximum_updates: int = 100_000
    target_decoded_bytes: int = 100_000_000
    report_every_decoded_bytes: int = 5_000_000
    validation_batches: int | None = 64
    validation_sampling: str = "prefix"
    validation_region_tokens: int = 0
    validation_regions: int = 0
    validation_domain_bytes: int = 0
    training_sampling: str = "blocks"
    training_region_tokens: int = 0
    training_decoded_bytes_per_update: int = 0
    training_micro_batch_size: int = 0
    run_source_decoded_bytes: int | None = None
    seed: int = 20260824
    device: str = "auto"
    threads: int = 11
    learning_rate: float = 3e-4
    embedding_learning_rate: float = 3e-4
    lr_decay_bytes: int = 0
    lr_decay_final_fraction: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    dimension: int = 224
    layers: int = 10
    heads: int = 7
    kv_heads: int = 1
    ff_dimension: int = 1328
    ff_extra_dimension: int = 0
    ff_extra_dimension_layers: int = 0
    feedforward_activation: str = "relu_squared"
    boolean_width: int = 1_280
    boolean_rounds: int = 5
    boolean_ffn_correction_features: int = 0
    boolean_ffn_correction_rank: int = 0
    boolean_ffn_correction_mode: str = "dense"
    boolean_ffn_correction_block_active_fraction: float = 1.0
    boolean_ffn_correction_taps: str = "final"
    boolean_ffn_pool_schedule: str = "legacy_mixed"
    boolean_ffn_routes: str = "byte_lane"
    boolean_ffn_thresholds: str = "linear"
    boolean_ffn_input_rounding: str = "none"
    boolean_backward: str = "hybrid"
    boolean_temperature: float = 1.0
    boolean_gradient_leak: float = 0.05
    boolean_program_init_margin: float = 0.02
    boolean_register_bits: int = 128
    boolean_balance_weight: float = 0.001
    boolean_gate_weight: float = 0.0001
    boolean_residual_state: str = "numerical"
    residual_boundary: str = "none"
    residual_normalizer: str = "rms"
    token_feature: str = "plain"
    token_smear: str = "none"
    value_embedding_layers: tuple[int, ...] = (1, 2, 7, 8, 9)
    bigram_hash_size: int = 16_384
    bigram_dimension: int = 56
    bigram_sign_mode: str = "none"
    bigram_learned_scale: bool = True
    ngram_orders: tuple[int, ...] = ()
    ngram_heads: int = 2
    ngram_table_rows: int = 4096
    ngram_dimension: int = 64
    rope_base: float = 10_000.0
    rope_fraction: float = 1.0
    partial_key_offset: str = "none"
    qk_norm: bool = True
    qk_cache_precision: str = "fp32"
    value_cache_precision: str = "fp32"
    qk_head_dimension: int = 0
    local_qk_head_dimension: int = 0
    kv_head_routing: str = "repeat"
    kv_head_sharing: str = "both"
    attention_qkv_projection_mode: str = "dense"
    qkv_squeeze_bitplanes: int = 8
    qkv_squeeze_polarities: int = 2
    qkv_squeeze_group_bits: int = 32
    qkv_squeeze_bypass: int = 32
    qkv_squeeze_input_rounding: str = "fp16"
    qkv_low_rank: int = 64
    qkv_shared_bases: int = 1
    qkv_local_width: int = 64
    qkv_global_rank: int = 16
    qkv_route_count: int = 16
    qkv_bitmix_rounds: int = 2
    qkv_bitmix_correction_features: int = 48
    qkv_bitmix_cell: str = "eor3"
    qkv_block_allocation: str = "uniform"
    qkv_block_rows: int = 16
    qkv_ticket_selection: str = "magnitude"
    qkv_ticket_temperature: float = 0.25
    qkv_head_input_groups: int = 0
    attention_qk_mode: str = "numerical"
    boolean_qk_width: int = 640
    boolean_qk_rounds: int = 5
    boolean_qk_bits_per_head: int = 64
    boolean_qk_correction_features: int = 0
    boolean_qk_correction_taps: str = "final"
    attention_value_mode: str = "numerical"
    attention_value_block_active_fraction: float = 0.5
    boolean_value_width: int = 640
    boolean_value_rounds: int = 5
    boolean_value_bits: int = 0
    boolean_value_correction_features: int = 0
    attention_output_mode: str = "numerical"
    boolean_output_width: int = 640
    boolean_output_rounds: int = 5
    boolean_output_correction_features: int = 0
    boolean_output_correction_taps: str = "final"
    boolean_output_routes: str = "byte_lane"
    boolean_output_cell: str = "lut3"
    attention_output_block_active_fraction: float = 0.5
    attention_output_codewords: int = 32
    attention_output_codebook_temperature: float = 0.25
    attention_output_codebook_response_shift: int = 10
    attention_output_head_local_rank: int = 0
    attention_output_squeeze_bitplanes: int = 8
    attention_output_squeeze_polarities: int = 2
    attention_output_squeeze_group_bits: int = 32
    attention_output_squeeze_bypass: int = 32
    attention_output_input_rounding: str = "none"
    attention_normalizer: str = "isa_exp2"
    clipped_linear_initial_slope: float = 0.5
    attention_window_policy: str = "global"
    local_attention_window: int = 64
    local_attention_heads: int = 4
    local_attention_layers: tuple[int, ...] = (0, 2, 4, 6, 8)
    attention_skip_layers: tuple[int, ...] = ()
    attention_shared_input_layers: tuple[int, ...] = ()
    attention_shared_qk_layers: tuple[int, ...] = ()
    attention_shared_qk_mode: str = "none"
    zero_init_residual: bool = True
    tie_embeddings: bool = True
    vocabulary_path: str = "numerical"
    boolean_vocab_code_width: int = 0
    boolean_vocab_pool_bits: int = 16
    boolean_vocab_pool_schedule: str = "uniform"
    boolean_vocab_correction_rank: int = 0
    boolean_vocab_factor_output: str = "fp16"
    vocabulary_hierarchy: str = "flat"
    boolean_token_expansion: str = "repeat"
    residual_scalars: bool = True
    residual_routing: str = "ordinary"
    u_skip_pairs: tuple[tuple[int, int], ...] = ((1, 8), (3, 6))
    dynamic_route_layers: tuple[int, ...] = (4, 7, 9)
    dynamic_route_dimension: int = 16
    dropout: float = 0.0
    vocabulary_padding_multiple: int = 128
    sample_every_report: bool = True
    sample_prompt: str = "Once upon a time, there was a little girl named Lily."
    sample_decoded_bytes: int = 256
    sample_temperature: float = 0.7
    sample_top_k: int = 40
    sample_seed: int = 20260824
    save_every_reports: int = 1
    save_gate_checkpoints: bool = False
    resume: str | None = None
    initialize_from: str | None = None
    projection_distill_updates: int = 0

    @classmethod
    def from_toml(cls, path: str | Path) -> "VocabLMConfig":
        with Path(path).open("rb") as handle:
            values = tomllib.load(handle)
        config = cls(**values.get("vocab_lm", values))
        config.validate()
        return config

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VocabLMConfig":
        config = cls(**values)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        positive = {
            "context_tokens": self.context_tokens,
            "stride_tokens": self.stride_tokens,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "maximum_updates": self.maximum_updates,
            "target_decoded_bytes": self.target_decoded_bytes,
            "report_every_decoded_bytes": self.report_every_decoded_bytes,
            "threads": self.threads,
            "dimension": self.dimension,
            "layers": self.layers,
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "ff_dimension": self.ff_dimension,
            "vocabulary_padding_multiple": self.vocabulary_padding_multiple,
            "sample_decoded_bytes": self.sample_decoded_bytes,
            "sample_top_k": self.sample_top_k,
            "save_every_reports": self.save_every_reports,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.validation_stride_tokens is not None and self.validation_stride_tokens <= 0:
            raise ValueError("validation_stride_tokens must be positive when set")
        if self.validation_batches is not None and self.validation_batches <= 0:
            raise ValueError("validation_batches must be positive when set")
        if self.validation_sampling not in {
            "prefix", "spread", "shared_regions", "domain_bytes"
        }:
            raise ValueError(
                "validation_sampling must be prefix, spread, shared_regions, "
                "or domain_bytes"
            )
        if self.validation_sampling == "shared_regions":
            if self.validation_region_tokens <= 0 or self.validation_regions <= 0:
                raise ValueError("shared-region validation needs positive region sizes")
            if self.validation_region_tokens % self.context_tokens:
                raise ValueError(
                    "validation_region_tokens must be divisible by context_tokens"
                )
        elif self.validation_region_tokens or self.validation_regions:
            raise ValueError(
                "validation region sizes require validation_sampling=shared_regions"
            )
        if self.validation_sampling == "domain_bytes":
            if self.validation_domain_bytes <= 0:
                raise ValueError("domain-byte validation needs a positive byte cap")
        elif self.validation_domain_bytes:
            raise ValueError(
                "validation_domain_bytes requires validation_sampling=domain_bytes"
            )
        if self.training_sampling not in {
            "blocks", "byte_proportional", "domain_balanced", "shared_regions"
        }:
            raise ValueError(
                "training_sampling must be blocks, byte_proportional, "
                "domain_balanced or shared_regions"
            )
        if self.training_sampling == "shared_regions":
            if (self.training_region_tokens <= 0
                    or self.training_region_tokens % self.context_tokens
                    or self.stride_tokens != self.context_tokens
                    or (self.batch_size * self.gradient_accumulation_steps * self.context_tokens)
                    % self.training_region_tokens):
                raise ValueError("shared training regions require whole regions per update and nonoverlapping windows")
        elif self.training_region_tokens:
            raise ValueError("training_region_tokens requires shared_regions sampling")
        if self.training_decoded_bytes_per_update < 0:
            raise ValueError("training_decoded_bytes_per_update cannot be negative")
        if self.training_micro_batch_size < 0:
            raise ValueError("training_micro_batch_size cannot be negative")
        if self.training_sampling == "byte_proportional":
            if self.training_decoded_bytes_per_update <= 0:
                raise ValueError(
                    "byte-proportional sampling needs a positive decoded-byte budget"
                )
            if self.gradient_accumulation_steps != 1:
                raise ValueError(
                    "byte-proportional sampling currently requires one accumulation step"
                )
        elif self.training_decoded_bytes_per_update:
            raise ValueError(
                "training_decoded_bytes_per_update requires byte_proportional sampling"
            )
        if self.training_sampling != "byte_proportional" and self.training_micro_batch_size:
            raise ValueError(
                "training_micro_batch_size requires byte_proportional sampling"
            )
        if self.run_source_decoded_bytes is not None and self.run_source_decoded_bytes < 0:
            raise ValueError("run_source_decoded_bytes cannot be negative")
        if self.dimension % self.heads:
            raise ValueError("heads must divide dimension")
        if self.heads % self.kv_heads:
            raise ValueError("kv_heads must divide heads")
        if self.kv_head_routing not in {
            "repeat", "cyclic", "learned_diagonal", "learned_full"
        }:
            raise ValueError("invalid KV head routing")
        if self.kv_head_sharing not in {"both", "key", "value"}:
            raise ValueError("invalid KV head sharing")
        if self.kv_heads == self.heads and self.kv_head_sharing != "both":
            raise ValueError("partial KV sharing requires fewer KV heads")
        if self.kv_head_routing == "cyclic" and (
            self.kv_heads != 1 or self.kv_head_sharing != "both"
        ):
            raise ValueError("cyclic KV routing requires one shared K/V head")
        if self.kv_head_routing in {"learned_diagonal", "learned_full"} and (
            self.kv_heads != 1 or self.kv_head_sharing != "value"
        ):
            raise ValueError(
                "learned KV routing requires one shared value head and independent keys"
            )
        if (self.dimension // self.heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        resolved_qk_dimension = (
            self.dimension // self.heads
            if self.qk_head_dimension == 0
            else self.qk_head_dimension
        )
        if (
            resolved_qk_dimension <= 0
            or resolved_qk_dimension > self.dimension // self.heads
            or resolved_qk_dimension % 8
        ):
            raise ValueError(
                "Q/K head dimension must be a positive multiple of 8 no larger than the value head"
            )
        if self.attention_qk_mode != "numerical" and self.qk_head_dimension != 0:
            raise ValueError("explicit Q/K head dimension currently requires numerical Q/K")
        if resolved_qk_dimension % 2:
            raise ValueError("RoPE Q/K head dimension must be even")
        resolved_local_qk_dimension = (
            resolved_qk_dimension
            if self.local_qk_head_dimension == 0
            else self.local_qk_head_dimension
        )
        if (
            resolved_local_qk_dimension <= 0
            or resolved_local_qk_dimension > resolved_qk_dimension
            or resolved_local_qk_dimension % 8
        ):
            raise ValueError(
                "local Q/K head dimension must be a positive multiple of 8 no larger than Q/K width"
            )
        if (
            self.local_qk_head_dimension != 0
            and self.attention_window_policy != "head_local"
        ):
            raise ValueError("local Q/K width requires head-local attention")
        if self.local_qk_head_dimension != 0 and (
            self.kv_heads != self.heads and self.kv_head_sharing in {"both", "key"}
        ):
            raise ValueError("local Q/K width currently requires independent key heads")
        if self.local_qk_head_dimension != 0 and self.partial_key_offset != "none":
            raise ValueError("local Q/K width does not support partial-key offset")
        if self.rope_fraction not in {0.5, 1.0}:
            raise ValueError("rope_fraction must be 0.5 or 1.0")
        if self.partial_key_offset not in {"none", "global_heads", "all_heads"}:
            raise ValueError(
                "partial_key_offset must be none, global_heads, or all_heads"
            )
        if self.partial_key_offset != "none":
            if self.rope_fraction != 0.5:
                raise ValueError("partial key offset requires half-width RoPE")
            if self.kv_heads != self.heads and self.kv_head_sharing in {"both", "key"}:
                raise ValueError("partial key offset currently requires full MHA")
            if (self.dimension // self.heads) % 4:
                raise ValueError("partial key offset requires head dimension divisible by 4")
        if self.qk_cache_precision not in {
            "fp32", "int8_fixed4", "int8_fixed4_fp16", "int8_dynamic"
        }:
            raise ValueError("invalid Q/K cache precision")
        if self.qk_cache_precision != "fp32" and self.attention_qk_mode != "numerical":
            raise ValueError("INT8 Q/K cache requires numerical Q/K")
        if self.value_cache_precision not in {
            "fp32", "fp16", "int8_weight_rms_scalar", "int8_weight_rms_channel"
        }:
            raise ValueError("invalid value-cache precision")
        if (
            self.value_cache_precision.startswith("int8_")
            and (
                self.attention_qk_mode != "numerical"
                or self.attention_value_mode != "numerical"
            )
        ):
            raise ValueError("INT8 value cache requires numerical Q/K/V")
        if self.value_cache_precision.startswith("int8_") and not (
            self.attention_normalizer.startswith("clipped_")
        ):
            raise ValueError("INT8 value cache requires a clipped attention normalizer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.embedding_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.lr_decay_bytes < 0 or not 0.0 <= self.lr_decay_final_fraction <= 1.0:
            raise ValueError("lr decay needs lr_decay_bytes >= 0 and a final fraction in [0, 1]")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.sample_temperature <= 0:
            raise ValueError("sample_temperature must be positive")
        if self.sample_seed < 0:
            raise ValueError("sample_seed cannot be negative")
        if self.attention_normalizer not in {
            "isa_exp2",
            "exact",
            "clipped_linear",
            "clipped_squared",
            "clipped_cubic",
            "clipped_dual_scale",
            "clipped_triple_scale",
        }:
            raise ValueError(
                "attention_normalizer must be isa_exp2, exact, clipped_linear, "
                "clipped_squared, clipped_cubic, clipped_dual_scale, or "
                "clipped_triple_scale"
            )
        if self.clipped_linear_initial_slope <= 0:
            raise ValueError("clipped_linear_initial_slope must be positive")
        if self.attention_window_policy not in {
            "global",
            "head_local",
            "layer_local",
        }:
            raise ValueError(
                "attention_window_policy must be global, head_local, or layer_local"
            )
        if self.attention_window_policy != "global":
            if not 0 < self.local_attention_window <= self.context_tokens:
                raise ValueError(
                    "local_attention_window must be in (0, context_tokens]"
                )
        if self.attention_window_policy == "head_local":
            if not 0 <= self.local_attention_heads <= self.heads:
                raise ValueError("local_attention_heads must be in [0, heads]")
        if self.attention_qk_mode not in {
            "numerical",
            "boolean_hamming",
            "boolean_hamming_calibrated",
            "boolean_hamming_grouped",
        }:
            raise ValueError("invalid attention_qk_mode")
        if self.attention_qkv_projection_mode not in {
            "dense", "signed_squeeze_hybrid", "low_rank_w8a8",
            "bitmix_trajectory_hybrid", "block_sparse_16x16", "shared_qk_basis",
            "local_global_qk", "cyclic_routes_qk", "tied_qk",
            "head_subspace_qk", "head_subspace_global_qk",
        }:
            raise ValueError("invalid QKV projection mode")
        if self.attention_qkv_projection_mode == "signed_squeeze_hybrid":
            squeeze_width = self.dimension * self.qkv_squeeze_bitplanes
            if self.attention_qk_mode != "numerical":
                raise ValueError("signed-squeeze QKV requires numerical Q/K")
            if self.attention_value_mode != "numerical" or self.kv_heads != self.heads:
                raise ValueError("signed-squeeze QKV currently requires numerical full MHA")
            if self.qkv_squeeze_bitplanes <= 0 or self.qkv_squeeze_polarities <= 0:
                raise ValueError("QKV signed squeeze needs positive planes and polarities")
            if self.qkv_squeeze_group_bits not in {16, 32, 64}:
                raise ValueError("QKV signed squeeze groups must be 16, 32, or 64 bits")
            if squeeze_width % self.qkv_squeeze_group_bits:
                raise ValueError("QKV signed squeeze width must contain whole groups")
            if not 0 < self.qkv_squeeze_bypass < self.dimension:
                raise ValueError("QKV signed squeeze bypass must be in (0, dimension)")
            if self.qkv_squeeze_bypass % 32:
                raise ValueError("QKV signed squeeze bypass must use whole 32-lane groups")
            if self.qkv_squeeze_input_rounding not in {"fp16", "fixed8", "fixed4"}:
                raise ValueError("invalid QKV signed-squeeze input rounding")
        if self.attention_qkv_projection_mode == "low_rank_w8a8":
            if self.attention_qk_mode != "numerical":
                raise ValueError("low-rank QKV requires numerical Q/K")
            if self.attention_value_mode != "numerical" or self.kv_heads != self.heads:
                raise ValueError("low-rank QKV currently requires numerical full MHA")
            if not 0 < self.qkv_low_rank < self.dimension:
                raise ValueError("QKV low rank must be in (0, dimension)")
            if self.qkv_low_rank % 16:
                raise ValueError("QKV low rank must be a multiple of 16 for SDOT")
        if self.attention_qkv_projection_mode == "shared_qk_basis":
            if self.attention_qk_mode != "numerical":
                raise ValueError("shared-basis QKV requires numerical Q/K")
            if (
                self.attention_value_mode != "numerical"
                or self.kv_heads != 1
                or self.kv_head_sharing != "value"
            ):
                raise ValueError(
                    "shared-basis QKV requires independent keys and one shared value"
                )
            if self.qkv_shared_bases not in {1, 2}:
                raise ValueError("shared-basis QKV requires one or two bases")
            if resolved_qk_dimension != 16:
                raise ValueError("shared-basis QKV currently requires 16-lane Q/K heads")
            if self.qkv_shared_bases == 2 and self.attention_window_policy != "head_local":
                raise ValueError("two Q/K bases require head-local attention groups")
        if self.attention_qkv_projection_mode == "local_global_qk":
            group_width = self.dimension // self.heads
            if self.attention_qk_mode != "numerical":
                raise ValueError("local-global QKV requires numerical Q/K")
            if (
                self.attention_value_mode != "numerical"
                or self.kv_heads != 1
                or self.kv_head_sharing != "value"
            ):
                raise ValueError(
                    "local-global QKV requires independent keys and one shared value"
                )
            if resolved_qk_dimension != 16 or self.dimension % self.heads:
                raise ValueError("local-global QKV requires 16-lane Q/K heads")
            if (
                self.qkv_local_width <= 0
                or self.qkv_local_width >= self.dimension
                or self.qkv_local_width % group_width
            ):
                raise ValueError(
                    "local-global QKV width must contain whole cyclic head groups"
                )
            if self.qkv_global_rank != 16:
                raise ValueError("local-global QKV currently requires rank 16")
        if self.attention_qkv_projection_mode == "cyclic_routes_qk":
            if self.attention_qk_mode != "numerical":
                raise ValueError("cyclic-routed QKV requires numerical Q/K")
            if (
                self.attention_value_mode != "numerical"
                or self.kv_heads != 1
                or self.kv_head_sharing != "value"
            ):
                raise ValueError(
                    "cyclic-routed QKV requires independent keys and one shared value"
                )
            if resolved_qk_dimension != 16:
                raise ValueError("cyclic-routed QKV requires 16-lane Q/K heads")
            if self.qkv_route_count not in {16, 32}:
                raise ValueError("cyclic-routed QKV requires 16 or 32 routes")
        if self.attention_qkv_projection_mode in {
            "tied_qk", "head_subspace_qk", "head_subspace_global_qk"
        }:
            if self.attention_qk_mode != "numerical":
                raise ValueError("shared-subspace QKV requires numerical Q/K")
            if (
                self.attention_value_mode != "numerical"
                or self.kv_heads != 1
                or self.kv_head_sharing != "value"
            ):
                raise ValueError(
                    "shared-subspace QKV requires independent keys and one shared value"
                )
            if resolved_qk_dimension != 16:
                raise ValueError("shared-subspace QKV requires 16-lane Q/K heads")
            if (
                self.attention_qkv_projection_mode == "head_subspace_global_qk"
                and self.qkv_global_rank != 8
            ):
                raise ValueError(
                    "corrected head-subspace QKV currently requires rank 8"
                )
        if self.attention_qkv_projection_mode == "bitmix_trajectory_hybrid":
            if self.attention_qk_mode != "numerical":
                raise ValueError("bitmix-trajectory QKV requires numerical Q/K")
            if self.attention_value_mode != "numerical" or self.kv_heads != self.heads:
                raise ValueError("bitmix-trajectory QKV currently requires full MHA")
            if self.qkv_squeeze_bitplanes <= 0:
                raise ValueError("bitmix-trajectory QKV needs positive bitplanes")
            if self.dimension * self.qkv_squeeze_bitplanes % 128:
                raise ValueError("bitmix-trajectory QKV needs whole NEON registers")
            if self.qkv_bitmix_rounds <= 0:
                raise ValueError("bitmix-trajectory QKV needs positive rounds")
            if self.qkv_bitmix_cell not in {"eor3", "bcax"}:
                raise ValueError("invalid bitmix-trajectory QKV cell")
            if self.qkv_bitmix_correction_features <= 0 or (
                self.qkv_bitmix_correction_features % (self.qkv_bitmix_rounds + 1)
            ):
                raise ValueError("bitmix QKV features must divide across trajectory")
            if not 0 < self.qkv_squeeze_bypass < self.dimension:
                raise ValueError("bitmix QKV bypass must be in (0, dimension)")
            if self.qkv_squeeze_bypass % 32:
                raise ValueError("bitmix QKV bypass must use aligned 32-channel groups")
            if self.qkv_squeeze_input_rounding != "fp16":
                raise ValueError("bitmix-trajectory QKV requires FP16 input rounding")
        if self.attention_qkv_projection_mode == "block_sparse_16x16":
            if self.attention_qk_mode != "numerical":
                raise ValueError("block-sparse QKV requires numerical Q/K")
            if self.attention_value_mode != "numerical":
                raise ValueError("block-sparse QKV requires numerical values")
            if self.dimension % 16:
                raise ValueError("block-sparse QKV requires dimension divisible by 16")
            if self.qkv_block_allocation not in {
                "uniform", "key_protected", "value_protected"
            }:
                raise ValueError("invalid QKV block allocation")
            if (
                self.qkv_block_allocation != "uniform"
                and self.attention_value_block_active_fraction != 0.75
            ):
                raise ValueError("protected QKV allocations require a 75% ticket budget")
            if self.qkv_head_input_groups:
                if not 1 <= self.qkv_head_input_groups <= self.heads:
                    raise ValueError("QKV head input groups must be in [1, heads]")
                if self.kv_head_sharing != "value" or self.kv_heads != 1:
                    raise ValueError(
                        "head-local QKV inputs require independent keys and one shared value"
                    )
                if (self.dimension // self.heads) % 16:
                    raise ValueError(
                        "head-local QKV input groups must align to 16 input columns"
                    )
        elif self.qkv_block_allocation != "uniform":
            raise ValueError("QKV block allocation requires block-sparse QKV")
        elif self.qkv_head_input_groups:
            raise ValueError("QKV head input groups require block-sparse QKV")
        if self.attention_qk_mode.startswith("boolean_"):
            if self.kv_heads != self.heads:
                raise ValueError("Boolean Q/K currently requires full multi-head attention")
            if self.boolean_qk_width <= 0 or self.boolean_qk_rounds <= 0:
                raise ValueError("Boolean Q/K width and rounds must be positive")
            if self.boolean_qk_width % self.dimension:
                raise ValueError("Boolean Q/K width must be a multiple of model dimension")
            if self.boolean_qk_width % self.boolean_register_bits:
                raise ValueError("Boolean Q/K width must contain whole execution registers")
            if self.boolean_qk_bits_per_head <= 0:
                raise ValueError("Boolean Q/K bits per head must be positive")
            if self.heads * self.boolean_qk_bits_per_head > self.boolean_qk_width:
                raise ValueError("Boolean Q/K output does not fit its packed width")
            if (
                self.attention_qk_mode == "boolean_hamming_grouped"
                and self.boolean_qk_bits_per_head % 4
            ):
                raise ValueError("grouped Boolean Q/K requires four equal bit groups")
            if self.boolean_qk_correction_features < 0:
                raise ValueError("Boolean Q/K correction features cannot be negative")
            if self.boolean_qk_correction_taps not in {
                "final", "equal_trajectory", "late_trajectory"
            }:
                raise ValueError("invalid Boolean Q/K correction taps")
            if (
                self.boolean_qk_correction_taps != "final"
                and self.boolean_qk_correction_features <= 0
            ):
                raise ValueError("multi-tap Q/K requires correction features")
        elif (
            self.boolean_qk_correction_features != 0
            or self.boolean_qk_correction_taps != "final"
        ):
            raise ValueError("Boolean Q/K correction requires Boolean Q/K")
        if self.attention_value_mode not in {
            "numerical",
            "block_sparse_16x16",
            "boolean_lut3",
            "boolean_lut3_bipolar",
            "boolean_lut3_corrected",
        }:
            raise ValueError("invalid attention_value_mode")
        if self.attention_value_mode.startswith("boolean_"):
            if self.attention_qk_mode != "numerical" or self.kv_heads != self.heads:
                raise ValueError("Boolean V currently requires numerical full-MHA Q/K")
            if self.boolean_value_width <= 0 or self.boolean_value_rounds <= 0:
                raise ValueError("Boolean V width and rounds must be positive")
            if self.boolean_value_width % self.dimension:
                raise ValueError("Boolean V width must be a multiple of model dimension")
            if self.boolean_value_width % self.boolean_register_bits:
                raise ValueError("Boolean V width must contain whole execution registers")
            if self.dimension > self.boolean_value_width:
                raise ValueError("Boolean V output does not fit its packed width")
            value_bits = (
                self.boolean_value_bits
                if self.boolean_value_bits > 0
                else self.dimension
            )
            if not 0 < value_bits <= self.dimension:
                raise ValueError("Boolean V bits must be in (0, dimension]")
            if value_bits % self.heads:
                raise ValueError("Boolean V bits must divide evenly across heads")
            if self.boolean_value_correction_features < 0:
                raise ValueError("Boolean V correction features cannot be negative")
            if self.attention_value_mode == "boolean_lut3_corrected":
                if value_bits >= self.dimension:
                    raise ValueError("corrected Boolean V must leave numerical channels")
                if self.boolean_value_correction_features <= 0:
                    raise ValueError("corrected Boolean V requires correction features")
                if (self.dimension - value_bits) % self.heads:
                    raise ValueError("Boolean V correction channels must divide across heads")
            elif self.boolean_value_correction_features != 0:
                raise ValueError("Boolean V correction requires corrected mode")
        elif self.boolean_value_bits != 0 or self.boolean_value_correction_features != 0:
            raise ValueError("Boolean V fields require a Boolean V mode")
        if not 0 < self.attention_value_block_active_fraction <= 1:
            raise ValueError("attention-value block fraction must be in (0, 1]")
        if self.attention_value_mode == "block_sparse_16x16":
            if self.attention_qk_mode != "numerical" or self.kv_heads != self.heads:
                raise ValueError("block-sparse V currently requires numerical full MHA")
            if self.dimension % 16:
                raise ValueError("block-sparse V requires dimension divisible by 16")
        if self.attention_output_mode not in {
            "numerical",
            "boolean_lut3_direct",
            "boolean_lut3_decoded",
            "block_sparse_16x16",
            "product_codebook_16",
            "head_local_hybrid",
            "signed_squeeze_hybrid",
        }:
            raise ValueError("invalid attention_output_mode")
        if self.attention_output_mode.startswith("boolean_"):
            if self.boolean_output_width <= 0 or self.boolean_output_rounds <= 0:
                raise ValueError("Boolean output width and rounds must be positive")
            if self.boolean_output_width % self.dimension:
                raise ValueError("Boolean output width must be a multiple of model dimension")
            if self.boolean_output_width % self.boolean_register_bits:
                raise ValueError("Boolean output width must contain whole execution registers")
            if self.dimension > self.boolean_output_width:
                raise ValueError("Boolean output does not fit its packed width")
            if self.boolean_output_correction_features < 0:
                raise ValueError("Boolean output correction features cannot be negative")
            if (
                self.boolean_output_correction_features > 0
                and self.attention_output_mode != "boolean_lut3_decoded"
            ):
                raise ValueError("Boolean output correction requires decoded mode")
            if self.boolean_output_correction_taps not in {
                "final", "equal_trajectory", "late_trajectory"
            }:
                raise ValueError("invalid Boolean output correction taps")
            if self.boolean_output_routes not in {
                "byte_lane", "register_mix", "bitmix", "xar"
            }:
                raise ValueError("invalid Boolean output route schedule")
            if self.boolean_output_cell not in {"lut3", "xar_lut2"}:
                raise ValueError("invalid Boolean output cell")
            if self.boolean_output_cell == "xar_lut2" and (
                self.attention_output_mode != "boolean_lut3_decoded"
                or self.boolean_output_routes != "xar"
            ):
                raise ValueError("XAR-LUT2 output requires decoded XAR routing")
            if (
                self.boolean_output_routes == "xar"
                and self.boolean_output_cell != "xar_lut2"
            ):
                raise ValueError("XAR output routing requires XAR-LUT2")
        elif self.boolean_output_correction_features != 0:
            raise ValueError("Boolean output correction requires decoded mode")
        elif (
            self.boolean_output_correction_taps != "final"
            or self.boolean_output_routes != "byte_lane"
            or self.boolean_output_cell != "lut3"
        ):
            raise ValueError("Boolean output interface fields require a Boolean mode")
        if not 0 < self.attention_output_block_active_fraction <= 1:
            raise ValueError("attention-output block fraction must be in (0, 1]")
        if self.attention_output_mode == "block_sparse_16x16" and self.dimension % 16:
            raise ValueError("block-sparse attention output requires dimension divisible by 16")
        if self.attention_output_mode == "product_codebook_16":
            if self.dimension % 16:
                raise ValueError("product-codebook output requires dimension divisible by 16")
            if self.attention_output_codewords not in {32, 64}:
                raise ValueError("product-codebook output requires 32 or 64 codewords")
            if self.attention_output_codebook_temperature <= 0:
                raise ValueError("product-codebook temperature must be positive")
            if not 1 <= self.attention_output_codebook_response_shift <= 16:
                raise ValueError("product-codebook response shift must be in [1, 16]")
        if self.attention_output_mode == "head_local_hybrid":
            if self.dimension % self.heads or (self.dimension // self.heads) % 16:
                raise ValueError(
                    "head-local output requires 16-lane-aligned head dimensions"
                )
            if self.attention_output_head_local_rank not in {0, 8}:
                raise ValueError("head-local output rank must be 0 or 8")
        if self.attention_output_mode == "signed_squeeze_hybrid":
            squeeze_width = self.dimension * self.attention_output_squeeze_bitplanes
            if self.attention_output_squeeze_bitplanes <= 0:
                raise ValueError("signed squeeze needs positive bitplanes")
            if self.attention_output_squeeze_polarities <= 0:
                raise ValueError("signed squeeze needs positive polarities")
            if self.attention_output_squeeze_group_bits not in {16, 32, 64}:
                raise ValueError("signed squeeze groups must contain 16, 32, or 64 bits")
            if squeeze_width % self.attention_output_squeeze_group_bits:
                raise ValueError("signed squeeze width must contain whole groups")
            if not 0 < self.attention_output_squeeze_bypass < self.dimension:
                raise ValueError("signed squeeze bypass must be in (0, dimension)")
        if self.attention_output_input_rounding not in {"none", "fp16"}:
            raise ValueError("invalid attention-output input rounding")
        if self.resume and self.initialize_from:
            raise ValueError("resume and initialize_from are mutually exclusive")
        if self.projection_distill_updates < 0:
            raise ValueError("projection_distill_updates cannot be negative")
        if self.projection_distill_updates and not self.initialize_from:
            raise ValueError("projection distillation requires initialize_from")
        if self.attention_window_policy == "layer_local" and any(
            not 0 <= layer < self.layers for layer in self.local_attention_layers
        ):
            raise ValueError("local_attention_layers contains an invalid layer")
        if self.feedforward_activation not in {
            "relu_squared",
            "reglu",
            "isa_swiglu",
            "boolean_lut3",
            "boolean_lut3_gated",
            "boolean_factorized_minterm",
            "boolean_dual3",
        }:
            raise ValueError(
                "feedforward_activation must be relu_squared, reglu, isa_swiglu, "
                "boolean_lut3, boolean_lut3_gated, "
                "boolean_factorized_minterm, or boolean_dual3"
            )
        if self.feedforward_activation.startswith("boolean_"):
            if self.boolean_width <= 0 or self.boolean_rounds <= 0:
                raise ValueError("Boolean width and rounds must be positive")
            if self.boolean_width % self.dimension:
                raise ValueError("Boolean width must be a multiple of model dimension")
            if self.boolean_width % self.boolean_register_bits:
                raise ValueError("Boolean width must contain whole execution registers")
            if self.boolean_backward not in {"ste", "multilinear", "softened", "hybrid"}:
                raise ValueError("invalid Boolean backward mode")
            if self.boolean_temperature <= 0:
                raise ValueError("Boolean temperature must be positive")
            if not 0 <= self.boolean_gradient_leak <= 1:
                raise ValueError("Boolean gradient leak must be in [0, 1]")
            if self.boolean_program_init_margin <= 0:
                raise ValueError("Boolean program initialization margin must be positive")
            if self.boolean_balance_weight < 0 or self.boolean_gate_weight < 0:
                raise ValueError("Boolean regularization weights cannot be negative")
            if self.boolean_ffn_correction_features < 0:
                raise ValueError("Boolean FFN correction features cannot be negative")
            if self.boolean_ffn_correction_mode not in {
                "dense", "product_codebook_32", "product_codebook_64"
            }:
                raise ValueError("invalid Boolean FFN correction mode")
            if self.boolean_ffn_correction_mode != "dense":
                if self.boolean_ffn_correction_features <= 0:
                    raise ValueError("FFN correction codebook requires correction features")
                if self.boolean_ffn_correction_rank:
                    raise ValueError("FFN correction codebook cannot also be low rank")
                if (
                    self.dimension % 16
                    or self.boolean_ffn_correction_features % 16
                ):
                    raise ValueError(
                        "FFN correction codebook requires 16-aligned dimensions"
                    )
            if self.boolean_ffn_correction_rank:
                if self.boolean_ffn_correction_features <= 0:
                    raise ValueError(
                        "factorized FFN correction requires correction features"
                    )
                if (
                    self.boolean_ffn_correction_rank % 16
                    or self.boolean_ffn_correction_rank
                    >= min(self.dimension, self.boolean_ffn_correction_features)
                ):
                    raise ValueError(
                        "FFN correction rank must be a multiple of 16 below both matrix dimensions"
                    )
            if not 0 < self.boolean_ffn_correction_block_active_fraction <= 1:
                raise ValueError("Boolean FFN correction block fraction must be in (0, 1]")
            if self.boolean_ffn_correction_block_active_fraction < 1:
                if (
                    self.boolean_ffn_correction_rank
                    or self.boolean_ffn_correction_mode != "dense"
                ):
                    raise ValueError(
                        "Boolean FFN correction blocks require a direct correction"
                    )
                if (
                    self.boolean_ffn_correction_features <= 0
                    or self.dimension % 16
                    or self.boolean_ffn_correction_features % 16
                ):
                    raise ValueError(
                        "Boolean FFN correction blocks require 16-aligned direct dimensions"
                    )
            if self.boolean_ffn_correction_taps not in {
                "final", "equal_trajectory", "late_trajectory"
            }:
                raise ValueError("invalid Boolean FFN correction taps")
            if self.boolean_ffn_pool_schedule not in {
                "legacy_mixed", "state_aligned", "state_rotated"
            }:
                raise ValueError("invalid Boolean FFN correction pool schedule")
            if (
                self.boolean_ffn_pool_schedule != "legacy_mixed"
                and self.boolean_ffn_correction_taps != "equal_trajectory"
            ):
                raise ValueError(
                    "state-aligned FFN pools require equal trajectory taps"
                )
            if self.boolean_ffn_routes not in {
                "byte_lane", "register_mix", "bitmix"
            }:
                raise ValueError("invalid Boolean FFN route schedule")
            if self.boolean_ffn_thresholds not in {
                "linear", "gaussian_quantile", "power2"
            }:
                raise ValueError("invalid Boolean FFN threshold schedule")
            if self.boolean_ffn_input_rounding not in {"none", "fp16", "fixed8"}:
                raise ValueError("invalid Boolean FFN input rounding")
            if (
                self.boolean_ffn_thresholds == "power2"
                and self.boolean_width // self.dimension != 8
            ):
                raise ValueError("power2 Boolean thresholds require eight bitplanes")
            if (
                self.boolean_ffn_correction_taps != "final"
                and self.boolean_ffn_correction_features <= 0
            ):
                raise ValueError("multi-tap correction requires correction features")
        elif (
            self.boolean_ffn_correction_features != 0
            or self.boolean_ffn_correction_rank != 0
            or self.boolean_ffn_correction_mode != "dense"
            or self.boolean_ffn_correction_block_active_fraction != 1.0
            or self.boolean_ffn_correction_taps != "final"
            or self.boolean_ffn_pool_schedule != "legacy_mixed"
            or self.boolean_ffn_routes != "byte_lane"
            or self.boolean_ffn_thresholds != "linear"
            or self.boolean_ffn_input_rounding != "none"
        ):
            raise ValueError("Boolean FFN correction requires a Boolean FFN")
        if self.boolean_residual_state not in {
            "numerical",
            "persistent_xor",
            "persistent_bcax",
            "persistent_bsl25",
            "persistent_bsl50",
        }:
            raise ValueError("invalid Boolean residual-state mode")
        if (
            self.boolean_residual_state != "numerical"
            and not self.feedforward_activation.startswith("boolean_")
        ):
            raise ValueError("persistent Boolean residual state requires a Boolean FFN")
        if self.residual_boundary not in {"none", "block_fp16", "dual_fp16"}:
            raise ValueError("invalid residual boundary")
        if self.qkv_block_rows not in {8, 16, 32}:
            raise ValueError("QKV ticket rows must be 8, 16, or 32")
        if self.qkv_ticket_selection not in {
            "magnitude", "trainable_global", "trainable_family"
        }:
            raise ValueError("invalid QKV ticket selection")
        if self.qkv_ticket_temperature <= 0:
            raise ValueError("QKV ticket temperature must be positive")
        if (
            self.qkv_ticket_selection != "magnitude"
            and self.attention_qkv_projection_mode != "block_sparse_16x16"
        ):
            raise ValueError("trainable QKV tickets require block-sparse QKV")
        if (
            self.attention_qkv_projection_mode == "block_sparse_16x16"
            and self.dimension % self.qkv_block_rows
        ):
            raise ValueError("QKV ticket rows must divide the model dimension")
        if self.residual_normalizer not in {"rms", "centered_rms", "centered_l1"}:
            raise ValueError("invalid residual normalizer")
        if (
            self.boolean_residual_state.startswith("persistent_bsl")
            and self.boolean_width // self.dimension < 4
        ):
            raise ValueError("selective BSL persistence requires at least four bitplanes")
        if self.ff_extra_dimension < 0 or self.ff_extra_dimension_layers < 0:
            raise ValueError("extra FF dimensions and layer count cannot be negative")
        if self.ff_extra_dimension_layers > self.layers:
            raise ValueError("ff_extra_dimension_layers cannot exceed layers")
        if self.token_feature not in {"plain", "value_embedding", "hashed_bigram"}:
            raise ValueError(
                "token_feature must be plain, value_embedding, or hashed_bigram"
            )
        if self.token_smear not in {"none", "static", "clipped_dynamic"}:
            raise ValueError(
                "token_smear must be none, static, or clipped_dynamic"
            )
        if self.token_feature == "value_embedding" and any(
            not 0 <= layer < self.layers for layer in self.value_embedding_layers
        ):
            raise ValueError("value_embedding_layers contains an invalid layer")
        if self.token_feature == "hashed_bigram":
            if (
                self.bigram_hash_size <= 0
                or self.bigram_hash_size & (self.bigram_hash_size - 1)
            ):
                raise ValueError("bigram_hash_size must be a positive power of two")
            if not 0 < self.bigram_dimension <= self.dimension:
                raise ValueError("bigram_dimension must be in (0, dimension]")
            if self.bigram_sign_mode not in {"none", "scalar", "vector"}:
                raise ValueError(
                    "bigram_sign_mode must be none, scalar, or vector"
                )
        elif self.bigram_sign_mode != "none":
            raise ValueError("bigram_sign_mode requires token_feature=hashed_bigram")
        if self.ngram_orders:
            orders = tuple(self.ngram_orders)
            if tuple(sorted(set(orders))) != orders or not all(
                2 <= order <= 16 for order in orders
            ):
                raise ValueError("ngram_orders must be strictly increasing within [2, 16]")
            if not 1 <= self.ngram_heads <= 4:
                raise ValueError("ngram_heads must be in [1, 4]")
            if self.ngram_table_rows <= 0 or self.ngram_table_rows & (
                self.ngram_table_rows - 1
            ):
                raise ValueError("ngram_table_rows must be a positive power of two")
            if (
                self.ngram_dimension <= 0
                or len(orders) * self.ngram_dimension > self.dimension
            ):
                raise ValueError("n-gram slices must fit inside the embedding width")
        if self.residual_routing not in {
            "ordinary",
            "u_skip",
            "dynamic_input",
        }:
            raise ValueError(
                "residual_routing must be ordinary, u_skip, or dynamic_input"
            )
        if self.vocabulary_path not in {
            "numerical",
            "boolean_hybrid",
            "boolean_tied",
        }:
            raise ValueError("invalid vocabulary_path")
        if self.vocabulary_path != "numerical" and not self.tie_embeddings:
            raise ValueError("Boolean vocabulary paths require tied token codes")
        if self.vocabulary_path == "boolean_hybrid" and self.dimension // 5 <= 1:
            raise ValueError("hybrid Boolean vocabulary path needs dimension >= 10")
        if self.vocabulary_path == "boolean_hybrid":
            code_width = (
                self.boolean_vocab_code_width
                if self.boolean_vocab_code_width > 0
                else self.dimension - (self.dimension // 5 - 1) - 1
            )
            if not 0 < code_width < self.dimension - 1:
                raise ValueError("hybrid Boolean token-code width must leave correction capacity")
            if self.boolean_vocab_pool_bits not in {8, 16, 32}:
                raise ValueError("Boolean vocabulary pool bits must be 8, 16, or 32")
            if self.boolean_vocab_pool_schedule not in {
                "uniform", "mixed_8_16_32", "mixed_16_32"
            }:
                raise ValueError("invalid Boolean vocabulary pool schedule")
            if self.boolean_vocab_correction_rank and (
                self.boolean_vocab_correction_rank % 16
                or self.boolean_vocab_correction_rank >= self.dimension - code_width - 1
            ):
                raise ValueError(
                    "Boolean vocabulary correction rank must be a multiple of 16 "
                    "below the correction feature count"
                )
            if self.boolean_vocab_factor_output not in {
                "fp16", "w8a8_row", "w8a8_group16"
            }:
                raise ValueError("invalid Boolean vocabulary factor output")
            if self.vocabulary_hierarchy not in {"flat", "balanced_32x64"}:
                raise ValueError("invalid vocabulary hierarchy")
            if (
                self.vocabulary_hierarchy != "flat"
                and not self.boolean_vocab_correction_rank
            ):
                raise ValueError(
                    "hierarchical vocabulary requires a factorized correction"
                )
            if (
                self.boolean_vocab_factor_output != "fp16"
                and not self.boolean_vocab_correction_rank
            ):
                raise ValueError(
                    "W8A8 Boolean vocabulary output requires a factorized correction"
                )
            if self.boolean_token_expansion not in {"repeat", "xor_pair", "bcax_triplet"}:
                raise ValueError("invalid Boolean token expansion")
            if (
                self.boolean_token_expansion != "repeat"
                and not 2 * code_width < self.dimension <= 3 * code_width
            ):
                raise ValueError("nonlinear token expansion needs two copies plus derived lanes")
        elif (
            self.boolean_vocab_code_width != 0
            or self.boolean_vocab_pool_bits != 16
            or self.boolean_vocab_pool_schedule != "uniform"
            or self.boolean_vocab_correction_rank != 0
            or self.boolean_vocab_factor_output != "fp16"
            or self.vocabulary_hierarchy != "flat"
            or self.boolean_token_expansion != "repeat"
        ):
            raise ValueError("Boolean vocabulary allocation applies only to boolean_hybrid")
        if self.residual_routing == "u_skip" and any(
            len(pair) != 2
            or not 0 <= pair[0] < pair[1] < self.layers
            for pair in self.u_skip_pairs
        ):
            raise ValueError("u_skip_pairs must contain valid forward layer pairs")
        if self.residual_routing == "dynamic_input" and any(
            not 0 <= layer < self.layers for layer in self.dynamic_route_layers
        ):
            raise ValueError("dynamic_route_layers contains an invalid layer")
        if self.residual_routing == "dynamic_input" and not (
            0 < self.dynamic_route_dimension <= self.dimension
        ):
            raise ValueError("dynamic_route_dimension must be in (0, dimension]")
        if (
            len(set(self.attention_skip_layers)) != len(self.attention_skip_layers)
            or any(not 0 <= layer < self.layers for layer in self.attention_skip_layers)
        ):
            raise ValueError("attention_skip_layers must contain unique valid layers")
        shared = tuple(self.attention_shared_input_layers)
        if shared and (
            len(shared) < 2
            or tuple(sorted(set(shared))) != shared
            or any(not 0 <= layer < self.layers for layer in shared)
            or any(right != left + 1 for left, right in zip(shared, shared[1:]))
            or any(layer in self.attention_skip_layers for layer in shared)
        ):
            raise ValueError(
                "attention_shared_input_layers must be a contiguous group of "
                "at least two active layers"
            )
        shared_qk = tuple(self.attention_shared_qk_layers)
        if self.attention_shared_qk_mode not in {"none", "key", "qk"}:
            raise ValueError("attention_shared_qk_mode must be none, key, or qk")
        if bool(shared_qk) != (self.attention_shared_qk_mode != "none"):
            raise ValueError(
                "attention_shared_qk_layers and attention_shared_qk_mode must "
                "be enabled together"
            )
        if shared_qk and (
            len(shared_qk) < 2
            or tuple(sorted(set(shared_qk))) != shared_qk
            or any(not 0 <= layer < self.layers for layer in shared_qk)
            or any(right != left + 1 for left, right in zip(shared_qk, shared_qk[1:]))
            or any(layer not in shared for layer in shared_qk)
            or self.attention_qk_mode != "numerical"
            or self.boolean_residual_state != "numerical"
            or self.attention_qkv_projection_mode
            not in {"dense", "block_sparse_16x16"}
        ):
            raise ValueError(
                "attention_shared_qk_layers must be a contiguous numerical-Q/K "
                "subset of attention_shared_input_layers"
            )
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")
