from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from .config import VocabLMConfig
from .data import DocumentBlockDataset, TokenCorpus, TokenWindowDataset, corpus_loaders
from .generation import generate
from .metrics import generation_metrics
from .model import (
    BooleanVocabularyReadout,
    VocabularyTransformerLM,
    build_model,
    capacity_report,
)
from .tokenizer import END_OF_TEXT, Vocabulary


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also invokes accelerator seed hooks. CPU runs must not.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def _optimizer(
    model: VocabularyTransformerLM, config: VocabLMConfig
) -> torch.optim.Optimizer:
    groups: dict[str, list[nn.Parameter]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if "ngram_table" in name:
            groups["ngram"].append(parameter)
        elif "token_embedding" in name or "readout" in name:
            groups["embedding"].append(parameter)
        else:
            groups["ordinary"].append(parameter)
    return torch.optim.AdamW(
        [
            {
                "name": name,
                "params": parameters,
                "lr": (
                    config.embedding_learning_rate
                    if name == "embedding"
                    else config.learning_rate
                ),
                # Table rows are hit rarely; decaying every row every step would
                # erase what a rare n-gram learned between its occurrences.
                "weight_decay": 0.0 if name == "ngram" else config.weight_decay,
            }
            for name, parameters in groups.items()
        ],
        weight_decay=config.weight_decay,
    )


def learning_rate_scale(config: VocabLMConfig, run_bytes: int) -> float:
    """Linear decay to ``lr_decay_final_fraction`` over the last ``lr_decay_bytes`` of a run."""
    if not config.lr_decay_bytes:
        return 1.0
    remaining = config.target_decoded_bytes - run_bytes
    progress = min(1.0, max(0.0, remaining / config.lr_decay_bytes))
    return config.lr_decay_final_fraction + (1.0 - config.lr_decay_final_fraction) * progress


def _deterministic_indices(
    dataset,
    batch_size: int,
    seed: int,
    update_index: int,
    sampling: str = "blocks",
    region_tokens: int = 0,
) -> list[int]:
    if sampling == "shared_regions":
        if (not isinstance(dataset, TokenWindowDataset)
                or dataset.stride != dataset.context
                or region_tokens <= 0 or region_tokens % dataset.context):
            raise ValueError("shared_regions requires contiguous TokenWindowDataset regions")
        windows = region_tokens // dataset.context
        if batch_size % windows:
            raise ValueError("batch must contain whole shared regions")
        regions = len(dataset) // windows
        if not regions:
            raise ValueError("dataset has no complete shared region")
        region_batch = batch_size // windows
        batches = math.ceil(regions / region_batch)
        epoch, batch_index = divmod(update_index, batches)
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(regions, generator=generator)
        # Repeat only to fill the final batch; every context sees the same repeats.
        order = order.repeat(math.ceil(batches * region_batch / regions))
        selected = order[batch_index * region_batch:(batch_index + 1) * region_batch]
        return [int(region) * windows + offset for region in selected for offset in range(windows)]
    batches_per_epoch = math.ceil(len(dataset) / batch_size)
    epoch, batch_index = divmod(update_index, batches_per_epoch)
    epoch_samples = batches_per_epoch * batch_size
    generator = torch.Generator().manual_seed(seed + epoch)
    if sampling == "domain_balanced":
        domains = getattr(dataset, "block_domains", None)
        if domains is None:
            raise ValueError(
                "domain-balanced sampling requires document domain metadata"
            )
        counts: dict[str, int] = defaultdict(int)
        for domain in domains:
            counts[domain] += 1
        weights = torch.tensor(
            [1.0 / counts[domain] for domain in domains], dtype=torch.float64
        )
        permutation = torch.multinomial(
            weights,
            epoch_samples,
            replacement=True,
            generator=generator,
        )
    elif sampling == "blocks":
        permutation = torch.randperm(len(dataset), generator=generator)
        if len(permutation) < epoch_samples:
            repeats = math.ceil(epoch_samples / len(permutation))
            permutation = permutation.repeat(repeats)[:epoch_samples]
    else:
        raise ValueError(f"unknown training sampling policy: {sampling}")
    start = batch_index * batch_size
    return permutation[start : start + batch_size].tolist()


def _deterministic_batch(
    dataset,
    batch_size: int,
    seed: int,
    update_index: int,
    sampling: str = "blocks",
    region_tokens: int = 0,
):
    indices = _deterministic_indices(
        dataset, batch_size, seed, update_index, sampling, region_tokens
    )
    examples = [dataset[index] for index in indices]
    return tuple(torch.stack([item[field] for item in examples]) for field in range(3))


def _deterministic_byte_batch(
    dataset: DocumentBlockDataset,
    decoded_bytes: int,
    seed: int,
    update_index: int,
) -> tuple[tuple[Tensor, Tensor, Tensor], dict[str, int], int]:
    """Sample whole document blocks uniformly, then mask to an exact byte budget.

    Uniform block draws followed by charging each real target byte make expected
    domain exposure proportional to the corpus bytes, including short terminal
    blocks. The final block is right-masked so contexts with different literal
    overhead consume identical supervised-byte budgets.
    """
    if decoded_bytes <= 0:
        raise ValueError("decoded byte budget must be positive")
    if not isinstance(dataset, DocumentBlockDataset):
        raise ValueError("byte-proportional sampling requires document blocks")
    generator = torch.Generator().manual_seed(seed * 1_000_003 + update_index)
    indices: list[int] = []
    available = 0
    while available < decoded_bytes:
        index = int(torch.randint(len(dataset), (1,), generator=generator))
        indices.append(index)
        available += dataset.block_lengths[index] - 1
    examples = [dataset[index] for index in indices]
    batch = tuple(
        torch.stack([item[field] for item in examples]) for field in range(3)
    )
    target_lengths = batch[2].clone()
    remaining = decoded_bytes
    domain_bytes: dict[str, int] = defaultdict(int)
    for row, index in enumerate(indices):
        real = int(target_lengths[row].sum())
        keep = min(real, remaining)
        if keep < target_lengths.shape[1]:
            target_lengths[row, keep:] = 0
        if dataset.block_domains is not None:
            domain_bytes[dataset.block_domains[index]] += keep
        remaining -= keep
        if remaining == 0:
            for later in range(row + 1, target_lengths.shape[0]):
                target_lengths[later].zero_()
            break
    if remaining:
        raise RuntimeError("failed to assemble the requested decoded-byte budget")
    return (batch[0], batch[1], target_lengths), dict(domain_bytes), len(indices)


@torch.inference_mode()
def evaluate(
    model: VocabularyTransformerLM,
    loader,
    device: torch.device,
    maximum_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    started = time.perf_counter()
    for batch_index, (tokens, targets, target_lengths) in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        logits = model(tokens.to(device))
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        losses = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none"
        )
        valid = target_lengths.flatten() > 0
        total_nll += float(losses[valid].sum())
        total_tokens += int(valid.sum())
        total_bytes += int(target_lengths.sum())
    elapsed = time.perf_counter() - started
    return {
        "loss_per_token": total_nll / max(total_tokens, 1),
        "bpb": total_nll / max(total_bytes, 1) / math.log(2.0),
        "tokens": total_tokens,
        "decoded_bytes": total_bytes,
        "seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "decoded_bytes_per_second": total_bytes / max(elapsed, 1e-9),
    }


def evaluate_document_domains(
    model: VocabularyTransformerLM,
    dataset,
    device: torch.device,
    batch_size: int,
    maximum_batches: int | None,
    target_domain_shares: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate the same frozen validation selection separately by domain."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        selected = list(dataset.indices)
        domains = getattr(base, "block_domains", None)
        if domains is None:
            return {}
        selected_domains = [domains[index] for index in selected]
    else:
        selected = None
        selected_domains = getattr(dataset, "block_domains", None)
        if selected_domains is None:
            return {}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, domain in enumerate(selected_domains):
        groups[domain].append(index)
    results: dict[str, dict[str, float | int]] = {}
    for domain, indices in sorted(groups.items()):
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
        )
        results[domain] = evaluate(model, loader, device, maximum_batches)
    if not results:
        return {}
    bpb = {domain: float(metrics["bpb"]) for domain, metrics in results.items()}
    bytes_by_domain = {
        domain: int(metrics["decoded_bytes"])
        for domain, metrics in results.items()
    }
    total_bytes = sum(bytes_by_domain.values())
    total_tokens = sum(int(metrics["tokens"]) for metrics in results.values())
    total_seconds = sum(float(metrics["seconds"]) for metrics in results.values())
    aggregate_bpb = sum(
        bpb[domain] * bytes_by_domain[domain] for domain in bpb
    ) / max(total_bytes, 1)
    aggregate_loss_per_token = sum(
        float(results[domain]["loss_per_token"]) * int(results[domain]["tokens"])
        for domain in results
    ) / max(total_tokens, 1)
    summary: dict[str, Any] = {
        "macro_average_bpb": sum(bpb.values()) / len(bpb),
        "validation_byte_weighted_bpb": aggregate_bpb,
        "worst_domain": max(bpb, key=bpb.get),
        "worst_domain_bpb": max(bpb.values()),
        "domain_count": len(bpb),
        "decoded_bytes": total_bytes,
    }
    if target_domain_shares:
        present_weight = sum(
            float(target_domain_shares.get(domain, 0.0)) for domain in bpb
        )
        if present_weight > 0:
            summary["target_mix_weighted_bpb"] = sum(
                bpb[domain] * float(target_domain_shares.get(domain, 0.0))
                for domain in bpb
            ) / present_weight
    return {
        "loss_per_token": aggregate_loss_per_token,
        "bpb": aggregate_bpb,
        "tokens": total_tokens,
        "decoded_bytes": total_bytes,
        "seconds": total_seconds,
        "tokens_per_second": total_tokens / max(total_seconds, 1e-9),
        "decoded_bytes_per_second": total_bytes / max(total_seconds, 1e-9),
        "domains": results,
        "domain_summary": summary,
    }


def _resize_ngram_table_for_mask(
    source: Tensor,
    *,
    regions: int,
    source_rows: int,
    target_rows: int,
    dimension: int,
) -> Tensor | None:
    """Resize power-of-two hash banks without changing a grown table's function."""
    table = source.view(regions, source_rows, dimension)
    if target_rows % source_rows == 0:
        return table.repeat(1, target_rows // source_rows, 1)
    if source_rows % target_rows == 0:
        return table.view(
            regions,
            source_rows // target_rows,
            target_rows,
            dimension,
        ).mean(dim=1)
    return None


def _initialize_from_dense_checkpoint(
    model: VocabularyTransformerLM,
    config: VocabLMConfig,
    manifest: dict[str, Any],
    device: torch.device,
) -> tuple[VocabularyTransformerLM, dict[str, Any]]:
    if config.initialize_from is None:
        raise ValueError("initialize_from checkpoint is missing")
    checkpoint = torch.load(
        config.initialize_from, map_location="cpu", weights_only=False
    )
    source_config = VocabLMConfig.from_dict(checkpoint["config"])
    if (
        source_config.corpus_dir != config.corpus_dir
        and manifest.get("tokenizer_kind") != "byte_identity"
    ):
        raise ValueError("cross-corpus warm starts require the identity byte vocabulary")
    output_scaffolds = {"numerical", "block_sparse_16x16", config.attention_output_mode}
    if source_config.attention_output_mode not in output_scaffolds:
        raise ValueError("projection scaffold has an incompatible output mode")
    if source_config.attention_value_mode not in {
        "numerical", config.attention_value_mode
    }:
        raise ValueError("projection scaffold has an incompatible value mode")
    qkv_scaffolds = {"dense", config.attention_qkv_projection_mode}
    if config.attention_qkv_projection_mode in {
        "shared_qk_basis", "local_global_qk", "cyclic_routes_qk",
        "tied_qk", "head_subspace_qk", "head_subspace_global_qk",
    }:
        qkv_scaffolds.add("block_sparse_16x16")
    if source_config.attention_qkv_projection_mode not in qkv_scaffolds:
        raise ValueError("projection scaffold has an incompatible QKV mode")
    ignored = {
        "corpus_dir", "output_dir", "batch_size", "gradient_accumulation_steps",
        "maximum_updates", "target_decoded_bytes", "seed",
        "report_every_decoded_bytes", "validation_batches", "validation_sampling",
        "validation_region_tokens", "validation_regions",
        "training_sampling", "training_region_tokens",
        "device", "threads", "learning_rate", "embedding_learning_rate",
        "lr_decay_bytes", "lr_decay_final_fraction", "weight_decay",
        "gradient_clip",
        "context_tokens", "stride_tokens", "validation_stride_tokens",
        "sample_every_report", "sample_prompt", "sample_decoded_bytes",
        "sample_temperature", "sample_top_k", "sample_seed", "save_every_reports",
        "save_gate_checkpoints",
        "resume", "initialize_from", "projection_distill_updates",
        "attention_output_mode", "attention_output_block_active_fraction",
        "attention_output_codewords", "attention_output_codebook_temperature",
        "attention_output_codebook_response_shift",
        "attention_output_head_local_rank",
        "boolean_output_correction_taps", "boolean_output_routes",
        "boolean_output_cell",
        "attention_output_squeeze_bitplanes", "attention_output_squeeze_polarities",
        "attention_output_squeeze_group_bits", "attention_output_squeeze_bypass",
        "attention_output_input_rounding",
        "attention_value_mode", "attention_value_block_active_fraction",
        "boolean_ffn_correction_rank",
        "boolean_ffn_correction_mode",
        "boolean_ffn_correction_block_active_fraction",
        "boolean_ffn_pool_schedule",
        "boolean_vocab_correction_rank",
        "boolean_vocab_factor_output",
        "vocabulary_hierarchy",
        "kv_heads", "kv_head_routing", "kv_head_sharing",
        "value_cache_precision",
        "qk_head_dimension",
        "local_qk_head_dimension",
        "layers",
        "attention_qkv_projection_mode", "qkv_squeeze_bitplanes",
        "qkv_squeeze_polarities", "qkv_squeeze_group_bits",
        "qkv_squeeze_bypass", "qkv_squeeze_input_rounding",
        "qkv_low_rank", "qkv_bitmix_rounds",
        "qkv_shared_bases", "qkv_local_width", "qkv_global_rank",
        "qkv_route_count",
        "qkv_bitmix_correction_features", "qkv_bitmix_cell",
        "qkv_block_allocation",
        "qkv_block_rows",
        "qkv_ticket_selection", "qkv_ticket_temperature",
        "qkv_head_input_groups",
        "ngram_table_rows",
        "residual_boundary",
        "attention_skip_layers",
        "attention_shared_input_layers",
        "attention_shared_qk_layers",
        "attention_shared_qk_mode",
        "local_attention_window",
        "residual_routing",
        "u_skip_pairs",
        "dynamic_route_layers",
        "dynamic_route_dimension",
    }
    mismatches = [
        field.name
        for field in __import__("dataclasses").fields(config)
        if field.name not in ignored
        and getattr(source_config, field.name) != getattr(config, field.name)
    ]
    if mismatches:
        raise ValueError(
            "initialization topology mismatch: " + ", ".join(mismatches)
        )
    if config.layers < source_config.layers or config.layers > source_config.layers + 1:
        raise ValueError("warm layer conversion permits exactly one appended layer")
    teacher = build_model(source_config, manifest)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    output_projection_changed = (
        source_config.attention_output_mode != config.attention_output_mode
        or source_config.attention_output_block_active_fraction
        != config.attention_output_block_active_fraction
        or source_config.attention_output_codewords
        != config.attention_output_codewords
        or source_config.attention_output_codebook_response_shift
        != config.attention_output_codebook_response_shift
        or source_config.attention_output_head_local_rank
        != config.attention_output_head_local_rank
    )
    qkv_projection_changed = (
        source_config.attention_qkv_projection_mode
        != config.attention_qkv_projection_mode
        or source_config.attention_value_block_active_fraction
        != config.attention_value_block_active_fraction
        or source_config.qkv_block_allocation != config.qkv_block_allocation
        or source_config.qkv_block_rows != config.qkv_block_rows
        or source_config.qkv_head_input_groups != config.qkv_head_input_groups
        or source_config.qkv_global_rank != config.qkv_global_rank
    )
    ffn_correction_blocks_changed = (
        source_config.boolean_ffn_correction_block_active_fraction
        != config.boolean_ffn_correction_block_active_fraction
    )
    ffn_correction_projection_changed = (
        source_config.boolean_ffn_correction_rank
        != config.boolean_ffn_correction_rank
        or source_config.boolean_ffn_correction_mode
        != config.boolean_ffn_correction_mode
    )
    ffn_pool_changed = (
        source_config.boolean_ffn_pool_schedule
        != config.boolean_ffn_pool_schedule
    )
    for name, value in source_state.items():
        if name not in target_state or target_state[name].shape != value.shape:
            continue
        if (
            output_projection_changed
            and ".attention.output." in name
        ):
            continue
        if (
            qkv_projection_changed
            and ".attention.qkv." in name
        ):
            continue
        if (
            ffn_correction_projection_changed
            and ".boolean_ffn.correction_" in name
        ):
            continue
        if ffn_pool_changed and name.endswith(
            ".boolean_ffn.correction_indices"
        ):
            continue
        target_state[name] = value
    model.load_state_dict(target_state)
    if (
        source_config.ngram_table_rows != config.ngram_table_rows
        and source_config.ngram_orders == config.ngram_orders
        and source_config.ngram_heads == config.ngram_heads
        and source_config.ngram_dimension == config.ngram_dimension
        and "ngram_table.weight" in source_state
        and model.ngram_table is not None
    ):
        regions = len(config.ngram_orders) * config.ngram_heads
        resized = _resize_ngram_table_for_mask(
            source_state["ngram_table.weight"],
            regions=regions,
            source_rows=source_config.ngram_table_rows,
            target_rows=config.ngram_table_rows,
            dimension=config.ngram_dimension,
        )
        if resized is None:
            raise ValueError("n-gram table resize requires an integer row multiple")
        with torch.no_grad():
            model.ngram_table.weight.copy_(resized.reshape_as(model.ngram_table.weight))
    if output_projection_changed:
        for target_layer, source_layer in zip(model.layers, teacher.layers, strict=True):
            source_weight = source_layer.attention.output.weight.detach()
            if hasattr(source_layer.attention.output, "expanded_mask"):
                source_weight = (
                    source_weight
                    * source_layer.attention.output.expanded_mask().detach()
                )
            target_layer.attention.output.initialize_from_dense(
                source_weight
            )
    if source_config.attention_value_mode != config.attention_value_mode:
        for target_layer, source_layer in zip(model.layers, teacher.layers, strict=True):
            target_layer.attention.qkv.initialize_from_dense(
                source_layer.attention.qkv.weight
            )
    if (
        qkv_projection_changed
    ):
        for target_layer, source_layer in zip(model.layers, teacher.layers, strict=True):
            source_projection = source_layer.attention.qkv
            source_weight = source_projection.weight.detach()
            if hasattr(source_projection, "expanded_mask"):
                source_weight = (
                    source_weight * source_projection.expanded_mask().detach()
                )
            target_layer.attention.qkv.initialize_from_dense(
                source_weight
            )
    if ffn_correction_projection_changed:
        for target_layer, source_layer in zip(model.layers, teacher.layers, strict=True):
            if target_layer.boolean_ffn is None or source_layer.boolean_ffn is None:
                raise ValueError("FFN correction conversion requires Boolean FFNs")
            source_weight = source_layer.boolean_ffn.correction_weight
            if source_weight is None:
                raise ValueError("FFN correction conversion requires a dense scaffold")
            target_layer.boolean_ffn.initialize_correction_from_dense(
                source_weight.detach()
            )
    elif ffn_correction_blocks_changed:
        for target_layer, source_layer in zip(model.layers, teacher.layers, strict=True):
            if target_layer.boolean_ffn is None or source_layer.boolean_ffn is None:
                raise ValueError("FFN block conversion requires Boolean FFNs")
            source_weight = source_layer.boolean_ffn.correction_weight
            if source_weight is None:
                raise ValueError("FFN block conversion requires a direct correction")
            target_layer.boolean_ffn.initialize_correction_block_mask_from_dense(
                source_weight.detach()
            )
    if (
        source_config.boolean_vocab_correction_rank
        != config.boolean_vocab_correction_rank
    ):
        if (
            teacher.readout.correction_input_weight is not None
            and teacher.readout.correction_output_weight is not None
        ):
            model.readout.initialize_correction_from_factorized(
                teacher.readout.correction_input_weight.detach(),
                teacher.readout.correction_output_weight.detach(),
            )
        else:
            source_weight = teacher.readout.correction_weight
            if source_weight is None:
                raise ValueError(
                    "readout correction conversion requires a dense or factorized scaffold"
                )
            model.readout.initialize_correction_from_dense(source_weight.detach())
    if source_config.vocabulary_hierarchy != config.vocabulary_hierarchy:
        if not isinstance(model.readout, BooleanVocabularyReadout):
            raise ValueError("hierarchy conversion requires a Boolean readout")
        if not isinstance(teacher.readout, BooleanVocabularyReadout):
            raise ValueError("hierarchy conversion requires a Boolean scaffold")
        if teacher.readout.correction_output_weight is None:
            raise ValueError("hierarchy conversion requires factorized source geometry")
        model.readout.initialize_hierarchy_from_flat(
            teacher.token_embedding.weight.detach(),
            teacher.readout.correction_output_weight.detach(),
        )
    source_qk_dimension = teacher.layers[0].attention.qk_head_dim
    target_qk_dimension = model.layers[0].attention.qk_head_dim
    source_head_qk_dimensions = teacher.layers[0].attention.head_qk_dims
    target_head_qk_dimensions = model.layers[0].attention.head_qk_dims
    if source_head_qk_dimensions != target_head_qk_dimensions:
        for target_layer, source_layer in zip(model.layers, teacher.layers, strict=True):
            source_attention = source_layer.attention
            target_attention = target_layer.attention
            if (
                source_attention.attention_qk_mode != "numerical"
                or target_attention.attention_qk_mode != "numerical"
                or source_attention.key_heads != source_attention.heads
                or target_attention.key_heads != target_attention.heads
                or source_attention.heads != target_attention.heads
                or source_attention.value_heads != target_attention.value_heads
            ):
                raise ValueError(
                    "reduced Q/K warm conversion requires matched numerical independent keys"
                )
            source_projection = source_attention.qkv
            target_projection = target_attention.qkv
            if source_projection is None or target_projection is None:
                raise ValueError("reduced Q/K warm conversion requires fused QKV projections")
            source_weight = source_projection.weight.detach()
            if hasattr(source_projection, "expanded_mask"):
                source_weight = source_weight * source_projection.expanded_mask().detach()
            heads = source_attention.heads
            source_query_rows = heads * source_qk_dimension
            source_key_rows = source_attention.key_heads * source_qk_dimension
            query = source_weight[:source_query_rows].reshape(
                heads, source_qk_dimension, -1
            )
            key = source_weight[
                source_query_rows : source_query_rows + source_key_rows
            ].reshape(heads, source_qk_dimension, -1)
            value = source_weight[source_query_rows + source_key_rows :]
            selected_queries = []
            selected_keys = []
            for head in range(heads):
                source_active = source_attention.head_qk_dims[head]
                target_active = target_attention.head_qk_dims[head]
                if target_active > source_active:
                    raise ValueError(
                        "warm Q/K width conversion cannot add uninitialized lanes"
                    )
                pair_scores = (
                    query[head, :source_active]
                    .square().sum(dim=-1).reshape(-1, 2).sum(dim=-1)
                    + key[head, :source_active]
                    .square().sum(dim=-1).reshape(-1, 2).sum(dim=-1)
                )
                pairs = pair_scores.topk(target_active // 2).indices.sort().values
                lanes = torch.stack((2 * pairs, 2 * pairs + 1), dim=-1).flatten()
                target_query = query.new_zeros(target_qk_dimension, query.shape[-1])
                target_key = key.new_zeros(target_qk_dimension, key.shape[-1])
                target_query[:target_active] = query[head].index_select(0, lanes)
                target_key[:target_active] = key[head].index_select(0, lanes)
                selected_queries.append(target_query)
                selected_keys.append(target_key)
            converted = torch.cat(
                (
                    torch.cat(selected_queries, dim=0),
                    torch.cat(selected_keys, dim=0),
                    value,
                ),
                dim=0,
            )
            if hasattr(target_projection, "initialize_from_dense"):
                target_projection.initialize_from_dense(converted)
            else:
                with torch.no_grad():
                    target_projection.weight.copy_(converted)
    source_attention = teacher.layers[0].attention
    target_attention = model.layers[0].attention
    if (
        source_attention.key_heads != target_attention.key_heads
        or source_attention.value_heads != target_attention.value_heads
    ):
        if source_attention.key_heads != source_config.heads or source_attention.value_heads != source_config.heads:
            raise ValueError("warm KV-head conversion currently requires an MHA scaffold")
        for target_layer, source_layer in zip(model.layers, teacher.layers, strict=True):
            source_projection = source_layer.attention.qkv
            source_weight = source_projection.weight.detach()
            if hasattr(source_projection, "deployed_mask"):
                mask = source_projection.deployed_mask().repeat_interleave(
                    source_projection.block_rows, 0
                ).repeat_interleave(16, 1)
                source_weight = source_weight * mask
            dimension = config.dimension
            head_dim = dimension // config.heads
            query = source_weight[:dimension]
            key = source_weight[dimension : 2 * dimension].reshape(
                source_layer.attention.key_heads, head_dim, dimension
            )
            value = source_weight[2 * dimension :].reshape(
                source_layer.attention.value_heads, head_dim, dimension
            )
            if target_layer.attention.key_heads == 1:
                key = key.mean(dim=0)
            elif target_layer.attention.key_heads == source_layer.attention.key_heads:
                key = key.flatten(0, 1)
            else:
                raise ValueError("unsupported key-head warm conversion")
            if target_layer.attention.value_heads == 1:
                value = value.mean(dim=0)
            elif target_layer.attention.value_heads == source_layer.attention.value_heads:
                value = value.flatten(0, 1)
            else:
                raise ValueError("unsupported value-head warm conversion")
            converted = torch.cat((query, key, value), dim=0)
            if hasattr(target_layer.attention.qkv, "initialize_from_dense"):
                target_layer.attention.qkv.initialize_from_dense(converted)
            else:
                with torch.no_grad():
                    target_layer.attention.qkv.weight.copy_(converted)
    if config.layers > source_config.layers:
        if (
            source_head_qk_dimensions != target_head_qk_dimensions
            or source_config.attention_qkv_projection_mode
            != config.attention_qkv_projection_mode
            or source_config.attention_output_mode != config.attention_output_mode
            or source_config.attention_value_mode != config.attention_value_mode
        ):
            raise ValueError(
                "appended-layer initialization requires an otherwise identical projection graph"
            )
        appended = model.layers[-1]
        appended.load_state_dict(teacher.layers[-1].state_dict())
        if not hasattr(appended, "residual_scales"):
            raise ValueError("appended-layer initialization requires residual scalars")
        with torch.no_grad():
            appended.residual_scales[1] = 0.0
            appended.residual_scales[3] = 0.0
    model.to(device)
    return teacher, checkpoint


def _distill_attention_outputs(
    model: VocabularyTransformerLM,
    teacher: VocabularyTransformerLM,
    dataset,
    config: VocabLMConfig,
    device: torch.device,
    source_update: int,
) -> dict[str, float | int]:
    distill_ffn = (
        model.layers[0].boolean_ffn is not None
        and teacher.layers[0].boolean_ffn is not None
        and (
            model.layers[0].boolean_ffn.correction_rank
            != teacher.layers[0].boolean_ffn.correction_rank
            or model.layers[0].boolean_ffn.correction_mode
            != teacher.layers[0].boolean_ffn.correction_mode
            or model.layers[0].boolean_ffn.pool_schedule
            != teacher.layers[0].boolean_ffn.pool_schedule
        )
    )
    distill_readout = (
        getattr(model.readout, "correction_rank", 0)
        != getattr(teacher.readout, "correction_rank", 0)
        or getattr(model.readout, "hierarchy", "flat")
        != getattr(teacher.readout, "hierarchy", "flat")
    )
    distill_hierarchy = (
        getattr(model.readout, "hierarchy", "flat")
        != getattr(teacher.readout, "hierarchy", "flat")
    )
    distill_qkv = (
        model.layers[0].attention.attention_qkv_projection_mode
        != teacher.layers[0].attention.attention_qkv_projection_mode
        or model.layers[0].attention.attention_value_mode
        != teacher.layers[0].attention.attention_value_mode
        or model.layers[0].attention.key_heads
        != teacher.layers[0].attention.key_heads
        or model.layers[0].attention.value_heads
        != teacher.layers[0].attention.value_heads
        or model.layers[0].attention.qk_head_dim
        != teacher.layers[0].attention.qk_head_dim
        or (
            hasattr(model.layers[0].attention.qkv, "deployed_mask")
            and hasattr(teacher.layers[0].attention.qkv, "deployed_mask")
            and not torch.equal(
                model.layers[0].attention.qkv.deployed_mask(),
                teacher.layers[0].attention.qkv.deployed_mask(),
            )
        )
    )
    distill_value_only = (
        not distill_readout
        and config.attention_value_mode == "block_sparse_16x16"
    )
    if distill_readout:
        module_layer_indices = [0]
        candidate_modules = [model.readout]
        teacher_modules = [teacher.readout]
        parameters = list(model.readout.parameters())
    elif distill_ffn:
        module_layer_indices = list(range(len(model.layers)))
        candidate_modules = [layer.boolean_ffn for layer in model.layers]
        teacher_modules = [layer.boolean_ffn for layer in teacher.layers]
        parameters = [
            parameter
            for module in candidate_modules
            for name, parameter in module.named_parameters()
            if name.startswith("correction_")
        ]
    else:
        module_layer_indices = [
            index
            for index in range(len(model.layers))
            if not (
                model.layers[index].skip_attention
                or teacher.layers[index].skip_attention
            )
        ]
        candidate_modules = [
            model.layers[index].attention.qkv
            if distill_qkv
            else model.layers[index].attention.output
            for index in module_layer_indices
        ]
        teacher_modules = [
            teacher.layers[index].attention.qkv
            if distill_qkv
            else teacher.layers[index].attention.output
            for index in module_layer_indices
        ]
        parameters = [
            parameter for module in candidate_modules for parameter in module.parameters()
        ]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=0.0
    )
    captured: list[tuple[tuple[Tensor, ...], Tensor] | None] = [
        None
    ] * len(teacher_modules)
    hooks = []
    for index, module in enumerate(teacher_modules):
        def capture(_module, arguments, result, *, index=index):
            captured[index] = (
                tuple(argument.detach() for argument in arguments),
                result.detach(),
            )
        hooks.append(module.register_forward_hook(capture))
    first_loss = final_loss = math.nan
    teacher.eval()
    model.train()
    try:
        for distill_update in range(config.projection_distill_updates):
            tokens, _, _ = _deterministic_batch(
                dataset,
                config.batch_size,
                config.seed,
                source_update + distill_update,
                config.training_sampling,
            )
            captured[:] = [None] * len(captured)
            # `no_grad` keeps captured activations usable by the candidate's
            # autograd graph; inference tensors cannot be saved for backward.
            with torch.no_grad():
                teacher(tokens.to(device))
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for module_index, (module, pair) in enumerate(
                zip(candidate_modules, captured, strict=True)
            ):
                if pair is None:
                    raise RuntimeError("teacher projection hook did not fire")
                inputs, targets = pair
                predicted = module(*inputs)
                if (
                    distill_qkv
                    and (
                        model.layers[0].attention.key_heads
                        != teacher.layers[0].attention.key_heads
                        or model.layers[0].attention.value_heads
                        != teacher.layers[0].attention.value_heads
                    )
                ):
                    dimension = model.dimension
                    layer_index = module_layer_indices[module_index]
                    source_attention = teacher.layers[layer_index].attention
                    target_attention = model.layers[layer_index].attention
                    source_key_heads = source_attention.key_heads
                    source_value_heads = source_attention.value_heads
                    head_dim = dimension // target_attention.heads
                    query = targets[..., :dimension]
                    key = targets[
                        ..., dimension : dimension + source_key_heads * head_dim
                    ]
                    value = targets[
                        ..., dimension + source_key_heads * head_dim :
                    ]
                    key = key.reshape(
                        *key.shape[:-1], source_key_heads, head_dim
                    )
                    value = value.reshape(
                        *value.shape[:-1], source_value_heads, head_dim
                    )
                    key = (
                        key.mean(-2)
                        if target_attention.key_heads == 1
                        else key.flatten(-2)
                    )
                    value = (
                        value.mean(-2)
                        if target_attention.value_heads == 1
                        else value.flatten(-2)
                    )
                    targets = torch.cat((query, key, value), dim=-1)
                if distill_value_only:
                    predicted = predicted[..., 2 * model.dimension :]
                    targets = targets[..., 2 * model.dimension :]
                if distill_hierarchy:
                    targets = F.log_softmax(targets, dim=-1)
                losses.append(F.mse_loss(predicted, targets))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
            value = float(loss.detach())
            if distill_update == 0:
                first_loss = value
            final_loss = value
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "updates": config.projection_distill_updates,
        "first_mse": first_loss,
        "final_mse": final_loss,
    }


def _checkpoint(
    model: VocabularyTransformerLM,
    optimizer: torch.optim.Optimizer,
    config: VocabLMConfig,
    *,
    update: int,
    decoded_bytes_processed: int,
    tokens_processed: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "config": config.to_dict(),
        "update": update,
        "decoded_bytes_processed": decoded_bytes_processed,
        "tokens_processed": tokens_processed,
        "validation": validation,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
    }


def run_training(config: VocabLMConfig) -> dict[str, Any]:
    config.validate()
    device = select_device(config.device)
    set_seed(config.seed, device)
    torch.set_num_threads(config.threads)
    corpus = TokenCorpus.load(config.corpus_dir)
    vocabulary = Vocabulary.load(config.corpus_dir)
    train_loader, validation_loader = corpus_loaders(
        corpus,
        context=config.context_tokens,
        stride=config.stride_tokens,
        validation_stride=config.validation_stride_tokens,
        batch_size=config.batch_size,
        seed=config.seed,
        validation_windows=(
            config.validation_batches * config.batch_size
            if config.validation_sampling == "spread" and config.validation_batches
            else None
        ),
        validation_region_tokens=(
            config.validation_region_tokens
            if config.validation_sampling == "shared_regions" else None
        ),
        validation_regions=(
            config.validation_regions
            if config.validation_sampling == "shared_regions" else None
        ),
        validation_domain_bytes=(
            config.validation_domain_bytes
            if config.validation_sampling == "domain_bytes" else None
        ),
    )
    model = build_model(config, corpus.manifest).to(device)
    optimizer = _optimizer(model, config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (output / "corpus_manifest.json").write_text(
        json.dumps(corpus.manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "isa_manifest.json").write_text(
        json.dumps(model.isa_manifest(), indent=2) + "\n", encoding="utf-8"
    )

    update = 0
    decoded_bytes_processed = 0
    tokens_processed = 0
    best_bpb = math.inf
    best_update = 0
    last_validation: dict[str, Any] = {}
    if config.resume:
        resumed = torch.load(config.resume, map_location="cpu", weights_only=False)
        source = VocabLMConfig.from_dict(resumed["config"])
        cross_corpus_resume = source.corpus_dir != config.corpus_dir
        if cross_corpus_resume:
            source_manifest_path = Path(source.corpus_dir) / "manifest.json"
            if not source_manifest_path.exists():
                raise ValueError(
                    "cross-corpus continuation cannot verify the source byte vocabulary"
                )
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            if (
                source_manifest.get("tokenizer_kind") != "byte_identity"
                or corpus.manifest.get("tokenizer_kind") != "byte_identity"
            ):
                raise ValueError(
                    "cross-corpus continuation requires identity byte vocabularies"
                )
        topology = (
            "corpus_dir",
            "context_tokens",
            "dimension",
            "layers",
            "heads",
            "kv_heads",
            "ff_dimension",
            "ff_extra_dimension",
            "ff_extra_dimension_layers",
            "feedforward_activation",
            "boolean_width",
            "boolean_rounds",
            "boolean_ffn_correction_features",
            "boolean_ffn_correction_rank",
            "boolean_ffn_correction_mode",
            "boolean_ffn_correction_block_active_fraction",
            "boolean_ffn_correction_taps",
            "boolean_ffn_pool_schedule",
            "boolean_ffn_routes",
            "boolean_ffn_thresholds",
            "boolean_ffn_input_rounding",
            "boolean_backward",
            "boolean_temperature",
            "boolean_gradient_leak",
            "boolean_program_init_margin",
            "boolean_register_bits",
            "boolean_residual_state",
            "residual_boundary",
            "residual_normalizer",
            "token_feature",
            "value_embedding_layers",
            "bigram_hash_size",
            "bigram_dimension",
            "rope_base",
            "rope_fraction",
            "partial_key_offset",
            "qk_norm",
            "value_cache_precision",
            "qk_head_dimension",
            "local_qk_head_dimension",
            "kv_head_routing",
            "kv_head_sharing",
            "attention_qkv_projection_mode",
            "qkv_squeeze_bitplanes",
            "qkv_squeeze_polarities",
            "qkv_squeeze_group_bits",
            "qkv_squeeze_bypass",
            "qkv_squeeze_input_rounding",
            "qkv_low_rank",
            "qkv_shared_bases",
            "qkv_local_width",
            "qkv_global_rank",
            "qkv_route_count",
            "qkv_bitmix_rounds",
            "qkv_bitmix_correction_features",
            "qkv_bitmix_cell",
            "qkv_block_allocation",
            "qkv_block_rows",
            "qkv_ticket_selection",
            "qkv_ticket_temperature",
            "qkv_head_input_groups",
            "attention_qk_mode",
            "boolean_qk_width",
            "boolean_qk_rounds",
            "boolean_qk_bits_per_head",
            "boolean_qk_correction_features",
            "boolean_qk_correction_taps",
            "attention_value_mode",
            "attention_value_block_active_fraction",
            "boolean_value_width",
            "boolean_value_rounds",
            "attention_output_mode",
            "boolean_output_width",
            "boolean_output_rounds",
            "boolean_output_correction_features",
            "boolean_output_correction_taps",
            "boolean_output_routes",
            "boolean_output_cell",
            "attention_output_block_active_fraction",
            "attention_output_squeeze_bitplanes",
            "attention_output_squeeze_polarities",
            "attention_output_squeeze_group_bits",
            "attention_output_squeeze_bypass",
            "attention_output_input_rounding",
            "attention_normalizer",
            "clipped_linear_initial_slope",
            "attention_window_policy",
            "local_attention_window",
            "local_attention_heads",
            "local_attention_layers",
            "attention_skip_layers",
            "zero_init_residual",
            "tie_embeddings",
            "vocabulary_path",
            "vocabulary_hierarchy",
            "boolean_vocab_code_width",
            "boolean_vocab_pool_bits",
            "boolean_vocab_pool_schedule",
            "boolean_token_expansion",
            "residual_scalars",
            "residual_routing",
            "u_skip_pairs",
            "dynamic_route_layers",
            "dynamic_route_dimension",
            "vocabulary_padding_multiple",
        )
        mismatches = [
            name
            for name in topology
            if not (name == "corpus_dir" and cross_corpus_resume)
            and getattr(source, name) != getattr(config, name)
        ]
        ffn_block_conversion = (
            source.boolean_ffn_correction_block_active_fraction
            != config.boolean_ffn_correction_block_active_fraction
        )
        if ffn_block_conversion:
            mismatches.remove("boolean_ffn_correction_block_active_fraction")
        vocab_rank_expansion = (
            0 < source.boolean_vocab_correction_rank
            < config.boolean_vocab_correction_rank
        )
        if (
            source.boolean_vocab_correction_rank
            != config.boolean_vocab_correction_rank
            and not vocab_rank_expansion
        ):
            mismatches.append("boolean_vocab_correction_rank")
        if mismatches:
            raise ValueError("resume topology mismatch: " + ", ".join(mismatches))
        if vocab_rank_expansion or ffn_block_conversion:
            source_state = resumed["model_state_dict"]
            target_state = model.state_dict()
            for name, value in source_state.items():
                if (
                    name in target_state
                    and target_state[name].shape == value.shape
                    and not (
                        ffn_block_conversion
                        and name.endswith("boolean_ffn.correction_block_mask")
                    )
                ):
                    target_state[name] = value
            model.load_state_dict(target_state)
            if vocab_rank_expansion:
                model.readout.initialize_correction_from_factorized(
                    source_state["readout.correction_input_weight"],
                    source_state["readout.correction_output_weight"],
                )
            if ffn_block_conversion:
                for layer_index, layer in enumerate(model.layers):
                    if layer.boolean_ffn is None:
                        raise ValueError("FFN block conversion requires Boolean FFNs")
                    weight = source_state[
                        f"layers.{layer_index}.boolean_ffn.correction_weight"
                    ]
                    layer.boolean_ffn.initialize_correction_block_mask_from_dense(
                        weight
                    )
        else:
            model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if vocab_rank_expansion:
            expanded_parameters = {
                model.readout.correction_input_weight,
                model.readout.correction_output_weight,
            }
            for parameter, state in optimizer.state.items():
                if parameter not in expanded_parameters:
                    continue
                for name, value in tuple(state.items()):
                    if not isinstance(value, Tensor) or value.ndim == 0:
                        continue
                    if value.shape == parameter.shape:
                        continue
                    expanded = torch.zeros_like(parameter, device=value.device)
                    overlap = tuple(
                        slice(0, min(source, target))
                        for source, target in zip(
                            value.shape, parameter.shape, strict=True
                        )
                    )
                    expanded[overlap] = value[overlap]
                    state[name] = expanded
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, Tensor):
                    state[name] = value.to(device)
        update = int(resumed["update"])
        decoded_bytes_processed = int(resumed["decoded_bytes_processed"])
        tokens_processed = int(resumed["tokens_processed"])
        last_validation = (
            {} if cross_corpus_resume else dict(resumed.get("validation", {}))
        )
        best_bpb = float(last_validation.get("bpb", math.inf))
        best_update = update if math.isfinite(best_bpb) else 0
        prior_best_path = output / "best.pt"
        if prior_best_path.exists() and not cross_corpus_resume:
            prior_best = torch.load(
                prior_best_path, map_location="cpu", weights_only=False
            )
            prior_validation = prior_best.get("validation", {})
            prior_bpb = float(prior_validation.get("bpb", math.inf))
            if prior_bpb < best_bpb:
                best_bpb = prior_bpb
                best_update = int(prior_best.get("update", 0))
        if "rng_state" in resumed:
            torch.set_rng_state(resumed["rng_state"])
    elif config.initialize_from:
        teacher, initialized = _initialize_from_dense_checkpoint(
            model, config, corpus.manifest, device
        )
        # Deployment bundles intentionally omit optimizer/RNG state and may
        # omit counters other than decoded bytes. They remain valid warm-start
        # scaffolds; use a fresh optimizer and preserve available exposure.
        update = int(initialized.get("update", 0))
        decoded_bytes_processed = int(initialized.get("decoded_bytes_processed", 0))
        tokens_processed = int(initialized.get("tokens_processed", 0))
        if config.projection_distill_updates:
            distillation = _distill_attention_outputs(
                model,
                teacher,
                train_loader.dataset,
                config,
                device,
                update,
            )
            (output / "projection_distillation.json").write_text(
                json.dumps(distillation, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"projection_distillation": distillation}), flush=True)
        del teacher
        optimizer = _optimizer(model, config)
        (output / "isa_manifest.json").write_text(
            json.dumps(model.isa_manifest(), indent=2) + "\n", encoding="utf-8"
        )

    start_update = update
    start_bytes = decoded_bytes_processed
    run_source_bytes = (
        start_bytes
        if config.run_source_decoded_bytes is None
        else config.run_source_decoded_bytes
    )
    if decoded_bytes_processed < run_source_bytes:
        raise ValueError(
            "checkpoint decoded-byte counter precedes the frozen run source"
        )
    run_bytes_before_attempt = decoded_bytes_processed - run_source_bytes
    if run_bytes_before_attempt > config.target_decoded_bytes:
        raise ValueError("checkpoint exceeds the frozen run byte target")
    next_report = (
        run_source_bytes
        + (
            run_bytes_before_attempt // config.report_every_decoded_bytes + 1
        )
        * config.report_every_decoded_bytes
    )
    metrics_path = output / "metrics.jsonl"
    interval_nll = 0.0
    interval_objective = 0.0
    interval_tokens = 0
    interval_bytes = 0
    interval_domain_bytes: dict[str, int] = defaultdict(int)
    report_count = 0
    started = time.perf_counter()

    while (
        update < config.maximum_updates
        and decoded_bytes_processed - run_source_bytes < config.target_decoded_bytes
    ):
        remaining_run_bytes = (
            config.target_decoded_bytes
            - (decoded_bytes_processed - run_source_bytes)
        )
        remaining_report_bytes = next_report - decoded_bytes_processed
        if config.training_sampling == "byte_proportional":
            update_byte_budget = min(
                config.training_decoded_bytes_per_update,
                remaining_run_bytes,
                remaining_report_bytes,
            )
            (
                (batch_tokens, batch_targets, batch_lengths),
                batch_domain_bytes,
                effective_batch,
            ) = _deterministic_byte_batch(
                train_loader.dataset,
                update_byte_budget,
                config.seed,
                update,
            )
            for domain, count in batch_domain_bytes.items():
                interval_domain_bytes[domain] += count
        else:
            effective_batch = config.batch_size * config.gradient_accumulation_steps
            batch_tokens, batch_targets, batch_lengths = _deterministic_batch(
                train_loader.dataset,
                effective_batch,
                config.seed,
                update,
                config.training_sampling,
                config.training_region_tokens,
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if config.training_sampling == "byte_proportional":
            micro_batch_size = config.training_micro_batch_size or len(batch_tokens)
            accumulation_steps = math.ceil(len(batch_tokens) / micro_batch_size)
            total_batch_targets = int((batch_lengths > 0).sum())
        else:
            micro_batch_size = config.batch_size
            accumulation_steps = config.gradient_accumulation_steps
            total_batch_targets = 0
        for accumulation_index in range(accumulation_steps):
            if config.training_sampling == "byte_proportional":
                begin = accumulation_index * micro_batch_size
                end = min(begin + micro_batch_size, len(batch_tokens))
            else:
                begin = accumulation_index * config.batch_size
                end = begin + config.batch_size
            tokens = batch_tokens[begin:end].to(device)
            targets = batch_targets[begin:end].to(device)
            target_lengths = batch_lengths[begin:end].to(device)
            logits = model(tokens)
            losses = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            )
            valid = target_lengths.flatten() > 0
            if not bool(valid.any()):
                raise ValueError("training batch contains no supervised targets")
            nll = losses[valid].mean()
            balance, gate = model.boolean_regularization()
            objective = (
                nll
                + config.boolean_balance_weight * balance
                + config.boolean_gate_weight * gate
            )
            count = int(valid.sum())
            objective_scale = (
                count / total_batch_targets
                if config.training_sampling == "byte_proportional"
                else 1.0 / accumulation_steps
            )
            (objective * objective_scale).backward()
            byte_count = int(target_lengths.sum())
            interval_nll += float(nll.detach()) * count
            interval_objective += float(objective.detach()) * count
            interval_tokens += count
            interval_bytes += byte_count
            tokens_processed += count
            decoded_bytes_processed += byte_count
        if config.lr_decay_bytes:
            # Set from the config's base rates each update, so a resumed optimizer's
            # stored learning rate cannot leak into the schedule.
            scale = learning_rate_scale(
                config, decoded_bytes_processed - run_source_bytes
            )
            for group in optimizer.param_groups:
                base = (
                    config.embedding_learning_rate
                    if group.get("name") == "embedding"
                    else config.learning_rate
                )
                group["lr"] = base * scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        update += 1

        should_report = (
            decoded_bytes_processed >= next_report
            or decoded_bytes_processed - run_source_bytes >= config.target_decoded_bytes
            or update == config.maximum_updates
        )
        if not should_report:
            continue
        validation = evaluate_document_domains(
            model,
            validation_loader.dataset,
            device,
            config.batch_size,
            config.validation_batches,
            corpus.manifest.get("target_domain_shares"),
        )
        if not validation:
            validation = evaluate(
                model, validation_loader, device, config.validation_batches
            )
        last_validation = validation
        record: dict[str, Any] = {
            "update": update,
            "run_decoded_bytes_processed": decoded_bytes_processed - run_source_bytes,
            "attempt_decoded_bytes_processed": decoded_bytes_processed - start_bytes,
            "decoded_bytes_processed": decoded_bytes_processed,
            "tokens_processed": tokens_processed,
            "effective_batch_size": effective_batch,
            "train_loss_per_token": interval_nll / max(interval_tokens, 1),
            "train_bpb": interval_nll / max(interval_bytes, 1) / math.log(2.0),
            "train_objective_per_token": interval_objective / max(interval_tokens, 1),
            "validation": validation,
            "learning_rate_scale": learning_rate_scale(
                config, decoded_bytes_processed - run_source_bytes
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if interval_domain_bytes:
            record["train_domain_bytes"] = dict(
                sorted(interval_domain_bytes.items())
            )
        boolean_diagnostics = model.boolean_diagnostics()
        if boolean_diagnostics is not None:
            record["boolean"] = boolean_diagnostics
        if config.sample_every_report:
            separator = (
                f"\n{END_OF_TEXT}\n"
                if vocabulary.manifest.get("end_of_text_id") is not None
                else ""
            )
            prompt = f"{separator}{config.sample_prompt}".encode("utf-8")
            sample = generate(
                model,
                vocabulary,
                prompt,
                max_new_bytes=config.sample_decoded_bytes,
                device=device,
                temperature=config.sample_temperature,
                top_k=config.sample_top_k,
                top_p=None,
                seed=config.sample_seed,
            )
            record["sample"] = {
                "prompt": config.sample_prompt,
                "method": f"temperature_{config.sample_temperature}_topk{config.sample_top_k}",
                "seed": config.sample_seed,
                "text": sample.data.decode("utf-8", errors="replace"),
                "generated_tokens": sample.token_count,
                "generated_bytes": len(sample.data),
                "tokens_per_second": sample.tokens_per_second,
                "stopped_at_end_of_text": sample.stopped_at_end_of_text,
                "metrics": generation_metrics(prompt, sample.data, sample.seconds),
            }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        report_count += 1
        interval_nll = interval_objective = 0.0
        interval_tokens = interval_bytes = 0
        interval_domain_bytes.clear()
        while next_report <= decoded_bytes_processed:
            next_report += config.report_every_decoded_bytes

        payload = _checkpoint(
            model,
            optimizer,
            config,
            update=update,
            decoded_bytes_processed=decoded_bytes_processed,
            tokens_processed=tokens_processed,
            validation=validation,
        )
        if float(validation["bpb"]) < best_bpb:
            best_bpb = float(validation["bpb"])
            best_update = update
            torch.save(payload, output / "best.pt")
        if report_count % config.save_every_reports == 0:
            torch.save(payload, output / "last.pt")
        if config.save_gate_checkpoints:
            gate_dir = output / "gate_checkpoints"
            gate_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                payload,
                gate_dir / f"gate_{report_count:04d}_{decoded_bytes_processed}.pt",
            )

    capacity = capacity_report(model)
    summary = {
        "best_validation_bpb": best_bpb,
        "best_update": best_update,
        "final_validation": last_validation,
        "start_update": start_update,
        "final_update": update,
        "run_decoded_bytes_processed": decoded_bytes_processed - run_source_bytes,
        "attempt_decoded_bytes_processed": decoded_bytes_processed - start_bytes,
        "decoded_bytes_processed": decoded_bytes_processed,
        "tokens_processed": tokens_processed,
        "parameter_count": capacity["total_scalar_parameters"],
        "training_parameter_count": capacity["training_scalar_parameters"],
        "capacity": capacity,
        "logical_vocabulary_size": vocabulary.logical_size,
        "padded_vocabulary_size": vocabulary.padded_size,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
