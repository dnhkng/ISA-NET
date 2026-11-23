from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TokenCorpus:
    directory: Path
    manifest: dict[str, Any]
    train: Tensor
    validation: Tensor
    token_byte_lengths: Tensor
    train_document_offsets: Tensor | None = None
    validation_document_offsets: Tensor | None = None
    train_document_domains: tuple[str, ...] | None = None
    validation_document_domains: tuple[str, ...] | None = None

    @classmethod
    def load(cls, directory: str | Path, *, verify: bool = True) -> "TokenCorpus":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("unit") != "token":
            raise ValueError("expected a token-corpus manifest")
        train_path = directory / "train.tokens"
        validation_path = directory / "validation.tokens"
        lengths_path = directory / "token_byte_lengths.npy"
        train_offsets_path = directory / "train_document_offsets.npy"
        validation_offsets_path = directory / "validation_document_offsets.npy"
        documents_path = directory / "documents.jsonl"
        if train_offsets_path.exists() != validation_offsets_path.exists():
            raise ValueError("train and validation document offsets must be paired")
        if verify:
            expected = {
                train_path: manifest["train_token_sha256"],
                validation_path: manifest["validation_token_sha256"],
                lengths_path: manifest["token_byte_lengths_sha256"],
            }
            for path, digest in expected.items():
                if _sha256_file(path) != digest:
                    raise ValueError(f"hash mismatch: {path.name}")
            optional = {
                train_offsets_path: manifest.get("train_document_offsets_sha256"),
                validation_offsets_path: manifest.get(
                    "validation_document_offsets_sha256"
                ),
            }
            for path, digest in optional.items():
                if path.exists():
                    if digest is None:
                        raise ValueError(f"missing manifest hash: {path.name}")
                    if _sha256_file(path) != digest:
                        raise ValueError(f"hash mismatch: {path.name}")
                elif digest is not None:
                    raise ValueError(f"missing document offsets: {path.name}")
            if documents_path.exists():
                expected_documents = manifest.get("documents_manifest_sha256")
                if expected_documents is None:
                    raise ValueError("missing manifest hash: documents.jsonl")
                if _sha256_file(documents_path) != expected_documents:
                    raise ValueError("hash mismatch: documents.jsonl")
        train_values = np.fromfile(train_path, dtype=np.uint16).astype(np.int64)
        validation_values = np.fromfile(validation_path, dtype=np.uint16).astype(np.int64)
        lengths = np.load(lengths_path).astype(np.int64)
        train_offsets = (
            np.load(train_offsets_path, allow_pickle=False).astype(np.int64)
            if train_offsets_path.exists()
            else None
        )
        validation_offsets = (
            np.load(validation_offsets_path, allow_pickle=False).astype(np.int64)
            if validation_offsets_path.exists()
            else None
        )
        train_domains: tuple[str, ...] | None = None
        validation_domains: tuple[str, ...] | None = None
        if documents_path.exists():
            records = [
                json.loads(line)
                for line in documents_path.read_text(encoding="utf-8").splitlines()
            ]
            train_domains = tuple(
                record["domain"] for record in records if record["split"] == "train"
            )
            validation_domains = tuple(
                record["domain"]
                for record in records
                if record["split"] == "validation"
            )
        if len(train_values) != manifest["train_tokens"]:
            raise ValueError("training token count does not match manifest")
        if len(validation_values) != manifest["validation_tokens"]:
            raise ValueError("validation token count does not match manifest")
        if len(lengths) != manifest["logical_vocabulary_size"]:
            raise ValueError("token-byte length table does not match vocabulary")
        for split, values, offsets in (
            ("train", train_values, train_offsets),
            ("validation", validation_values, validation_offsets),
        ):
            if offsets is None:
                continue
            if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(values):
                raise ValueError(f"invalid {split} document offsets")
            if np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f"non-monotonic {split} document offsets")
        for split, offsets, domains in (
            ("train", train_offsets, train_domains),
            ("validation", validation_offsets, validation_domains),
        ):
            if offsets is not None and domains is None:
                raise ValueError(f"missing {split} document domains")
            if domains is not None and (
                offsets is None or len(domains) != len(offsets) - 1
            ):
                raise ValueError(f"{split} document domains and offsets disagree")
        return cls(
            directory,
            manifest,
            torch.from_numpy(train_values),
            torch.from_numpy(validation_values),
            torch.from_numpy(lengths),
            None if train_offsets is None else torch.from_numpy(train_offsets),
            None
            if validation_offsets is None
            else torch.from_numpy(validation_offsets),
            train_domains,
            validation_domains,
        )


class TokenWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self, values: Tensor, token_byte_lengths: Tensor, context: int, stride: int
    ) -> None:
        if context <= 0 or stride <= 0:
            raise ValueError("context and stride must be positive")
        self.values = values
        self.token_byte_lengths = token_byte_lengths
        self.context = context
        self.stride = stride
        self.window_count = max(0, (len(values) - context - 1) // stride + 1)

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0 or index >= self.window_count:
            raise IndexError(index)
        start = index * self.stride
        window = self.values[start : start + self.context + 1]
        targets = window[1:]
        return window[:-1], targets, self.token_byte_lengths[targets]


class DocumentBlockDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Fixed-shape examples that match independently coded deployment blocks.

    The first token of every block is a literal in the native codec. A block of
    ``n`` tokens therefore supplies ``n - 1`` supervised targets. Short final
    blocks are padded on the right; zero target lengths mask those positions,
    and causal attention ensures that padding cannot affect real predictions.
    No example crosses a document or deployment-block boundary.
    """

    def __init__(
        self,
        values: Tensor,
        token_byte_lengths: Tensor,
        document_offsets: Tensor,
        context: int,
        document_domains: tuple[str, ...] | None = None,
    ) -> None:
        if context <= 0:
            raise ValueError("context must be positive")
        if len(document_offsets) < 2:
            raise ValueError("document offsets must contain a start and end")
        self.values = values
        self.token_byte_lengths = token_byte_lengths
        self.context = context
        starts: list[int] = []
        lengths: list[int] = []
        document_indices: list[int] = []
        offsets = document_offsets.tolist()
        if document_domains is not None and len(document_domains) != len(offsets) - 1:
            raise ValueError("document domains and offsets disagree")
        for document_index, (document_start, document_end) in enumerate(
            zip(offsets, offsets[1:])
        ):
            for start in range(document_start, document_end, context):
                length = min(context, document_end - start)
                if length >= 2:
                    starts.append(start)
                    lengths.append(length)
                    document_indices.append(document_index)
        self.block_starts = starts
        self.block_lengths = lengths
        self.block_document_indices = document_indices
        self.block_domains = (
            None
            if document_domains is None
            else [document_domains[index] for index in document_indices]
        )

    def __len__(self) -> int:
        return len(self.block_starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = self.block_starts[index]
        length = self.block_lengths[index]
        real_targets = length - 1
        tokens = torch.zeros(self.context, dtype=self.values.dtype)
        targets = torch.zeros(self.context, dtype=self.values.dtype)
        target_lengths = torch.zeros(self.context, dtype=torch.int64)
        tokens[:real_targets] = self.values[start : start + real_targets]
        targets[:real_targets] = self.values[start + 1 : start + length]
        target_lengths[:real_targets] = self.token_byte_lengths[
            targets[:real_targets]
        ]
        return tokens, targets, target_lengths


def corpus_loaders(
    corpus: TokenCorpus,
    *,
    context: int,
    stride: int,
    validation_stride: int | None,
    batch_size: int,
    seed: int,
    validation_windows: int | None = None,
    validation_region_tokens: int | None = None,
    validation_regions: int | None = None,
    validation_domain_bytes: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    has_document_offsets = (
        corpus.train_document_offsets is not None
        and corpus.validation_document_offsets is not None
    )
    if has_document_offsets:
        if corpus.manifest.get("tokenizer_kind") != "byte_identity":
            raise ValueError("document-block training currently requires byte tokens")
        if stride != context or validation_stride not in {None, context}:
            raise ValueError(
                "document-block training requires context-aligned train and validation strides"
            )
        train = DocumentBlockDataset(
            corpus.train,
            corpus.token_byte_lengths,
            corpus.train_document_offsets,
            context=context,
            document_domains=corpus.train_document_domains,
        )
        validation = DocumentBlockDataset(
            corpus.validation,
            corpus.token_byte_lengths,
            corpus.validation_document_offsets,
            context=context,
            document_domains=corpus.validation_document_domains,
        )
    else:
        train = TokenWindowDataset(
            corpus.train, corpus.token_byte_lengths, context=context, stride=stride
        )
        validation = TokenWindowDataset(
            corpus.validation,
            corpus.token_byte_lengths,
            context=context,
            stride=validation_stride or context,
        )
    if not len(train) or not len(validation):
        requirement = (
            "at least one document with two tokens"
            if has_document_offsets
            else f"at least {context + 1} tokens"
        )
        raise ValueError(f"each token split must contain {requirement}")
    if validation_windows is not None and validation_windows < len(validation):
        # Evenly spaced across the whole split: a contiguous prefix measures one
        # region's text (enwik9's validation opens with a highly repetitive stretch).
        picks = torch.linspace(0, len(validation) - 1, validation_windows).round().long()
        validation = Subset(validation, picks.unique().tolist())
    if validation_region_tokens is not None or validation_regions is not None:
        if has_document_offsets:
            raise ValueError("shared-region validation is for contiguous token splits")
        if validation_windows is not None:
            raise ValueError("validation windows and shared regions are mutually exclusive")
        if validation_stride not in {None, context}:
            raise ValueError("shared-region validation requires context stride")
        if (
            validation_region_tokens is None
            or validation_regions is None
            or validation_region_tokens <= 0
            or validation_regions <= 0
            or validation_region_tokens % context
        ):
            raise ValueError("invalid shared validation-region geometry")
        windows_per_region = validation_region_tokens // context
        available_regions = len(validation) // windows_per_region
        if validation_regions > available_regions:
            raise ValueError(
                f"requested {validation_regions} validation regions, "
                f"but only {available_regions} are available"
            )
        region_picks = torch.linspace(
            0, available_regions - 1, validation_regions
        ).round().long().unique().tolist()
        if len(region_picks) != validation_regions:
            raise ValueError("shared validation-region selection produced duplicates")
        window_picks = [
            region * windows_per_region + offset
            for region in region_picks
            for offset in range(windows_per_region)
        ]
        validation = Subset(validation, window_picks)
    if validation_domain_bytes is not None:
        if not has_document_offsets or not isinstance(validation, DocumentBlockDataset):
            raise ValueError("domain-byte validation requires document-block data")
        if validation.block_domains is None:
            raise ValueError("domain-byte validation requires document domains")
        if validation_domain_bytes <= 0:
            raise ValueError("validation_domain_bytes must be positive")
        domain_indices: dict[str, list[int]] = {}
        for index, domain in enumerate(validation.block_domains):
            domain_indices.setdefault(domain, []).append(index)
        selected: list[int] = []
        for domain in sorted(domain_indices):
            indices = domain_indices[domain]
            # Spread the fixed slice over the complete domain partition rather
            # than selecting one source/group-heavy prefix.
            desired_blocks = min(
                len(indices),
                max(1, (validation_domain_bytes + context - 2) // (context - 1)),
            )
            positions = (
                torch.linspace(0, len(indices) - 1, desired_blocks)
                .round()
                .long()
                .unique()
                .tolist()
            )
            domain_selected = [indices[position] for position in positions]
            # Short document tails mean the estimate can undershoot. Add more
            # deterministic blocks until the byte target is reached.
            picked = set(domain_selected)
            decoded = sum(validation.block_lengths[index] - 1 for index in picked)
            if decoded < validation_domain_bytes:
                for index in indices:
                    if index in picked:
                        continue
                    picked.add(index)
                    decoded += validation.block_lengths[index] - 1
                    if decoded >= validation_domain_bytes:
                        break
            selected.extend(sorted(picked))
        validation = Subset(validation, sorted(selected))
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(validation, batch_size=batch_size, shuffle=False),
    )
