from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

END_OF_TEXT = "<|endoftext|>"


@dataclass(frozen=True, slots=True)
class CachedCorpus:
    """Raw train/validation byte corpus used while constructing the tokenizer."""

    directory: Path
    manifest: dict[str, Any]
    train: bytes
    validation: bytes

    @classmethod
    def load(cls, directory: str | Path, *, verify: bool = True) -> "CachedCorpus":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        train = (directory / "train.bin").read_bytes()
        validation = (directory / "validation.bin").read_bytes()
        if verify:
            for split, data in (("train", train), ("validation", validation)):
                if len(data) != manifest[f"{split}_bytes"]:
                    raise ValueError(f"{split} byte count does not match manifest")
                if hashlib.sha256(data).hexdigest() != manifest[f"{split}_sha256"]:
                    raise ValueError(f"{split} hash does not match manifest")
        return cls(directory, manifest, train, validation)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte alphabet used by the tokenizers ByteLevel component."""
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters), strict=True))


def _token_bytes(tokenizer: Tokenizer) -> list[bytes]:
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    logical_size = tokenizer.get_vocab_size(with_added_tokens=True)
    reverse = {character: value for value, character in _bytes_to_unicode().items()}
    by_id = [b""] * logical_size
    for token, token_id in vocabulary.items():
        if token == END_OF_TEXT:
            data = END_OF_TEXT.encode("utf-8")
        else:
            try:
                data = bytes(reverse[character] for character in token)
            except KeyError as error:
                raise ValueError(f"non-byte-level vocabulary token: {token!r}") from error
        by_id[token_id] = data
    if any(not value for value in by_id):
        missing = [index for index, value in enumerate(by_id) if not value]
        raise ValueError(f"vocabulary contains empty or missing token rows: {missing[:8]}")
    return by_id


def _story_iterator(data: bytes) -> Iterable[str]:
    separator_bytes = f"\n{END_OF_TEXT}\n".encode("utf-8")
    separator_text = separator_bytes.decode("utf-8")
    cursor = 0
    while cursor < len(data):
        boundary = data.find(separator_bytes, cursor)
        if boundary < 0:
            if cursor < len(data):
                yield data[cursor:].decode("utf-8")
            break
        if boundary > cursor:
            yield data[cursor:boundary].decode("utf-8")
        yield separator_text
        cursor = boundary + len(separator_bytes)


def _file_story_iterator(
    path: Path, *, maximum_source_bytes: int | None = None
) -> Iterable[str]:
    """Stream complete story records without retaining the corpus in memory."""
    separator = f"\n{END_OF_TEXT}\n".encode("utf-8")
    buffer = bytearray()
    emitted = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            buffer.extend(chunk)
            while True:
                boundary = buffer.find(separator)
                if boundary < 0:
                    break
                end = boundary + len(separator)
                record = bytes(buffer[:end])
                del buffer[:end]
                yield record.decode("utf-8")
                emitted += len(record)
                if maximum_source_bytes is not None and emitted >= maximum_source_bytes:
                    return
    if buffer:
        yield bytes(buffer).decode("utf-8")


def _encode_stream(tokenizer: Tokenizer, data: bytes) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for text in _story_iterator(data):
        ids = tokenizer.encode(text).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError("tokenization chunk is not exactly reversible")
        chunks.append(np.asarray(ids, dtype=np.uint16))
    if not chunks:
        return np.empty(0, dtype=np.uint16)
    return np.concatenate(chunks)


def _encode_file(
    tokenizer: Tokenizer,
    source_path: Path,
    output_path: Path,
    *,
    logical_size: int,
) -> tuple[int, int, np.ndarray]:
    token_count = 0
    decoded_bytes = 0
    counts = np.zeros(logical_size, dtype=np.int64)
    with output_path.open("wb") as output:
        for text in _file_story_iterator(source_path):
            ids = tokenizer.encode(text).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != text:
                raise ValueError("streaming tokenization is not exactly reversible")
            values = np.asarray(ids, dtype=np.uint16)
            values.tofile(output)
            token_count += len(values)
            decoded_bytes += len(text.encode("utf-8"))
            counts += np.bincount(values, minlength=logical_size)
    if decoded_bytes != source_path.stat().st_size:
        raise ValueError("streaming tokenizer did not consume every source byte")
    return token_count, decoded_bytes, counts


def _distribution(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _vocabulary_audit(
    token_bytes: list[bytes], train_ids: np.ndarray, validation_ids: np.ndarray
) -> dict[str, Any]:
    train_counts = np.bincount(train_ids, minlength=len(token_bytes))
    validation_counts = np.bincount(validation_ids, minlength=len(token_bytes))
    byte_lengths = np.asarray([len(value) for value in token_bytes], dtype=np.int64)
    decoded = [value.decode("utf-8", errors="ignore") for value in token_bytes]
    word_counts = np.asarray(
        [len(re.findall(r"[A-Za-z]+", value)) for value in decoded], dtype=np.int64
    )
    punctuation = np.asarray(
        [bool(re.search(r"[^A-Za-z\s]", value)) for value in decoded], dtype=np.bool_
    )
    return {
        "token_byte_length": _distribution(byte_lengths),
        "train_bytes_per_token": float(
            (byte_lengths * train_counts).sum() / max(train_counts.sum(), 1)
        ),
        "validation_bytes_per_token": float(
            (byte_lengths * validation_counts).sum() / max(validation_counts.sum(), 1)
        ),
        "active_train_token_fraction": float((train_counts > 0).mean()),
        "active_validation_token_fraction": float((validation_counts > 0).mean()),
        "train_tokens_with_fewer_than_100_uses": int(
            ((train_counts > 0) & (train_counts < 100)).sum()
        ),
        "single_byte_token_fraction": float((byte_lengths == 1).mean()),
        "one_word_token_fraction": float((word_counts == 1).mean()),
        "multiword_token_fraction": float((word_counts > 1).mean()),
        "punctuation_token_fraction": float(punctuation.mean()),
    }


def _vocabulary_audit_from_counts(
    token_bytes: list[bytes], train_counts: np.ndarray, validation_counts: np.ndarray
) -> dict[str, Any]:
    byte_lengths = np.asarray([len(value) for value in token_bytes], dtype=np.int64)
    decoded = [value.decode("utf-8", errors="ignore") for value in token_bytes]
    word_counts = np.asarray(
        [len(re.findall(r"[A-Za-z]+", value)) for value in decoded], dtype=np.int64
    )
    punctuation = np.asarray(
        [bool(re.search(r"[^A-Za-z\s]", value)) for value in decoded], dtype=np.bool_
    )
    return {
        "token_byte_length": _distribution(byte_lengths),
        "train_bytes_per_token": float(
            (byte_lengths * train_counts).sum() / max(train_counts.sum(), 1)
        ),
        "validation_bytes_per_token": float(
            (byte_lengths * validation_counts).sum()
            / max(validation_counts.sum(), 1)
        ),
        "active_train_token_fraction": float((train_counts > 0).mean()),
        "active_validation_token_fraction": float((validation_counts > 0).mean()),
        "train_tokens_with_fewer_than_100_uses": int(
            ((train_counts > 0) & (train_counts < 100)).sum()
        ),
        "single_byte_token_fraction": float((byte_lengths == 1).mean()),
        "one_word_token_fraction": float((word_counts == 1).mean()),
        "multiword_token_fraction": float((word_counts > 1).mean()),
        "punctuation_token_fraction": float(punctuation.mean()),
    }


def prepare_bpe_corpus_streaming(
    source_corpus_dir: str | Path,
    output_dir: str | Path,
    *,
    vocabulary_size: int = 4096,
    padding_multiple: int = 128,
    minimum_frequency: int = 2,
    vocabulary_training_bytes: int = 200_000_000,
) -> dict[str, Any]:
    """Prepare a large corpus with bounded-memory vocabulary induction/encoding."""
    if vocabulary_size < 512:
        raise ValueError("vocabulary_size must leave room beyond the byte alphabet")
    if padding_multiple <= 0 or vocabulary_training_bytes <= 0:
        raise ValueError("padding_multiple and vocabulary_training_bytes must be positive")
    source_dir = Path(source_corpus_dir)
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    train_source = source_dir / "train.bin"
    validation_source = source_dir / "validation.bin"
    if train_source.stat().st_size != int(source_manifest["train_bytes"]):
        raise ValueError("training source size does not match its manifest")
    if validation_source.stat().st_size != int(source_manifest["validation_bytes"]):
        raise ValueError("validation source size does not match its manifest")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(
        [AddedToken(END_OF_TEXT, single_word=False, normalized=False, special=True)]
    )
    trainer = BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=minimum_frequency,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[END_OF_TEXT],
    )
    tokenizer.train_from_iterator(
        _file_story_iterator(
            train_source, maximum_source_bytes=vocabulary_training_bytes
        ),
        trainer=trainer,
    )
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    token_bytes = _token_bytes(tokenizer)
    logical_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if logical_size > np.iinfo(np.uint16).max:
        raise ValueError("uint16 token cache supports at most 65,535 vocabulary rows")

    train_path = output / "train.tokens"
    validation_path = output / "validation.tokens"
    train_tokens, train_bytes, train_counts = _encode_file(
        tokenizer, train_source, train_path, logical_size=logical_size
    )
    validation_tokens, validation_bytes, validation_counts = _encode_file(
        tokenizer, validation_source, validation_path, logical_size=logical_size
    )
    lengths = np.asarray([len(value) for value in token_bytes], dtype=np.uint16)
    if int(lengths.astype(np.int64) @ train_counts) != train_bytes:
        raise ValueError("training token byte lengths do not reconstruct source")
    if int(lengths.astype(np.int64) @ validation_counts) != validation_bytes:
        raise ValueError("validation token byte lengths do not reconstruct source")
    lengths_path = output / "token_byte_lengths.npy"
    np.save(lengths_path, lengths)
    (output / "token_bytes.json").write_text(
        json.dumps([value.hex() for value in token_bytes]) + "\n", encoding="utf-8"
    )
    padded_size = math.ceil(logical_size / padding_multiple) * padding_multiple
    manifest: dict[str, Any] = {
        "format_version": 1,
        "unit": "token",
        "tokenizer_kind": "byte_level_bpe",
        "tokenizers_version": __import__("tokenizers").__version__,
        "logical_vocabulary_size": logical_size,
        "padded_vocabulary_size": padded_size,
        "vocabulary_padding_multiple": padding_multiple,
        "end_of_text": END_OF_TEXT,
        "end_of_text_id": tokenizer.token_to_id(END_OF_TEXT),
        "normalization": "none",
        "byte_fallback": True,
        "streaming_preparation": True,
        "vocabulary_training_bytes_requested": vocabulary_training_bytes,
        "source_corpus_dir": str(source_dir.resolve()),
        "source_train_sha256": source_manifest["train_sha256"],
        "source_validation_sha256": source_manifest["validation_sha256"],
        "train_bytes": train_bytes,
        "validation_bytes": validation_bytes,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_token_sha256": _sha256_file(train_path),
        "validation_token_sha256": _sha256_file(validation_path),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "token_byte_lengths_sha256": _sha256_file(lengths_path),
        "audit": _vocabulary_audit_from_counts(
            token_bytes, train_counts, validation_counts
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_bpe_prefix_corpus(
    source_corpus_dir: str | Path,
    source_tokenizer_dir: str | Path,
    output_dir: str | Path,
    *,
    vocabulary_size: int,
    padding_multiple: int = 128,
) -> dict[str, Any]:
    """Derive an exact smaller BPE by truncating a trained merge sequence."""
    if vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be positive")
    if padding_multiple <= 0:
        raise ValueError("padding_multiple must be positive")
    source_dir = Path(source_corpus_dir)
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    train_source = source_dir / "train.bin"
    validation_source = source_dir / "validation.bin"
    if train_source.stat().st_size != int(source_manifest["train_bytes"]):
        raise ValueError("training source size does not match its manifest")
    if validation_source.stat().st_size != int(source_manifest["validation_bytes"]):
        raise ValueError("validation source size does not match its manifest")

    source_tokenizer_path = Path(source_tokenizer_dir) / "tokenizer.json"
    specification = json.loads(source_tokenizer_path.read_text(encoding="utf-8"))
    model_specification = specification.get("model") or {}
    vocabulary = model_specification.get("vocab") or {}
    merges = model_specification.get("merges") or []
    source_size = len(vocabulary)
    base_rows = source_size - len(merges)
    if not base_rows <= vocabulary_size <= source_size:
        raise ValueError(
            f"vocabulary_size must be in [{base_rows}, {source_size}] for this merge prefix"
        )
    retained = {
        token: int(token_id)
        for token, token_id in vocabulary.items()
        if int(token_id) < vocabulary_size
    }
    if sorted(retained.values()) != list(range(vocabulary_size)):
        raise ValueError("source tokenizer IDs are not a contiguous merge sequence")
    model_specification["vocab"] = retained
    model_specification["merges"] = merges[: vocabulary_size - base_rows]
    tokenizer = Tokenizer.from_str(json.dumps(specification))
    if tokenizer.get_vocab_size(with_added_tokens=True) != vocabulary_size:
        raise ValueError("derived tokenizer has the wrong vocabulary size")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    token_bytes = _token_bytes(tokenizer)
    logical_size = tokenizer.get_vocab_size(with_added_tokens=True)
    train_path = output / "train.tokens"
    validation_path = output / "validation.tokens"
    train_tokens, train_bytes, train_counts = _encode_file(
        tokenizer, train_source, train_path, logical_size=logical_size
    )
    validation_tokens, validation_bytes, validation_counts = _encode_file(
        tokenizer, validation_source, validation_path, logical_size=logical_size
    )
    lengths = np.asarray([len(value) for value in token_bytes], dtype=np.uint16)
    if int(lengths.astype(np.int64) @ train_counts) != train_bytes:
        raise ValueError("training token byte lengths do not reconstruct source")
    if int(lengths.astype(np.int64) @ validation_counts) != validation_bytes:
        raise ValueError("validation token byte lengths do not reconstruct source")
    lengths_path = output / "token_byte_lengths.npy"
    np.save(lengths_path, lengths)
    (output / "token_bytes.json").write_text(
        json.dumps([value.hex() for value in token_bytes]) + "\n", encoding="utf-8"
    )
    padded_size = math.ceil(logical_size / padding_multiple) * padding_multiple
    source_tokenizer_manifest_path = Path(source_tokenizer_dir) / "manifest.json"
    manifest: dict[str, Any] = {
        "format_version": 1,
        "unit": "token",
        "tokenizer_kind": "byte_level_bpe_merge_prefix",
        "tokenizers_version": __import__("tokenizers").__version__,
        "logical_vocabulary_size": logical_size,
        "padded_vocabulary_size": padded_size,
        "vocabulary_padding_multiple": padding_multiple,
        "end_of_text": END_OF_TEXT,
        "end_of_text_id": tokenizer.token_to_id(END_OF_TEXT),
        "normalization": "none",
        "byte_fallback": True,
        "streaming_preparation": True,
        "source_tokenizer_dir": str(Path(source_tokenizer_dir).resolve()),
        "source_tokenizer_sha256": _sha256_file(source_tokenizer_path),
        "source_tokenizer_manifest_sha256": _sha256_file(
            source_tokenizer_manifest_path
        ),
        "source_tokenizer_logical_vocabulary_size": source_size,
        "retained_merge_count": vocabulary_size - base_rows,
        "source_corpus_dir": str(source_dir.resolve()),
        "source_train_sha256": source_manifest["train_sha256"],
        "source_validation_sha256": source_manifest["validation_sha256"],
        "train_bytes": train_bytes,
        "validation_bytes": validation_bytes,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_token_sha256": _sha256_file(train_path),
        "validation_token_sha256": _sha256_file(validation_path),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "token_byte_lengths_sha256": _sha256_file(lengths_path),
        "audit": _vocabulary_audit_from_counts(
            token_bytes, train_counts, validation_counts
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_bpe_corpus(
    source_corpus_dir: str | Path,
    output_dir: str | Path,
    *,
    vocabulary_size: int = 4096,
    padding_multiple: int = 128,
    minimum_frequency: int = 2,
) -> dict[str, Any]:
    if vocabulary_size < 512:
        raise ValueError("vocabulary_size must leave room beyond the byte alphabet")
    if padding_multiple <= 0:
        raise ValueError("padding_multiple must be positive")
    source = CachedCorpus.load(source_corpus_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(
        [AddedToken(END_OF_TEXT, single_word=False, normalized=False, special=True)]
    )
    trainer = BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=minimum_frequency,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[END_OF_TEXT],
    )
    tokenizer.train_from_iterator(_story_iterator(source.train), trainer=trainer)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    train_ids = _encode_stream(tokenizer, source.train)
    validation_ids = _encode_stream(tokenizer, source.validation)
    logical_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if logical_size > np.iinfo(np.uint16).max:
        raise ValueError("uint16 token cache supports at most 65,535 vocabulary rows")
    if train_ids.max(initial=0) >= logical_size or validation_ids.max(initial=0) >= logical_size:
        raise AssertionError("token cache contains an out-of-vocabulary id")

    token_bytes = _token_bytes(tokenizer)
    lengths = np.asarray([len(value) for value in token_bytes], dtype=np.uint16)
    if int(lengths[train_ids].sum()) != len(source.train):
        raise ValueError("training token byte lengths do not reconstruct source byte count")
    if int(lengths[validation_ids].sum()) != len(source.validation):
        raise ValueError("validation token byte lengths do not reconstruct source byte count")
    train_path = output / "train.tokens"
    validation_path = output / "validation.tokens"
    lengths_path = output / "token_byte_lengths.npy"
    train_ids.tofile(train_path)
    validation_ids.tofile(validation_path)
    np.save(lengths_path, lengths)
    (output / "token_bytes.json").write_text(
        json.dumps([value.hex() for value in token_bytes]) + "\n", encoding="utf-8"
    )
    padded_size = math.ceil(logical_size / padding_multiple) * padding_multiple
    manifest: dict[str, Any] = {
        "format_version": 1,
        "unit": "token",
        "tokenizer_kind": "byte_level_bpe",
        "tokenizers_version": __import__("tokenizers").__version__,
        "logical_vocabulary_size": logical_size,
        "padded_vocabulary_size": padded_size,
        "vocabulary_padding_multiple": padding_multiple,
        "end_of_text": END_OF_TEXT,
        "end_of_text_id": tokenizer.token_to_id(END_OF_TEXT),
        "normalization": "none",
        "byte_fallback": True,
        "source_corpus_dir": str(Path(source_corpus_dir).resolve()),
        "source_train_sha256": source.manifest["train_sha256"],
        "source_validation_sha256": source.manifest["validation_sha256"],
        "train_bytes": len(source.train),
        "validation_bytes": len(source.validation),
        "train_tokens": int(len(train_ids)),
        "validation_tokens": int(len(validation_ids)),
        "train_token_sha256": _sha256_file(train_path),
        "validation_token_sha256": _sha256_file(validation_path),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "token_byte_lengths_sha256": _sha256_file(lengths_path),
        "audit": _vocabulary_audit(token_bytes, train_ids, validation_ids),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True, slots=True)
class Vocabulary:
    directory: Path
    tokenizer: Tokenizer
    manifest: dict[str, Any]
    token_bytes: tuple[bytes, ...]

    @classmethod
    def load(cls, directory: str | Path) -> "Vocabulary | ByteVocabulary":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("tokenizer_kind") == BYTE_TOKENIZER_KIND:
            return ByteVocabulary(directory, manifest)
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        values = json.loads((directory / "token_bytes.json").read_text(encoding="utf-8"))
        token_bytes = tuple(bytes.fromhex(value) for value in values)
        if len(token_bytes) != manifest["logical_vocabulary_size"]:
            raise ValueError("token byte table does not match logical vocabulary size")
        return cls(directory, tokenizer, manifest, token_bytes)

    @property
    def logical_size(self) -> int:
        return int(self.manifest["logical_vocabulary_size"])

    @property
    def padded_size(self) -> int:
        return int(self.manifest["padded_vocabulary_size"])

    def encode(self, data: bytes | str) -> list[int]:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: Iterable[int]) -> bytes:
        return self.tokenizer.decode(
            list(token_ids), skip_special_tokens=False
        ).encode("utf-8")


BYTE_TOKENIZER_KIND = "byte_identity"
_BYTE_TOKENS = tuple(bytes((value,)) for value in range(256))


@dataclass(frozen=True, slots=True)
class ByteVocabulary:
    """Identity vocabulary: token id ``i`` is the single byte ``i``.

    There is no end-of-text token; a byte corpus is one continuous stream.
    """

    directory: Path
    manifest: dict[str, Any]

    @property
    def token_bytes(self) -> tuple[bytes, ...]:
        return _BYTE_TOKENS

    @property
    def logical_size(self) -> int:
        return int(self.manifest["logical_vocabulary_size"])

    @property
    def padded_size(self) -> int:
        return int(self.manifest["padded_vocabulary_size"])

    def encode(self, data: bytes | str) -> list[int]:
        return list(data.encode("utf-8") if isinstance(data, str) else data)

    def decode(self, token_ids: Iterable[int]) -> bytes:
        return bytes(token_ids)


def _copy_as_byte_tokens(source: Any, count: int, path: Path, source_digest: Any) -> str:
    """Widen ``count`` source bytes to uint16 token ids and return the file hash."""
    digest = hashlib.sha256()
    remaining = count
    with path.open("wb") as handle:
        while remaining:
            chunk = source.read(min(remaining, 64 << 20))
            if not chunk:
                raise ValueError("source ended before the requested split size")
            source_digest.update(chunk)
            widened = np.frombuffer(chunk, dtype=np.uint8).astype(np.uint16).tobytes()
            digest.update(widened)
            handle.write(widened)
            remaining -= len(chunk)
    return digest.hexdigest()


def prepare_byte_corpus(
    source_file: str | Path,
    output_dir: str | Path,
    *,
    validation_bytes: int,
    test_bytes: int,
    padding_multiple: int = 128,
) -> dict[str, Any]:
    """Cache a raw byte stream as identity tokens with contiguous splits.

    Train comes first and test last, so held-out text is never seen in
    training and every split keeps its original byte order.
    """
    if validation_bytes <= 0 or test_bytes <= 0 or padding_multiple <= 0:
        raise ValueError("split sizes and padding_multiple must be positive")
    source_path = Path(source_file)
    total_bytes = source_path.stat().st_size
    train_bytes = total_bytes - validation_bytes - test_bytes
    if train_bytes <= 0:
        raise ValueError("held-out splits leave no training bytes")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_bytes, "validation": validation_bytes, "test": test_bytes}
    source_digest = hashlib.sha256()
    token_digests: dict[str, str] = {}
    with source_path.open("rb") as source:
        for split, count in splits.items():
            token_digests[split] = _copy_as_byte_tokens(
                source, count, output / f"{split}.tokens", source_digest
            )
    lengths_path = output / "token_byte_lengths.npy"
    np.save(lengths_path, np.ones(256, dtype=np.uint16))
    (output / "token_bytes.json").write_text(
        json.dumps([value.hex() for value in _BYTE_TOKENS]) + "\n", encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "unit": "token",
        "tokenizer_kind": BYTE_TOKENIZER_KIND,
        "logical_vocabulary_size": 256,
        "padded_vocabulary_size": math.ceil(256 / padding_multiple) * padding_multiple,
        "vocabulary_padding_multiple": padding_multiple,
        "end_of_text": None,
        "end_of_text_id": None,
        "normalization": "none",
        "source_file": str(source_path.resolve()),
        "source_bytes": total_bytes,
        "source_sha256": source_digest.hexdigest(),
        "token_byte_lengths_sha256": _sha256_file(lengths_path),
    }
    for split, count in splits.items():
        manifest[f"{split}_bytes"] = count
        manifest[f"{split}_tokens"] = count
        manifest[f"{split}_token_sha256"] = token_digests[split]
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
