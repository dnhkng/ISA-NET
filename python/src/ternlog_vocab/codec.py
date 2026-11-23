"""Lossless byte compression driven by a byte-level language model.

Each byte is range-coded against a next-byte distribution quantized to integer
frequencies. The distribution is the model's, optionally mixed with a long-range match
model: a greedy predictor that proposes the byte which followed the most recent earlier
occurrence of the trailing bytes. The model only sees its context window; the match
model reaches back through the whole bounded history, where most repeats in text like
Wikipedia live.

The coder is the byte-oriented 32-bit design used by LZMA (carry propagation through a
cached byte), chosen because its inner loop is a handful of integer operations that
port directly to a native single-core decoder.

Encoder and decoder walk the identical ``prefill``/``decode_step`` path on CPU with one
thread and replay the same integer match model, so both sides compute bit-identical
probabilities. The header carries a model fingerprint, the mixing parameters, and a
CRC of the original data, so a mismatched checkpoint or any numerical drift fails
loudly instead of producing wrong output.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .model import VocabularyTransformerLM

MAGIC = b"TLZ2"
_HEADER = struct.Struct("<4sQI16sI")
_MATCH_HEADER = struct.Struct("<IBB4d4d")
HEADER_BYTES = _HEADER.size + _MATCH_HEADER.size
TOTAL_BITS = 16
_SYMBOLS = 256
_TOP = 1 << 24
_MASK32 = (1 << 32) - 1
MATCH_BUCKETS = (6, 16, 32, 64)
"""Lower edges of the match-length buckets; the last bucket is open-ended."""
_MIX_GRID = np.linspace(0.0, 0.99, 100)


class RangeEncoder:
    def __init__(self) -> None:
        self._low = 0
        self._range = _MASK32
        self._cache = 0
        self._cache_size = 1
        self._output = bytearray()

    def encode(self, start: int, size: int, total: int) -> None:
        if not (0 <= start and 0 < size and start + size <= total <= 1 << TOTAL_BITS):
            raise ValueError("invalid frequency interval")
        scale = self._range // total
        self._low += start * scale
        self._range = size * scale
        while self._range < _TOP:
            self._range <<= 8
            self._shift_low()

    def _shift_low(self) -> None:
        if self._low < 0xFF000000 or self._low >= 1 << 32:
            carry = self._low >> 32
            pending = self._cache
            while True:
                self._output.append((pending + carry) & 0xFF)
                pending = 0xFF
                self._cache_size -= 1
                if self._cache_size == 0:
                    break
            self._cache = (self._low >> 24) & 0xFF
        self._cache_size += 1
        self._low = (self._low << 8) & _MASK32

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self._output)


class RangeDecoder:
    def __init__(self, data: bytes, offset: int = 0) -> None:
        self._data = data
        self._position = offset
        self._range = _MASK32
        self._code = 0
        self._scale = 0
        for _ in range(5):
            self._code = ((self._code << 8) | self._next_byte()) & _MASK32

    def _next_byte(self) -> int:
        if self._position >= len(self._data):
            return 0
        value = self._data[self._position]
        self._position += 1
        return value

    def decode_frequency(self, total: int) -> int:
        self._scale = self._range // total
        return min(self._code // self._scale, total - 1)

    def consume(self, start: int, size: int) -> None:
        self._code -= start * self._scale
        self._range = size * self._scale
        while self._range < _TOP:
            self._code = ((self._code << 8) | self._next_byte()) & _MASK32
            self._range <<= 8


def quantize(probabilities: np.ndarray, total_bits: int = TOTAL_BITS) -> np.ndarray:
    """Integer frequencies with every symbol >= 1 and a sum <= ``2**total_bits``."""
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("distribution contains non-finite values")
    budget = (1 << total_bits) - len(probabilities)
    if budget <= 0:
        raise ValueError("too many symbols for the frequency precision")
    return np.floor(probabilities * budget).astype(np.int64) + 1


class MatchModel:
    """Greedy long-range match predictor.

    Proposes the byte that followed the most recent earlier occurrence of the trailing
    ``context`` bytes, if that occurrence is within ``history`` bytes, and keeps
    following it while predictions stay correct. Integer-only and deterministic, so an
    encoder and decoder replaying the same bytes stay in lockstep. This reference keeps
    the whole stream; a native build keeps a ``history``-byte ring buffer plus a
    ``2**table_bits`` table of 32-bit positions.
    """

    def __init__(self, history: int = 1 << 16, table_bits: int = 16, context: int = 6) -> None:
        if history <= 0 or not 1 <= table_bits <= 30 or context <= 0:
            raise ValueError("invalid match model geometry")
        self.history = history
        self.table_bits = table_bits
        self.context = context
        self._mask = (1 << table_bits) - 1
        self._table = [-1] * (1 << table_bits)
        self._data = bytearray()
        self._pointer = -1
        self._length = 0

    def predict(self) -> tuple[int, int]:
        """The proposed next byte (-1 if none) and the current match length."""
        if self._pointer < 0:
            return -1, 0
        return self._data[self._pointer], self._length

    def update(self, byte: int) -> None:
        if self._pointer >= 0:
            if self._data[self._pointer] == byte:
                self._pointer += 1
                self._length += 1
            else:
                self._pointer, self._length = -1, 0
        self._data.append(byte)
        end = len(self._data)
        if end < self.context:
            return
        key = bytes(self._data[end - self.context : end])
        slot = zlib.crc32(key) & self._mask
        if self._pointer < 0:
            candidate = self._table[slot]
            if (
                candidate >= 0
                and end - candidate <= self.history
                and self._data[candidate - self.context : candidate] == key
            ):
                self._pointer, self._length = candidate, self.context
        self._table[slot] = end


def match_trace(
    data: bytes, history: int = 1 << 16, table_bits: int = 16, context: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Match prediction (-1 if none) and match length in force at every position."""
    matcher = MatchModel(history, table_bits, context)
    predicted = np.full(len(data), -1, dtype=np.int16)
    lengths = np.zeros(len(data), dtype=np.int32)
    for index, byte in enumerate(data):
        predicted[index], lengths[index] = matcher.predict()
        matcher.update(byte)
    return predicted, lengths


def _bucket_members(bucket: int, predicted: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    low = MATCH_BUCKETS[bucket]
    high = MATCH_BUCKETS[bucket + 1] if bucket + 1 < len(MATCH_BUCKETS) else np.iinfo(np.int64).max
    return (predicted >= 0) & (lengths >= low) & (lengths < high)


@dataclass(frozen=True, slots=True)
class MatchMixing:
    """Per-bucket linear mixing of the model with the match prediction.

    For a live match in bucket ``b`` proposing byte ``s``:
    ``p = (1 - w_b) * p_model + w_b * q_b`` with ``q_b(s) = c_b`` and
    ``q_b(other) = (1 - c_b) / 255``.
    """

    weights: tuple[float, float, float, float]
    confidences: tuple[float, float, float, float]
    history: int = 1 << 16
    table_bits: int = 16
    context: int = 6

    def __post_init__(self) -> None:
        if len(self.weights) != len(MATCH_BUCKETS) or len(self.confidences) != len(MATCH_BUCKETS):
            raise ValueError("one weight and one confidence per match bucket")
        if not all(0.0 <= value < 1.0 for value in (*self.weights, *self.confidences)):
            raise ValueError("mixing weights and confidences must be in [0, 1)")

    def mix(self, probabilities: np.ndarray, predicted: int, length: int) -> np.ndarray:
        bucket = -1
        for index, edge in enumerate(MATCH_BUCKETS):
            if length >= edge:
                bucket = index
        if predicted < 0 or bucket < 0 or self.weights[bucket] == 0.0:
            return probabilities
        weight, confidence = self.weights[bucket], self.confidences[bucket]
        other = (1.0 - confidence) / (_SYMBOLS - 1)
        mixed = (1.0 - weight) * probabilities + weight * other
        mixed[predicted] += weight * (confidence - other)
        return mixed

    def probability_of_actual(
        self,
        p_actual: np.ndarray,
        predicted: np.ndarray,
        lengths: np.ndarray,
        actual: np.ndarray,
    ) -> np.ndarray:
        """Vectorized ``mix(...)[actual]`` given only the model's probability of the actual byte."""
        result = np.asarray(p_actual, dtype=np.float64).copy()
        correct = predicted == actual
        for bucket in range(len(MATCH_BUCKETS)):
            members = _bucket_members(bucket, predicted, lengths)
            weight, confidence = self.weights[bucket], self.confidences[bucket]
            q = np.where(correct[members], confidence, (1.0 - confidence) / (_SYMBOLS - 1))
            result[members] = (1.0 - weight) * result[members] + weight * q
        return result


def fit_match_mixing(
    p_actual: np.ndarray,
    predicted: np.ndarray,
    lengths: np.ndarray,
    actual: np.ndarray,
    *,
    history: int = 1 << 16,
    table_bits: int = 16,
    context: int = 6,
) -> MatchMixing:
    """Choose each bucket's confidence and weight to minimize code length on a fitting slice."""
    p_actual = np.clip(np.asarray(p_actual, dtype=np.float64), 1e-12, 1.0)
    correct = predicted == actual
    weights, confidences = [], []
    for bucket in range(len(MATCH_BUCKETS)):
        members = _bucket_members(bucket, predicted, lengths)
        if not members.any():
            weights.append(0.0)
            confidences.append(0.0)
            continue
        confidence = min(float(correct[members].mean()), 0.9999)
        q = np.where(correct[members], confidence, (1.0 - confidence) / (_SYMBOLS - 1))
        costs = [-np.log2((1.0 - w) * p_actual[members] + w * q).sum() for w in _MIX_GRID]
        weights.append(float(_MIX_GRID[int(np.argmin(costs))]))
        confidences.append(confidence)
    return MatchMixing(tuple(weights), tuple(confidences), history, table_bits, context)


def _probabilities(logits: Tensor) -> np.ndarray:
    if logits.shape[-1] != _SYMBOLS:
        raise ValueError("the byte codec requires a 256-symbol model")
    values = logits.detach().to(torch.float64).cpu().numpy()
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def model_fingerprint(model: VocabularyTransformerLM) -> bytes:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.digest()[:16]


@contextmanager
def _deterministic(model: VocabularyTransformerLM) -> Iterator[None]:
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("the codec runs on CPU so both sides compute identical probabilities")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.set_num_threads(previous)


class _Predictor:
    """Next-byte distributions along the cached decode path."""

    def __init__(self, model: VocabularyTransformerLM) -> None:
        self._model = model
        self._caches: list[tuple[Tensor, Tensor]] = []
        self._position = 0
        self._tokens: list[int] = []
        orders = getattr(model, "ngram_orders", ())
        self._history = orders[-1] - 1 if orders else 0

    def start(self, first: int) -> np.ndarray:
        logits, self._caches = self._model.prefill(torch.tensor([[first]]))
        self._position = 1
        self._tokens = [first]
        return _probabilities(logits[0, -1])

    def advance(self, token: int) -> np.ndarray:
        # n-gram tables hash the bytes before ``token`` within this block, matching
        # the cold-start windows the model was trained on
        history = torch.tensor([self._tokens[-self._history :]]) if self._history else None
        logits, self._caches = self._model.decode_step(
            torch.tensor([token]),
            self._caches,
            position=self._position,
            previous_token=torch.tensor([self._tokens[-1]]),
            history=history,
        )
        self._position += 1
        self._tokens.append(token)
        return _probabilities(logits[0, -1])


_UNIFORM = np.full(_SYMBOLS, 1.0 / _SYMBOLS)


@dataclass(slots=True)
class CodecStats:
    input_bytes: int = 0
    output_bytes: int = 0
    model_bits: float = 0.0
    """Code length under the model alone (before mixing, unquantized)."""
    quantized_bits: float = 0.0
    """Ideal code length under the quantized frequencies the coder actually uses."""
    match_predictions: int = 0
    match_correct: int = 0

    @property
    def bits_per_byte(self) -> float:
        return 8 * self.output_bytes / max(self.input_bytes, 1)


def _block_lengths(total: int, block_size: int) -> list[int]:
    return [min(block_size, total - start) for start in range(0, total, block_size)]


def _next_distribution(predictor: _Predictor, block: bytes | bytearray, base: int, index: int) -> np.ndarray:
    if index == 0:
        return _UNIFORM
    if index == 1:
        return predictor.start(block[base])
    return predictor.advance(block[base + index - 1])


def compress_with_stats(
    model: VocabularyTransformerLM,
    data: bytes,
    *,
    block_size: int | None = None,
    match: MatchMixing | None = None,
) -> tuple[bytes, CodecStats]:
    block_size = model.context if block_size is None else block_size
    if not 1 <= block_size <= model.context:
        raise ValueError("block_size must be in [1, model context]")
    stats = CodecStats(input_bytes=len(data))
    encoder = RangeEncoder()
    matcher = None if match is None else MatchModel(match.history, match.table_bits, match.context)
    with _deterministic(model):
        fingerprint = model_fingerprint(model)
        base = 0
        for length in _block_lengths(len(data), block_size):
            predictor = _Predictor(model)
            for index in range(length):
                probabilities = _next_distribution(predictor, data, base, index)
                symbol = data[base + index]
                stats.model_bits -= math.log2(probabilities[symbol])
                if matcher is not None:
                    predicted, match_length = matcher.predict()
                    if predicted >= 0:
                        stats.match_predictions += 1
                        stats.match_correct += predicted == symbol
                    probabilities = match.mix(probabilities, predicted, match_length)
                    matcher.update(symbol)
                frequencies = quantize(probabilities)
                cumulative = np.cumsum(frequencies)
                size = int(frequencies[symbol])
                total = int(cumulative[-1])
                encoder.encode(int(cumulative[symbol]) - size, size, total)
                stats.quantized_bits -= math.log2(size / total)
            base += length
    match_fields = (
        (0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if match is None
        else (match.history, match.table_bits, match.context, *match.weights, *match.confidences)
    )
    header = _HEADER.pack(MAGIC, len(data), block_size, fingerprint, zlib.crc32(data))
    blob = header + _MATCH_HEADER.pack(*match_fields) + encoder.finish()
    stats.output_bytes = len(blob)
    return blob, stats


def compress(
    model: VocabularyTransformerLM,
    data: bytes,
    *,
    block_size: int | None = None,
    match: MatchMixing | None = None,
) -> bytes:
    return compress_with_stats(model, data, block_size=block_size, match=match)[0]


def decompress(model: VocabularyTransformerLM, blob: bytes) -> bytes:
    if len(blob) < HEADER_BYTES:
        raise ValueError("truncated stream")
    magic, total_bytes, block_size, fingerprint, crc = _HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("not a TLZ2 stream")
    if not 1 <= block_size <= model.context:
        raise ValueError("stream block size exceeds the model context")
    fields = _MATCH_HEADER.unpack_from(blob, _HEADER.size)
    match = (
        None
        if fields[0] == 0
        else MatchMixing(tuple(fields[3:7]), tuple(fields[7:11]), fields[0], fields[1], fields[2])
    )
    matcher = None if match is None else MatchModel(match.history, match.table_bits, match.context)
    output = bytearray()
    decoder = RangeDecoder(blob, HEADER_BYTES)
    with _deterministic(model):
        if model_fingerprint(model) != fingerprint:
            raise ValueError("model fingerprint does not match the stream")
        for length in _block_lengths(total_bytes, block_size):
            base = len(output)
            predictor = _Predictor(model)
            for index in range(length):
                probabilities = _next_distribution(predictor, output, base, index)
                if matcher is not None:
                    predicted, match_length = matcher.predict()
                    probabilities = match.mix(probabilities, predicted, match_length)
                frequencies = quantize(probabilities)
                cumulative = np.cumsum(frequencies)
                value = decoder.decode_frequency(int(cumulative[-1]))
                symbol = int(np.searchsorted(cumulative, value, side="right"))
                size = int(frequencies[symbol])
                decoder.consume(int(cumulative[symbol]) - size, size)
                output.append(symbol)
                if matcher is not None:
                    matcher.update(symbol)
    result = bytes(output)
    if zlib.crc32(result) != crc:
        raise ValueError("checksum mismatch: decoder probabilities diverged from the encoder")
    return result
