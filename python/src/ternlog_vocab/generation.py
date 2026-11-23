from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import VocabularyTransformerLM, isa_softmax
from .tokenizer import Vocabulary


def _sample_token(
    logits: Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator,
) -> int:
    values = logits.float().cpu()
    if greedy:
        return int(values.argmax())
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = values / temperature
    if top_k is not None:
        if not 1 <= top_k <= values.numel():
            raise ValueError("top_k must not exceed vocabulary size")
        threshold = torch.topk(values, top_k).values[-1]
        values = values.masked_fill(values < threshold, -torch.inf)
    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_values, sorted_indices = torch.sort(values, descending=True)
        sorted_probabilities = isa_softmax(
            sorted_values, torch.isfinite(sorted_values)
        )
        remove = sorted_probabilities.cumsum(dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        values[sorted_indices[remove]] = -torch.inf
    mask = torch.isfinite(values)
    probabilities = isa_softmax(values, mask)
    return int(torch.multinomial(probabilities, 1, generator=generator))


@dataclass(frozen=True, slots=True)
class GenerationResult:
    data: bytes
    token_count: int
    seconds: float
    stopped_at_end_of_text: bool

    @property
    def bytes_per_second(self) -> float:
        return len(self.data) / max(self.seconds, 1e-9)

    @property
    def tokens_per_second(self) -> float:
        return self.token_count / max(self.seconds, 1e-9)


@torch.inference_mode()
def generate(
    model: VocabularyTransformerLM,
    vocabulary: Vocabulary,
    prompt: bytes,
    *,
    max_new_bytes: int,
    device: torch.device,
    greedy: bool = False,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = 0.9,
    seed: int = 20260824,
    suppress_end_of_text: bool = False,
) -> GenerationResult:
    if max_new_bytes <= 0:
        raise ValueError("max_new_bytes must be positive")
    prompt_ids = vocabulary.encode(prompt or b" ")
    visible = prompt_ids[-model.context :]
    offset = len(prompt_ids) - len(visible)
    tokens = torch.tensor(visible, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    logits, caches = model.prefill(tokens, position_offset=offset)
    next_logits = logits[0, -1]
    next_position = len(prompt_ids)
    previous_token_id = visible[-1]
    generated_ids: list[int] = []
    generated = bytearray()
    stopped = False
    raw_end_id = vocabulary.manifest.get("end_of_text_id")
    end_id = None if raw_end_id is None else int(raw_end_id)
    maximum_tokens = max_new_bytes * 4 + 32
    while len(generated) < max_new_bytes and len(generated_ids) < maximum_tokens:
        sampling_logits = next_logits
        if suppress_end_of_text and end_id is not None:
            sampling_logits = next_logits.clone()
            sampling_logits[end_id] = -torch.inf
        token_id = _sample_token(
            sampling_logits,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        if token_id == end_id:
            stopped = True
            break
        if token_id >= vocabulary.logical_size:
            raise AssertionError("model sampled a padded vocabulary row")
        generated_ids.append(token_id)
        generated.extend(vocabulary.token_bytes[token_id])
        if len(generated) >= max_new_bytes:
            break
        token = torch.tensor([token_id], dtype=torch.long, device=device)
        previous = torch.tensor(
            [previous_token_id], dtype=torch.long, device=device
        )
        longest = model.ngram_orders[-1] if model.ngram_orders else 0
        history = (
            torch.tensor(
                [(visible + generated_ids[:-1])[-(longest - 1) :]],
                dtype=torch.long,
                device=device,
            )
            if longest
            else None
        )
        logits, caches = model.decode_step(
            token, caches, position=next_position, previous_token=previous, history=history
        )
        next_logits = logits[0, -1]
        next_position += 1
        previous_token_id = token_id
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return GenerationResult(
        bytes(generated), len(generated_ids), time.perf_counter() - started, stopped
    )
