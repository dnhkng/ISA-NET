from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any


def _distinct_ngram_fraction(values: Sequence[Any], width: int) -> float:
    count = max(len(values) - width + 1, 0)
    if not count:
        return 0.0
    return len({tuple(values[index : index + width]) for index in range(count)}) / count


def _maximum_repeated_run(data: bytes, maximum_unit: int = 16) -> int:
    best = 1
    for unit in range(1, min(maximum_unit, len(data) // 2) + 1):
        for start in range(0, len(data) - unit * 2 + 1):
            pattern = data[start : start + unit]
            run = 1
            while data[start + run * unit : start + (run + 1) * unit] == pattern:
                run += 1
            best = max(best, run)
    return best


_PROMPT_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "little", "named",
    "of", "on", "once", "one", "she", "that", "the", "there", "they", "this",
    "time", "to", "upon", "was", "were", "with",
}


def _prompt_retention(prompt: bytes, generated_text: str) -> tuple[float, float]:
    prompt_text = prompt.decode("utf-8", errors="ignore")
    generated_words = set(re.findall(r"[A-Za-z']+", generated_text.lower()))
    content_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z']+", prompt_text)
        if len(word) >= 3 and word.lower() not in _PROMPT_STOPWORDS
    }
    names = {
        word.lower()
        for word in re.findall(r"\b[A-Z][a-z]+\b", prompt_text)
        if word.lower() not in _PROMPT_STOPWORDS
    }
    content_recall = (
        len(content_words & generated_words) / len(content_words) if content_words else 1.0
    )
    name_recall = len(names & generated_words) / len(names) if names else 1.0
    return content_recall, name_recall


def generation_metrics(
    prompt: bytes,
    generated: bytes,
    seconds: float,
    reference_word_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    try:
        text = generated.decode("utf-8")
        utf8_valid = True
    except UnicodeDecodeError:
        text = generated.decode("utf-8", errors="replace")
        utf8_valid = False
    words = re.findall(r"[A-Za-z']+", text.lower())
    byte_windows = max(len(generated) - 7, 0)
    repeated = (
        byte_windows
        - len({generated[index : index + 8] for index in range(byte_windows)})
        if byte_windows
        else 0
    )
    prompt_windows = {
        prompt[index : index + 8] for index in range(max(len(prompt) - 7, 0))
    }
    copied = sum(
        generated[index : index + 8] in prompt_windows for index in range(byte_windows)
    )
    content_recall, name_recall = _prompt_retention(prompt, text)
    metrics: dict[str, Any] = {
        "generated_bytes": len(generated),
        "seconds": seconds,
        "bytes_per_second": len(generated) / max(seconds, 1e-9),
        "utf8_valid": utf8_valid,
        "replacement_character_count": text.count("\ufffd"),
        "separator_count": generated.count(b"\n<|endoftext|>\n"),
        "repeated_8byte_window_fraction": repeated / max(byte_windows, 1),
        "prompt_copy_8byte_fraction": copied / max(byte_windows, 1),
        "distinct_byte_2_fraction": _distinct_ngram_fraction(generated, 2),
        "distinct_byte_4_fraction": _distinct_ngram_fraction(generated, 4),
        "distinct_word_2_fraction": _distinct_ngram_fraction(words, 2),
        "distinct_word_3_fraction": _distinct_ngram_fraction(words, 3),
        "maximum_repeated_run": _maximum_repeated_run(generated),
        "word_count": len(words),
        "prompt_content_word_recall": content_recall,
        "prompt_name_recall": name_recall,
    }
    if reference_word_counts is not None:
        prompt_words = set(re.findall(r"[A-Za-z']+", prompt.decode("utf-8", errors="ignore").lower()))
        denominator = max(len(words), 1)
        metrics["training_vocabulary_word_fraction"] = sum(
            word in prompt_words or reference_word_counts[word] > 0 for word in words
        ) / denominator
        metrics["common_training_word_fraction"] = sum(
            word in prompt_words or reference_word_counts[word] >= 5 for word in words
        ) / denominator
    return metrics
