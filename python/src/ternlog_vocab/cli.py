from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from .codec import MatchMixing, compress_with_stats, decompress
from .config import VocabLMConfig
from .curves import load_curve, summarize_curve, summarize_seed_curves
from .generation import generate
from .model import VocabularyTransformerLM, build_model, capacity_report
from .tokenizer import (
    END_OF_TEXT,
    ByteVocabulary,
    Vocabulary,
    prepare_bpe_corpus,
    prepare_bpe_prefix_corpus,
    prepare_bpe_corpus_streaming,
    prepare_byte_corpus,
)
from .train import run_training, select_device


def _prepare(args: argparse.Namespace) -> None:
    if args.streaming:
        result = prepare_bpe_corpus_streaming(
            args.source_corpus_dir,
            args.output_dir,
            vocabulary_size=args.vocabulary_size,
            padding_multiple=args.padding_multiple,
            minimum_frequency=args.minimum_frequency,
            vocabulary_training_bytes=args.vocabulary_training_bytes,
        )
    else:
        result = prepare_bpe_corpus(
            args.source_corpus_dir,
            args.output_dir,
            vocabulary_size=args.vocabulary_size,
            padding_multiple=args.padding_multiple,
            minimum_frequency=args.minimum_frequency,
        )
    print(json.dumps(result, indent=2))


def _train(args: argparse.Namespace) -> None:
    config = VocabLMConfig.from_toml(args.config)
    if args.target_decoded_bytes is not None:
        config = dataclasses.replace(
            config, target_decoded_bytes=args.target_decoded_bytes
        )
    if args.report_every_decoded_bytes is not None:
        config = dataclasses.replace(
            config, report_every_decoded_bytes=args.report_every_decoded_bytes
        )
    if args.resume is not None:
        config = dataclasses.replace(
            config,
            resume=str(args.resume),
            initialize_from=None,
            projection_distill_updates=0,
        )
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)
    if args.output_dir is not None:
        config = dataclasses.replace(config, output_dir=str(args.output_dir))
    if args.initialize_from is not None:
        config = dataclasses.replace(
            config,
            initialize_from=str(args.initialize_from),
            resume=None,
            projection_distill_updates=args.projection_distill_updates,
        )
    if args.qkv_active_fraction is not None:
        config = dataclasses.replace(
            config,
            attention_value_block_active_fraction=args.qkv_active_fraction,
        )
    if args.qkv_head_input_groups is not None:
        config = dataclasses.replace(
            config,
            qkv_head_input_groups=args.qkv_head_input_groups,
        )
    if args.boolean_ffn_correction_rank is not None:
        config = dataclasses.replace(
            config,
            boolean_ffn_correction_rank=args.boolean_ffn_correction_rank,
        )
    if args.boolean_ffn_correction_block_active_fraction is not None:
        config = dataclasses.replace(
            config,
            boolean_ffn_correction_block_active_fraction=(
                args.boolean_ffn_correction_block_active_fraction
            ),
        )
    if args.boolean_ffn_pool_schedule is not None:
        config = dataclasses.replace(
            config, boolean_ffn_pool_schedule=args.boolean_ffn_pool_schedule
        )
    if args.boolean_vocab_correction_rank is not None:
        config = dataclasses.replace(
            config,
            boolean_vocab_correction_rank=args.boolean_vocab_correction_rank,
        )
    if args.boolean_vocab_factor_output is not None:
        config = dataclasses.replace(
            config,
            boolean_vocab_factor_output=args.boolean_vocab_factor_output,
        )
    config.validate()
    print(json.dumps(run_training(config), indent=2))


def _prepare_prefix(args: argparse.Namespace) -> None:
    result = prepare_bpe_prefix_corpus(
        args.source_corpus_dir,
        args.source_tokenizer_dir,
        args.output_dir,
        vocabulary_size=args.vocabulary_size,
        padding_multiple=args.padding_multiple,
    )
    print(json.dumps(result, indent=2))


def _prepare_bytes(args: argparse.Namespace) -> None:
    result = prepare_byte_corpus(
        args.source_file,
        args.output_dir,
        validation_bytes=args.validation_bytes,
        test_bytes=args.test_bytes,
        padding_multiple=args.padding_multiple,
    )
    print(json.dumps(result, indent=2))


def _load_model(
    checkpoint_path: Path,
) -> tuple[Vocabulary | ByteVocabulary, VocabularyTransformerLM]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = VocabLMConfig.from_dict(checkpoint["config"])
    vocabulary = Vocabulary.load(config.corpus_dir)
    model = build_model(config, vocabulary.manifest)
    model.load_state_dict(checkpoint["model_state_dict"])
    return vocabulary, model.eval()


def _generate(args: argparse.Namespace) -> None:
    vocabulary, model = _load_model(args.checkpoint)
    device = select_device(args.device)
    model = model.to(device)
    has_separator = vocabulary.manifest.get("end_of_text_id") is not None
    prefix = f"\n{END_OF_TEXT}\n" if args.prefix_separator and has_separator else ""
    result = generate(
        model,
        vocabulary,
        (prefix + args.prompt).encode("utf-8"),
        max_new_bytes=args.max_new_bytes,
        device=device,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        suppress_end_of_text=args.suppress_end_of_text,
    )
    print(
        json.dumps(
            {
                "text": result.data.decode("utf-8", errors="replace"),
                "generated_bytes": len(result.data),
                "generated_tokens": result.token_count,
                "bytes_per_second": result.bytes_per_second,
                "tokens_per_second": result.tokens_per_second,
                "stopped_at_end_of_text": result.stopped_at_end_of_text,
                "capacity": capacity_report(model),
            },
            indent=2,
        )
    )


def _compress(args: argparse.Namespace) -> None:
    _, model = _load_model(args.checkpoint)
    data = args.input.read_bytes()
    match = None
    if args.match_mixing is not None:
        fields = json.loads(args.match_mixing.read_text())
        match = MatchMixing(
            tuple(fields["weights"]),
            tuple(fields["confidences"]),
            fields["history"],
            fields["table_bits"],
            fields["context"],
        )
    started = time.perf_counter()
    blob, stats = compress_with_stats(model, data, block_size=args.block_size, match=match)
    seconds = time.perf_counter() - started
    args.output.write_bytes(blob)
    print(
        json.dumps(
            {
                "input_bytes": stats.input_bytes,
                "output_bytes": stats.output_bytes,
                "bits_per_byte": stats.bits_per_byte,
                "model_bits_per_byte": stats.model_bits / max(stats.input_bytes, 1),
                "quantized_bits_per_byte": stats.quantized_bits / max(stats.input_bytes, 1),
                "match_predictions": stats.match_predictions,
                "match_correct": stats.match_correct,
                "seconds": seconds,
                "bytes_per_second": stats.input_bytes / max(seconds, 1e-9),
            },
            indent=2,
        )
    )


def _decompress(args: argparse.Namespace) -> None:
    _, model = _load_model(args.checkpoint)
    started = time.perf_counter()
    data = decompress(model, args.input.read_bytes())
    seconds = time.perf_counter() - started
    args.output.write_bytes(data)
    print(
        json.dumps(
            {
                "output_bytes": len(data),
                "seconds": seconds,
                "bytes_per_second": len(data) / max(seconds, 1e-9),
            },
            indent=2,
        )
    )


def _curve(args: argparse.Namespace) -> None:
    result = summarize_curve(
        load_curve(args.metrics), forecast_decoded_bytes=args.forecast_decoded_bytes
    )
    print(json.dumps(result, indent=2))


def _seed_curve(args: argparse.Namespace) -> None:
    result = summarize_seed_curves([load_curve(path) for path in args.metrics])
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ternlog-vocab")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="train a lossless byte-BPE vocabulary and cache tokens")
    prepare.add_argument("source_corpus_dir", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--vocabulary-size", type=int, default=4096)
    prepare.add_argument("--padding-multiple", type=int, default=128)
    prepare.add_argument("--minimum-frequency", type=int, default=2)
    prepare.add_argument("--streaming", action="store_true")
    prepare.add_argument("--vocabulary-training-bytes", type=int, default=200_000_000)
    prepare.set_defaults(function=_prepare)

    prepare_prefix = commands.add_parser(
        "prepare-prefix", help="derive and stream-encode a smaller BPE merge prefix"
    )
    prepare_prefix.add_argument("source_corpus_dir", type=Path)
    prepare_prefix.add_argument("source_tokenizer_dir", type=Path)
    prepare_prefix.add_argument("output_dir", type=Path)
    prepare_prefix.add_argument("--vocabulary-size", type=int, required=True)
    prepare_prefix.add_argument("--padding-multiple", type=int, default=128)
    prepare_prefix.set_defaults(function=_prepare_prefix)

    prepare_bytes = commands.add_parser(
        "prepare-bytes", help="cache a raw byte stream as identity tokens (no tokenizer)"
    )
    prepare_bytes.add_argument("source_file", type=Path)
    prepare_bytes.add_argument("output_dir", type=Path)
    prepare_bytes.add_argument("--validation-bytes", type=int, required=True)
    prepare_bytes.add_argument("--test-bytes", type=int, required=True)
    prepare_bytes.add_argument("--padding-multiple", type=int, default=128)
    prepare_bytes.set_defaults(function=_prepare_bytes)

    train = commands.add_parser("train", help="train the vocabulary-native model")
    train.add_argument("config", type=Path)
    train.add_argument("--target-decoded-bytes", type=int)
    train.add_argument("--report-every-decoded-bytes", type=int)
    train.add_argument("--resume", type=Path)
    train.add_argument("--seed", type=int)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--initialize-from", type=Path)
    train.add_argument("--projection-distill-updates", type=int, default=0)
    train.add_argument("--qkv-active-fraction", type=float)
    train.add_argument("--qkv-head-input-groups", type=int)
    train.add_argument("--boolean-ffn-correction-rank", type=int)
    train.add_argument("--boolean-ffn-correction-block-active-fraction", type=float)
    train.add_argument(
        "--boolean-ffn-pool-schedule",
        choices=("legacy_mixed", "state_aligned", "state_rotated"),
    )
    train.add_argument("--boolean-vocab-correction-rank", type=int)
    train.add_argument(
        "--boolean-vocab-factor-output",
        choices=("fp16", "w8a8_row", "w8a8_group16"),
    )
    train.set_defaults(function=_train)

    generation = commands.add_parser("generate", help="generate from a vocabulary-native checkpoint")
    generation.add_argument("checkpoint", type=Path)
    generation.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily.")
    generation.add_argument("--max-new-bytes", type=int, default=256)
    generation.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    generation.add_argument("--seed", type=int, default=20260824)
    generation.add_argument("--greedy", action="store_true")
    generation.add_argument("--temperature", type=float, default=0.7)
    generation.add_argument("--top-k", type=int, default=40)
    generation.add_argument("--top-p", type=float)
    generation.add_argument("--suppress-end-of-text", action="store_true")
    generation.add_argument(
        "--prefix-separator", action=argparse.BooleanOptionalAction, default=True
    )
    generation.set_defaults(function=_generate)

    compress_command = commands.add_parser(
        "compress", help="losslessly compress a file with a byte-level checkpoint"
    )
    compress_command.add_argument("checkpoint", type=Path)
    compress_command.add_argument("input", type=Path)
    compress_command.add_argument("output", type=Path)
    compress_command.add_argument(
        "--block-size", type=int, help="bytes per independently coded block (default: model context)"
    )
    compress_command.add_argument(
        "--match-mixing",
        type=Path,
        help="JSON with weights, confidences, history, table_bits, context for the match model",
    )
    compress_command.set_defaults(function=_compress)

    decompress_command = commands.add_parser(
        "decompress", help="restore a file compressed with the same checkpoint"
    )
    decompress_command.add_argument("checkpoint", type=Path)
    decompress_command.add_argument("input", type=Path)
    decompress_command.add_argument("output", type=Path)
    decompress_command.set_defaults(function=_decompress)

    curve = commands.add_parser(
        "curve", help="summarize and backtest a byte-normalized validation curve"
    )
    curve.add_argument("metrics", type=Path)
    curve.add_argument("--forecast-decoded-bytes", type=int, default=100_000_000)
    curve.set_defaults(function=_curve)

    seed_curve = commands.add_parser(
        "seed-curve", help="measure level and interval dispersion across seed curves"
    )
    seed_curve.add_argument("metrics", type=Path, nargs="+")
    seed_curve.set_defaults(function=_seed_curve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)
