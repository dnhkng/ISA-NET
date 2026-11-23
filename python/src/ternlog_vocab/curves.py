from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class CurvePoint:
    decoded_bytes: int
    validation_bpb: float


def load_curve(path: str | Path) -> list[CurvePoint]:
    points: dict[int, CurvePoint] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        validation = record.get("validation") or {}
        decoded_bytes = record.get(
            "decoded_bytes_processed", record.get("bytes_processed")
        )
        bpb = validation.get("bpb", record.get("validation_bpb"))
        if decoded_bytes is not None and bpb is not None:
            point = CurvePoint(int(decoded_bytes), float(bpb))
            points[point.decoded_bytes] = point
    return [points[key] for key in sorted(points)]


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0.0:
        return mean_y, 0.0
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    ) / denominator
    return mean_y - slope * mean_x, slope


def _fit_transformed(
    points: list[CurvePoint],
    *,
    floor: float,
    x_transform: Callable[[float], float],
) -> tuple[float, float, float]:
    xs = [x_transform(point.decoded_bytes / 1_000_000.0) for point in points]
    ys = [math.log(point.validation_bpb - floor) for point in points]
    intercept, slope = _linear_fit(xs, ys)
    predictions = [floor + math.exp(intercept + slope * value) for value in xs]
    rmse = math.sqrt(
        sum(
            (point.validation_bpb - prediction) ** 2
            for point, prediction in zip(points, predictions, strict=True)
        )
        / len(points)
    )
    return intercept, slope, rmse


def _fit_family(
    points: list[CurvePoint], family: str
) -> dict[str, float | str | bool]:
    if family not in {"power", "exponential"}:
        raise ValueError(f"unknown curve family: {family}")
    transform = math.log if family == "power" else (lambda value: value)
    minimum = min(point.validation_bpb for point in points)
    best: tuple[float, float, float, float] | None = None
    # The zero lower bound is intentionally broad. A boundary solution is
    # reported as unidentifiable rather than interpreted as model capacity.
    for index in range(1001):
        floor = (minimum - 1e-5) * index / 1001
        intercept, slope, rmse = _fit_transformed(
            points, floor=floor, x_transform=transform
        )
        candidate = (rmse, floor, intercept, slope)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    rmse, floor, intercept, slope = best
    return {
        "family": family,
        "floor": floor,
        "floor_at_lower_search_boundary": floor <= minimum / 1001 * 2,
        "floor_interpretation": "fit parameter only; not an identified capacity limit",
        "intercept": intercept,
        "slope": slope,
        "rmse": rmse,
    }


def _predict(fit: dict[str, float | str | bool], decoded_bytes: int) -> float:
    x = decoded_bytes / 1_000_000.0
    transformed = math.log(x) if fit["family"] == "power" else x
    return float(fit["floor"]) + math.exp(
        float(fit["intercept"]) + float(fit["slope"]) * transformed
    )


def summarize_curve(
    points: list[CurvePoint], *, forecast_decoded_bytes: int
) -> dict[str, object]:
    if len(points) < 4:
        raise ValueError("at least four validation points are required")
    fits = [_fit_family(points, family) for family in ("power", "exponential")]
    split = max(4, len(points) // 2)
    backtest_points = points[split:]
    for fit, family in zip(fits, ("power", "exponential"), strict=True):
        early_fit = _fit_family(points[:split], family)
        errors = [
            abs(_predict(early_fit, point.decoded_bytes) - point.validation_bpb)
            for point in backtest_points
        ]
        fit["backtest_mae"] = sum(errors) / len(errors) if errors else math.nan
        fit["forecast_bpb"] = _predict(fit, forecast_decoded_bytes)

    late = points[-4:]
    _, late_slope = _linear_fit(
        [point.decoded_bytes / 1_000_000.0 for point in late],
        [point.validation_bpb for point in late],
    )
    latest = points[-1]
    forecasts = [float(fit["forecast_bpb"]) for fit in fits]
    preferred = min(fits, key=lambda fit: float(fit["backtest_mae"]))
    return {
        "points": len(points),
        "latest_decoded_bytes": latest.decoded_bytes,
        "latest_validation_bpb": latest.validation_bpb,
        "latest_interval_gain_bpb": (
            points[-2].validation_bpb - latest.validation_bpb
        ),
        "late_slope_bpb_per_million_bytes": late_slope,
        "forecast_decoded_bytes": forecast_decoded_bytes,
        "forecast_bpb_range": [min(forecasts), max(forecasts)],
        "preferred_family_by_backtest": preferred["family"],
        "preferred_forecast_bpb": preferred["forecast_bpb"],
        "fits": fits,
    }


def _sample_standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def summarize_seed_curves(curves: list[list[CurvePoint]]) -> dict[str, object]:
    if len(curves) < 2:
        raise ValueError("at least two seed curves are required")
    if any(not curve for curve in curves):
        raise ValueError("seed curves cannot be empty")
    common_points = min(len(curve) for curve in curves)
    gates: list[dict[str, object]] = []
    for index in range(common_points):
        byte_counts = [curve[index].decoded_bytes for curve in curves]
        values = [curve[index].validation_bpb for curve in curves]
        gate: dict[str, object] = {
            "point": index + 1,
            "mean_decoded_bytes": round(sum(byte_counts) / len(byte_counts)),
            "decoded_bytes_range": [min(byte_counts), max(byte_counts)],
            "mean_validation_bpb": sum(values) / len(values),
            "sample_standard_deviation_bpb": _sample_standard_deviation(values),
            "range_bpb": max(values) - min(values),
            "seed_validation_bpb": values,
        }
        if index:
            gains = [
                curve[index - 1].validation_bpb - curve[index].validation_bpb
                for curve in curves
            ]
            gate["mean_interval_gain_bpb"] = sum(gains) / len(gains)
            gate["interval_gain_sample_standard_deviation_bpb"] = (
                _sample_standard_deviation(gains)
            )
            gate["seed_interval_gain_bpb"] = gains
        gates.append(gate)
    return {"seeds": len(curves), "common_points": common_points, "gates": gates}
