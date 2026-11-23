from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternlog_mnist.gates import bcax, eor3, hard_bit, lut2, lut3, minterm, mux, residual_xor
from ternlog_mnist.routing import RouteSpec, route_indices


def _apply_rope(
    state: Tensor, positions: Tensor, base: float, fraction: float = 1.0
) -> Tensor:
    half = state.shape[-1] // 2
    rotated_half = int(half * fraction)
    if rotated_half <= 0 or rotated_half > half:
        raise ValueError("invalid RoPE fraction")
    frequencies = torch.exp(
        -math.log(base)
        * torch.arange(0, rotated_half, device=state.device, dtype=torch.float32)
        / rotated_half
    )
    angles = positions.to(torch.float32)[:, None] * frequencies[None]
    cosine = angles.cos().to(state.dtype)[None, None]
    sine = angles.sin().to(state.dtype)[None, None]
    left, right = state[..., :half], state[..., half:]
    rotated_left = left[..., :rotated_half] * cosine - right[..., :rotated_half] * sine
    rotated_right = right[..., :rotated_half] * cosine + left[..., :rotated_half] * sine
    return torch.cat(
        (
            rotated_left,
            left[..., rotated_half:],
            rotated_right,
            right[..., rotated_half:],
        ),
        dim=-1,
    )


def isa_exp2(values: Tensor) -> Tensor:
    """Quartic exp2 approximation with an M4 bit-construction lowering.

    Native inference constructs ``2**integer`` from IEEE-754 exponent bits.
    ``torch.exp2`` is used only to express that exact factor during training.
    """
    integer = torch.floor(values)
    fraction = values - integer
    ln2 = math.log(2.0)
    polynomial = 1.0 + fraction * (
        ln2
        + fraction
        * (
            ln2**2 / 2.0
            + fraction * (ln2**3 / 6.0 + fraction * (ln2**4 / 24.0))
        )
    )
    return torch.exp2(integer) * polynomial


def isa_sigmoid(values: Tensor) -> Tensor:
    clamped = values.clamp(min=-16.0, max=16.0)
    exponential = isa_exp2(-clamped * math.log2(math.e))
    return 1.0 / (1.0 + exponential)


def isa_softmax(scores: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    centered = scores - scores.amax(dim=-1, keepdim=True)
    centered = centered.clamp(min=-16.0, max=0.0)
    exponentials = isa_exp2(centered * math.log2(math.e))
    if mask is not None:
        exponentials = exponentials.masked_fill(~mask, 0.0)
    return exponentials / exponentials.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class ISAResidualNorm(nn.Module):
    """Parameter-matched residual normalizers with direct M4 reductions."""

    def __init__(self, dimension: int, mode: str, epsilon: float = 1e-6) -> None:
        super().__init__()
        if mode not in {"centered_rms", "centered_l1"}:
            raise ValueError("ISAResidualNorm requires a centered mode")
        self.mode = mode
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, state: Tensor) -> Tensor:
        centered = state - state.mean(dim=-1, keepdim=True)
        if self.mode == "centered_rms":
            inverse = torch.rsqrt(
                centered.square().mean(dim=-1, keepdim=True) + self.epsilon
            )
        else:
            gaussian_rms_ratio = math.sqrt(math.pi / 2.0)
            denominator = (
                centered.abs().mean(dim=-1, keepdim=True) * gaussian_rms_ratio
                + self.epsilon
            )
            inverse = denominator.reciprocal()
        return centered * inverse * self.weight


def residual_norm(dimension: int, mode: str) -> nn.Module:
    return nn.RMSNorm(dimension) if mode == "rms" else ISAResidualNorm(dimension, mode)


def exact_softmax(scores: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return torch.softmax(scores, dim=-1)


def clipped_linear_softmax(
    scores: Tensor,
    slope: Tensor,
    mask: Tensor | None = None,
    *,
    power: int = 1,
    quantize_int8: bool = False,
) -> Tensor:
    if power not in (1, 2, 3):
        raise ValueError("clipped attention power must be 1, 2, or 3")
    if mask is not None:
        maximum = scores.masked_fill(~mask, -torch.inf).amax(
            dim=-1, keepdim=True
        )
        centered = (scores - maximum).masked_fill(~mask, 0.0)
    else:
        centered = scores - scores.amax(dim=-1, keepdim=True)
    weights = F.relu(1.0 + slope * centered)
    if power >= 2:
        squared = weights * weights
        weights = squared if power == 2 else squared * weights
    if mask is not None:
        weights = weights.masked_fill(~mask, 0.0)
    if quantize_int8:
        # The max-centred clipped numerator has an exact maximum of one.
        # Quantize it before the reciprocal normalization, matching the native
        # fixed-scale U8/SDOT path and avoiding a cache-shape-dependent scale.
        hard_weights = (weights * 127.0).round().clamp(0, 127) / 127.0
        weights = weights + (hard_weights - weights).detach()
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def clipped_multiscale_softmax(
    scores: Tensor,
    slope: Tensor,
    scales: tuple[int, ...],
    mask: Tensor | None = None,
    *,
    quantize_int8: bool = False,
) -> Tensor:
    if scales not in {(1, 2), (1, 2, 4)}:
        raise ValueError("clipped multiscale attention requires 1/2 or 1/2/4 slopes")
    if mask is not None:
        maximum = scores.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
        centered = (scores - maximum).masked_fill(~mask, 0.0)
    else:
        centered = scores - scores.amax(dim=-1, keepdim=True)
    weights = sum(F.relu(1.0 + scale * slope * centered) for scale in scales)
    if mask is not None:
        weights = weights.masked_fill(~mask, 0.0)
    if quantize_int8:
        maximum = float(len(scales))
        hard_weights = (
            (weights * (127.0 / maximum)).round().clamp(0, 127)
            * (maximum / 127.0)
        )
        weights = weights + (hard_weights - weights).detach()
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class BooleanLUT3Projection(nn.Module):
    """Packed hard LUT3 projection used for Boolean attention Q/K."""

    def __init__(
        self,
        dimension: int,
        width: int,
        rounds: int,
        output_bits: int,
        *,
        backward_mode: str,
        temperature: float,
        gradient_leak: float,
        program_init_margin: float,
        register_bits: int,
        correction_features: int = 0,
        correction_output_bits: int = 0,
        correction_taps: str = "final",
    ) -> None:
        super().__init__()
        if width % dimension or width % register_bits or output_bits > width:
            raise ValueError("invalid packed Boolean projection dimensions")
        self.dimension = dimension
        self.width = width
        self.rounds = rounds
        self.output_bits = output_bits
        self.bitplanes = width // dimension
        self.backward_mode = backward_mode
        self.temperature = temperature
        self.gradient_leak = gradient_leak
        self.register_bits = register_bits
        self.correction_features = correction_features
        self.correction_output_bits = correction_output_bits
        self.correction_taps = correction_taps
        if correction_taps not in {"final", "equal_trajectory", "late_trajectory"}:
            raise ValueError("invalid Boolean projection correction taps")
        self.program_logits = nn.Parameter(torch.empty(rounds, 8, width))
        nn.init.normal_(
            self.program_logits, mean=-program_init_margin, std=0.01
        )
        thresholds = torch.linspace(-1.5, 1.5, self.bitplanes)
        self.register_buffer("encoder_thresholds", thresholds, persistent=True)
        routes = []
        for round_index in range(rounds):
            routes.append(
                torch.stack(
                    (
                        route_indices(width, RouteSpec("identity"), register_bits),
                        route_indices(
                            width,
                            RouteSpec("byte_rotate", 1 << (round_index % 5)),
                            register_bits,
                        ),
                        route_indices(
                            width,
                            RouteSpec("lane64_rotate", 1 << (round_index % 5)),
                            register_bits,
                        ),
                    )
                )
            )
        self.register_buffer("route_indices", torch.stack(routes), persistent=True)
        output_indices = torch.div(
            torch.arange(output_bits) * width,
            output_bits,
            rounding_mode="floor",
        )
        self.register_buffer("output_indices", output_indices, persistent=True)
        if correction_features > 0:
            if correction_output_bits <= 0:
                raise ValueError("Boolean correction needs output dimensions")
            self.correction_weight = nn.Parameter(
                torch.empty(correction_output_bits, correction_features)
            )
            nn.init.normal_(self.correction_weight, mean=0.0, std=0.01)
            groups = torch.arange(correction_features)
            starts = torch.where(
                groups < width // 16,
                groups * 16,
                (groups * 37) % width,
            )
            self.register_buffer(
                "correction_indices",
                (
                    starts[:, None] + torch.arange(16)[None]
                ).remainder(width).flatten(),
                persistent=True,
            )
            if correction_taps == "equal_trajectory":
                if correction_features % (rounds + 1):
                    raise ValueError("equal Q/K trajectory features must divide across states")
                tap_indices = torch.arange(correction_features).div(
                    correction_features // (rounds + 1), rounding_mode="floor"
                )
            elif correction_taps == "late_trajectory":
                if rounds < 2 or correction_features % (2 * (rounds - 1)):
                    raise ValueError("late Q/K trajectory features do not divide across rounds")
                early_features = correction_features // 2
                tap_indices = torch.cat(
                    (
                        1 + torch.arange(early_features).div(
                            early_features // (rounds - 1), rounding_mode="floor"
                        ),
                        torch.full((correction_features - early_features,), rounds),
                    )
                )
            else:
                tap_indices = torch.full((correction_features,), rounds)
            self.register_buffer("correction_tap_indices", tap_indices, persistent=True)
        else:
            self.register_parameter("correction_weight", None)
            self.register_buffer("correction_indices", None, persistent=True)
            self.register_buffer("correction_tap_indices", None, persistent=True)
        self._regularization: Tensor | None = None
        self._last_diagnostics: dict[str, Tensor] = {}

    def forward(self, state: Tensor) -> Tensor:
        boolean_state = hard_bit(
            state.unsqueeze(-1)
            - self.encoder_thresholds.view(
                *([1] * (state.ndim - 1)), 1, self.bitplanes
            ),
            self.temperature,
        ).flatten(-2)
        states = [boolean_state] if self.correction_taps != "final" else []
        occupancies = [boolean_state.mean()]
        branches: list[Tensor] = []
        for round_index in range(self.rounds):
            a, b, c = (
                boolean_state.index_select(
                    -1, self.route_indices[round_index, route_index]
                )
                for route_index in range(3)
            )
            program = hard_bit(
                self.program_logits[round_index], self.temperature
            ).view(8, *([1] * (state.ndim - 1)), self.width)
            branch = lut3(
                a,
                b,
                c,
                program,
                self.backward_mode,
                self.temperature,
                self.gradient_leak,
            )
            boolean_state = residual_xor(boolean_state, branch)
            if self.correction_taps != "final":
                states.append(boolean_state)
            branches.append(branch.mean())
            occupancies.append(boolean_state.mean())
        occupancy = torch.stack(occupancies)
        self._regularization = (occupancy - 0.5).square().mean()
        discrete = (self.program_logits.detach() >= 0).to(torch.uint8)
        self._last_diagnostics = {
            "encoded_occupancy": occupancies[0].detach(),
            "final_occupancy": occupancies[-1].detach(),
            "final_balance_error": (occupancies[-1] - 0.5).abs().detach(),
            "mean_branch_occupancy": torch.stack(branches).mean().detach(),
            "active_program_fraction": discrete.float().mean(),
            "noop_program_fraction": (discrete.sum(dim=1) == 0).float().mean(),
        }
        output = boolean_state.index_select(-1, self.output_indices)
        if self.correction_weight is not None:
            assert self.correction_indices is not None
            if self.correction_taps == "final":
                selected = boolean_state.index_select(-1, self.correction_indices)
            else:
                assert self.correction_tap_indices is not None
                combined = torch.cat(states, dim=-1)
                routed_indices = self.correction_indices.reshape(
                    self.correction_features, 16
                ) + self.correction_tap_indices[:, None] * self.width
                selected = combined.index_select(-1, routed_indices.flatten())
            pooled = (2 * selected - 1).reshape(
                *selected.shape[:-1], self.correction_features, 16
            ).sum(dim=-1) / 4.0
            correction = pooled @ self.correction_weight.T
            output = torch.cat((output, correction), dim=-1)
        return output

    def regularization(self) -> Tensor:
        return (
            self.program_logits.new_zeros(())
            if self._regularization is None
            else self._regularization
        )

    def diagnostics(self) -> dict[str, float]:
        result = {name: float(value) for name, value in self._last_diagnostics.items()}
        gradient = self.program_logits.grad
        result["program_gradient_nonzero_fraction"] = (
            0.0 if gradient is None else float((gradient != 0).float().mean())
        )
        result["program_gradient_mean_abs"] = (
            0.0 if gradient is None else float(gradient.abs().mean())
        )
        return result

    def discrete_program(self) -> Tensor:
        return (self.program_logits >= 0).to(torch.uint8)


def fake_w8a8_linear(state: Tensor, weight: Tensor) -> Tensor:
    """Hard-forward per-vector/per-row W8A8 with straight-through gradients."""
    activation_scale = (
        state.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 127
    )
    weight_scale = (
        weight.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 127
    )
    activation_quantized = torch.round(state / activation_scale).clamp(-127, 127)
    weight_quantized = torch.round(weight / weight_scale).clamp(-127, 127)
    activation_hard = state + (
        activation_quantized * activation_scale - state
    ).detach()
    weight_hard = weight + (weight_quantized * weight_scale - weight).detach()
    return F.linear(activation_hard, weight_hard)


def fake_w8a8_grouped_weight_linear(
    state: Tensor, weight: Tensor, *, group_size: int
) -> Tensor:
    """W8A8 with one activation scale and independent weight-group scales."""
    if weight.shape[-1] % group_size:
        raise ValueError("W8A8 weight group must divide the input dimension")
    activation_scale = (
        state.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 127
    )
    activation_quantized = torch.round(state / activation_scale).clamp(-127, 127)
    activation_hard = state + (
        activation_quantized * activation_scale - state
    ).detach()
    grouped = weight.reshape(*weight.shape[:-1], -1, group_size)
    weight_scale = (
        grouped.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 127
    )
    weight_quantized = torch.round(grouped / weight_scale).clamp(-127, 127)
    weight_hard = grouped + (
        weight_quantized * weight_scale - grouped
    ).detach()
    return F.linear(activation_hard, weight_hard.reshape_as(weight))


def fake_w8_linear(state: Tensor, weight: Tensor) -> Tensor:
    """Hard-forward per-row W8 with an exact retained activation domain."""
    weight_scale = (
        weight.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 127
    )
    weight_quantized = torch.round(weight / weight_scale).clamp(-127, 127)
    weight_hard = weight + (weight_quantized * weight_scale - weight).detach()
    return F.linear(state, weight_hard)


class LowRankW8A8Linear(nn.Module):
    """Two row-major SDOT projections with a learned narrow interface."""

    def __init__(self, dimension: int, output_dimension: int, rank: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.rank = rank
        self.input_weight = nn.Parameter(torch.empty(rank, dimension))
        self.output_weight = nn.Parameter(torch.empty(output_dimension, rank))
        nn.init.normal_(self.input_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_weight, mean=0.0, std=0.02)

    def initialize_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.output_dimension, self.dimension):
            raise ValueError("dense low-rank source shape mismatch")
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(
                weight.detach().float(), full_matrices=False
            )
            root = singular[: self.rank].sqrt()
            self.input_weight.copy_(root[:, None] * right[: self.rank])
            self.output_weight.copy_(left[:, : self.rank] * root[None, :])

    def forward(self, state: Tensor) -> Tensor:
        hidden = fake_w8a8_linear(state, self.input_weight)
        return fake_w8a8_linear(hidden, self.output_weight)


class LocalGlobalQKVLinear(nn.Module):
    """Head-local W8 Q/K plus one shared global W8/FP16 correction basis."""

    def __init__(
        self,
        dimension: int,
        heads: int,
        qk_dimension: int,
        value_dimension: int,
        local_width: int,
        global_rank: int,
    ) -> None:
        super().__init__()
        group_width = dimension // heads
        if (
            dimension % heads
            or local_width % group_width
            or not 0 < local_width < dimension
            or global_rank % 16
        ):
            raise ValueError("invalid local-global QKV geometry")
        self.dimension = dimension
        self.heads = heads
        self.qk_dimension = qk_dimension
        self.value_dimension = value_dimension
        self.local_width = local_width
        self.global_rank = global_rank
        self.local_weight = nn.Parameter(
            torch.empty(2, heads, qk_dimension, local_width)
        )
        self.global_weight = nn.Parameter(torch.empty(global_rank, dimension))
        self.global_transform = nn.Parameter(
            torch.empty(2, heads, qk_dimension, global_rank)
        )
        self.value_weight = nn.Parameter(torch.empty(value_dimension, dimension))
        nn.init.normal_(self.local_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.global_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.global_transform, mean=0.0, std=0.02)
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)
        groups = local_width // group_width
        indices = []
        for head in range(heads):
            indices.append(
                torch.cat(
                    [
                        torch.arange(
                            ((head + offset) % heads) * group_width,
                            ((head + offset) % heads + 1) * group_width,
                        )
                        for offset in range(groups)
                    ]
                )
            )
        self.register_buffer(
            "local_indices", torch.stack(indices), persistent=True
        )

    @staticmethod
    def _shared_activation_boundary(state: Tensor) -> Tensor:
        scale = state.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 127
        quantized = torch.round(state / scale).clamp(-127, 127) * scale
        return state + (quantized - state).detach()

    @staticmethod
    def _fp16_boundary(state: Tensor) -> Tensor:
        hard = state.to(torch.float16).to(state.dtype)
        return state + (hard - state).detach()

    def initialize_from_dense(self, weight: Tensor) -> None:
        expected = 2 * self.heads * self.qk_dimension + self.value_dimension
        if tuple(weight.shape) != (expected, self.dimension):
            raise ValueError("dense local-global QKV source shape mismatch")
        qk = weight[: 2 * self.heads * self.qk_dimension].detach().float().reshape(
            2, self.heads, self.qk_dimension, self.dimension
        )
        residual = qk.clone()
        for family in range(2):
            for head in range(self.heads):
                residual[family, head, :, self.local_indices[head]] = 0
        matrix = residual.reshape(-1, self.dimension)
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        root = singular[: self.global_rank].sqrt()
        global_weight = root[:, None] * right[: self.global_rank]
        global_transform = (left[:, : self.global_rank] * root[None, :]).reshape(
            2, self.heads, self.qk_dimension, self.global_rank
        )
        reconstructed = torch.einsum(
            "fhor,rd->fhod", global_transform, global_weight
        )
        with torch.no_grad():
            self.global_weight.copy_(global_weight)
            self.global_transform.copy_(global_transform)
            for family in range(2):
                for head in range(self.heads):
                    indices = self.local_indices[head]
                    self.local_weight[family, head].copy_(
                        (qk[family, head] - reconstructed[family, head]).index_select(
                            -1, indices
                        )
                    )
            self.value_weight.copy_(weight[2 * self.heads * self.qk_dimension :])

    def forward(self, state: Tensor) -> Tensor:
        activation = self._shared_activation_boundary(state)
        global_features = fake_w8_linear(activation, self.global_weight)
        global_features = self._fp16_boundary(global_features)
        transform = self._fp16_boundary(self.global_transform)
        global_qk = torch.einsum("...r,fhor->...fho", global_features, transform)
        local = []
        for family in range(2):
            heads = []
            for head in range(self.heads):
                local_input = activation.index_select(-1, self.local_indices[head])
                heads.append(
                    fake_w8_linear(local_input, self.local_weight[family, head])
                )
            local.append(torch.stack(heads, dim=-2))
        qk = torch.stack(local, dim=-3) + global_qk
        value = fake_w8_linear(activation, self.value_weight)
        return torch.cat(
            (qk[..., 0, :, :].flatten(-2), qk[..., 1, :, :].flatten(-2), value),
            dim=-1,
        )

    def active_weight_values(self) -> int:
        return (
            self.local_weight.numel()
            + self.global_weight.numel()
            + self.global_transform.numel()
            + self.value_weight.numel()
        )


class HeadSubspaceQKVLinear(nn.Module):
    """One complete residual subspace per head, shared by Q/K roles."""

    def __init__(
        self,
        dimension: int,
        heads: int,
        qk_dimension: int,
        value_dimension: int,
        transformed: bool,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        if 2 * heads * qk_dimension != dimension:
            raise ValueError("head-subspace QKV requires combined Q/K rows equal dimension")
        self.dimension = dimension
        self.heads = heads
        self.qk_dimension = qk_dimension
        self.qk_rows = heads * qk_dimension
        self.value_dimension = value_dimension
        self.transformed = transformed
        self.global_rank = global_rank
        if global_rank and (not transformed or global_rank % 8):
            raise ValueError(
                "head-subspace global correction requires transformed 8-lane groups"
            )
        self.base_weight = nn.Parameter(
            torch.empty(self.qk_rows + global_rank + value_dimension, dimension)
        )
        nn.init.normal_(self.base_weight, mean=0.0, std=0.02)
        if transformed:
            self.query_transform = nn.Parameter(
                torch.cat(
                    (
                        torch.eye(qk_dimension),
                        torch.zeros(qk_dimension, global_rank),
                    ),
                    dim=-1,
                ).repeat(heads, 1, 1)
            )
            self.key_transform = nn.Parameter(
                torch.cat(
                    (
                        torch.eye(qk_dimension),
                        torch.zeros(qk_dimension, global_rank),
                    ),
                    dim=-1,
                ).repeat(heads, 1, 1)
            )
        else:
            self.register_parameter("query_transform", None)
            self.register_parameter("key_transform", None)

    @staticmethod
    def _fp16_boundary(state: Tensor) -> Tensor:
        hard = state.to(torch.float16).to(state.dtype)
        return state + (hard - state).detach()

    def initialize_from_dense(self, weight: Tensor) -> None:
        expected = 2 * self.qk_rows + self.value_dimension
        if tuple(weight.shape) != (expected, self.dimension):
            raise ValueError("dense head-subspace QKV source shape mismatch")
        query = weight[: self.qk_rows].reshape(
            self.heads, self.qk_dimension, self.dimension
        )
        key = weight[self.qk_rows : 2 * self.qk_rows].reshape(
            self.heads, self.qk_dimension, self.dimension
        )
        with torch.no_grad():
            if not self.transformed:
                self.base_weight[: self.qk_rows].copy_(
                    ((query + key) * 0.5).reshape(self.qk_rows, self.dimension)
                )
            else:
                reconstructed_query = []
                reconstructed_key = []
                for head in range(self.heads):
                    matrix = torch.cat((query[head], key[head]), dim=0).float()
                    left, singular, right = torch.linalg.svd(
                        matrix, full_matrices=False
                    )
                    root = singular[: self.qk_dimension].sqrt()
                    base = root[:, None] * right[: self.qk_dimension]
                    transforms = left[:, : self.qk_dimension] * root[None, :]
                    start = head * self.qk_dimension
                    self.base_weight[start : start + self.qk_dimension].copy_(base)
                    self.query_transform[head].copy_(
                        torch.cat(
                            (
                                transforms[: self.qk_dimension],
                                torch.zeros(
                                    self.qk_dimension,
                                    self.global_rank,
                                    device=transforms.device,
                                    dtype=transforms.dtype,
                                ),
                            ),
                            dim=-1,
                        )
                    )
                    self.key_transform[head].copy_(
                        torch.cat(
                            (
                                transforms[self.qk_dimension :],
                                torch.zeros(
                                    self.qk_dimension,
                                    self.global_rank,
                                    device=transforms.device,
                                    dtype=transforms.dtype,
                                ),
                            ),
                            dim=-1,
                        )
                    )
                    reconstructed_query.append(
                        transforms[: self.qk_dimension] @ base
                    )
                    reconstructed_key.append(
                        transforms[self.qk_dimension :] @ base
                    )
                if self.global_rank:
                    residual = torch.cat(
                        (
                            query - torch.stack(reconstructed_query),
                            key - torch.stack(reconstructed_key),
                        ),
                        dim=0,
                    ).reshape(2 * self.qk_rows, self.dimension)
                    left, singular, right = torch.linalg.svd(
                        residual.float(), full_matrices=False
                    )
                    root = singular[: self.global_rank].sqrt()
                    global_base = root[:, None] * right[: self.global_rank]
                    global_transform = (
                        left[:, : self.global_rank] * root[None, :]
                    ).reshape(
                        2, self.heads, self.qk_dimension, self.global_rank
                    )
                    self.base_weight[
                        self.qk_rows : self.qk_rows + self.global_rank
                    ].copy_(global_base)
                    self.query_transform[..., self.qk_dimension :].copy_(
                        global_transform[0]
                    )
                    self.key_transform[..., self.qk_dimension :].copy_(
                        global_transform[1]
                    )
            value_start = self.qk_rows + self.global_rank
            self.base_weight[value_start:].copy_(weight[2 * self.qk_rows :])

    def forward(self, state: Tensor) -> Tensor:
        projected = fake_w8a8_linear(state, self.base_weight)
        bases = projected[..., : self.qk_rows].reshape(
            *projected.shape[:-1], self.heads, self.qk_dimension
        )
        global_features = projected[
            ..., self.qk_rows : self.qk_rows + self.global_rank
        ]
        value = projected[..., self.qk_rows + self.global_rank :]
        if self.transformed:
            bases = self._fp16_boundary(bases)
            if self.global_rank:
                global_features = self._fp16_boundary(global_features)
                global_features = global_features.unsqueeze(-2).expand(
                    *bases.shape[:-2], self.heads, self.global_rank
                )
                bases = torch.cat((bases, global_features), dim=-1)
            query = torch.einsum(
                "...hi,hoi->...ho",
                bases,
                self._fp16_boundary(self.query_transform),
            )
            key = torch.einsum(
                "...hi,hoi->...ho",
                bases,
                self._fp16_boundary(self.key_transform),
            )
        else:
            query = key = bases
        return torch.cat((query.flatten(-2), key.flatten(-2), value), dim=-1)

    def active_weight_values(self) -> int:
        return self.base_weight.numel() + sum(
            parameter.numel()
            for parameter in (self.query_transform, self.key_transform)
            if parameter is not None
        )


class CyclicRoutedQKVLinear(nn.Module):
    """FP16 cyclic Q/K routes plus one direct shared-value W8 projection."""

    def __init__(
        self,
        dimension: int,
        qk_output_dimension: int,
        value_dimension: int,
        routes: int,
    ) -> None:
        super().__init__()
        if qk_output_dimension != dimension or routes not in {16, 32}:
            raise ValueError("invalid cyclic-routed QKV geometry")
        self.dimension = dimension
        self.qk_output_dimension = qk_output_dimension
        self.value_dimension = value_dimension
        self.routes = routes
        self.route_weight = nn.Parameter(torch.empty(qk_output_dimension, routes))
        self.value_weight = nn.Parameter(torch.empty(value_dimension, dimension))
        nn.init.normal_(self.route_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)
        self.register_buffer(
            "route_offsets",
            torch.tensor(
                [(route * 53) % dimension for route in range(routes)],
                dtype=torch.long,
            ),
            persistent=True,
        )

    @staticmethod
    def _fp16_boundary(state: Tensor) -> Tensor:
        hard = state.to(torch.float16).to(state.dtype)
        return state + (hard - state).detach()

    def initialize_from_dense(self, weight: Tensor) -> None:
        expected = self.qk_output_dimension + self.value_dimension
        if tuple(weight.shape) != (expected, self.dimension):
            raise ValueError("dense cyclic-routed QKV source shape mismatch")
        rows = torch.arange(self.qk_output_dimension, device=weight.device)
        indices = (
            rows[:, None] + self.route_offsets.to(weight.device)[None, :]
        ) % self.dimension
        with torch.no_grad():
            self.route_weight.copy_(
                weight[: self.qk_output_dimension].gather(1, indices)
            )
            self.value_weight.copy_(weight[self.qk_output_dimension :])

    def forward(self, state: Tensor) -> Tensor:
        rounded_state = self._fp16_boundary(state)
        rounded_weight = self._fp16_boundary(self.route_weight)
        qk = torch.zeros(
            *state.shape[:-1],
            self.qk_output_dimension,
            device=state.device,
            dtype=state.dtype,
        )
        for route, offset in enumerate(self.route_offsets.tolist()):
            routed = torch.roll(rounded_state, shifts=-offset, dims=-1)
            qk = qk + routed * rounded_weight[:, route]
        value = fake_w8a8_linear(state, self.value_weight)
        return torch.cat((qk, value), dim=-1)

    def active_weight_values(self) -> int:
        return self.route_weight.numel() + self.value_weight.numel()


class SharedBasisQKVLinear(nn.Module):
    """ISA-shaped Q/K bases with tiny head-specific FP16 transforms."""

    def __init__(
        self,
        dimension: int,
        heads: int,
        qk_dimension: int,
        value_dimension: int,
        bases: int,
        head_contexts: Sequence[int],
        context: int,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.qk_dimension = qk_dimension
        self.value_dimension = value_dimension
        self.bases = bases
        self.base_rows = 2 * bases * qk_dimension + value_dimension
        self.base_weight = nn.Parameter(torch.empty(self.base_rows, dimension))
        self.query_transform = nn.Parameter(
            torch.eye(qk_dimension).repeat(heads, 1, 1)
        )
        self.key_transform = nn.Parameter(
            torch.eye(qk_dimension).repeat(heads, 1, 1)
        )
        nn.init.normal_(self.base_weight, mean=0.0, std=0.02)
        self.register_buffer(
            "head_base_indices",
            torch.tensor(
                [
                    0 if bases == 1 or head_context < context else 1
                    for head_context in head_contexts
                ],
                dtype=torch.long,
            ),
            persistent=True,
        )

    @staticmethod
    def _fp16_boundary(state: Tensor) -> Tensor:
        hard = state.to(torch.float16).to(state.dtype)
        return state + (hard - state).detach()

    @staticmethod
    def _factor_group(weight: Tensor) -> tuple[Tensor, Tensor]:
        heads, rows, columns = weight.shape
        left, singular, right = torch.linalg.svd(
            weight.detach().float().reshape(heads * rows, columns),
            full_matrices=False,
        )
        root = singular[:rows].sqrt()
        base = root[:, None] * right[:rows]
        transforms = (left[:, :rows] * root[None, :]).reshape(heads, rows, rows)
        return base, transforms

    def initialize_from_dense(self, weight: Tensor) -> None:
        expected_rows = 2 * self.heads * self.qk_dimension + self.value_dimension
        if tuple(weight.shape) != (expected_rows, self.dimension):
            raise ValueError("dense shared-basis source shape mismatch")
        query = weight[: self.heads * self.qk_dimension].reshape(
            self.heads, self.qk_dimension, self.dimension
        )
        key = weight[
            self.heads * self.qk_dimension : 2 * self.heads * self.qk_dimension
        ].reshape(self.heads, self.qk_dimension, self.dimension)
        value = weight[2 * self.heads * self.qk_dimension :]
        with torch.no_grad():
            for family_index, (family, transforms) in enumerate(
                (
                    (query, self.query_transform),
                    (key, self.key_transform),
                )
            ):
                for base_index in range(self.bases):
                    selected = (self.head_base_indices == base_index).nonzero().flatten()
                    if selected.numel() == 0:
                        raise ValueError("shared Q/K basis has no assigned heads")
                    base, group_transforms = self._factor_group(
                        family.index_select(0, selected)
                    )
                    offset = (family_index * self.bases + base_index) * self.qk_dimension
                    self.base_weight[offset : offset + self.qk_dimension].copy_(base)
                    transforms.index_copy_(0, selected, group_transforms)
            self.base_weight[2 * self.bases * self.qk_dimension :].copy_(value)

    def forward(self, state: Tensor) -> Tensor:
        projected = fake_w8a8_linear(state, self.base_weight)
        qk_rows = self.bases * self.qk_dimension
        query_bases = projected[..., :qk_rows].reshape(
            *projected.shape[:-1], self.bases, self.qk_dimension
        )
        key_bases = projected[..., qk_rows : 2 * qk_rows].reshape(
            *projected.shape[:-1], self.bases, self.qk_dimension
        )
        value = projected[..., 2 * qk_rows :]
        query_input = self._fp16_boundary(
            query_bases.index_select(-2, self.head_base_indices)
        )
        key_input = self._fp16_boundary(
            key_bases.index_select(-2, self.head_base_indices)
        )
        query_weight = self._fp16_boundary(self.query_transform)
        key_weight = self._fp16_boundary(self.key_transform)
        query = torch.einsum("...hi,hoi->...ho", query_input, query_weight)
        key = torch.einsum("...hi,hoi->...ho", key_input, key_weight)
        return torch.cat((query.flatten(-2), key.flatten(-2), value), dim=-1)


class QKVBlockSparseLinear16(nn.Module):
    """Checkpoint-static rectangular SDOT tickets for all Q/K/V rows."""

    def __init__(
        self,
        dimension: int,
        active_fraction: float,
        output_dimension: int | None = None,
        region_dimensions: tuple[int, int, int] | None = None,
        allocation: str = "uniform",
        block_rows: int = 16,
        ticket_selection: str = "magnitude",
        ticket_temperature: float = 0.25,
        qk_head_dimensions: tuple[int, ...] | None = None,
        head_input_groups: int = 0,
    ) -> None:
        super().__init__()
        if dimension % 16 or dimension % block_rows or block_rows not in {8, 16, 32}:
            raise ValueError("QKV ticket dimensions must align to 8x16 SDOT tiles")
        self.dimension = dimension
        self.output_dimension = 3 * dimension if output_dimension is None else output_dimension
        self.region_dimensions = (
            (dimension, dimension, dimension)
            if region_dimensions is None else region_dimensions
        )
        if sum(self.region_dimensions) != self.output_dimension or any(
            value % block_rows for value in self.region_dimensions
        ):
            raise ValueError("QKV ticket regions must align to the row block")
        self.region_row_blocks = tuple(
            value // block_rows for value in self.region_dimensions
        )
        self.active_fraction = active_fraction
        self.allocation = allocation
        self.block_rows = block_rows
        self.ticket_selection = ticket_selection
        self.ticket_temperature = ticket_temperature
        self.head_input_groups = head_input_groups
        self.weight = nn.Parameter(torch.empty(self.output_dimension, dimension))
        self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        row_blocks = self.output_dimension // block_rows
        column_blocks = dimension // 16
        permitted_rows = torch.ones(row_blocks, dtype=torch.bool)
        if qk_head_dimensions is not None:
            maximum = self.region_dimensions[0] // len(qk_head_dimensions)
            if (
                self.region_dimensions[1] != self.region_dimensions[0]
                or any(
                    value <= 0 or value > maximum or value % block_rows
                    for value in qk_head_dimensions
                )
            ):
                raise ValueError("invalid per-head Q/K ticket geometry")
            allowed = []
            for _ in range(2):
                for value in qk_head_dimensions:
                    allowed.extend(
                        [True] * (value // block_rows)
                        + [False] * ((maximum - value) // block_rows)
                    )
            allowed.extend(
                [True] * (self.region_dimensions[2] // block_rows)
            )
            permitted_rows = torch.tensor(allowed, dtype=torch.bool)
        permitted = permitted_rows[:, None].expand(row_blocks, column_blocks).clone()
        if head_input_groups:
            if qk_head_dimensions is None:
                raise ValueError("head-local inputs require per-head Q/K geometry")
            heads = len(qk_head_dimensions)
            if not 1 <= head_input_groups <= heads or dimension % heads:
                raise ValueError("invalid head-local QKV input grouping")
            columns_per_group = dimension // heads // 16
            if columns_per_group <= 0 or dimension // heads % 16:
                raise ValueError("head-local QKV groups must align to input blocks")
            rows_per_head = self.region_row_blocks[0] // heads
            if rows_per_head * heads != self.region_row_blocks[0]:
                raise ValueError("Q/K rows do not divide across input groups")
            for region in range(2):
                region_begin = region * self.region_row_blocks[0]
                for head in range(heads):
                    row_begin = region_begin + head * rows_per_head
                    row_end = row_begin + rows_per_head
                    permitted[row_begin:row_end] = False
                    for offset in range(head_input_groups):
                        group = (head + offset) % heads
                        column_begin = group * columns_per_group
                        column_end = column_begin + columns_per_group
                        permitted[row_begin:row_end, column_begin:column_end] = (
                            permitted_rows[row_begin:row_end, None]
                        )
        self.register_buffer(
            "permitted_block_mask", permitted, persistent=False
        )
        active = max(
            1, round(int(self.permitted_block_mask.sum()) * active_fraction)
        )
        mask = torch.zeros(row_blocks * column_blocks)
        permitted_indices = self.permitted_block_mask.flatten().nonzero().flatten()
        mask[permitted_indices[:active]] = 1
        self.register_buffer(
            "block_mask", mask.reshape(row_blocks, column_blocks), persistent=True
        )
        self.register_buffer(
            "initial_block_mask", self.block_mask.clone(), persistent=False
        )
        self.register_load_state_dict_post_hook(self._reset_initial_mask_after_load)
        if ticket_selection == "magnitude":
            self.register_parameter("ticket_logits", None)
        else:
            self.ticket_logits = nn.Parameter(torch.zeros_like(self.block_mask))

    @staticmethod
    def _reset_initial_mask_after_load(module, _incompatible) -> None:
        module.initial_block_mask.copy_(module.block_mask)

    def _hard_mask(self, scores: Tensor) -> Tensor:
        mask = torch.zeros_like(scores)
        if self.ticket_selection == "trainable_family":
            begin = 0
            for region_rows in self.region_row_blocks:
                end = begin + region_rows
                permitted = self.permitted_block_mask[begin:end]
                region_scores = scores[begin:end].masked_fill(~permitted, -torch.inf)
                active = max(
                    1, round(int(permitted.sum()) * self.active_fraction)
                )
                selected = region_scores.flatten().topk(active).indices
                mask[begin:end].flatten()[selected] = 1
                begin = end
            return mask
        active = max(
            1, round(int(self.permitted_block_mask.sum()) * self.active_fraction)
        )
        eligible = scores.masked_fill(~self.permitted_block_mask, -torch.inf)
        mask.flatten()[eligible.flatten().topk(active).indices] = 1
        return mask

    def deployed_mask(self) -> Tensor:
        return (
            self.block_mask
            if self.ticket_logits is None
            else self._hard_mask(self.ticket_logits.detach())
        )

    def _training_mask(self) -> Tensor:
        if self.ticket_logits is None:
            return self.block_mask
        if not self.training:
            hard = self.deployed_mask()
            with torch.no_grad():
                self.block_mask.copy_(hard)
            return hard
        hard = self._hard_mask(self.ticket_logits)
        soft = torch.empty_like(self.ticket_logits)
        if self.ticket_selection == "trainable_family":
            regions = self.region_row_blocks
        else:
            regions = (self.ticket_logits.shape[0],)
        begin = 0
        for region_rows in regions:
            end = begin + region_rows
            values = self.ticket_logits[begin:end]
            active = max(1, round(values.numel() * self.active_fraction))
            threshold = values.flatten().topk(active).values[-1].detach()
            soft[begin:end] = torch.sigmoid(
                (values - threshold) / self.ticket_temperature
            )
            begin = end
        with torch.no_grad():
            self.block_mask.copy_(hard)
        return (hard - soft).detach() + soft

    def initialize_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.output_dimension, self.dimension):
            raise ValueError("dense QKV block-ticket source shape mismatch")
        row_blocks, column_blocks = self.block_mask.shape
        scores = weight.detach().reshape(
            row_blocks, self.block_rows, column_blocks, 16
        ).square().sum(dim=(1, 3)).sqrt()
        scores = scores.masked_fill(~self.permitted_block_mask, -torch.inf)
        mask = torch.zeros_like(scores)
        if self.allocation == "uniform":
            active = max(
                1, round(int(self.permitted_block_mask.sum()) * self.active_fraction)
            )
            selected = scores.flatten().topk(active).indices
            mask.flatten()[selected] = 1
            fractions = None
        elif self.allocation == "key_protected":
            fractions = (0.75, 1.0, 0.5)
        elif self.allocation == "value_protected":
            fractions = (0.75, 0.5, 1.0)
        else:
            raise ValueError(f"invalid QKV ticket allocation: {self.allocation}")
        if fractions is not None:
            begin = 0
            for region_rows, fraction in zip(
                self.region_row_blocks, fractions, strict=True
            ):
                end = begin + region_rows
                region_scores = scores[begin:end]
                permitted = self.permitted_block_mask[begin:end]
                active = max(1, round(int(permitted.sum()) * fraction))
                selected = region_scores.flatten().topk(active).indices
                region_mask = mask[begin:end].flatten()
                region_mask[selected] = 1
                begin = end
        with torch.no_grad():
            self.weight.copy_(weight)
            self.block_mask.copy_(mask)
            if self.ticket_logits is not None:
                centered = scores - scores.mean()
                self.ticket_logits.copy_(centered / scores.std().clamp_min(1e-6))
                self.block_mask.copy_(self._hard_mask(self.ticket_logits))
            self.initial_block_mask.copy_(self.block_mask)

    def expanded_mask(self) -> Tensor:
        return self._training_mask().repeat_interleave(
            self.block_rows, 0
        ).repeat_interleave(16, 1)

    def forward(self, state: Tensor) -> Tensor:
        return fake_w8a8_linear(state, self.weight * self.expanded_mask())

    def active_weight_values(self) -> int:
        return int(self.deployed_mask().sum().item()) * self.block_rows * 16

    def prepacked_metadata_bytes(self) -> int:
        return (self.block_mask.shape[0] + 1) * 2 + int(
            self.deployed_mask().sum().item()
        )

    def training_candidate_program_values(self) -> int:
        return 0 if self.ticket_logits is None else self.ticket_logits.numel()


class ValueBlockSparseQKVLinear16(nn.Module):
    """Fused dense Q/K plus checkpoint-static 16x16 block-sparse V."""

    def __init__(self, dimension: int, active_fraction: float) -> None:
        super().__init__()
        if dimension % 16:
            raise ValueError("ValueBlockSparseQKVLinear16 requires a multiple of 16")
        self.dimension = dimension
        self.active_fraction = active_fraction
        self.weight = nn.Parameter(torch.empty(3 * dimension, dimension))
        self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        blocks = dimension // 16
        active = max(1, round(blocks * blocks * active_fraction))
        mask = torch.zeros(blocks * blocks)
        mask[:active] = 1
        self.register_buffer(
            "value_block_mask", mask.reshape(blocks, blocks), persistent=True
        )

    def initialize_from_dense(self, weight: Tensor) -> None:
        expected = (3 * self.dimension, self.dimension)
        if tuple(weight.shape) != expected:
            raise ValueError("dense QKV shape mismatch")
        blocks = self.dimension // 16
        value = weight.detach()[2 * self.dimension :]
        scores = value.reshape(blocks, 16, blocks, 16)
        scores = scores.square().sum(dim=(1, 3)).sqrt()
        active = max(1, round(scores.numel() * self.active_fraction))
        selected = scores.flatten().topk(active).indices
        mask = torch.zeros_like(scores).flatten()
        mask[selected] = 1
        with torch.no_grad():
            self.weight.copy_(weight)
            self.value_block_mask.copy_(mask.reshape_as(scores))

    def expanded_mask(self) -> Tensor:
        dense_qk = self.weight.new_ones(2 * self.dimension, self.dimension)
        value = self.value_block_mask.repeat_interleave(16, 0).repeat_interleave(16, 1)
        return torch.cat((dense_qk, value), dim=0)

    def forward(self, state: Tensor) -> Tensor:
        return fake_w8a8_linear(state, self.weight * self.expanded_mask())

    def active_weight_values(self) -> int:
        return (
            2 * self.dimension * self.dimension
            + int(self.value_block_mask.sum().item()) * 16 * 16
        )

    def prepacked_metadata_bytes(self) -> int:
        dense_rows = 2 * self.dimension
        value_rows = self.value_block_mask.shape[0]
        return dense_rows * 0 + (value_rows + 1) * 2 + int(
            self.value_block_mask.sum().item()
        )


class BlockSparseLinear16(nn.Module):
    """Dense training scaffold with checkpoint-static 16x16 deployed blocks."""

    def __init__(
        self, dimension: int, active_fraction: float, input_rounding: str
    ) -> None:
        super().__init__()
        if dimension % 16:
            raise ValueError("BlockSparseLinear16 requires a multiple of 16")
        self.dimension = dimension
        self.active_fraction = active_fraction
        self.input_rounding = input_rounding
        self.weight = nn.Parameter(torch.empty(dimension, dimension))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        blocks = dimension // 16
        active = max(1, round(blocks * blocks * active_fraction))
        mask = torch.zeros(blocks * blocks)
        mask[:active] = 1
        self.register_buffer("block_mask", mask.reshape(blocks, blocks), persistent=True)

    def initialize_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.dimension, self.dimension):
            raise ValueError("dense attention-output shape mismatch")
        blocks = self.dimension // 16
        scores = weight.detach().reshape(blocks, 16, blocks, 16)
        scores = scores.square().sum(dim=(1, 3)).sqrt()
        active = max(1, round(scores.numel() * self.active_fraction))
        selected = scores.flatten().topk(active).indices
        mask = torch.zeros_like(scores).flatten()
        mask[selected] = 1
        with torch.no_grad():
            self.weight.copy_(weight)
            self.block_mask.copy_(mask.reshape_as(scores))

    def expanded_mask(self) -> Tensor:
        return self.block_mask.repeat_interleave(16, 0).repeat_interleave(16, 1)

    def forward(self, state: Tensor) -> Tensor:
        if self.input_rounding == "fp16":
            state = state + (
                state.to(torch.float16).to(state.dtype) - state
            ).detach()
        return fake_w8a8_linear(state, self.weight * self.expanded_mask())

    def active_weight_values(self) -> int:
        return int(self.block_mask.sum().item()) * 16 * 16

    def prepacked_metadata_bytes(self) -> int:
        return (self.block_mask.shape[0] + 1) * 2 + int(
            self.block_mask.sum().item()
        )


class HeadLocalHybridLinear(nn.Module):
    """Head-local W8A8 blocks plus an optional rank-8 W8/FP16 correction."""

    def __init__(
        self,
        dimension: int,
        heads: int,
        rank: int,
        input_rounding: str,
    ) -> None:
        super().__init__()
        if dimension % heads or (dimension // heads) % 16 or rank not in {0, 8}:
            raise ValueError("invalid head-local output geometry")
        self.dimension = dimension
        self.heads = heads
        self.head_dimension = dimension // heads
        self.rank = rank
        self.input_rounding = input_rounding
        self.local_weight = nn.Parameter(
            torch.empty(heads, self.head_dimension, self.head_dimension)
        )
        nn.init.kaiming_uniform_(self.local_weight, a=math.sqrt(5))
        if rank:
            self.global_input_weight = nn.Parameter(torch.empty(rank, dimension))
            self.global_output_weight = nn.Parameter(torch.empty(dimension, rank))
            nn.init.normal_(self.global_input_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.global_output_weight, mean=0.0, std=0.02)
        else:
            self.register_parameter("global_input_weight", None)
            self.register_parameter("global_output_weight", None)

    @staticmethod
    def _fp16_boundary(state: Tensor) -> Tensor:
        hard = state.to(torch.float16).to(state.dtype)
        return state + (hard - state).detach()

    def expanded_local_weight(self) -> Tensor:
        return torch.block_diag(*self.local_weight.unbind(0))

    def initialize_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.dimension, self.dimension):
            raise ValueError("dense head-local output shape mismatch")
        source = weight.detach().float()
        local = torch.stack(
            [
                source[
                    head * self.head_dimension : (head + 1) * self.head_dimension,
                    head * self.head_dimension : (head + 1) * self.head_dimension,
                ]
                for head in range(self.heads)
            ]
        )
        with torch.no_grad():
            self.local_weight.copy_(local)
        if not self.rank:
            return
        residual = source - self.expanded_local_weight().detach()
        left, singular, right = torch.linalg.svd(residual, full_matrices=False)
        root = singular[: self.rank].sqrt()
        input_weight = root[:, None] * right[: self.rank]
        output_weight = left[:, : self.rank] * root[None, :]
        reconstructed = output_weight @ input_weight
        refined_local = torch.stack(
            [
                (source - reconstructed)[
                    head * self.head_dimension : (head + 1) * self.head_dimension,
                    head * self.head_dimension : (head + 1) * self.head_dimension,
                ]
                for head in range(self.heads)
            ]
        )
        with torch.no_grad():
            self.local_weight.copy_(refined_local)
            self.global_input_weight.copy_(input_weight)
            self.global_output_weight.copy_(output_weight)

    def forward(self, state: Tensor) -> Tensor:
        if self.input_rounding == "fp16":
            state = self._fp16_boundary(state)
        local = fake_w8a8_linear(state, self.expanded_local_weight())
        if not self.rank:
            return local
        hidden = fake_w8a8_linear(state, self.global_input_weight)
        hidden = self._fp16_boundary(hidden)
        output_weight = self._fp16_boundary(self.global_output_weight)
        return local + F.linear(hidden, output_weight)

    def active_weight_values(self) -> int:
        return self.local_weight.numel() + (
            0
            if not self.rank
            else self.global_input_weight.numel()
            + self.global_output_weight.numel()
        )

    def deployed_numeric_bytes(self) -> int:
        local = self.local_weight.numel()
        if not self.rank:
            return local + 4 * self.dimension
        return (
            local
            + self.global_input_weight.numel()
            + 2 * self.global_output_weight.numel()
            + 4 * (self.dimension + self.rank)
        )


class ProductCodebookLinear16(nn.Module):
    """Shared 16-lane SDOT codebooks with hard TBL-selected responses."""

    def __init__(
        self,
        dimension: int,
        *,
        output_dimension: int | None = None,
        codewords: int,
        temperature: float,
        response_shift: int,
        input_rounding: str,
    ) -> None:
        super().__init__()
        output_dimension = dimension if output_dimension is None else output_dimension
        if dimension % 16 or output_dimension % 16 or codewords not in {32, 64}:
            raise ValueError("invalid product-codebook geometry")
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.blocks = dimension // 16
        self.codewords = codewords
        self.temperature = temperature
        self.response_shift = response_shift
        self.input_rounding = input_rounding
        self.codebook = nn.Parameter(torch.empty(self.blocks, codewords, 16))
        self.row_scale = nn.Parameter(torch.ones(output_dimension))
        self.code_index_logits = nn.Parameter(
            torch.empty(self.blocks, output_dimension, codewords)
        )
        nn.init.uniform_(self.codebook, -8.0, 8.0)
        nn.init.normal_(self.code_index_logits, std=0.02)

    @staticmethod
    def _cluster(values: Tensor, clusters: int) -> tuple[Tensor, Tensor]:
        chosen = [int(values.square().sum(dim=-1).argmax())]
        distance = (values - values[chosen[0]]).square().sum(dim=-1)
        for _ in range(1, clusters):
            index = int(distance.argmax())
            chosen.append(index)
            distance = torch.minimum(
                distance,
                (values - values[index]).square().sum(dim=-1),
            )
        centers = values[chosen].clone()
        assignment = torch.zeros(
            values.shape[0], dtype=torch.long, device=values.device
        )
        for _ in range(24):
            distances = torch.cdist(values.float(), centers.float()).square()
            updated_assignment = distances.argmin(dim=-1)
            updated = centers.clone()
            for cluster in range(clusters):
                members = values[updated_assignment == cluster]
                if members.numel():
                    updated[cluster] = members.mean(dim=0)
            assignment = updated_assignment
            if torch.equal(updated, centers):
                break
            centers = updated
        return centers, assignment

    def initialize_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.output_dimension, self.dimension):
            raise ValueError("dense product-codebook source shape mismatch")
        scales = weight.detach().abs().amax(dim=-1).clamp_min(1.0E-12) / 127.0
        normalized = weight.detach() / scales[:, None]
        codebooks = []
        logits = []
        for block in range(self.blocks):
            values = normalized[:, block * 16 : (block + 1) * 16]
            centers, _ = self._cluster(values, self.codewords)
            centers = centers.round().clamp(-127, 127)
            distances = torch.cdist(values.float(), centers.float()).square()
            standardized = distances / distances.std(dim=-1, keepdim=True).clamp_min(
                1.0E-6
            )
            codebooks.append(centers)
            logits.append(-standardized)
        with torch.no_grad():
            self.row_scale.copy_(scales)
            self.codebook.copy_(torch.stack(codebooks))
            self.code_index_logits.copy_(torch.stack(logits))

    def deployed_indices(self) -> Tensor:
        return self.code_index_logits.detach().argmax(dim=-1)

    def deployed_codebook(self) -> Tensor:
        return self.codebook.detach().round().clamp(-127, 127).to(torch.int8)

    def _selectors(self) -> Tensor:
        indices = self.code_index_logits.argmax(dim=-1)
        hard = F.one_hot(indices, self.codewords).to(self.code_index_logits.dtype)
        if not self.training:
            return hard
        soft = F.softmax(
            self.code_index_logits / self.temperature, dim=-1
        )
        return soft + (hard - soft).detach()

    def forward(self, state: Tensor) -> Tensor:
        if self.input_rounding == "fp16":
            rounded = state.to(torch.float16).to(state.dtype)
            state = state + (rounded - state).detach()
        activation_scale = (
            state.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0E-12)
            / 127.0
        )
        activation_integer = state / activation_scale
        hard_activation = activation_integer.round().clamp(-127, 127)
        activation_integer = activation_integer + (
            hard_activation - activation_integer
        ).detach()
        activation_blocks = activation_integer.reshape(
            *activation_integer.shape[:-1], self.blocks, 16
        )
        hard_codebook = self.codebook.round().clamp(-127, 127)
        codebook = self.codebook + (hard_codebook - self.codebook).detach()
        responses = torch.einsum("...bi,bki->...bk", activation_blocks, codebook)
        divisor = float(1 << self.response_shift)
        scaled_responses = responses / divisor
        hard_responses = scaled_responses.round().clamp(-128, 127)
        responses = scaled_responses + (
            hard_responses - scaled_responses
        ).detach()
        selected = torch.einsum("...bk,brk->...r", responses, self._selectors())
        return selected * (divisor * activation_scale) * self.row_scale

    def deployed_program_bits(self) -> int:
        return self.blocks * self.output_dimension * int(math.log2(self.codewords))

    def training_candidate_program_values(self) -> int:
        return self.code_index_logits.numel()

    def deployed_parameter_bytes(self) -> int:
        return (
            self.codebook.numel()
            + math.ceil(self.deployed_program_bits() / 8)
            + self.row_scale.numel() * 4
        )


class SignedSqueezeHybridLinear(nn.Module):
    """FCMGE bitplanes, learned XOR polarities, CNT squeeze, and dense bypass."""

    def __init__(
        self,
        dimension: int,
        *,
        output_dimension: int | None = None,
        bitplanes: int,
        polarities: int,
        group_bits: int,
        bypass_features: int,
        temperature: float,
        input_rounding: str = "fp16",
    ) -> None:
        super().__init__()
        width = dimension * bitplanes
        if width % group_bits:
            raise ValueError("signed-squeeze groups do not cover the encoded width")
        self.dimension = dimension
        self.output_dimension = (
            dimension if output_dimension is None else output_dimension
        )
        self.bitplanes = bitplanes
        self.polarities = polarities
        self.group_bits = group_bits
        self.groups = width // group_bits
        self.squeeze_features = self.groups * polarities
        self.bypass_features = bypass_features
        self.temperature = temperature
        self.input_rounding = input_rounding
        self.encoder_offsets = nn.Parameter(torch.zeros(dimension))
        self.register_buffer(
            "encoder_thresholds",
            torch.linspace(-1.75, 1.75, bitplanes),
            persistent=True,
        )
        polarity = torch.empty(polarities, self.groups, group_bits)
        for plane in range(polarities):
            index = torch.arange(self.groups * group_bits).reshape(
                self.groups, group_bits
            )
            bits = ((index * (37 + 16 * plane) + 11 + 23 * plane) % 127) < 63
            polarity[plane].copy_(torch.where(bits, 1.0, -1.0))
        self.polarity_logits = nn.Parameter(polarity)
        self.squeeze_weight = nn.Parameter(
            torch.empty(self.output_dimension, self.squeeze_features)
        )
        nn.init.normal_(self.squeeze_weight, mean=0.0, std=0.01)
        self.bypass_weight = nn.Parameter(
            torch.empty(self.output_dimension, bypass_features)
        )
        nn.init.zeros_(self.bypass_weight)
        self.register_buffer(
            "bypass_indices", torch.arange(bypass_features), persistent=True
        )
        self._last_occupancy: Tensor | None = None

    def initialize_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.output_dimension, self.dimension):
            raise ValueError("dense signed-squeeze source shape mismatch")
        importance = weight.detach().square().sum(dim=0)
        if self.bypass_features % 32:
            # Small diagnostic output squeezes predate the QKV native ABI and
            # retain their magnitude-selected gathered bypass.
            indices = importance.topk(self.bypass_features).indices.sort().values
        else:
            starts = range(0, self.dimension - self.bypass_features + 1, 32)
            start = max(
                starts,
                key=lambda value: float(
                    importance[value : value + self.bypass_features].sum()
                ),
            )
            indices = torch.arange(start, start + self.bypass_features)
        with torch.no_grad():
            self.bypass_indices.copy_(indices)
            self.bypass_weight.copy_(weight[:, indices])

    def _features(self, state: Tensor) -> Tensor:
        encoded = hard_bit(
            state.unsqueeze(-1)
            - self.encoder_offsets.view(*([1] * (state.ndim - 1)), -1, 1)
            - self.encoder_thresholds.view(*([1] * state.ndim), -1),
            self.temperature,
        )
        # Threshold-major groups map to contiguous CNT words at export.
        encoded = encoded.transpose(-1, -2).flatten(-2)
        grouped = (2 * encoded - 1).reshape(
            *encoded.shape[:-1], 1, self.groups, self.group_bits
        )
        polarity = 2 * hard_bit(self.polarity_logits, self.temperature) - 1
        features = (grouped * polarity).sum(dim=-1) / math.sqrt(self.group_bits)
        self._last_occupancy = encoded.mean().detach()
        return features.flatten(-2)

    def _round_input(self, state: Tensor) -> Tensor:
        if self.input_rounding == "fp16":
            hard = state.to(torch.float16).to(state.dtype)
        elif self.input_rounding in {"fixed8", "fixed4"}:
            scale = 8.0 if self.input_rounding == "fixed8" else 4.0
            hard = torch.round(state * scale) / scale
        else:
            raise ValueError(f"invalid signed-squeeze rounding: {self.input_rounding}")
        return state + (hard - state).detach()

    def forward(self, state: Tensor) -> Tensor:
        rounded = self._round_input(state)
        squeeze = fake_w8_linear(self._features(rounded), self.squeeze_weight)
        bypass = fake_w8a8_linear(
            rounded.index_select(-1, self.bypass_indices), self.bypass_weight
        )
        return squeeze + bypass

    def regularization(self) -> tuple[Tensor, Tensor]:
        occupancy = (
            self.encoder_offsets.new_tensor(0.5)
            if self._last_occupancy is None
            else self._last_occupancy
        )
        return (occupancy - 0.5).square(), self.encoder_offsets.new_zeros(())

    def diagnostics(self) -> dict[str, float]:
        polarity = (self.polarity_logits.detach() >= 0).float()
        gradient = self.polarity_logits.grad
        return {
            "encoded_occupancy": 0.5 if self._last_occupancy is None else float(self._last_occupancy),
            "polarity_one_fraction": float(polarity.mean()),
            "polarity_gradient_nonzero_fraction": (
                0.0 if gradient is None else float((gradient != 0).float().mean())
            ),
        }

    def deployed_program_bits(self) -> int:
        return self.polarity_logits.numel()


class BitmixTrajectoryHybridLinear(nn.Module):
    """Packed EOR3/BCAX trajectory counts plus an aligned numerical bypass."""

    def __init__(
        self,
        dimension: int,
        output_dimension: int,
        *,
        bitplanes: int,
        rounds: int,
        correction_features: int,
        bypass_features: int,
        cell: str,
        backward_mode: str,
        temperature: float,
        gradient_leak: float,
        program_init_margin: float,
        input_rounding: str,
    ) -> None:
        super().__init__()
        if cell not in {"eor3", "bcax"}:
            raise ValueError("bitmix trajectory cell must be EOR3 or BCAX")
        width = dimension * bitplanes
        if width % 128 or correction_features % (rounds + 1):
            raise ValueError("invalid bitmix trajectory dimensions")
        if bypass_features % 32:
            raise ValueError("bitmix bypass must use aligned 32-channel groups")
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.bitplanes = bitplanes
        self.width = width
        self.rounds = rounds
        self.correction_features = correction_features
        self.bypass_features = bypass_features
        self.cell = cell
        self.backward_mode = backward_mode
        self.temperature = temperature
        self.gradient_leak = gradient_leak
        self.input_rounding = input_rounding
        self.encoder_offsets = nn.Parameter(torch.zeros(dimension))
        self.register_buffer(
            "encoder_thresholds",
            torch.linspace(-1.75, 1.75, bitplanes),
            persistent=True,
        )
        self.gate_logits = nn.Parameter(torch.empty(rounds, width))
        nn.init.normal_(
            self.gate_logits, mean=-program_init_margin, std=0.01
        )
        features_per_tap = correction_features // (rounds + 1)
        windows = width // 16
        feature = torch.arange(correction_features)
        taps = torch.div(feature, features_per_tap, rounding_mode="floor")
        within = feature.remainder(features_per_tap)
        starts = ((within * 5 + taps * 17) % windows) * 16
        self.register_buffer("correction_taps", taps, persistent=True)
        self.register_buffer(
            "correction_indices",
            (starts[:, None] + torch.arange(16)[None]).flatten(),
            persistent=True,
        )
        polarity = torch.empty(correction_features, 16)
        index = torch.arange(correction_features * 16).reshape(
            correction_features, 16
        )
        polarity.copy_(torch.where(((index * 37 + 11) % 127) < 63, 1.0, -1.0))
        self.polarity_logits = nn.Parameter(polarity)
        self.correction_weight = nn.Parameter(
            torch.empty(output_dimension, correction_features)
        )
        nn.init.normal_(self.correction_weight, mean=0.0, std=0.01)
        self.bypass_weight = nn.Parameter(
            torch.zeros(output_dimension, bypass_features)
        )
        self.register_buffer(
            "bypass_indices", torch.arange(bypass_features), persistent=True
        )
        self._last_diagnostics: dict[str, Tensor] = {}
        self._regularization: Tensor | None = None

    def initialize_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.output_dimension, self.dimension):
            raise ValueError("dense bitmix-trajectory source shape mismatch")
        importance = weight.detach().square().sum(dim=0)
        starts = range(0, self.dimension - self.bypass_features + 1, 32)
        start = max(
            starts,
            key=lambda value: float(
                importance[value : value + self.bypass_features].sum()
            ),
        )
        indices = torch.arange(start, start + self.bypass_features)
        with torch.no_grad():
            self.bypass_indices.copy_(indices)
            self.bypass_weight.copy_(weight[:, indices])

    def _round_input(self, state: Tensor) -> Tensor:
        if self.input_rounding != "fp16":
            raise ValueError("bitmix trajectory currently requires FP16 rounding")
        hard = state.to(torch.float16).to(state.dtype)
        return state + (hard - state).detach()

    def _states(self, state: Tensor) -> list[Tensor]:
        encoded = hard_bit(
            state.unsqueeze(-1)
            - self.encoder_offsets.view(*([1] * (state.ndim - 1)), -1, 1)
            - self.encoder_thresholds.view(*([1] * state.ndim), -1),
            self.temperature,
        ).transpose(-1, -2).flatten(-2)
        states = [encoded]
        branches = []
        registers = self.width // 128
        for round_index in range(self.rounds):
            viewed = encoded.view(*encoded.shape[:-1], registers, 128)
            a = viewed
            b = torch.roll(viewed, shifts=-1, dims=-2)
            c = torch.roll(viewed, shifts=-3, dims=-2)
            branch = (
                eor3(
                    a, b, c, self.backward_mode, self.temperature,
                    self.gradient_leak,
                )
                if self.cell == "eor3"
                else bcax(
                    a, b, c, self.backward_mode, self.temperature,
                    self.gradient_leak,
                )
            ).flatten(-2)
            gate = hard_bit(self.gate_logits[round_index], self.temperature)
            encoded = mux(
                gate,
                branch,
                encoded,
                self.backward_mode,
                self.temperature,
                self.gradient_leak,
            )
            branches.append(branch.mean())
            states.append(encoded)
        occupancy = torch.stack([value.mean() for value in states])
        self._regularization = (occupancy - 0.5).square().mean()
        hard_gates = (self.gate_logits.detach() >= 0).float()
        self._last_diagnostics = {
            "encoded_occupancy": occupancy[0].detach(),
            "final_occupancy": occupancy[-1].detach(),
            "final_balance_error": (occupancy[-1] - 0.5).abs().detach(),
            "mean_branch_occupancy": torch.stack(branches).mean().detach(),
            "active_gate_fraction": hard_gates.mean(),
        }
        return states

    def _features(self, states: list[Tensor]) -> Tensor:
        combined = torch.cat(states, dim=-1)
        routed = self.correction_indices.reshape(
            self.correction_features, 16
        ) + self.correction_taps[:, None] * self.width
        selected = combined.index_select(-1, routed.flatten())
        polarity = hard_bit(self.polarity_logits, self.temperature)
        different = residual_xor(
            selected.reshape(*selected.shape[:-1], self.correction_features, 16),
            polarity,
            identity_backward=False,
        )
        return (2 * different - 1).sum(dim=-1) / 4.0

    def forward(self, state: Tensor) -> Tensor:
        rounded = self._round_input(state)
        correction = fake_w8_linear(
            self._features(self._states(rounded)), self.correction_weight
        )
        bypass = fake_w8a8_linear(
            rounded.index_select(-1, self.bypass_indices), self.bypass_weight
        )
        return correction + bypass

    def regularization(self) -> tuple[Tensor, Tensor]:
        zero = self.encoder_offsets.new_zeros(())
        return (
            zero if self._regularization is None else self._regularization,
            zero,
        )

    def diagnostics(self) -> dict[str, float]:
        result = {
            name: float(value) for name, value in self._last_diagnostics.items()
        }
        for name, parameter in (
            ("gate", self.gate_logits),
            ("polarity", self.polarity_logits),
        ):
            gradient = parameter.grad
            result[f"{name}_gradient_nonzero_fraction"] = (
                0.0 if gradient is None else float((gradient != 0).float().mean())
            )
            result[f"{name}_gradient_mean_abs"] = (
                0.0 if gradient is None else float(gradient.abs().mean())
            )
        return result

    def deployed_program_bits(self) -> int:
        return self.gate_logits.numel() + self.polarity_logits.numel()


class VocabularyAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        kv_heads: int,
        *,
        context: int,
        rope_base: float,
        rope_fraction: float,
        partial_key_offset: str,
        qk_norm: bool,
        qk_cache_precision: str,
        value_cache_precision: str,
        qk_head_dimension: int,
        local_qk_head_dimension: int,
        kv_head_routing: str,
        kv_head_sharing: str,
        attention_qkv_projection_mode: str,
        qkv_squeeze_bitplanes: int,
        qkv_squeeze_polarities: int,
        qkv_squeeze_group_bits: int,
        qkv_squeeze_bypass: int,
        qkv_squeeze_input_rounding: str,
        qkv_low_rank: int,
        qkv_shared_bases: int,
        qkv_local_width: int,
        qkv_global_rank: int,
        qkv_route_count: int,
        qkv_bitmix_rounds: int,
        qkv_bitmix_correction_features: int,
        qkv_bitmix_cell: str,
        qkv_block_allocation: str,
        qkv_block_rows: int,
        qkv_ticket_selection: str,
        qkv_ticket_temperature: float,
        qkv_head_input_groups: int,
        attention_qk_mode: str,
        boolean_qk_width: int,
        boolean_qk_rounds: int,
        boolean_qk_bits_per_head: int,
        boolean_qk_correction_features: int,
        boolean_qk_correction_taps: str,
        attention_value_mode: str,
        attention_value_block_active_fraction: float,
        boolean_value_width: int,
        boolean_value_rounds: int,
        boolean_value_bits: int,
        boolean_value_correction_features: int,
        attention_output_mode: str,
        boolean_output_width: int,
        boolean_output_rounds: int,
        boolean_output_correction_features: int,
        boolean_output_correction_taps: str,
        boolean_output_routes: str,
        boolean_output_cell: str,
        attention_output_block_active_fraction: float,
        attention_output_codewords: int,
        attention_output_codebook_temperature: float,
        attention_output_codebook_response_shift: int,
        attention_output_head_local_rank: int,
        attention_output_squeeze_bitplanes: int,
        attention_output_squeeze_polarities: int,
        attention_output_squeeze_group_bits: int,
        attention_output_squeeze_bypass: int,
        attention_output_input_rounding: str,
        boolean_backward: str,
        boolean_temperature: float,
        boolean_gradient_leak: float,
        boolean_program_init_margin: float,
        boolean_register_bits: int,
        attention_normalizer: str,
        clipped_linear_initial_slope: float,
        head_contexts: Sequence[int],
        use_value_embedding: bool,
        dropout: float,
        zero_init_residual: bool,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.kv_heads = kv_heads
        self.kv_head_sharing = kv_head_sharing
        self.key_heads = kv_heads if kv_head_sharing in {"both", "key"} else heads
        self.value_heads = kv_heads if kv_head_sharing in {"both", "value"} else heads
        self.head_dim = dimension // heads
        self.qk_head_dim = (
            self.head_dim if qk_head_dimension == 0 else qk_head_dimension
        )
        self.local_qk_head_dim = (
            self.qk_head_dim
            if local_qk_head_dimension == 0
            else local_qk_head_dimension
        )
        self.context = context
        self.head_contexts = tuple(int(value) for value in head_contexts)
        if len(self.head_contexts) != heads:
            raise ValueError("one attention context is required per query head")
        self.head_qk_dims = tuple(
            self.local_qk_head_dim if value < context else self.qk_head_dim
            for value in self.head_contexts
        )
        self.rope_base = rope_base
        self.rope_fraction = rope_fraction
        self.partial_key_offset = partial_key_offset
        self.qk_norm = qk_norm
        self.qk_cache_precision = qk_cache_precision
        self.value_cache_precision = value_cache_precision
        self.kv_head_routing = kv_head_routing
        if kv_head_routing == "learned_diagonal":
            self.value_head_transform = nn.Parameter(
                torch.ones(heads, self.head_dim)
            )
        elif kv_head_routing == "learned_full":
            self.value_head_transform = nn.Parameter(
                torch.eye(self.head_dim).repeat(heads, 1, 1)
            )
        else:
            self.register_parameter("value_head_transform", None)
        self.attention_qkv_projection_mode = attention_qkv_projection_mode
        self.attention_qk_mode = attention_qk_mode
        self.boolean_qk_bits_per_head = boolean_qk_bits_per_head
        self.boolean_qk_correction_features = boolean_qk_correction_features
        self.boolean_qk_correction_dims_per_head = (
            self.head_dim if boolean_qk_correction_features > 0 else 0
        )
        self.attention_value_mode = attention_value_mode
        self.boolean_value_bits = (
            boolean_value_bits if boolean_value_bits > 0 else dimension
        )
        self.boolean_value_correction_dims = (
            dimension - self.boolean_value_bits
            if attention_value_mode == "boolean_lut3_corrected"
            else 0
        )
        self.attention_output_mode = attention_output_mode
        self.attention_normalizer = attention_normalizer
        if attention_qk_mode == "numerical":
            if attention_value_mode in {"numerical", "block_sparse_16x16"}:
                if attention_qkv_projection_mode == "signed_squeeze_hybrid":
                    self.qkv = SignedSqueezeHybridLinear(
                        dimension,
                        output_dimension=3 * dimension,
                        bitplanes=qkv_squeeze_bitplanes,
                        polarities=qkv_squeeze_polarities,
                        group_bits=qkv_squeeze_group_bits,
                        bypass_features=qkv_squeeze_bypass,
                        temperature=boolean_temperature,
                        input_rounding=qkv_squeeze_input_rounding,
                    )
                elif attention_qkv_projection_mode == "low_rank_w8a8":
                    self.qkv = LowRankW8A8Linear(
                        dimension, 3 * dimension, qkv_low_rank
                    )
                elif attention_qkv_projection_mode == "shared_qk_basis":
                    self.qkv = SharedBasisQKVLinear(
                        dimension,
                        self.heads,
                        self.qk_head_dim,
                        self.value_heads * self.head_dim,
                        qkv_shared_bases,
                        self.head_contexts,
                        context,
                    )
                elif attention_qkv_projection_mode == "local_global_qk":
                    self.qkv = LocalGlobalQKVLinear(
                        dimension,
                        self.heads,
                        self.qk_head_dim,
                        self.value_heads * self.head_dim,
                        qkv_local_width,
                        qkv_global_rank,
                    )
                elif attention_qkv_projection_mode == "cyclic_routes_qk":
                    self.qkv = CyclicRoutedQKVLinear(
                        dimension,
                        2 * self.heads * self.qk_head_dim,
                        self.value_heads * self.head_dim,
                        qkv_route_count,
                    )
                elif attention_qkv_projection_mode in {
                    "tied_qk", "head_subspace_qk", "head_subspace_global_qk"
                }:
                    self.qkv = HeadSubspaceQKVLinear(
                        dimension,
                        self.heads,
                        self.qk_head_dim,
                        self.value_heads * self.head_dim,
                        transformed=(
                            attention_qkv_projection_mode
                            in {"head_subspace_qk", "head_subspace_global_qk"}
                        ),
                        global_rank=(
                            qkv_global_rank
                            if attention_qkv_projection_mode
                            == "head_subspace_global_qk"
                            else 0
                        ),
                    )
                elif attention_qkv_projection_mode == "bitmix_trajectory_hybrid":
                    self.qkv = BitmixTrajectoryHybridLinear(
                        dimension,
                        3 * dimension,
                        bitplanes=qkv_squeeze_bitplanes,
                        rounds=qkv_bitmix_rounds,
                        correction_features=qkv_bitmix_correction_features,
                        bypass_features=qkv_squeeze_bypass,
                        cell=qkv_bitmix_cell,
                        backward_mode=boolean_backward,
                        temperature=boolean_temperature,
                        gradient_leak=boolean_gradient_leak,
                        program_init_margin=boolean_program_init_margin,
                        input_rounding=qkv_squeeze_input_rounding,
                    )
                elif attention_qkv_projection_mode == "block_sparse_16x16":
                    qkv_dimensions = (
                        self.heads * self.qk_head_dim,
                        self.key_heads * self.qk_head_dim,
                        self.value_heads * self.head_dim,
                    )
                    self.qkv = QKVBlockSparseLinear16(
                        dimension,
                        attention_value_block_active_fraction,
                        sum(qkv_dimensions),
                        qkv_dimensions,
                        qkv_block_allocation,
                        qkv_block_rows,
                        qkv_ticket_selection,
                        qkv_ticket_temperature,
                        (
                            self.head_qk_dims
                            if (
                                self.local_qk_head_dim != self.qk_head_dim
                                or qkv_head_input_groups
                            )
                            else None
                        ),
                        qkv_head_input_groups,
                    )
                else:
                    self.qkv = (
                        nn.Linear(
                            dimension,
                            (self.heads + self.key_heads) * self.qk_head_dim
                            + self.value_heads * self.head_dim,
                        bias=False,
                    )
                    if attention_value_mode == "numerical"
                    else ValueBlockSparseQKVLinear16(
                        dimension, attention_value_block_active_fraction
                    )
                    )
                self.register_module("qk_projection", None)
                self.register_module("boolean_value", None)
            else:
                self.register_module("qkv", None)
                self.qk_projection = nn.Linear(
                    dimension, 2 * self.heads * self.qk_head_dim, bias=False
                )
                self.boolean_value = BooleanLUT3Projection(
                    dimension=dimension,
                    width=boolean_value_width,
                    rounds=boolean_value_rounds,
                    output_bits=self.boolean_value_bits,
                    backward_mode=boolean_backward,
                    temperature=boolean_temperature,
                    gradient_leak=boolean_gradient_leak,
                    program_init_margin=boolean_program_init_margin,
                    register_bits=boolean_register_bits,
                    correction_features=boolean_value_correction_features,
                    correction_output_bits=self.boolean_value_correction_dims,
                )
            self.register_module("value_projection", None)
            self.register_module("boolean_query", None)
            self.register_module("boolean_key", None)
        else:
            self.register_module("qkv", None)
            self.register_module("qk_projection", None)
            self.register_module("boolean_value", None)
            self.value_projection = nn.Linear(
                dimension, self.value_heads * self.head_dim, bias=False
            )
            projection_arguments = {
                "dimension": dimension,
                "width": boolean_qk_width,
                "rounds": boolean_qk_rounds,
                "output_bits": heads * boolean_qk_bits_per_head,
                "backward_mode": boolean_backward,
                "temperature": boolean_temperature,
                "gradient_leak": boolean_gradient_leak,
                "program_init_margin": boolean_program_init_margin,
                "register_bits": boolean_register_bits,
                "correction_features": boolean_qk_correction_features,
                "correction_taps": boolean_qk_correction_taps,
                "correction_output_bits": (
                    dimension if boolean_qk_correction_features > 0 else 0
                ),
            }
            self.boolean_query = BooleanLUT3Projection(**projection_arguments)
            self.boolean_key = BooleanLUT3Projection(**projection_arguments)
        if attention_output_mode == "numerical":
            self.output = nn.Linear(dimension, dimension, bias=False)
            self.register_parameter("boolean_output_strength", None)
        elif attention_output_mode == "block_sparse_16x16":
            self.output = BlockSparseLinear16(
                dimension,
                attention_output_block_active_fraction,
                attention_output_input_rounding,
            )
            self.register_parameter("boolean_output_strength", None)
        elif attention_output_mode == "product_codebook_16":
            self.output = ProductCodebookLinear16(
                dimension,
                codewords=attention_output_codewords,
                temperature=attention_output_codebook_temperature,
                response_shift=attention_output_codebook_response_shift,
                input_rounding=attention_output_input_rounding,
            )
            self.register_parameter("boolean_output_strength", None)
        elif attention_output_mode == "head_local_hybrid":
            self.output = HeadLocalHybridLinear(
                dimension,
                heads,
                attention_output_head_local_rank,
                attention_output_input_rounding,
            )
            self.register_parameter("boolean_output_strength", None)
        elif attention_output_mode == "signed_squeeze_hybrid":
            self.output = SignedSqueezeHybridLinear(
                dimension,
                bitplanes=attention_output_squeeze_bitplanes,
                polarities=attention_output_squeeze_polarities,
                group_bits=attention_output_squeeze_group_bits,
                bypass_features=attention_output_squeeze_bypass,
                temperature=boolean_temperature,
            )
            self.register_parameter("boolean_output_strength", None)
        elif attention_output_mode == "boolean_lut3_direct":
            self.output = BooleanLUT3Projection(
                dimension=dimension,
                width=boolean_output_width,
                rounds=boolean_output_rounds,
                output_bits=dimension,
                backward_mode=boolean_backward,
                temperature=boolean_temperature,
                gradient_leak=boolean_gradient_leak,
                program_init_margin=boolean_program_init_margin,
                register_bits=boolean_register_bits,
            )
            self.boolean_output_strength = nn.Parameter(
                torch.tensor(0.0 if zero_init_residual else 1.0)
            )
        else:
            self.output = BooleanLUT3FFN(
                dimension,
                boolean_output_width,
                boolean_output_rounds,
                backward_mode=boolean_backward,
                temperature=boolean_temperature,
                gradient_leak=boolean_gradient_leak,
                program_init_margin=boolean_program_init_margin,
                register_bits=boolean_register_bits,
                block_gates=False,
                cell=boolean_output_cell,
                correction_features=boolean_output_correction_features,
                correction_taps=boolean_output_correction_taps,
                route_schedule=boolean_output_routes,
                input_rounding=attention_output_input_rounding,
            )
            self.register_parameter("boolean_output_strength", None)
        self.dropout = nn.Dropout(dropout)
        if attention_normalizer.startswith("clipped_"):
            self.clipped_linear_log_slope = nn.Parameter(
                torch.full((1, heads, 1, 1), math.log(clipped_linear_initial_slope))
            )
        else:
            self.register_parameter("clipped_linear_log_slope", None)
        if attention_qk_mode in {
            "boolean_hamming_calibrated",
            "boolean_hamming_grouped",
        }:
            scale_shape = (
                (heads, 4)
                if attention_qk_mode == "boolean_hamming_grouped"
                else (heads,)
            )
            self.boolean_qk_log_scale = nn.Parameter(torch.zeros(scale_shape))
            self.boolean_relative_bias = nn.Parameter(
                torch.tensor((0.0, -0.05, -0.15, -0.35)).repeat(heads, 1)
            )
        else:
            self.register_parameter("boolean_qk_log_scale", None)
            self.register_parameter("boolean_relative_bias", None)
        fixed_slopes = torch.linspace(0.0, 0.02, heads)
        self.register_buffer("boolean_fixed_distance_slopes", fixed_slopes, persistent=True)
        if use_value_embedding:
            if self.value_heads != heads:
                raise ValueError("value embeddings currently require full MHA")
            self.value_embedding_scale = nn.Parameter(torch.ones(heads))
        else:
            self.register_parameter("value_embedding_scale", None)
        if zero_init_residual and attention_output_mode == "numerical":
            nn.init.zeros_(self.output.weight)
        elif zero_init_residual and attention_output_mode == "boolean_lut3_decoded":
            nn.init.zeros_(self.output.decoder_weight)

    def _output(self, attended: Tensor) -> Tensor:
        projected = self.output(attended)
        if self.attention_output_mode == "boolean_lut3_direct":
            assert self.boolean_output_strength is not None
            return (2 * projected - 1) * self.boolean_output_strength
        return projected

    def _causal_mask(self, positions: Tensor) -> Tensor:
        relative = positions[:, None] - positions[None, :]
        contexts = torch.tensor(
            self.head_contexts, device=positions.device, dtype=relative.dtype
        )[None, :, None, None]
        return (relative[None, None] >= 0) & (relative[None, None] < contexts)

    def _cached_mask(self, key_length: int, device: torch.device) -> Tensor:
        distance = key_length - 1 - torch.arange(key_length, device=device)
        contexts = torch.tensor(
            self.head_contexts, device=device, dtype=distance.dtype
        )[None, :, None, None]
        return distance[None, None, None, :] < contexts

    def _normalize(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.attention_normalizer == "isa_exp2":
            return isa_softmax(scores, mask)
        if self.attention_normalizer == "exact":
            return exact_softmax(scores, mask)
        if self.clipped_linear_log_slope is None:
            raise RuntimeError("clipped-linear attention is missing its slopes")
        # Exponentiation is training-only parameterization. Native export folds it
        # into one positive scalar per head before inference.
        if self.attention_normalizer in {
            "clipped_dual_scale", "clipped_triple_scale"
        }:
            return clipped_multiscale_softmax(
                scores,
                self.clipped_linear_log_slope.exp(),
                (1, 2) if self.attention_normalizer == "clipped_dual_scale" else (1, 2, 4),
                mask,
                quantize_int8=self.value_cache_precision.startswith("int8_"),
            )
        return clipped_linear_softmax(
            scores,
            self.clipped_linear_log_slope.exp(),
            mask,
            power={
                "clipped_linear": 1,
                "clipped_squared": 2,
                "clipped_cubic": 3,
            }[self.attention_normalizer],
            quantize_int8=self.value_cache_precision.startswith("int8_"),
        )

    def _project(
        self,
        state: Tensor,
        positions: Tensor,
        auxiliary_value: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = state.shape
        if self.attention_qk_mode == "numerical":
            if self.qkv is not None:
                query, key, value = self.qkv(state).split(
                    (
                        self.heads * self.qk_head_dim,
                        self.key_heads * self.qk_head_dim,
                        self.value_heads * self.head_dim,
                    ),
                    dim=-1,
                )
            else:
                assert self.qk_projection is not None
                assert self.boolean_value is not None
                query, key = self.qk_projection(state).chunk(2, dim=-1)
                value = self.boolean_value(state)
                if self.attention_value_mode == "boolean_lut3_bipolar":
                    value = 2 * value - 1
                elif self.attention_value_mode == "boolean_lut3_corrected":
                    boolean = value[..., : self.boolean_value_bits]
                    correction = value[..., self.boolean_value_bits :]
                    boolean = (2 * boolean - 1).view(
                        batch,
                        length,
                        self.heads,
                        self.boolean_value_bits // self.heads,
                    )
                    correction = correction.view(
                        batch,
                        length,
                        self.heads,
                        self.boolean_value_correction_dims // self.heads,
                    )
                    value = torch.cat((boolean, correction), dim=-1).flatten(-2)
            query = query.view(
                batch, length, self.heads, self.qk_head_dim
            ).transpose(1, 2)
            key = key.view(
                batch, length, self.key_heads, self.qk_head_dim
            ).transpose(1, 2)
        else:
            assert self.boolean_query is not None
            assert self.boolean_key is not None
            assert self.value_projection is not None
            qk_dimensions = (
                self.boolean_qk_bits_per_head
                + self.boolean_qk_correction_dims_per_head
            )
            query = self.boolean_query(state).view(
                batch, length, self.heads, qk_dimensions
            ).transpose(1, 2)
            key = self.boolean_key(state).view(
                batch, length, self.heads, qk_dimensions
            ).transpose(1, 2)
            value = self.value_projection(state)
        value = value.view(batch, length, self.value_heads, self.head_dim).transpose(1, 2)
        if auxiliary_value is not None:
            if self.value_embedding_scale is None:
                raise RuntimeError("auxiliary value supplied to an unconfigured layer")
            auxiliary = auxiliary_value.view(
                batch, length, self.heads, self.head_dim
            ).transpose(1, 2)
            value = value + auxiliary * self.value_embedding_scale[None, :, None, None]
        value = self._quantize_value_cache(value)
        if self.attention_qk_mode != "numerical":
            return query, key, value
        if self.qk_norm:
            if self.local_qk_head_dim == self.qk_head_dim:
                query = F.rms_norm(query, (self.qk_head_dim,), eps=1e-6)
                key = F.rms_norm(key, (self.qk_head_dim,), eps=1e-6)
                query = _apply_rope(
                    query, positions, self.rope_base, self.rope_fraction
                )
                key = _apply_rope(
                    key, positions, self.rope_base, self.rope_fraction
                )
                query, key = self._quantize_qk_cache(query, key)
                return query, key, value
            if self.key_heads != self.heads:
                raise RuntimeError("mixed Q/K width requires independent keys")
            normalized_query = []
            normalized_key = []
            for head, active_dimension in enumerate(self.head_qk_dims):
                head_query = query[:, head, :, :active_dimension]
                head_key = key[:, head, :, :active_dimension]
                head_query = F.rms_norm(
                    head_query, (active_dimension,), eps=1e-6
                )
                head_key = F.rms_norm(head_key, (active_dimension,), eps=1e-6)
                head_query = _apply_rope(
                    head_query[:, None], positions, self.rope_base,
                    self.rope_fraction,
                )[:, 0]
                head_key = _apply_rope(
                    head_key[:, None], positions, self.rope_base,
                    self.rope_fraction,
                )[:, 0]
                normalized_query.append(
                    F.pad(head_query, (0, self.qk_head_dim - active_dimension))
                )
                normalized_key.append(
                    F.pad(head_key, (0, self.qk_head_dim - active_dimension))
                )
            query = torch.stack(normalized_query, dim=1)
            key = torch.stack(normalized_key, dim=1)
            query, key = self._quantize_qk_cache(query, key)
            return query, key, value
        query = _apply_rope(query, positions, self.rope_base, self.rope_fraction)
        key = _apply_rope(key, positions, self.rope_base, self.rope_fraction)
        query, key = self._quantize_qk_cache(query, key)
        return query, key, value

    def _quantize_qk_cache(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        if self.qk_cache_precision == "fp32":
            return query, key
        if self.qk_cache_precision in {"int8_fixed4", "int8_fixed4_fp16"}:
            if self.qk_cache_precision == "int8_fixed4_fp16":
                rounded_query = query.to(torch.float16).to(query.dtype)
                rounded_key = key.to(torch.float16).to(key.dtype)
                query = query + (rounded_query - query).detach()
                key = key + (rounded_key - key).detach()
            scale = 4.0 / 127.0
            hard_query = torch.round(query / scale).clamp(-127, 127) * scale
            hard_key = torch.round(key / scale).clamp(-127, 127) * scale
        elif self.qk_cache_precision == "int8_dynamic":
            query_scale = query.detach().abs().amax(dim=-1, keepdim=True).clamp_min(
                1.0e-12
            ) / 127.0
            key_scale = key.detach().abs().amax(dim=-1, keepdim=True).clamp_min(
                1.0e-12
            ) / 127.0
            hard_query = torch.round(query / query_scale).clamp(-127, 127) * query_scale
            hard_key = torch.round(key / key_scale).clamp(-127, 127) * key_scale
        else:
            raise ValueError("invalid Q/K cache precision")
        return (
            query + (hard_query - query).detach(),
            key + (hard_key - key).detach(),
        )

    def _quantize_value_cache(self, value: Tensor) -> Tensor:
        if self.value_cache_precision == "fp32":
            return value
        if self.value_cache_precision == "fp16":
            hard_value = value.to(torch.float16).to(value.dtype)
            return value + (hard_value - value).detach()
        if not self.value_cache_precision.startswith("int8_weight_rms_"):
            raise ValueError("invalid value-cache precision")
        if self.qkv is None or not hasattr(self.qkv, "weight"):
            raise RuntimeError(
                "fixed INT8 value-cache calibration requires a numerical QKV weight"
            )
        value_rows = self.value_heads * self.head_dim
        weight = self.qkv.weight[-value_rows:]
        if hasattr(self.qkv, "expanded_mask"):
            weight = weight * self.qkv.expanded_mask()[-value_rows:]
        scales = (
            4.0
            * weight.detach().float().square().sum(dim=-1).sqrt()
            / 127.0
        ).clamp_min(1.0E-12)
        if self.value_cache_precision == "int8_weight_rms_scalar":
            scales = scales.amax().expand_as(scales)
        scales = scales.reshape(1, self.value_heads, 1, self.head_dim).to(value)
        integer = value / scales
        hard_integer = integer.round().clamp(-127, 127)
        hard_value = hard_integer * scales
        return value + (hard_value - value).detach()

    def _scores(
        self,
        query: Tensor,
        key: Tensor,
        query_positions: Tensor,
        key_positions: Tensor,
    ) -> Tensor:
        if self.attention_qk_mode == "numerical":
            scales = query.new_tensor(
                [1.0 / math.sqrt(value) for value in self.head_qk_dims]
            )[None, :, None, None]
            if self.qk_cache_precision in {"int8_fixed4", "int8_fixed4_fp16"}:
                cache_scale = 4.0 / 127.0
                query_integer = query / cache_scale
                key_integer = key / cache_scale
                hard_query = torch.round(query_integer).clamp(-127, 127)
                hard_key = torch.round(key_integer).clamp(-127, 127)
                query_integer = query_integer + (
                    hard_query - query_integer
                ).detach()
                key_integer = key_integer + (hard_key - key_integer).detach()
                return (query_integer @ key_integer.transpose(-2, -1)) * (
                    cache_scale * cache_scale * scales
                )
            return (query @ key.transpose(-2, -1)) * scales
        query_bits = query[..., : self.boolean_qk_bits_per_head]
        key_bits = key[..., : self.boolean_qk_bits_per_head]
        different = residual_xor(
            query_bits.unsqueeze(-2),
            key_bits.unsqueeze(-3),
            identity_backward=False,
        )
        if self.attention_qk_mode == "boolean_hamming_grouped":
            grouped_mismatch = different.view(
                *different.shape[:-1], 4, self.boolean_qk_bits_per_head // 4
            ).sum(-1)
            grouped_centered = self.boolean_qk_bits_per_head // 4 - 2 * grouped_mismatch
            assert self.boolean_qk_log_scale is not None
            scores = (
                grouped_centered
                * self.boolean_qk_log_scale.exp()[None, :, None, None, :]
            ).sum(-1) / math.sqrt(self.boolean_qk_bits_per_head)
        else:
            mismatch = different.sum(-1)
            centered = self.boolean_qk_bits_per_head - 2 * mismatch
            if self.boolean_qk_log_scale is None:
                scale = 1.0 / math.sqrt(self.boolean_qk_bits_per_head)
            else:
                scale = (
                    self.boolean_qk_log_scale.exp()
                    / math.sqrt(self.boolean_qk_bits_per_head)
                )[None, :, None, None]
            scores = centered * scale
        if self.boolean_qk_correction_dims_per_head > 0:
            query_correction = query[..., self.boolean_qk_bits_per_head :]
            key_correction = key[..., self.boolean_qk_bits_per_head :]
            scores = scores + (
                query_correction @ key_correction.transpose(-2, -1)
                / math.sqrt(self.boolean_qk_correction_dims_per_head)
            )
        distance = (query_positions[:, None] - key_positions[None, :]).clamp_min(0)
        if self.boolean_relative_bias is None:
            return scores - (
                self.boolean_fixed_distance_slopes[None, :, None, None]
                * distance[None, None]
            )
        buckets = (
            (distance > 8).to(torch.long)
            + (distance > 32).to(torch.long)
            + (distance > 64).to(torch.long)
        )
        bias = self.boolean_relative_bias[:, buckets]
        return scores + bias[None]

    def _expand(self, state: Tensor, *, value: bool = False) -> Tensor:
        source_heads = state.shape[1]
        if source_heads == self.heads:
            return state
        expanded = state.repeat_interleave(self.heads // source_heads, dim=1)
        if self.kv_head_routing in {
            "repeat", "learned_diagonal", "learned_full"
        }:
            return expanded
        stride = 5 if value else 7
        route_dimension = self.head_dim if value else self.qk_head_dim
        return torch.stack(
            [
                torch.roll(
                    expanded[:, head], (head * stride) % route_dimension, dims=-1
                )
                for head in range(self.heads)
            ],
            dim=1,
        )

    def _route_attended_value(self, attended: Tensor) -> Tensor:
        if self.kv_head_routing == "learned_diagonal":
            assert self.value_head_transform is not None
            return attended * self.value_head_transform[None, :, None, :]
        if self.kv_head_routing == "learned_full":
            assert self.value_head_transform is not None
            hard_input = attended.to(torch.float16).to(attended.dtype)
            routed_input = attended + (hard_input - attended).detach()
            hard_weight = self.value_head_transform.to(torch.float16).to(
                self.value_head_transform.dtype
            )
            routed_weight = self.value_head_transform + (
                hard_weight - self.value_head_transform
            ).detach()
            return torch.einsum(
                "bhld,hde->bhle", routed_input, routed_weight
            )
        return attended

    def _offset_key(self, key: Tensor) -> Tensor:
        if self.partial_key_offset == "none" or key.shape[2] <= 1:
            return key
        quarter = self.qk_head_dim // 4
        stationary = torch.zeros(
            self.qk_head_dim, dtype=torch.bool, device=key.device
        )
        stationary[quarter : 2 * quarter] = True
        stationary[3 * quarter :] = True
        selected_heads = torch.ones(self.heads, dtype=torch.bool, device=key.device)
        if self.partial_key_offset == "global_heads":
            selected_heads = torch.tensor(
                [value == self.context for value in self.head_contexts],
                dtype=torch.bool,
                device=key.device,
            )
        replace = selected_heads[:, None] & stationary[None, :]
        previous = torch.cat((key[:, :, :1], key[:, :, :-1]), dim=2)
        return torch.where(replace[None, :, None, :], previous, key)

    def forward(
        self,
        state: Tensor,
        positions: Tensor,
        auxiliary_value: Tensor | None = None,
    ) -> Tensor:
        output, _ = self.forward_with_shared_qk(
            state, positions, auxiliary_value, None, "none"
        )
        return output

    def forward_with_shared_qk(
        self,
        state: Tensor,
        positions: Tensor,
        auxiliary_value: Tensor | None,
        shared_qk: tuple[Tensor, Tensor] | None,
        shared_qk_mode: str,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        query, key, value = self._project(state, positions, auxiliary_value)
        projected_qk = (query, key)
        if shared_qk is not None:
            if shared_qk_mode == "qk":
                query = shared_qk[0]
            if shared_qk_mode in {"key", "qk"}:
                key = shared_qk[1]
            else:
                raise ValueError("shared Q/K tensors require key or qk mode")
        key = self._offset_key(self._expand(key))
        value = self._expand(value, value=True)
        scores = self._scores(query, key, positions, positions)
        mask = self._causal_mask(positions)
        weights = self.dropout(self._normalize(scores, mask))
        attended = self._route_attended_value(weights @ value)
        attended = attended.transpose(1, 2).contiguous().flatten(-2)
        return self._output(attended), projected_qk

    def forward_cached(
        self,
        state: Tensor,
        positions: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        auxiliary_value: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        output, updated, _ = self.forward_cached_with_shared_qk(
            state, positions, cache, auxiliary_value, None, "none"
        )
        return output, updated

    def forward_cached_with_shared_qk(
        self,
        state: Tensor,
        positions: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        auxiliary_value: Tensor | None,
        shared_qk: tuple[Tensor, Tensor] | None,
        shared_qk_mode: str,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        query, key, value = self._project(state, positions, auxiliary_value)
        projected_qk = (query, key)
        if shared_qk is not None:
            if shared_qk_mode == "qk":
                query = shared_qk[0]
            if shared_qk_mode in {"key", "qk"}:
                key = shared_qk[1]
            else:
                raise ValueError("shared Q/K tensors require key or qk mode")
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        key = key[:, :, -self.context :]
        value = value[:, :, -self.context :]
        expanded_key = self._offset_key(self._expand(key))
        key_positions = positions[-1] - torch.arange(
            expanded_key.shape[2] - 1,
            -1,
            -1,
            device=positions.device,
        )
        scores = self._scores(query, expanded_key, positions, key_positions)
        weights = self._normalize(scores, self._cached_mask(key.shape[2], key.device))
        attended = self._route_attended_value(
            weights @ self._expand(value, value=True)
        ).transpose(1, 2).contiguous().flatten(-2)
        return self._output(attended), (key, value), projected_qk


class BooleanLUT3FFN(nn.Module):
    """Hard-forward ISA Boolean Transformer FFN with fixed M4 routes."""

    def __init__(
        self,
        dimension: int,
        width: int,
        rounds: int,
        *,
        backward_mode: str,
        temperature: float,
        gradient_leak: float,
        program_init_margin: float,
        register_bits: int,
        block_gates: bool,
        cell: str = "lut3",
        correction_features: int = 0,
        correction_rank: int = 0,
        correction_mode: str = "dense",
        correction_block_active_fraction: float = 1.0,
        correction_taps: str = "final",
        pool_schedule: str = "legacy_mixed",
        route_schedule: str = "byte_lane",
        threshold_schedule: str = "linear",
        input_rounding: str = "none",
    ) -> None:
        super().__init__()
        if width % dimension or width % register_bits:
            raise ValueError("Boolean width must divide into dimensions and registers")
        self.dimension = dimension
        self.width = width
        self.rounds = rounds
        self.bitplanes = width // dimension
        self.backward_mode = backward_mode
        self.temperature = temperature
        self.gradient_leak = gradient_leak
        self.register_bits = register_bits
        self.cell = cell
        self.correction_features = correction_features
        self.correction_rank = correction_rank
        self.correction_mode = correction_mode
        self.correction_block_active_fraction = correction_block_active_fraction
        self.correction_taps = correction_taps
        self.pool_schedule = pool_schedule
        self.route_schedule = route_schedule
        self.threshold_schedule = threshold_schedule
        self.input_rounding = input_rounding
        self.encoder_offsets = nn.Parameter(torch.zeros(dimension))
        training_planes = (
            9 if cell == "factorized_minterm"
            else 2 if cell == "dual3"
            else 4 if cell == "xar_lut2"
            else 8
        )
        self.program_logits = nn.Parameter(
            torch.empty(rounds, training_planes, width)
        )
        if cell == "factorized_minterm":
            nn.init.normal_(
                self.program_logits[:, 0], mean=-program_init_margin, std=0.01
            )
            nn.init.normal_(self.program_logits[:, 1:], mean=0.0, std=0.01)
            polarities = torch.tensor(
                [
                    [(choice >> 2) & 1, (choice >> 1) & 1, choice & 1]
                    for choice in range(8)
                ],
                dtype=torch.float32,
            )
            self.register_buffer("factorized_polarities", polarities, persistent=True)
        else:
            nn.init.normal_(
                self.program_logits, mean=-program_init_margin, std=0.01
            )
            self.register_buffer("factorized_polarities", None, persistent=True)
        self.decoder_weight = nn.Parameter(torch.empty(dimension, self.bitplanes))
        nn.init.normal_(self.decoder_weight, mean=0.0, std=0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(dimension))
        if correction_features > 0:
            if correction_mode.startswith("product_codebook_"):
                self.register_parameter("correction_weight", None)
                self.register_parameter("correction_input_weight", None)
                self.register_parameter("correction_output_weight", None)
                self.correction_codebook = ProductCodebookLinear16(
                    correction_features,
                    output_dimension=dimension,
                    codewords=int(correction_mode.rsplit("_", 1)[1]),
                    temperature=0.25,
                    response_shift=10,
                    input_rounding="none",
                )
            elif correction_rank:
                self.register_parameter("correction_weight", None)
                self.correction_input_weight = nn.Parameter(
                    torch.zeros(correction_rank, correction_features)
                )
                self.correction_output_weight = nn.Parameter(
                    torch.zeros(dimension, correction_rank)
                )
                self.correction_codebook = None
            else:
                self.correction_weight = nn.Parameter(
                    torch.zeros(dimension, correction_features)
                )
                self.register_parameter("correction_input_weight", None)
                self.register_parameter("correction_output_weight", None)
                self.correction_codebook = None
            if correction_taps == "equal_trajectory":
                if correction_features % (rounds + 1):
                    raise ValueError("equal trajectory features must divide across states")
                tap_indices = torch.arange(correction_features).div(
                    correction_features // (rounds + 1), rounding_mode="floor"
                )
            elif correction_taps == "late_trajectory":
                if rounds < 2 or correction_features % (2 * (rounds - 1)):
                    raise ValueError("late trajectory features do not divide across rounds")
                early_features = correction_features // 2
                tap_indices = torch.cat(
                    (
                        1 + torch.arange(early_features).div(
                            early_features // (rounds - 1), rounding_mode="floor"
                        ),
                        torch.full((correction_features - early_features,), rounds),
                    )
                )
            else:
                tap_indices = torch.full((correction_features,), rounds)
            groups = torch.arange(correction_features)
            if pool_schedule == "legacy_mixed":
                starts = torch.where(
                    groups < width // 16,
                    groups * 16,
                    (groups * 37) % width,
                )
            elif pool_schedule in {"state_aligned", "state_rotated"}:
                if correction_taps != "equal_trajectory":
                    raise ValueError(
                        "state-aligned FFN pools require equal trajectory taps"
                    )
                features_per_state = correction_features // (rounds + 1)
                local_groups = groups.remainder(features_per_state)
                block_count = width // 16
                offsets = (
                    torch.zeros_like(tap_indices)
                    if pool_schedule == "state_aligned"
                    else tap_indices * (block_count // (rounds + 1))
                )
                starts = (local_groups + offsets).remainder(block_count) * 16
            else:
                raise ValueError("invalid Boolean FFN correction pool schedule")
            self.register_buffer(
                "correction_indices",
                (starts[:, None] + torch.arange(16)[None]).remainder(width).flatten(),
                persistent=True,
            )
            self.register_buffer("correction_tap_indices", tap_indices, persistent=True)
        else:
            self.register_parameter("correction_weight", None)
            self.register_parameter("correction_input_weight", None)
            self.register_parameter("correction_output_weight", None)
            self.correction_codebook = None
            self.register_buffer("correction_indices", None, persistent=True)
            self.register_buffer("correction_tap_indices", None, persistent=True)
        if correction_block_active_fraction < 1.0:
            if correction_rank or dimension % 16 or correction_features % 16:
                raise ValueError(
                    "sparse FFN correction requires 16-aligned direct correction weights"
                )
            row_blocks = dimension // 16
            column_blocks = correction_features // 16
            active = max(
                1,
                round(
                    row_blocks
                    * column_blocks
                    * correction_block_active_fraction
                ),
            )
            mask = torch.zeros(row_blocks * column_blocks)
            mask[:active] = 1
            self.register_buffer(
                "correction_block_mask",
                mask.reshape(row_blocks, column_blocks),
                persistent=True,
            )
        else:
            self.register_buffer("correction_block_mask", None, persistent=True)
        if block_gates:
            self.block_gate_logits = nn.Parameter(
                torch.full((rounds, width // register_bits), 0.1)
            )
        else:
            self.register_parameter("block_gate_logits", None)

        if threshold_schedule == "linear":
            thresholds = torch.linspace(-1.75, 1.75, self.bitplanes)
        elif threshold_schedule == "gaussian_quantile":
            probabilities = (
                torch.arange(self.bitplanes, dtype=torch.float32) + 1
            ) / (self.bitplanes + 1)
            thresholds = math.sqrt(2.0) * torch.erfinv(2 * probabilities - 1)
        elif threshold_schedule == "power2" and self.bitplanes == 8:
            thresholds = torch.tensor((-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0))
        else:
            raise ValueError("invalid Boolean FFN threshold schedule")
        self.register_buffer("encoder_thresholds", thresholds, persistent=True)
        if route_schedule not in {"byte_lane", "register_mix", "bitmix", "xar"}:
            raise ValueError("invalid Boolean FFN route schedule")
        register_count = width // register_bits
        butterfly_stages = max(1, int(math.log2(register_count)))
        routes = []
        for round_index in range(rounds):
            shift = 1 << (round_index % 5)
            third_route = (
                RouteSpec("lane64_rotate", shift)
                if route_schedule == "byte_lane"
                else RouteSpec("register_rotate", shift)
                if route_schedule == "register_mix"
                else RouteSpec(
                    "register_rotate",
                    1 << (round_index % butterfly_stages),
                )
                if route_schedule == "xar"
                else RouteSpec("bit_rotate", shift)
            )
            routes.append(
                torch.stack(
                    (
                        route_indices(width, RouteSpec("identity"), register_bits),
                        route_indices(
                            width,
                            RouteSpec("byte_rotate", shift),
                            register_bits,
                        ),
                        route_indices(width, third_route, register_bits),
                    )
                )
            )
        self.register_buffer("route_indices", torch.stack(routes), persistent=True)
        self.register_buffer(
            "xar_rotation_indices",
            torch.stack(
                [
                    route_indices(
                        width,
                        RouteSpec("lane64_rotate", 1 << (index % 5)),
                        register_bits,
                    )
                    for index in range(rounds)
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "carrier_route_indices",
            route_indices(width, RouteSpec("byte_rotate", 1), register_bits),
            persistent=True,
        )
        carrier_planes = torch.arange(width).remainder(self.bitplanes)
        self.register_buffer(
            "carrier_bsl25_mask",
            (carrier_planes >= (self.bitplanes - 2) // 2)
            & (carrier_planes < (self.bitplanes - 2) // 2 + 2),
            persistent=False,
        )
        self.register_buffer(
            "carrier_bsl50_mask",
            (carrier_planes >= (self.bitplanes - 4) // 2)
            & (carrier_planes < (self.bitplanes - 4) // 2 + 4),
            persistent=False,
        )
        self._regularization: Tensor | None = None
        self._gate_regularization: Tensor | None = None
        self._last_diagnostics: dict[str, Tensor | float] = {}

    def _program(self, round_index: int, state: Tensor) -> Tensor:
        if self.cell == "factorized_minterm":
            assert self.factorized_polarities is not None
            enable = hard_bit(
                self.program_logits[round_index, 0], self.temperature
            )
            scores = self.program_logits[round_index, 1:]
            probability = torch.softmax(scores / self.temperature, dim=0)
            hard = F.one_hot(scores.argmax(dim=0), num_classes=8).T.to(scores.dtype)
            selection = hard.detach() - probability.detach() + probability
            polarity = torch.einsum(
                "kw,kp->pw", selection, self.factorized_polarities
            )
            bits = torch.cat((enable.unsqueeze(0), polarity), dim=0)
        else:
            bits = hard_bit(self.program_logits[round_index], self.temperature)
        return bits.view(bits.shape[0], *([1] * (state.ndim - 1)), self.width)

    def _cell(
        self, round_index: int, a: Tensor, b: Tensor, c: Tensor, program: Tensor
    ) -> Tensor:
        arguments = (
            self.backward_mode,
            self.temperature,
            self.gradient_leak,
        )
        if self.cell == "lut3":
            return lut3(a, b, c, program, *arguments)
        if self.cell == "factorized_minterm":
            return minterm(a, b, c, program, *arguments)
        if self.cell == "dual3":
            nonlinear_a = bcax(a, b, c, *arguments)
            nonlinear_b = mux(a, b, c, *arguments)
            selected = mux(program[0], nonlinear_a, nonlinear_b, *arguments)
            return eor3(
                selected, program[1], torch.zeros_like(a), *arguments
            )
        if self.cell == "xar_lut2":
            mixed = eor3(a, b, torch.zeros_like(a), *arguments)
            mixed = mixed.index_select(-1, self.xar_rotation_indices[round_index])
            return lut2(mixed, c, program, *arguments)
        raise AssertionError(f"unhandled Boolean cell: {self.cell}")

    def _encode(self, state: Tensor) -> Tensor:
        if self.input_rounding == "fp16":
            state = state.to(torch.float16).to(state.dtype)
        elif self.input_rounding == "fixed8":
            state = torch.round(state * 256.0) / 256.0
        elif self.input_rounding != "none":
            raise ValueError("invalid Boolean FFN input rounding")
        return hard_bit(
            state.unsqueeze(-1)
            - self.encoder_offsets.view(1, 1, self.dimension, 1)
            - self.encoder_thresholds.view(1, 1, 1, self.bitplanes),
            self.temperature,
        ).flatten(-2)

    def _transform(
        self, boolean_state: Tensor, *, capture_states: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        states = [boolean_state] if capture_states else []
        occupancies = [boolean_state.mean()]
        branch_occupancies: list[Tensor] = []
        gate_probabilities: list[Tensor] = []
        for round_index in range(self.rounds):
            a, b, c = (
                boolean_state.index_select(
                    -1, self.route_indices[round_index, route_index]
                )
                for route_index in range(3)
            )
            branch = self._cell(
                round_index, a, b, c, self._program(round_index, boolean_state)
            )
            if self.block_gate_logits is not None:
                gates = hard_bit(
                    self.block_gate_logits[round_index], self.temperature
                )
                branch = branch * gates.repeat_interleave(self.register_bits).view(
                    1, 1, self.width
                )
                gate_probabilities.append(
                    torch.sigmoid(
                        self.block_gate_logits[round_index] / self.temperature
                    ).mean()
                )
            boolean_state = residual_xor(boolean_state, branch)
            if capture_states:
                states.append(boolean_state)
            branch_occupancies.append(branch.mean())
            occupancies.append(boolean_state.mean())

        occupancy_tensor = torch.stack(occupancies)
        self._regularization = (occupancy_tensor - 0.5).square().mean()
        self._gate_regularization = (
            torch.stack(gate_probabilities).mean()
            if gate_probabilities
            else boolean_state.new_zeros(())
        )
        discrete_program = self.discrete_program().detach()
        noop_fraction = (
            (discrete_program[:, 0] == 0).float().mean()
            if self.cell == "factorized_minterm"
            else discrete_program.new_zeros((), dtype=torch.float32)
            if self.cell == "dual3"
            else (discrete_program.sum(dim=1) == 0).float().mean()
        )
        self._last_diagnostics = {
            "encoded_occupancy": occupancies[0].detach(),
            "final_occupancy": occupancies[-1].detach(),
            "final_balance_error": (occupancies[-1] - 0.5).abs().detach(),
            "mean_branch_occupancy": torch.stack(branch_occupancies).mean().detach(),
            "active_program_fraction": discrete_program.float().mean(),
            "noop_program_fraction": noop_fraction,
            "active_hardware_block_fraction": (
                1.0
                if self.block_gate_logits is None
                else (self.block_gate_logits.detach() >= 0).float().mean()
            ),
        }
        return (boolean_state, states) if capture_states else boolean_state

    def _decode(
        self, boolean_state: Tensor, correction_states: Sequence[Tensor] | None = None
    ) -> Tensor:
        signed = 2 * boolean_state.reshape(
            *boolean_state.shape[:-1], self.dimension, self.bitplanes
        ) - 1
        decoded = (signed * self.decoder_weight).sum(dim=-1)
        if (
            self.correction_weight is not None
            or self.correction_input_weight is not None
            or self.correction_codebook is not None
        ):
            assert self.correction_indices is not None
            if correction_states is None or self.correction_taps == "final":
                selected = boolean_state.index_select(-1, self.correction_indices)
            else:
                assert self.correction_tap_indices is not None
                combined = torch.cat(tuple(correction_states), dim=-1)
                routed_indices = self.correction_indices.reshape(
                    self.correction_features, 16
                ) + self.correction_tap_indices[:, None] * self.width
                selected = combined.index_select(-1, routed_indices.flatten())
            pooled = (2 * selected - 1).reshape(
                *selected.shape[:-1], self.correction_features, 16
            ).sum(dim=-1) / 4.0
            if self.correction_codebook is not None:
                correction = self.correction_codebook(pooled)
            elif self.correction_weight is not None:
                correction_weight = self.correction_weight
                if self.correction_block_mask is not None:
                    correction_weight = correction_weight * (
                        self.correction_block_mask.repeat_interleave(16, 0)
                        .repeat_interleave(16, 1)
                    )
                correction = pooled @ correction_weight.T
            else:
                assert self.correction_input_weight is not None
                assert self.correction_output_weight is not None
                hidden = fake_w8_linear(pooled, self.correction_input_weight)
                hard_hidden = hidden.to(torch.float16).to(hidden.dtype)
                hidden = hidden + (hard_hidden - hidden).detach()
                hard_output = self.correction_output_weight.to(torch.float16).to(
                    self.correction_output_weight.dtype
                )
                output_weight = self.correction_output_weight + (
                    hard_output - self.correction_output_weight
                ).detach()
                correction = F.linear(hidden, output_weight)
            decoded = decoded + correction
        return decoded + self.decoder_bias

    def forward(self, state: Tensor) -> Tensor:
        encoded = self._encode(state)
        if self.correction_taps == "final":
            return self._decode(self._transform(encoded))
        updated, states = self._transform(encoded, capture_states=True)
        return self._decode(updated, states)

    def forward_with_carrier(
        self,
        state: Tensor,
        carrier: Tensor | None,
        mode: str,
    ) -> tuple[Tensor, Tensor]:
        """Apply the FFN while carrying its packed state between blocks.

        The numerical residual remains available to attention during this
        bounded experiment. Only the Boolean FFN state crosses the layer
        boundary, allowing the persistent representation to be evaluated
        before removing numerical residual storage.
        """
        encoded = self._encode(state)
        if carrier is not None:
            if carrier.shape != encoded.shape:
                raise ValueError("Boolean carrier shape does not match encoded state")
            if mode == "persistent_xor":
                encoded = residual_xor(encoded, carrier)
            elif mode == "persistent_bcax":
                mask = encoded.index_select(-1, self.carrier_route_indices)
                encoded = bcax(
                    encoded,
                    carrier,
                    mask,
                    self.backward_mode,
                    self.temperature,
                    self.gradient_leak,
                )
            elif mode in {"persistent_bsl25", "persistent_bsl50"}:
                mask = (
                    self.carrier_bsl25_mask
                    if mode == "persistent_bsl25"
                    else self.carrier_bsl50_mask
                )
                encoded = torch.where(mask, carrier, encoded)
            else:
                raise ValueError(f"invalid Boolean carrier mode: {mode}")
        if self.correction_taps == "final":
            updated = self._transform(encoded)
            return self._decode(updated), updated
        updated, states = self._transform(encoded, capture_states=True)
        return self._decode(updated, states), updated

    def regularization(self) -> tuple[Tensor, Tensor]:
        if self._regularization is None or self._gate_regularization is None:
            zero = self.program_logits.new_zeros(())
            return zero, zero
        return self._regularization, self._gate_regularization

    def diagnostics(self) -> dict[str, float]:
        result = {
            name: float(value)
            for name, value in self._last_diagnostics.items()
        }

        def record_gradient_family(name: str, parameter: Tensor | None) -> None:
            gradient = None if parameter is None else parameter.grad
            result[f"{name}_gradient_nonzero_fraction"] = (
                0.0 if gradient is None else float((gradient != 0).float().mean())
            )
            result[f"{name}_gradient_mean_abs"] = (
                0.0 if gradient is None else float(gradient.abs().mean())
            )

        record_gradient_family("program", self.program_logits)
        record_gradient_family("gate", self.block_gate_logits)
        record_gradient_family("encoder", self.encoder_offsets)
        record_gradient_family("decoder", self.decoder_weight)
        record_gradient_family("correction", self.correction_weight)
        record_gradient_family("correction_input", self.correction_input_weight)
        record_gradient_family("correction_output", self.correction_output_weight)
        if self.correction_codebook is not None:
            record_gradient_family(
                "correction_codebook", self.correction_codebook.codebook
            )
            record_gradient_family(
                "correction_code_index", self.correction_codebook.code_index_logits
            )
        return result

    def initialize_correction_from_dense(self, weight: Tensor) -> None:
        if tuple(weight.shape) != (self.dimension, self.correction_features):
            raise ValueError("dense FFN correction source shape mismatch")
        with torch.no_grad():
            if self.correction_codebook is not None:
                self.correction_codebook.initialize_from_dense(weight)
                return
            if self.correction_weight is not None:
                self.correction_weight.copy_(weight)
                self.initialize_correction_block_mask_from_dense(weight)
                return
            if self.correction_input_weight is None or self.correction_output_weight is None:
                raise ValueError("FFN correction is disabled")
            left, singular, right = torch.linalg.svd(
                weight.detach().float(), full_matrices=False
            )
            root = singular[: self.correction_rank].sqrt()
            self.correction_input_weight.copy_(
                root[:, None] * right[: self.correction_rank]
            )
            self.correction_output_weight.copy_(
                left[:, : self.correction_rank] * root[None, :]
            )

    def initialize_correction_block_mask_from_dense(self, weight: Tensor) -> None:
        if self.correction_block_mask is None:
            return
        if tuple(weight.shape) != (self.dimension, self.correction_features):
            raise ValueError("dense FFN correction block source shape mismatch")
        row_blocks, column_blocks = self.correction_block_mask.shape
        scores = weight.detach().reshape(
            row_blocks, 16, column_blocks, 16
        ).square().sum(dim=(1, 3))
        active = max(
            1,
            round(
                scores.numel() * self.correction_block_active_fraction
            ),
        )
        mask = torch.zeros_like(scores).flatten()
        mask[scores.flatten().topk(active).indices] = 1
        with torch.no_grad():
            self.correction_block_mask.copy_(mask.reshape_as(scores))

    def correction_active_weight_values(self) -> int:
        if self.correction_codebook is not None:
            return (
                self.correction_codebook.codebook.numel()
                + self.correction_codebook.row_scale.numel()
            )
        if self.correction_weight is None:
            return 0
        if self.correction_block_mask is None:
            return self.correction_weight.numel()
        return int(self.correction_block_mask.sum().item()) * 16 * 16

    def correction_prepacked_metadata_bytes(self) -> int:
        if self.correction_block_mask is None:
            return 0
        return (self.correction_block_mask.shape[0] + 1) * 2 + int(
            self.correction_block_mask.sum().item()
        )

    def initialize_from_dense(self, weight: Tensor) -> None:
        """Start a decoded Boolean projection from a stable dense scaffold."""
        if tuple(weight.shape) != (self.dimension, self.dimension):
            raise ValueError("dense decoded-Boolean source shape mismatch")
        samples = torch.linspace(
            -4.0, 4.0, 4097, device=weight.device, dtype=weight.dtype
        )
        signed = 2 * (
            samples[:, None] >= self.encoder_thresholds.to(weight).view(1, -1)
        ).to(weight.dtype) - 1
        design = torch.cat((signed, torch.ones_like(samples[:, None])), dim=1)
        staircase = torch.linalg.lstsq(design, samples).solution
        with torch.no_grad():
            # All-zero truth tables make every residual Boolean round an
            # identity at the hard forward boundary.
            self.program_logits.fill_(-0.02)
            self.encoder_offsets.zero_()
            self.decoder_weight.zero_()
            diagonal = weight.diagonal()
            self.decoder_weight.copy_(diagonal[:, None] * staircase[:-1])
            self.decoder_bias.copy_(diagonal * staircase[-1])
            if self.correction_weight is not None:
                self.correction_weight.zero_()
            if self.correction_input_weight is not None:
                self.correction_input_weight.zero_()
            if self.correction_output_weight is not None:
                self.correction_output_weight.zero_()
            if self.correction_codebook is not None:
                self.correction_codebook.codebook.zero_()
                self.correction_codebook.row_scale.zero_()
                self.correction_codebook.code_index_logits.zero_()

    def discrete_program(self) -> Tensor:
        if self.cell != "factorized_minterm":
            return (self.program_logits >= 0).to(torch.uint8)
        assert self.factorized_polarities is not None
        enable = (self.program_logits[:, 0] >= 0).to(torch.uint8)
        choices = self.program_logits[:, 1:].argmax(dim=1)
        polarity = self.factorized_polarities[choices].permute(0, 2, 1)
        return torch.cat((enable.unsqueeze(1), polarity.to(torch.uint8)), dim=1)

    def deployed_program_bits(self) -> int:
        return self.discrete_program().numel() + (
            0 if self.block_gate_logits is None else self.block_gate_logits.numel()
        )

    def training_candidate_program_values(self) -> int:
        return self.program_logits.numel() - self.discrete_program().numel()


class VocabularyBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        kv_heads: int,
        ff_dimension: int,
        feedforward_activation: str,
        boolean_width: int,
        boolean_rounds: int,
        boolean_ffn_correction_features: int,
        boolean_ffn_correction_rank: int,
        boolean_ffn_correction_mode: str,
        boolean_ffn_correction_block_active_fraction: float,
        boolean_ffn_correction_taps: str,
        boolean_ffn_pool_schedule: str,
        boolean_ffn_routes: str,
        boolean_ffn_thresholds: str,
        boolean_ffn_input_rounding: str,
        boolean_backward: str,
        boolean_temperature: float,
        boolean_gradient_leak: float,
        boolean_program_init_margin: float,
        boolean_register_bits: int,
        *,
        context: int,
        rope_base: float,
        rope_fraction: float,
        partial_key_offset: str,
        qk_norm: bool,
        qk_cache_precision: str,
        value_cache_precision: str,
        qk_head_dimension: int,
        local_qk_head_dimension: int,
        kv_head_routing: str,
        kv_head_sharing: str,
        attention_qkv_projection_mode: str,
        qkv_squeeze_bitplanes: int,
        qkv_squeeze_polarities: int,
        qkv_squeeze_group_bits: int,
        qkv_squeeze_bypass: int,
        qkv_squeeze_input_rounding: str,
        qkv_low_rank: int,
        qkv_shared_bases: int,
        qkv_local_width: int,
        qkv_global_rank: int,
        qkv_route_count: int,
        qkv_bitmix_rounds: int,
        qkv_bitmix_correction_features: int,
        qkv_bitmix_cell: str,
        qkv_block_allocation: str,
        qkv_block_rows: int,
        qkv_ticket_selection: str,
        qkv_ticket_temperature: float,
        qkv_head_input_groups: int,
        attention_qk_mode: str,
        boolean_qk_width: int,
        boolean_qk_rounds: int,
        boolean_qk_bits_per_head: int,
        boolean_qk_correction_features: int,
        boolean_qk_correction_taps: str,
        attention_value_mode: str,
        attention_value_block_active_fraction: float,
        boolean_value_width: int,
        boolean_value_rounds: int,
        boolean_value_bits: int,
        boolean_value_correction_features: int,
        attention_output_mode: str,
        boolean_output_width: int,
        boolean_output_rounds: int,
        boolean_output_correction_features: int,
        boolean_output_correction_taps: str,
        boolean_output_routes: str,
        boolean_output_cell: str,
        attention_output_block_active_fraction: float,
        attention_output_codewords: int,
        attention_output_codebook_temperature: float,
        attention_output_codebook_response_shift: int,
        attention_output_head_local_rank: int,
        attention_output_squeeze_bitplanes: int,
        attention_output_squeeze_polarities: int,
        attention_output_squeeze_group_bits: int,
        attention_output_squeeze_bypass: int,
        attention_output_input_rounding: str,
        attention_normalizer: str,
        clipped_linear_initial_slope: float,
        head_contexts: Sequence[int],
        use_value_embedding: bool,
        skip_attention: bool,
        dropout: float,
        zero_init_residual: bool,
        residual_scalars: bool,
        boolean_residual_state: str,
        residual_boundary: str,
        residual_normalizer: str,
    ) -> None:
        super().__init__()
        self.skip_attention = skip_attention
        self.attention_norm = residual_norm(dimension, residual_normalizer)
        self.attention = VocabularyAttention(
            dimension,
            heads,
            kv_heads,
            context=context,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
            partial_key_offset=partial_key_offset,
            qk_norm=qk_norm,
            qk_cache_precision=qk_cache_precision,
            value_cache_precision=value_cache_precision,
            qk_head_dimension=qk_head_dimension,
            local_qk_head_dimension=local_qk_head_dimension,
            kv_head_routing=kv_head_routing,
            kv_head_sharing=kv_head_sharing,
            attention_qkv_projection_mode=attention_qkv_projection_mode,
            qkv_squeeze_bitplanes=qkv_squeeze_bitplanes,
            qkv_squeeze_polarities=qkv_squeeze_polarities,
            qkv_squeeze_group_bits=qkv_squeeze_group_bits,
            qkv_squeeze_bypass=qkv_squeeze_bypass,
            qkv_squeeze_input_rounding=qkv_squeeze_input_rounding,
            qkv_low_rank=qkv_low_rank,
            qkv_shared_bases=qkv_shared_bases,
            qkv_local_width=qkv_local_width,
            qkv_global_rank=qkv_global_rank,
            qkv_route_count=qkv_route_count,
            qkv_bitmix_rounds=qkv_bitmix_rounds,
            qkv_bitmix_correction_features=qkv_bitmix_correction_features,
            qkv_bitmix_cell=qkv_bitmix_cell,
            qkv_block_allocation=qkv_block_allocation,
            qkv_block_rows=qkv_block_rows,
            qkv_ticket_selection=qkv_ticket_selection,
            qkv_ticket_temperature=qkv_ticket_temperature,
            qkv_head_input_groups=qkv_head_input_groups,
            attention_qk_mode=attention_qk_mode,
            boolean_qk_width=boolean_qk_width,
            boolean_qk_rounds=boolean_qk_rounds,
            boolean_qk_bits_per_head=boolean_qk_bits_per_head,
            boolean_qk_correction_features=boolean_qk_correction_features,
            boolean_qk_correction_taps=boolean_qk_correction_taps,
            attention_value_mode=attention_value_mode,
            attention_value_block_active_fraction=attention_value_block_active_fraction,
            boolean_value_width=boolean_value_width,
            boolean_value_rounds=boolean_value_rounds,
            boolean_value_bits=boolean_value_bits,
            boolean_value_correction_features=boolean_value_correction_features,
            attention_output_mode=attention_output_mode,
            boolean_output_width=boolean_output_width,
            boolean_output_rounds=boolean_output_rounds,
            boolean_output_correction_features=boolean_output_correction_features,
            boolean_output_correction_taps=boolean_output_correction_taps,
            boolean_output_routes=boolean_output_routes,
            boolean_output_cell=boolean_output_cell,
            attention_output_block_active_fraction=attention_output_block_active_fraction,
            attention_output_codewords=attention_output_codewords,
            attention_output_codebook_temperature=attention_output_codebook_temperature,
            attention_output_codebook_response_shift=(
                attention_output_codebook_response_shift
            ),
            attention_output_head_local_rank=attention_output_head_local_rank,
            attention_output_squeeze_bitplanes=attention_output_squeeze_bitplanes,
            attention_output_squeeze_polarities=attention_output_squeeze_polarities,
            attention_output_squeeze_group_bits=attention_output_squeeze_group_bits,
            attention_output_squeeze_bypass=attention_output_squeeze_bypass,
            attention_output_input_rounding=attention_output_input_rounding,
            boolean_backward=boolean_backward,
            boolean_temperature=boolean_temperature,
            boolean_gradient_leak=boolean_gradient_leak,
            boolean_program_init_margin=boolean_program_init_margin,
            boolean_register_bits=boolean_register_bits,
            attention_normalizer=attention_normalizer,
            clipped_linear_initial_slope=clipped_linear_initial_slope,
            head_contexts=head_contexts,
            use_value_embedding=use_value_embedding,
            dropout=dropout,
            zero_init_residual=zero_init_residual,
        )
        self.feedforward_norm = residual_norm(dimension, residual_normalizer)
        self.feedforward_activation = feedforward_activation
        self.boolean_residual_state = boolean_residual_state
        self.residual_boundary = residual_boundary
        if feedforward_activation.startswith("boolean_"):
            self.boolean_ffn = BooleanLUT3FFN(
                dimension,
                boolean_width,
                boolean_rounds,
                backward_mode=boolean_backward,
                temperature=boolean_temperature,
                gradient_leak=boolean_gradient_leak,
                program_init_margin=boolean_program_init_margin,
                register_bits=boolean_register_bits,
                block_gates=feedforward_activation == "boolean_lut3_gated",
                cell=(
                    "factorized_minterm"
                    if feedforward_activation == "boolean_factorized_minterm"
                    else "dual3"
                    if feedforward_activation == "boolean_dual3"
                    else "lut3"
                ),
                correction_features=boolean_ffn_correction_features,
                correction_rank=boolean_ffn_correction_rank,
                correction_mode=boolean_ffn_correction_mode,
                correction_block_active_fraction=(
                    boolean_ffn_correction_block_active_fraction
                ),
                correction_taps=boolean_ffn_correction_taps,
                pool_schedule=boolean_ffn_pool_schedule,
                route_schedule=boolean_ffn_routes,
                threshold_schedule=boolean_ffn_thresholds,
                input_rounding=boolean_ffn_input_rounding,
            )
            self.register_module("ff_up", None)
            self.register_module("ff_down", None)
        else:
            self.register_module("boolean_ffn", None)
            up_dimension = (
                ff_dimension if feedforward_activation == "relu_squared" else 2 * ff_dimension
            )
            self.ff_up = nn.Linear(dimension, up_dimension, bias=False)
            self.ff_down = nn.Linear(ff_dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)
        if zero_init_residual and self.ff_down is not None:
            nn.init.zeros_(self.ff_down.weight)
        if residual_scalars:
            self.residual_scales = nn.Parameter(torch.ones(4))
        else:
            self.register_buffer("residual_scales", torch.ones(4), persistent=False)

    def _feedforward(self, state: Tensor) -> Tensor:
        if self.boolean_ffn is not None:
            return self.dropout(self.boolean_ffn(state))
        if self.ff_up is None or self.ff_down is None:
            raise RuntimeError("numerical FFN weights are missing")
        projected = self.ff_up(state)
        if self.feedforward_activation == "relu_squared":
            hidden = F.relu(projected).square()
        else:
            value, gate = projected.chunk(2, dim=-1)
            if self.feedforward_activation == "reglu":
                hidden = value * F.relu(gate)
            else:
                hidden = value * gate * isa_sigmoid(gate)
        return self.dropout(self.ff_down(hidden))

    @staticmethod
    def _fp16_boundary(state: Tensor) -> Tensor:
        hard = state.to(torch.float16).to(state.dtype)
        return state + (hard - state).detach()

    def _after_attention_boundary(self, state: Tensor) -> Tensor:
        return (
            self._fp16_boundary(state)
            if self.residual_boundary == "dual_fp16"
            else state
        )

    def _after_block_boundary(self, state: Tensor) -> Tensor:
        return (
            self._fp16_boundary(state)
            if self.residual_boundary in {"block_fp16", "dual_fp16"}
            else state
        )

    def forward(
        self,
        state: Tensor,
        positions: Tensor,
        auxiliary_value: Tensor | None = None,
        attention_source: Tensor | None = None,
    ) -> Tensor:
        scales = self.residual_scales
        attention_input = state if attention_source is None else attention_source
        state = (
            scales[0] * state
            if self.skip_attention
            else scales[0] * state + scales[1] * self.attention(
                self.attention_norm(attention_input), positions, auxiliary_value
            )
        )
        state = self._after_attention_boundary(state)
        state = scales[2] * state + scales[3] * self._feedforward(
            self.feedforward_norm(state)
        )
        return self._after_block_boundary(state)

    def forward_with_shared_qk(
        self,
        state: Tensor,
        positions: Tensor,
        auxiliary_value: Tensor | None,
        attention_source: Tensor,
        shared_qk: tuple[Tensor, Tensor] | None,
        shared_qk_mode: str,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.skip_attention:
            raise RuntimeError("a shared-Q/K layer cannot skip attention")
        scales = self.residual_scales
        branch, projected_qk = self.attention.forward_with_shared_qk(
            self.attention_norm(attention_source),
            positions,
            auxiliary_value,
            shared_qk,
            shared_qk_mode,
        )
        state = scales[0] * state + scales[1] * branch
        state = self._after_attention_boundary(state)
        state = scales[2] * state + scales[3] * self._feedforward(
            self.feedforward_norm(state)
        )
        return self._after_block_boundary(state), projected_qk

    def forward_cached(
        self,
        state: Tensor,
        positions: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        auxiliary_value: Tensor | None = None,
        attention_source: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        scales = self.residual_scales
        if self.skip_attention:
            updated = cache
            state = scales[0] * state
        else:
            attention_input = state if attention_source is None else attention_source
            branch, updated = self.attention.forward_cached(
                self.attention_norm(attention_input), positions, cache, auxiliary_value
            )
            state = scales[0] * state + scales[1] * branch
        state = self._after_attention_boundary(state)
        state = scales[2] * state + scales[3] * self._feedforward(
            self.feedforward_norm(state)
        )
        return self._after_block_boundary(state), updated

    def forward_cached_with_shared_qk(
        self,
        state: Tensor,
        positions: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        auxiliary_value: Tensor | None,
        attention_source: Tensor,
        shared_qk: tuple[Tensor, Tensor] | None,
        shared_qk_mode: str,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if self.skip_attention:
            raise RuntimeError("a shared-Q/K layer cannot skip attention")
        scales = self.residual_scales
        branch, updated, projected_qk = (
            self.attention.forward_cached_with_shared_qk(
                self.attention_norm(attention_source),
                positions,
                cache,
                auxiliary_value,
                shared_qk,
                shared_qk_mode,
            )
        )
        state = scales[0] * state + scales[1] * branch
        state = self._after_attention_boundary(state)
        state = scales[2] * state + scales[3] * self._feedforward(
            self.feedforward_norm(state)
        )
        return self._after_block_boundary(state), updated, projected_qk

    def forward_with_boolean_carrier(
        self,
        state: Tensor,
        positions: Tensor,
        carrier: Tensor | None,
        auxiliary_value: Tensor | None = None,
        attention_source: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.boolean_ffn is None:
            raise RuntimeError("Boolean carrier requires a Boolean FFN")
        scales = self.residual_scales
        attention_input = state if attention_source is None else attention_source
        state = (
            scales[0] * state
            if self.skip_attention
            else scales[0] * state + scales[1] * self.attention(
                self.attention_norm(attention_input), positions, auxiliary_value
            )
        )
        state = self._after_attention_boundary(state)
        branch, carrier = self.boolean_ffn.forward_with_carrier(
            self.feedforward_norm(state), carrier, self.boolean_residual_state
        )
        state = scales[2] * state + scales[3] * self.dropout(branch)
        return self._after_block_boundary(state), carrier

    def forward_cached_with_boolean_carrier(
        self,
        state: Tensor,
        positions: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        carrier: Tensor | None,
        auxiliary_value: Tensor | None = None,
        attention_source: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        if self.boolean_ffn is None:
            raise RuntimeError("Boolean carrier requires a Boolean FFN")
        scales = self.residual_scales
        if self.skip_attention:
            updated = cache
            state = scales[0] * state
        else:
            attention_input = state if attention_source is None else attention_source
            branch, updated = self.attention.forward_cached(
                self.attention_norm(attention_input), positions, cache, auxiliary_value
            )
            state = scales[0] * state + scales[1] * branch
        state = self._after_attention_boundary(state)
        ff_branch, carrier = self.boolean_ffn.forward_with_carrier(
            self.feedforward_norm(state), carrier, self.boolean_residual_state
        )
        state = scales[2] * state + scales[3] * self.dropout(ff_branch)
        return self._after_block_boundary(state), updated, carrier


class BooleanTokenEmbedding(nn.Module):
    """Contiguous learned token codes with a fixed bipolar input expansion."""

    def __init__(
        self,
        vocabulary_size: int,
        dimension: int,
        code_width: int,
        expansion_mode: str = "repeat",
    ) -> None:
        super().__init__()
        if not 0 < code_width < dimension:
            raise ValueError("Boolean token-code width must be in (0, dimension)")
        self.dimension = dimension
        self.code_width = code_width
        self.expansion_mode = expansion_mode
        if expansion_mode not in {"repeat", "xor_pair", "bcax_triplet"}:
            raise ValueError("invalid Boolean token expansion")
        self.derived_width = dimension - 2 * code_width
        if expansion_mode != "repeat" and not 0 < self.derived_width <= code_width:
            raise ValueError("nonlinear token expansion needs two copies plus derived lanes")
        self.weight = nn.Parameter(torch.empty(vocabulary_size, code_width))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        self.register_buffer(
            "expansion_indices",
            torch.arange(dimension) % code_width,
            persistent=True,
        )
        derived = torch.arange(max(0, self.derived_width))
        self.register_buffer("derived_a_indices", derived, persistent=True)
        self.register_buffer(
            "derived_b_indices", (derived + code_width // 2).remainder(code_width), persistent=True
        )
        self.register_buffer(
            "derived_c_indices", (derived + code_width // 4).remainder(code_width), persistent=True
        )

    def forward(self, tokens: Tensor) -> Tensor:
        codes = hard_bit(F.embedding(tokens, self.weight))
        expanded = codes.index_select(-1, self.expansion_indices)
        if self.expansion_mode != "repeat":
            a = codes.index_select(-1, self.derived_a_indices)
            b = codes.index_select(-1, self.derived_b_indices)
            if self.expansion_mode == "xor_pair":
                derived = a + b - 2 * a * b
            else:
                c = codes.index_select(-1, self.derived_c_indices)
                branch = b * (1 - c)
                derived = a + branch - 2 * a * branch
            expanded = torch.cat((expanded[..., : 2 * self.code_width], derived), dim=-1)
        return (2 * expanded - 1) * 0.02

    def discrete_codes(self) -> Tensor:
        return (self.weight >= 0).to(torch.uint8)


class BooleanVocabularyReadout(nn.Module):
    """Tied Hamming readout with an optional signed-popcount correction."""

    def __init__(
        self,
        vocabulary_size: int,
        dimension: int,
        code_width: int,
        mode: str,
        *,
        temperature: float,
        pool_bits: int = 16,
        pool_schedule: str = "uniform",
        correction_rank: int = 0,
        factor_output: str = "fp16",
        hierarchy: str = "flat",
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        self.code_width = code_width
        self.mode = mode
        self.temperature = temperature
        self.pool_bits = pool_bits
        self.pool_schedule = pool_schedule
        self.correction_rank = correction_rank
        self.factor_output = factor_output
        self.hierarchy = hierarchy
        if factor_output not in {"fp16", "w8a8_row", "w8a8_group16"}:
            raise ValueError("invalid Boolean vocabulary factor output")
        if factor_output != "fp16" and not correction_rank:
            raise ValueError("W8A8 factor output requires a correction rank")
        self.correction_features = (
            dimension - code_width - 1 if mode == "boolean_hybrid" else 0
        )
        if mode == "boolean_hybrid":
            if pool_bits not in {8, 16, 32}:
                raise ValueError("Boolean vocabulary pool bits must be 8, 16, or 32")
            if pool_schedule == "uniform":
                windows = torch.full((self.correction_features,), pool_bits)
                local_features = torch.arange(self.correction_features)
            elif pool_schedule == "mixed_8_16_32":
                windows = torch.tensor(
                    [8] * 32 + [16] * 32 + [32] * (self.correction_features - 64)
                )
                local_features = torch.cat(
                    (
                        torch.arange(32), torch.arange(32),
                        torch.arange(self.correction_features - 64),
                    )
                )
            elif pool_schedule == "mixed_16_32":
                windows = torch.tensor(
                    [16] * 32 + [32] * (self.correction_features - 32)
                )
                local_features = torch.cat(
                    (torch.arange(32), torch.arange(self.correction_features - 32))
                )
            else:
                raise ValueError("invalid Boolean vocabulary pool schedule")
            if self.correction_features <= 0:
                raise ValueError("hybrid vocabulary path needs correction features")
            if correction_rank:
                self.register_parameter("correction_weight", None)
                self.correction_input_weight = nn.Parameter(
                    torch.empty(correction_rank, self.correction_features)
                )
                self.correction_output_weight = nn.Parameter(
                    torch.empty(vocabulary_size, correction_rank)
                )
                nn.init.normal_(self.correction_input_weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.correction_output_weight)
            else:
                self.correction_weight = nn.Parameter(
                    torch.zeros(vocabulary_size, self.correction_features)
                )
                self.register_parameter("correction_input_weight", None)
                self.register_parameter("correction_output_weight", None)
            thresholds = torch.tensor((-1.5, -0.5, 0.5, 1.5))
            self.register_buffer("correction_thresholds", thresholds, persistent=True)
            starts = torch.where(
                local_features < (dimension * 4) // windows,
                local_features * windows,
                (torch.arange(self.correction_features) * 37) % (dimension * 4),
            )
            maximum_window = int(windows.max())
            offsets = torch.arange(maximum_window)[None]
            self.register_buffer(
                "correction_indices",
                (starts[:, None] + offsets).remainder(dimension * 4).flatten(),
                persistent=True,
            )
            self.register_buffer("correction_windows", windows, persistent=True)
            self.register_buffer(
                "correction_window_mask", offsets < windows[:, None], persistent=True
            )
        else:
            self.register_parameter("correction_weight", None)
            self.register_parameter("correction_input_weight", None)
            self.register_parameter("correction_output_weight", None)
            self.register_buffer("correction_thresholds", None, persistent=True)
            self.register_buffer("correction_indices", None, persistent=True)
            self.register_buffer("correction_windows", None, persistent=True)
            self.register_buffer("correction_window_mask", None, persistent=True)
        if hierarchy == "balanced_32x64":
            if (
                vocabulary_size != 2048
                or mode != "boolean_hybrid"
                or correction_rank <= 0
            ):
                raise ValueError(
                    "balanced 32x64 hierarchy requires a 2K factorized hybrid vocabulary"
                )
            self.cluster_code_logits = nn.Parameter(
                torch.empty(32, code_width)
            )
            self.cluster_correction_output_weight = nn.Parameter(
                torch.empty(32, correction_rank)
            )
            self.cluster_bias = nn.Parameter(torch.zeros(32))
            nn.init.normal_(self.cluster_code_logits, mean=0.0, std=0.02)
            nn.init.zeros_(self.cluster_correction_output_weight)
            order = torch.arange(2048, dtype=torch.long).reshape(32, 64)
            self.register_buffer("cluster_token_ids", order, persistent=True)
            self.register_buffer(
                "token_hierarchical_index",
                torch.arange(2048, dtype=torch.long),
                persistent=True,
            )
        elif hierarchy == "flat":
            self.register_parameter("cluster_code_logits", None)
            self.register_parameter("cluster_correction_output_weight", None)
            self.register_parameter("cluster_bias", None)
            self.register_buffer("cluster_token_ids", None, persistent=True)
            self.register_buffer(
                "token_hierarchical_index", None, persistent=True
            )
        else:
            raise ValueError("invalid vocabulary hierarchy")
        self.bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.register_buffer(
            "query_indices", torch.arange(code_width) % dimension, persistent=True
        )

    def forward(self, features: Tensor, code_logits: Tensor) -> Tensor:
        query = hard_bit(
            features.index_select(-1, self.query_indices), self.temperature
        )
        codes = hard_bit(code_logits, self.temperature)
        query_signed = 2 * query - 1
        code_signed = 2 * codes - 1
        logits = 0.25 * (query_signed @ code_signed.T) / math.sqrt(self.code_width)
        hidden: Tensor | None = None
        if self.correction_weight is not None or self.correction_input_weight is not None:
            assert self.correction_thresholds is not None
            assert self.correction_indices is not None
            assert self.correction_windows is not None
            assert self.correction_window_mask is not None
            encoded = hard_bit(
                features.unsqueeze(-1)
                - self.correction_thresholds.view(
                    *([1] * features.ndim), 4
                ),
                self.temperature,
            ).flatten(-2)
            selected = encoded.index_select(-1, self.correction_indices)
            pooled = (
                (2 * selected - 1).reshape(
                    *selected.shape[:-1], self.correction_features, -1
                ) * self.correction_window_mask
            ).sum(dim=-1) / self.correction_windows.sqrt()
            if self.correction_weight is not None:
                correction = pooled @ self.correction_weight.T
            else:
                assert self.correction_input_weight is not None
                assert self.correction_output_weight is not None
                hidden = fake_w8_linear(pooled, self.correction_input_weight)
                hard_hidden = hidden.to(torch.float16).to(hidden.dtype)
                hidden = hidden + (hard_hidden - hidden).detach()
                if self.factor_output == "fp16":
                    hard_output = self.correction_output_weight.to(torch.float16).to(
                        self.correction_output_weight.dtype
                    )
                    output_weight = self.correction_output_weight + (
                        hard_output - self.correction_output_weight
                    ).detach()
                    correction = F.linear(hidden, output_weight)
                elif self.factor_output == "w8a8_row":
                    correction = fake_w8a8_linear(
                        hidden, self.correction_output_weight
                    )
                elif self.factor_output == "w8a8_group16":
                    correction = fake_w8a8_grouped_weight_linear(
                        hidden, self.correction_output_weight, group_size=16
                    )
                else:
                    raise RuntimeError("invalid Boolean vocabulary factor output")
            logits = logits + correction
        logits = logits + self.bias
        if self.hierarchy == "flat":
            return logits
        if (
            hidden is None
            or self.cluster_code_logits is None
            or self.cluster_correction_output_weight is None
            or self.cluster_bias is None
            or self.cluster_token_ids is None
            or self.token_hierarchical_index is None
        ):
            raise RuntimeError("hierarchical vocabulary parameters are incomplete")
        cluster_codes = hard_bit(self.cluster_code_logits, self.temperature)
        cluster_main = 0.25 * (
            query_signed @ (2 * cluster_codes - 1).T
        ) / math.sqrt(self.code_width)
        if self.factor_output == "fp16":
            hard_cluster_output = self.cluster_correction_output_weight.to(
                torch.float16
            ).to(self.cluster_correction_output_weight.dtype)
            cluster_output = self.cluster_correction_output_weight + (
                hard_cluster_output - self.cluster_correction_output_weight
            ).detach()
            cluster_correction = F.linear(hidden, cluster_output)
        elif self.factor_output == "w8a8_row":
            cluster_correction = fake_w8a8_linear(
                hidden, self.cluster_correction_output_weight
            )
        elif self.factor_output == "w8a8_group16":
            cluster_correction = fake_w8a8_grouped_weight_linear(
                hidden,
                self.cluster_correction_output_weight,
                group_size=16,
            )
        else:
            raise RuntimeError("invalid hierarchical factor output")
        cluster_logits = cluster_main + cluster_correction + self.cluster_bias
        hierarchical_order = self.cluster_token_ids.flatten()
        ordered_leaf = logits.index_select(-1, hierarchical_order).reshape(
            *logits.shape[:-1], 32, 64
        )
        joint = F.log_softmax(cluster_logits, dim=-1).unsqueeze(-1) + (
            F.log_softmax(ordered_leaf, dim=-1)
        )
        return joint.flatten(-2).index_select(
            -1, self.token_hierarchical_index
        )

    @staticmethod
    def _balanced_clusters(geometry: Tensor) -> Tensor:
        values = F.normalize(geometry.detach().float().cpu(), dim=-1)
        chosen = [int((values - values.mean(dim=0)).square().sum(-1).argmax())]
        distance = (values - values[chosen[0]]).square().sum(-1)
        for _ in range(1, 32):
            index = int(distance.argmax())
            chosen.append(index)
            distance = torch.minimum(
                distance, (values - values[index]).square().sum(-1)
            )
        centers = values[chosen].clone()
        assignment = torch.empty(2048, dtype=torch.long)
        for _ in range(16):
            distances = torch.cdist(values, centers).square()
            preferences = distances.argsort(dim=-1, stable=True)
            sorted_distances = distances.gather(1, preferences[:, :2])
            confidence = sorted_distances[:, 1] - sorted_distances[:, 0]
            token_order = confidence.argsort(descending=True, stable=True)
            capacity = torch.full((32,), 64, dtype=torch.long)
            for token in token_order.tolist():
                for cluster in preferences[token].tolist():
                    if capacity[cluster] > 0:
                        assignment[token] = cluster
                        capacity[cluster] -= 1
                        break
            updated = torch.stack(
                [values[assignment == cluster].mean(dim=0) for cluster in range(32)]
            )
            if torch.equal(
                assignment,
                torch.cdist(values, updated).square().argmin(dim=-1),
            ):
                centers = updated
                break
            centers = updated
        return torch.stack(
            [
                torch.nonzero(assignment == cluster, as_tuple=False)
                .flatten()
                .sort()
                .values
                for cluster in range(32)
            ]
        )

    @torch.no_grad()
    def initialize_hierarchy_from_flat(
        self,
        source_code_logits: Tensor,
        geometry_output_weight: Tensor,
    ) -> None:
        if (
            self.hierarchy != "balanced_32x64"
            or self.cluster_code_logits is None
            or self.cluster_correction_output_weight is None
            or self.cluster_bias is None
            or self.cluster_token_ids is None
            or self.token_hierarchical_index is None
            or self.correction_output_weight is None
        ):
            raise ValueError("target readout is not a balanced hierarchy")
        if (
            source_code_logits.shape != (2048, self.code_width)
            or geometry_output_weight.shape[0] != 2048
        ):
            raise ValueError("hierarchy source geometry has the wrong shape")
        signed_codes = 2 * (source_code_logits.detach().float() >= 0) - 1
        normalized_output = F.normalize(
            geometry_output_weight.detach().float(), dim=-1
        )
        order = self._balanced_clusters(
            torch.cat((signed_codes / math.sqrt(self.code_width), normalized_output), dim=-1)
        ).to(self.cluster_token_ids.device)
        inverse = torch.empty(2048, dtype=torch.long, device=order.device)
        inverse[order.flatten()] = torch.arange(2048, device=order.device)
        self.cluster_token_ids.copy_(order)
        self.token_hierarchical_index.copy_(inverse)
        source_codes = source_code_logits.to(
            device=self.cluster_code_logits.device,
            dtype=self.cluster_code_logits.dtype,
        )
        for cluster in range(32):
            members = order[cluster]
            weights = torch.softmax(self.bias.index_select(0, members), dim=0)
            self.cluster_code_logits[cluster].copy_(
                weights @ source_codes.index_select(0, members)
            )
            self.cluster_correction_output_weight[cluster].copy_(
                weights @ self.correction_output_weight.index_select(0, members)
            )
            self.cluster_bias[cluster].copy_(
                torch.logsumexp(self.bias.index_select(0, members), dim=0)
            )

    @torch.no_grad()
    def initialize_correction_from_dense(self, weight: Tensor) -> None:
        if self.correction_input_weight is None or self.correction_output_weight is None:
            raise ValueError("target readout correction is not factorized")
        if weight.shape != (self.vocabulary_size, self.correction_features):
            raise ValueError("dense readout correction shape mismatch")
        left, singular, right = torch.linalg.svd(weight.float(), full_matrices=False)
        root = singular[: self.correction_rank].sqrt()
        self.correction_input_weight.copy_(
            root[:, None] * right[: self.correction_rank]
        )
        self.correction_output_weight.copy_(
            left[:, : self.correction_rank] * root[None, :]
        )

    @torch.no_grad()
    def initialize_correction_from_factorized(
        self, input_weight: Tensor, output_weight: Tensor
    ) -> None:
        """Expand a trained factorization as an exact no-op in new rank lanes."""
        if self.correction_input_weight is None or self.correction_output_weight is None:
            raise ValueError("target readout correction is not factorized")
        if input_weight.ndim != 2 or output_weight.ndim != 2:
            raise ValueError("source readout factors must be matrices")
        source_rank = input_weight.shape[0]
        if (
            input_weight.shape[1] != self.correction_features
            or output_weight.shape != (self.vocabulary_size, source_rank)
        ):
            raise ValueError("source readout factor shape mismatch")
        input_weight = input_weight.to(
            device=self.correction_input_weight.device,
            dtype=self.correction_input_weight.dtype,
        )
        output_weight = output_weight.to(
            device=self.correction_output_weight.device,
            dtype=self.correction_output_weight.dtype,
        )
        if source_rank > self.correction_rank:
            self.initialize_correction_from_dense(output_weight @ input_weight)
            return
        self.correction_input_weight[:source_rank].copy_(input_weight)
        self.correction_output_weight[:, :source_rank].copy_(output_weight)
        if source_rank == self.correction_rank:
            return
        extra = self.correction_input_weight[source_rank:]
        basis = torch.linalg.qr(input_weight.float().T, mode="reduced").Q
        orthogonal = extra.float() - (extra.float() @ basis) @ basis.T
        source_norm = input_weight.float().norm(dim=1).mean().clamp_min(1.0e-8)
        orthogonal = orthogonal / orthogonal.norm(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8) * source_norm
        extra.copy_(orthogonal.to(extra.dtype))
        self.correction_output_weight[:, source_rank:].zero_()

    def deployed_code_bits(self, code_logits: Tensor) -> int:
        return code_logits.numel()


class VocabularyTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        logical_vocabulary_size: int,
        padded_vocabulary_size: int,
        context: int,
        dimension: int,
        layers: int,
        heads: int,
        kv_heads: int,
        ff_dimension: int,
        ff_extra_dimension: int = 0,
        ff_extra_dimension_layers: int = 0,
        feedforward_activation: str = "relu_squared",
        boolean_width: int = 1_280,
        boolean_rounds: int = 5,
        boolean_ffn_correction_features: int = 0,
        boolean_ffn_correction_rank: int = 0,
        boolean_ffn_correction_mode: str = "dense",
        boolean_ffn_correction_block_active_fraction: float = 1.0,
        boolean_ffn_correction_taps: str = "final",
        boolean_ffn_pool_schedule: str = "legacy_mixed",
        boolean_ffn_routes: str = "byte_lane",
        boolean_ffn_thresholds: str = "linear",
        boolean_ffn_input_rounding: str = "none",
        boolean_backward: str = "hybrid",
        boolean_temperature: float = 1.0,
        boolean_gradient_leak: float = 0.05,
        boolean_program_init_margin: float = 0.02,
        boolean_register_bits: int = 128,
        boolean_residual_state: str = "numerical",
        residual_boundary: str = "none",
        residual_normalizer: str = "rms",
        token_feature: str = "plain",
        token_smear: str = "none",
        value_embedding_layers: Sequence[int] = (1, 2, 7, 8, 9),
        bigram_hash_size: int = 16_384,
        bigram_dimension: int = 56,
        bigram_sign_mode: str = "none",
        bigram_learned_scale: bool = True,
        ngram_orders: Sequence[int] = (),
        ngram_heads: int = 2,
        ngram_table_rows: int = 4096,
        ngram_dimension: int = 64,
        rope_base: float = 10_000.0,
        rope_fraction: float = 1.0,
        partial_key_offset: str = "none",
        qk_norm: bool = True,
        qk_cache_precision: str = "fp32",
        value_cache_precision: str = "fp32",
        qk_head_dimension: int = 0,
        local_qk_head_dimension: int = 0,
        kv_head_routing: str = "repeat",
        kv_head_sharing: str = "both",
        attention_qkv_projection_mode: str = "dense",
        qkv_squeeze_bitplanes: int = 8,
        qkv_squeeze_polarities: int = 2,
        qkv_squeeze_group_bits: int = 32,
        qkv_squeeze_bypass: int = 32,
        qkv_squeeze_input_rounding: str = "fp16",
        qkv_low_rank: int = 64,
        qkv_shared_bases: int = 1,
        qkv_local_width: int = 64,
        qkv_global_rank: int = 16,
        qkv_route_count: int = 16,
        qkv_bitmix_rounds: int = 2,
        qkv_bitmix_correction_features: int = 48,
        qkv_bitmix_cell: str = "eor3",
        qkv_block_allocation: str = "uniform",
        qkv_block_rows: int = 16,
        qkv_ticket_selection: str = "magnitude",
        qkv_ticket_temperature: float = 0.25,
        qkv_head_input_groups: int = 0,
        attention_qk_mode: str = "numerical",
        boolean_qk_width: int = 640,
        boolean_qk_rounds: int = 5,
        boolean_qk_bits_per_head: int = 64,
        boolean_qk_correction_features: int = 0,
        boolean_qk_correction_taps: str = "final",
        attention_value_mode: str = "numerical",
        attention_value_block_active_fraction: float = 0.5,
        boolean_value_width: int = 640,
        boolean_value_rounds: int = 5,
        boolean_value_bits: int = 0,
        boolean_value_correction_features: int = 0,
        attention_output_mode: str = "numerical",
        boolean_output_width: int = 640,
        boolean_output_rounds: int = 5,
        boolean_output_correction_features: int = 0,
        boolean_output_correction_taps: str = "final",
        boolean_output_routes: str = "byte_lane",
        boolean_output_cell: str = "lut3",
        attention_output_block_active_fraction: float = 0.5,
        attention_output_codewords: int = 32,
        attention_output_codebook_temperature: float = 0.25,
        attention_output_codebook_response_shift: int = 10,
        attention_output_head_local_rank: int = 0,
        attention_output_squeeze_bitplanes: int = 8,
        attention_output_squeeze_polarities: int = 2,
        attention_output_squeeze_group_bits: int = 32,
        attention_output_squeeze_bypass: int = 32,
        attention_output_input_rounding: str = "none",
        attention_normalizer: str = "isa_exp2",
        clipped_linear_initial_slope: float = 0.5,
        attention_window_policy: str = "global",
        local_attention_window: int = 64,
        local_attention_heads: int = 4,
        local_attention_layers: Sequence[int] = (0, 2, 4, 6, 8),
        attention_skip_layers: Sequence[int] = (),
        attention_shared_input_layers: Sequence[int] = (),
        attention_shared_qk_layers: Sequence[int] = (),
        attention_shared_qk_mode: str = "none",
        zero_init_residual: bool = True,
        tie_embeddings: bool = True,
        vocabulary_path: str = "numerical",
        boolean_vocab_code_width: int = 0,
        boolean_vocab_pool_bits: int = 16,
        boolean_vocab_pool_schedule: str = "uniform",
        boolean_vocab_correction_rank: int = 0,
        boolean_vocab_factor_output: str = "fp16",
        vocabulary_hierarchy: str = "flat",
        boolean_token_expansion: str = "repeat",
        residual_scalars: bool = True,
        residual_routing: str = "ordinary",
        u_skip_pairs: Sequence[Sequence[int]] = ((1, 8), (3, 6)),
        dynamic_route_layers: Sequence[int] = (4, 7, 9),
        dynamic_route_dimension: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if logical_vocabulary_size > padded_vocabulary_size:
            raise ValueError("padded vocabulary must cover the logical vocabulary")
        self.logical_vocabulary_size = logical_vocabulary_size
        self.padded_vocabulary_size = padded_vocabulary_size
        self.context = context
        self.dimension = dimension
        self.token_feature = token_feature
        self.vocabulary_path = vocabulary_path
        self.vocabulary_hierarchy = vocabulary_hierarchy
        self.token_smear = token_smear
        self.attention_window_policy = attention_window_policy
        self.attention_skip_layers = tuple(int(layer) for layer in attention_skip_layers)
        self.attention_shared_input_layers = tuple(
            int(layer) for layer in attention_shared_input_layers
        )
        self.attention_shared_qk_layers = tuple(
            int(layer) for layer in attention_shared_qk_layers
        )
        self.attention_shared_qk_mode = attention_shared_qk_mode
        self.boolean_residual_state = boolean_residual_state
        self.residual_boundary = residual_boundary
        local_layers = frozenset(local_attention_layers)

        def head_contexts(layer_index: int) -> tuple[int, ...]:
            if attention_window_policy == "head_local":
                return (
                    (local_attention_window,) * local_attention_heads
                    + (context,) * (heads - local_attention_heads)
                )
            if attention_window_policy == "layer_local" and layer_index in local_layers:
                return (local_attention_window,) * heads
            return (context,) * heads
        self.residual_routing = residual_routing
        self.u_skip_pairs = tuple((int(source), int(target)) for source, target in u_skip_pairs)
        self.dynamic_route_layers = tuple(int(layer) for layer in dynamic_route_layers)
        self.dynamic_route_dimension = dynamic_route_dimension
        self.bigram_hash_size = bigram_hash_size
        self.bigram_dimension = bigram_dimension
        self.bigram_sign_mode = bigram_sign_mode
        self.bigram_learned_scale = bigram_learned_scale
        if vocabulary_path == "numerical":
            self.token_embedding = nn.Embedding(padded_vocabulary_size, dimension)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        else:
            correction_features = dimension // 5 - 1
            code_width = (
                boolean_vocab_code_width
                if vocabulary_path == "boolean_hybrid" and boolean_vocab_code_width > 0
                else dimension - correction_features - 1
                if vocabulary_path == "boolean_hybrid"
                else dimension - 1
            )
            self.token_embedding = BooleanTokenEmbedding(
                padded_vocabulary_size, dimension, code_width, boolean_token_expansion
            )
        if token_smear != "none":
            self.smear_strength = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("smear_strength", None)
        if token_smear == "clipped_dynamic":
            self.smear_scale = nn.Parameter(torch.tensor(0.1))
            self.smear_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("smear_scale", None)
            self.register_parameter("smear_bias", None)
        value_layers = frozenset(value_embedding_layers)
        if token_feature == "value_embedding":
            self.value_embedding = nn.Embedding(padded_vocabulary_size, dimension)
            nn.init.normal_(self.value_embedding.weight, mean=0.0, std=0.01)
        else:
            self.register_module("value_embedding", None)
        self.ngram_orders = tuple(int(order) for order in ngram_orders)
        self.ngram_heads = ngram_heads
        self.ngram_table_rows = ngram_table_rows
        self.ngram_dimension = ngram_dimension
        if self.ngram_orders:
            # One region per (order, head). Rows start at zero, so the tables
            # begin as a no-op and have to earn their contribution.
            self.ngram_table = nn.Embedding(
                len(self.ngram_orders) * ngram_heads * ngram_table_rows, ngram_dimension
            )
            nn.init.zeros_(self.ngram_table.weight)
        else:
            self.register_module("ngram_table", None)
        if token_feature == "hashed_bigram":
            self.bigram_embedding = nn.Embedding(bigram_hash_size, bigram_dimension)
            nn.init.zeros_(self.bigram_embedding.weight)
            if bigram_learned_scale:
                self.bigram_scale = nn.Parameter(torch.ones(()))
            else:
                self.register_parameter("bigram_scale", None)
            if bigram_sign_mode == "vector":
                sign_rows = torch.arange(256, dtype=torch.int64)[:, None]
                sign_columns = torch.arange(bigram_dimension, dtype=torch.int64)[None, :]
                sign_bits = (
                    (sign_rows * 0x45D9F3B) ^ (sign_columns * 0x119DE1F3) ^
                    ((sign_rows + sign_columns) * 0x27D4EB2D)
                ) & 1
                self.register_buffer(
                    "bigram_sign_table",
                    (1 - 2 * sign_bits).to(torch.float32),
                    persistent=False,
                )
            else:
                self.register_buffer("bigram_sign_table", None, persistent=False)
        else:
            self.register_module("bigram_embedding", None)
            self.register_parameter("bigram_scale", None)
            self.register_buffer("bigram_sign_table", None, persistent=False)
        if residual_routing == "u_skip":
            self.u_skip_gates = nn.Parameter(torch.zeros(len(self.u_skip_pairs)))
        else:
            self.register_parameter("u_skip_gates", None)
        if residual_routing == "dynamic_input":
            route_count = len(self.dynamic_route_layers)
            self.dynamic_route_strength = nn.Parameter(torch.zeros(route_count))
            self.dynamic_route_scale = nn.Parameter(torch.full((route_count,), 0.1))
            self.dynamic_route_bias = nn.Parameter(torch.zeros(route_count))
        else:
            self.register_parameter("dynamic_route_strength", None)
            self.register_parameter("dynamic_route_scale", None)
            self.register_parameter("dynamic_route_bias", None)
        self.residual_normalizer = residual_normalizer
        self.input_norm = residual_norm(dimension, residual_normalizer)
        skip_layers = frozenset(self.attention_skip_layers)
        self.layers = nn.ModuleList(
            VocabularyBlock(
                dimension,
                heads,
                kv_heads,
                ff_dimension
                + (ff_extra_dimension if index < ff_extra_dimension_layers else 0),
                feedforward_activation,
                boolean_width,
                boolean_rounds,
                boolean_ffn_correction_features,
                boolean_ffn_correction_rank,
                boolean_ffn_correction_mode,
                boolean_ffn_correction_block_active_fraction,
                boolean_ffn_correction_taps,
                boolean_ffn_pool_schedule,
                boolean_ffn_routes,
                boolean_ffn_thresholds,
                boolean_ffn_input_rounding,
                boolean_backward,
                boolean_temperature,
                boolean_gradient_leak,
                boolean_program_init_margin,
                boolean_register_bits,
                context=context,
                rope_base=rope_base,
                rope_fraction=rope_fraction,
                partial_key_offset=partial_key_offset,
                qk_norm=qk_norm,
                qk_cache_precision=qk_cache_precision,
                value_cache_precision=value_cache_precision,
                qk_head_dimension=qk_head_dimension,
                local_qk_head_dimension=local_qk_head_dimension,
                kv_head_routing=kv_head_routing,
                kv_head_sharing=kv_head_sharing,
                attention_qkv_projection_mode=attention_qkv_projection_mode,
                qkv_squeeze_bitplanes=qkv_squeeze_bitplanes,
                qkv_squeeze_polarities=qkv_squeeze_polarities,
                qkv_squeeze_group_bits=qkv_squeeze_group_bits,
                qkv_squeeze_bypass=qkv_squeeze_bypass,
                qkv_squeeze_input_rounding=qkv_squeeze_input_rounding,
                qkv_low_rank=qkv_low_rank,
                qkv_shared_bases=qkv_shared_bases,
                qkv_local_width=qkv_local_width,
                qkv_global_rank=qkv_global_rank,
                qkv_route_count=qkv_route_count,
                qkv_bitmix_rounds=qkv_bitmix_rounds,
                qkv_bitmix_correction_features=qkv_bitmix_correction_features,
                qkv_bitmix_cell=qkv_bitmix_cell,
                qkv_block_allocation=qkv_block_allocation,
                qkv_block_rows=qkv_block_rows,
                qkv_ticket_selection=qkv_ticket_selection,
                qkv_ticket_temperature=qkv_ticket_temperature,
                qkv_head_input_groups=qkv_head_input_groups,
                attention_qk_mode=attention_qk_mode,
                boolean_qk_width=boolean_qk_width,
                boolean_qk_rounds=boolean_qk_rounds,
                boolean_qk_bits_per_head=boolean_qk_bits_per_head,
                boolean_qk_correction_features=boolean_qk_correction_features,
                boolean_qk_correction_taps=boolean_qk_correction_taps,
                attention_value_mode=attention_value_mode,
                attention_value_block_active_fraction=attention_value_block_active_fraction,
                boolean_value_width=boolean_value_width,
                boolean_value_rounds=boolean_value_rounds,
                boolean_value_bits=boolean_value_bits,
                boolean_value_correction_features=boolean_value_correction_features,
                attention_output_mode=attention_output_mode,
                boolean_output_width=boolean_output_width,
                boolean_output_rounds=boolean_output_rounds,
                boolean_output_correction_features=boolean_output_correction_features,
                boolean_output_correction_taps=boolean_output_correction_taps,
                boolean_output_routes=boolean_output_routes,
                boolean_output_cell=boolean_output_cell,
                attention_output_block_active_fraction=attention_output_block_active_fraction,
                attention_output_codewords=attention_output_codewords,
                attention_output_codebook_temperature=(
                    attention_output_codebook_temperature
                ),
                attention_output_codebook_response_shift=(
                    attention_output_codebook_response_shift
                ),
                attention_output_head_local_rank=(
                    attention_output_head_local_rank
                ),
                attention_output_squeeze_bitplanes=attention_output_squeeze_bitplanes,
                attention_output_squeeze_polarities=attention_output_squeeze_polarities,
                attention_output_squeeze_group_bits=attention_output_squeeze_group_bits,
                attention_output_squeeze_bypass=attention_output_squeeze_bypass,
                attention_output_input_rounding=attention_output_input_rounding,
                attention_normalizer=attention_normalizer,
                clipped_linear_initial_slope=clipped_linear_initial_slope,
                head_contexts=head_contexts(index),
                use_value_embedding=(
                    token_feature == "value_embedding" and index in value_layers
                ),
                skip_attention=index in skip_layers,
                dropout=dropout,
                zero_init_residual=zero_init_residual,
                residual_scalars=residual_scalars,
                boolean_residual_state=boolean_residual_state,
                residual_boundary=residual_boundary,
                residual_normalizer=residual_normalizer,
            )
            for index in range(layers)
        )
        self.final_norm = residual_norm(dimension, residual_normalizer)
        if vocabulary_path == "numerical":
            self.readout = nn.Linear(dimension, padded_vocabulary_size, bias=False)
            if tie_embeddings:
                self.readout.weight = self.token_embedding.weight
        else:
            if not isinstance(self.token_embedding, BooleanTokenEmbedding):
                raise RuntimeError("Boolean vocabulary embedding is missing")
            self.readout = BooleanVocabularyReadout(
                padded_vocabulary_size,
                dimension,
                self.token_embedding.code_width,
                vocabulary_path,
                temperature=boolean_temperature,
                pool_bits=boolean_vocab_pool_bits,
                pool_schedule=boolean_vocab_pool_schedule,
                correction_rank=boolean_vocab_correction_rank,
                factor_output=boolean_vocab_factor_output,
                hierarchy=vocabulary_hierarchy,
            )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        incompatible = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        deterministic_suffixes = (
            "token_embedding.derived_a_indices",
            "token_embedding.derived_b_indices",
            "token_embedding.derived_c_indices",
            "readout.correction_windows",
            "readout.correction_window_mask",
        )
        missing = [
            key for key in incompatible.missing_keys
            if not key.endswith(deterministic_suffixes)
        ]
        if strict and (missing or incompatible.unexpected_keys):
            raise RuntimeError(
                "state_dict mismatch: missing=" + repr(missing)
                + ", unexpected=" + repr(incompatible.unexpected_keys)
            )
        return incompatible

    def _bigram_hash(
        self, tokens: Tensor, previous_token: Tensor | None = None
    ) -> Tensor:
        previous = torch.empty_like(tokens)
        if previous_token is None:
            previous[:, 0] = 0
        else:
            previous[:, 0] = previous_token.reshape(-1)
        previous[:, 1:] = tokens[:, :-1]
        current = tokens.to(torch.int64)
        previous = previous.to(torch.int64)
        return ((36_313 * current) ^ (27_191 * previous)) & (
            self.bigram_hash_size - 1
        )

    _NGRAM_PAD = 256
    _NGRAM_MULTIPLIERS = (0x5BD1E995, 0x27D4EB2F, 0x165667B1, 0x61C88647)

    def _ngram_indices(self, tokens: Tensor) -> Tensor:
        """Table rows for each position's trailing n-grams: [batch, time, orders, heads].

        A 31-bit multiply-XOR hash over the last ``n`` bytes. Positions before
        the window start see a sentinel outside the byte range. Pure integer
        arithmetic, so a native decoder reproduces it exactly.
        """
        longest = self.ngram_orders[-1]
        history = F.pad(tokens.to(torch.int64), (longest - 1, 0), value=self._NGRAM_PAD)
        length = tokens.shape[1]
        orders = []
        for order_index, order in enumerate(self.ngram_orders):
            heads = []
            for head in range(self.ngram_heads):
                multiplier = self._NGRAM_MULTIPLIERS[head]
                value = torch.full_like(history[:, :length], order)
                for offset in range(longest - order, longest):
                    byte = history[:, offset : offset + length] + 1
                    value = ((value * multiplier) ^ byte) & 0x7FFFFFFF
                value = value ^ (value >> 15)
                region = order_index * self.ngram_heads + head
                heads.append(
                    (value & (self.ngram_table_rows - 1)) + region * self.ngram_table_rows
                )
            orders.append(torch.stack(heads, dim=-1))
        return torch.stack(orders, dim=-2)

    def _input_state(
        self,
        tokens: Tensor,
        previous_token: Tensor | None = None,
        ngram_history: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        embedded = self.token_embedding(tokens)
        if self.smear_strength is not None:
            first_previous = (
                torch.zeros_like(embedded[:, :1])
                if previous_token is None
                else self.token_embedding(previous_token.reshape(-1, 1))
            )
            previous_embedded = torch.cat(
                (first_previous, embedded[:, :-1]), dim=1
            )
            if self.smear_scale is None or self.smear_bias is None:
                gate: Tensor | float = 1.0
            else:
                gate = (
                    0.5
                    + self.smear_scale * embedded[..., :16].mean(dim=-1, keepdim=True)
                    + self.smear_bias
                ).clamp(0.0, 1.0)
            embedded = embedded + self.smear_strength * gate * previous_embedded
        if self.bigram_embedding is not None:
            previous = torch.empty_like(tokens)
            if previous_token is None:
                previous[:, 0] = 0
            else:
                previous[:, 0] = previous_token.reshape(-1)
            previous[:, 1:] = tokens[:, :-1]
            bigram = self.bigram_embedding(self._bigram_hash(tokens, previous_token))
            if self.bigram_sign_mode == "scalar":
                sign_bit = (
                    (tokens.to(torch.int64) * 1_103_515_245)
                    ^ (previous.to(torch.int64) * 12_345)
                ) >> 16
                bigram = bigram * (1 - 2 * (sign_bit & 1)).to(bigram.dtype)[..., None]
            elif self.bigram_sign_mode == "vector":
                if self.bigram_sign_table is None:
                    raise RuntimeError("vector bigram signs are missing their table")
                sign_index = (tokens.to(torch.int64) ^ previous.to(torch.int64)) & 255
                bigram = bigram * self.bigram_sign_table[sign_index].to(bigram.dtype)
            embedded = embedded.clone()
            scale: Tensor | float = (
                self.bigram_scale if self.bigram_scale is not None else 1.0
            )
            embedded[..., : self.bigram_dimension] += scale * bigram
        if self.ngram_table is not None:
            width = len(self.ngram_orders) * self.ngram_dimension
            source = (
                tokens
                if ngram_history is None
                else torch.cat((ngram_history.to(tokens), tokens), dim=1)
            )
            indices = self._ngram_indices(source)[:, -tokens.shape[1] :]
            features = self.ngram_table(indices).sum(dim=-2).flatten(-2)
            embedded = torch.cat(
                (embedded[..., :-width], embedded[..., -width:] + features), dim=-1
            )
        auxiliary = (
            self.value_embedding(tokens) if self.value_embedding is not None else None
        )
        return self.input_norm(embedded), auxiliary

    def _masked_logits(self, features: Tensor) -> Tensor:
        logits = (
            self.readout(features, self.token_embedding.weight)
            if isinstance(self.readout, BooleanVocabularyReadout)
            else self.readout(features)
        )
        if self.logical_vocabulary_size < self.padded_vocabulary_size:
            logits = logits.clone()
            logits[..., self.logical_vocabulary_size :] = -torch.inf
        return logits

    def _route_before_layer(
        self,
        state: Tensor,
        input_state: Tensor,
        snapshots: dict[int, Tensor],
        layer_index: int,
    ) -> Tensor:
        if self.u_skip_gates is not None:
            for route_index, (source, target) in enumerate(self.u_skip_pairs):
                if target == layer_index:
                    state = state + self.u_skip_gates[route_index] * snapshots[source]
        if (
            self.dynamic_route_strength is not None
            and layer_index in self.dynamic_route_layers
        ):
            route_index = self.dynamic_route_layers.index(layer_index)
            signal = state[..., : self.dynamic_route_dimension].mean(
                dim=-1, keepdim=True
            )
            gate = (
                0.5
                + self.dynamic_route_scale[route_index] * signal
                + self.dynamic_route_bias[route_index]
            ).clamp(0.0, 1.0)
            state = (
                state
                + self.dynamic_route_strength[route_index] * gate * input_state
            )
        return state

    def _record_routing_snapshot(
        self, snapshots: dict[int, Tensor], layer_index: int, state: Tensor
    ) -> None:
        if self.u_skip_gates is not None and any(
            source == layer_index for source, _ in self.u_skip_pairs
        ):
            snapshots[layer_index] = state

    def forward_features(self, tokens: Tensor, *, position_offset: int = 0) -> Tensor:
        positions = torch.arange(
            position_offset,
            position_offset + tokens.shape[1],
            device=tokens.device,
        )
        state, auxiliary = self._input_state(tokens)
        input_state = state
        snapshots: dict[int, Tensor] = {}
        boolean_carrier: Tensor | None = None
        shared_attention_source: Tensor | None = None
        shared_attention_qk: tuple[Tensor, Tensor] | None = None
        for layer_index, layer in enumerate(self.layers):
            state = self._route_before_layer(
                state, input_state, snapshots, layer_index
            )
            if (
                self.attention_shared_input_layers
                and layer_index == self.attention_shared_input_layers[0]
            ):
                shared_attention_source = state
            attention_source = (
                shared_attention_source
                if layer_index in self.attention_shared_input_layers
                else None
            )
            layer_auxiliary = (
                auxiliary
                if layer.attention.value_embedding_scale is not None
                else None
            )
            if layer_index in self.attention_shared_qk_layers:
                if attention_source is None:
                    raise RuntimeError("shared Q/K requires a shared attention source")
                state, projected_qk = layer.forward_with_shared_qk(
                    state,
                    positions,
                    layer_auxiliary,
                    attention_source,
                    shared_attention_qk,
                    self.attention_shared_qk_mode,
                )
                if shared_attention_qk is None:
                    shared_attention_qk = projected_qk
            elif self.boolean_residual_state == "numerical":
                state = layer(
                    state, positions, layer_auxiliary, attention_source
                )
            else:
                state, boolean_carrier = layer.forward_with_boolean_carrier(
                    state, positions, boolean_carrier, layer_auxiliary,
                    attention_source,
                )
            self._record_routing_snapshot(snapshots, layer_index, state)
        return self.final_norm(state)

    def forward(self, tokens: Tensor, *, position_offset: int = 0) -> Tensor:
        return self._masked_logits(
            self.forward_features(tokens, position_offset=position_offset)
        )

    def prefill(
        self, tokens: Tensor, *, position_offset: int = 0
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if tokens.shape[1] > self.context:
            raise ValueError("prefill exceeds model context")
        positions = torch.arange(
            position_offset,
            position_offset + tokens.shape[1],
            device=tokens.device,
        )
        state, auxiliary = self._input_state(tokens)
        input_state = state
        snapshots: dict[int, Tensor] = {}
        boolean_carrier: Tensor | None = None
        shared_attention_source: Tensor | None = None
        shared_attention_qk: tuple[Tensor, Tensor] | None = None
        caches: list[tuple[Tensor, Tensor]] = []
        for layer_index, layer in enumerate(self.layers):
            state = self._route_before_layer(
                state, input_state, snapshots, layer_index
            )
            if (
                self.attention_shared_input_layers
                and layer_index == self.attention_shared_input_layers[0]
            ):
                shared_attention_source = state
            attention_source = (
                shared_attention_source
                if layer_index in self.attention_shared_input_layers
                else state
            )
            layer_auxiliary = (
                auxiliary
                if layer.attention.value_embedding_scale is not None
                else None
            )
            scales = layer.residual_scales
            if layer.skip_attention:
                state = scales[0] * state
                empty = state.new_empty(0)
                layer_cache = (empty, empty)
            else:
                normalized = layer.attention_norm(attention_source)
                query, key, value = layer.attention._project(
                    normalized, positions, layer_auxiliary
                )
                if layer_index in self.attention_shared_qk_layers:
                    if shared_attention_qk is None:
                        shared_attention_qk = (query, key)
                    else:
                        if self.attention_shared_qk_mode == "qk":
                            query = shared_attention_qk[0]
                        key = shared_attention_qk[1]
                scores = layer.attention._scores(
                    query, layer.attention._expand(key), positions, positions
                )
                mask = layer.attention._causal_mask(positions)
                attended = layer.attention._route_attended_value(
                    layer.attention._normalize(scores, mask)
                    @ layer.attention._expand(value, value=True)
                ).transpose(1, 2).contiguous().flatten(-2)
                state = (
                    scales[0] * state
                    + scales[1] * layer.attention._output(attended)
                )
                layer_cache = (key, value)
            state = layer._after_attention_boundary(state)
            if self.boolean_residual_state == "numerical":
                state = scales[2] * state + scales[3] * layer._feedforward(
                    layer.feedforward_norm(state)
                )
            else:
                if layer.boolean_ffn is None:
                    raise RuntimeError("Boolean carrier requires a Boolean FFN")
                branch, boolean_carrier = layer.boolean_ffn.forward_with_carrier(
                    layer.feedforward_norm(state),
                    boolean_carrier,
                    self.boolean_residual_state,
                )
                state = scales[2] * state + scales[3] * layer.dropout(branch)
            state = layer._after_block_boundary(state)
            self._record_routing_snapshot(snapshots, layer_index, state)
            caches.append(layer_cache)
        return self._masked_logits(self.final_norm(state)), caches

    def decode_step(
        self,
        token: Tensor,
        caches: Sequence[tuple[Tensor, Tensor]],
        *,
        position: int,
        previous_token: Tensor | None = None,
        history: Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if token.ndim == 1:
            token = token[:, None]
        if token.shape[1] != 1:
            raise ValueError("decode_step accepts one token per batch")
        if self.ngram_table is not None and history is None:
            raise ValueError("n-gram tables need the preceding tokens: pass history")
        if len(caches) != len(self.layers):
            raise ValueError("one KV cache is required per layer")
        positions = torch.tensor([position], device=token.device)
        state, auxiliary = self._input_state(token, previous_token, history)
        input_state = state
        snapshots: dict[int, Tensor] = {}
        boolean_carrier: Tensor | None = None
        shared_attention_source: Tensor | None = None
        shared_attention_qk: tuple[Tensor, Tensor] | None = None
        updated: list[tuple[Tensor, Tensor]] = []
        for layer_index, (layer, cache) in enumerate(
            zip(self.layers, caches, strict=True)
        ):
            state = self._route_before_layer(
                state, input_state, snapshots, layer_index
            )
            if (
                self.attention_shared_input_layers
                and layer_index == self.attention_shared_input_layers[0]
            ):
                shared_attention_source = state
            attention_source = (
                shared_attention_source
                if layer_index in self.attention_shared_input_layers
                else None
            )
            layer_auxiliary = (
                auxiliary
                if layer.attention.value_embedding_scale is not None
                else None
            )
            if layer_index in self.attention_shared_qk_layers:
                if attention_source is None:
                    raise RuntimeError("shared Q/K requires a shared attention source")
                state, layer_cache, projected_qk = (
                    layer.forward_cached_with_shared_qk(
                        state,
                        positions,
                        cache,
                        layer_auxiliary,
                        attention_source,
                        shared_attention_qk,
                        self.attention_shared_qk_mode,
                    )
                )
                if shared_attention_qk is None:
                    shared_attention_qk = projected_qk
            elif self.boolean_residual_state == "numerical":
                state, layer_cache = layer.forward_cached(
                    state, positions, cache, layer_auxiliary, attention_source
                )
            else:
                state, layer_cache, boolean_carrier = (
                    layer.forward_cached_with_boolean_carrier(
                        state,
                        positions,
                        cache,
                        boolean_carrier,
                        layer_auxiliary,
                        attention_source,
                    )
                )
            self._record_routing_snapshot(snapshots, layer_index, state)
            updated.append(layer_cache)
        return self._masked_logits(self.final_norm(state)), updated

    def boolean_regularization(self) -> tuple[Tensor, Tensor]:
        zero = self.token_embedding.weight.new_zeros(())
        values = [
            layer.boolean_ffn.regularization()
            for layer in self.layers
            if layer.boolean_ffn is not None
        ]
        for layer in self.layers:
            if layer.skip_attention:
                continue
            if layer.attention.attention_qkv_projection_mode in {
                "signed_squeeze_hybrid", "bitmix_trajectory_hybrid"
            }:
                values.append(layer.attention.qkv.regularization())
            if layer.attention.boolean_query is not None:
                values.append((layer.attention.boolean_query.regularization(), zero))
                values.append((layer.attention.boolean_key.regularization(), zero))
            if layer.attention.boolean_value is not None:
                values.append((layer.attention.boolean_value.regularization(), zero))
            if layer.attention.attention_output_mode == "boolean_lut3_direct":
                values.append((layer.attention.output.regularization(), zero))
            elif layer.attention.attention_output_mode == "boolean_lut3_decoded":
                values.append(layer.attention.output.regularization())
            elif layer.attention.attention_output_mode == "signed_squeeze_hybrid":
                values.append(layer.attention.output.regularization())
        if not values:
            return zero, zero
        return (
            torch.stack([balance for balance, _ in values]).mean(),
            torch.stack([gate for _, gate in values]).mean(),
        )

    def boolean_diagnostics(self) -> dict[str, Any] | None:
        boolean_layers = [
            layer.boolean_ffn
            for layer in self.layers
            if layer.boolean_ffn is not None
        ]
        qk_layers = [
            layer.attention
            for layer in self.layers
            if not layer.skip_attention and layer.attention.boolean_query is not None
        ]
        value_layers = [
            layer.attention
            for layer in self.layers
            if not layer.skip_attention and layer.attention.boolean_value is not None
        ]
        output_layers = [
            layer.attention
            for layer in self.layers
            if not layer.skip_attention
            and layer.attention.attention_output_mode in {
                "boolean_lut3_direct",
                "boolean_lut3_decoded",
                "signed_squeeze_hybrid",
            }
        ]
        qkv_squeeze_layers = [
            layer.attention.qkv
            for layer in self.layers
            if not layer.skip_attention
            and layer.attention.attention_qkv_projection_mode in {
                "signed_squeeze_hybrid", "bitmix_trajectory_hybrid"
            }
        ]
        if not boolean_layers and not qk_layers and not value_layers and not output_layers:
            return None
        layers = [layer.diagnostics() for layer in boolean_layers]
        result: dict[str, Any] = {
            "layers": layers,
            "mean": (
                {}
                if not layers
                else {
                    name: sum(layer[name] for layer in layers) / len(layers)
                    for name in tuple(layers[0])
                }
            ),
            "deployed_program_bits": sum(
                layer.deployed_program_bits()
                for layer in boolean_layers
            ) + sum(
                attention.boolean_query.program_logits.numel()
                + attention.boolean_key.program_logits.numel()
                for attention in qk_layers
            ) + sum(
                attention.boolean_value.program_logits.numel()
                for attention in value_layers
            ) + sum(
                (
                    attention.output.deployed_program_bits()
                    if attention.attention_output_mode in {
                        "boolean_lut3_decoded", "signed_squeeze_hybrid"
                    }
                    else attention.output.program_logits.numel()
                )
                for attention in output_layers
            ) + sum(
                projection.deployed_program_bits()
                for projection in qkv_squeeze_layers
            ),
            "training_candidate_program_values": sum(
                layer.training_candidate_program_values()
                for layer in boolean_layers
            ),
        }
        if qk_layers:
            qk = [
                {
                    "query": attention.boolean_query.diagnostics(),
                    "key": attention.boolean_key.diagnostics(),
                }
                for attention in qk_layers
            ]
            result["qk_layers"] = qk
        if value_layers:
            result["value_layers"] = [
                attention.boolean_value.diagnostics()
                for attention in value_layers
            ]
        if output_layers:
            result["output_layers"] = [
                attention.output.diagnostics()
                for attention in output_layers
            ]
        if qkv_squeeze_layers:
            result["qkv_squeeze_layers"] = [
                projection.diagnostics() for projection in qkv_squeeze_layers
            ]
        block_layers = [
            layer.attention.output
            for layer in self.layers
            if layer.attention.attention_output_mode == "block_sparse_16x16"
        ]
        if block_layers:
            result["deployed_program_bits"] += sum(
                layer.block_mask.numel() for layer in block_layers
            )
            result["attention_output_blocks"] = {
                "active_fraction": sum(
                    float(layer.block_mask.mean()) for layer in block_layers
                ) / len(block_layers),
                "active_weight_values": sum(
                    layer.active_weight_values() for layer in block_layers
                ),
            }
        value_block_layers = [
            layer.attention.qkv
            for layer in self.layers
            if layer.attention.attention_value_mode == "block_sparse_16x16"
        ]
        if value_block_layers:
            result["deployed_program_bits"] += sum(
                layer.value_block_mask.numel() for layer in value_block_layers
            )
            result["attention_value_blocks"] = {
                "active_fraction": sum(
                    float(layer.value_block_mask.mean()) for layer in value_block_layers
                ) / len(value_block_layers),
                "active_weight_values": sum(
                    layer.active_weight_values() for layer in value_block_layers
                ),
            }
        qkv_block_layers = [
            layer.attention.qkv
            for layer in self.layers
            if layer.attention.attention_qkv_projection_mode == "block_sparse_16x16"
        ]
        if qkv_block_layers:
            result["deployed_program_bits"] += sum(
                int(layer.permitted_block_mask.sum()) for layer in qkv_block_layers
            )
            result["training_candidate_program_values"] += sum(
                layer.training_candidate_program_values()
                for layer in qkv_block_layers
            )
            first_mask = qkv_block_layers[0].block_mask
            q_rows, k_rows, v_rows = qkv_block_layers[0].region_row_blocks
            result["qkv_blocks"] = {
                "active_fraction": sum(
                    float(layer.block_mask.mean()) for layer in qkv_block_layers
                ) / len(qkv_block_layers),
                "active_permitted_fraction": sum(
                    float(layer.deployed_mask().sum())
                    / int(layer.permitted_block_mask.sum())
                    for layer in qkv_block_layers
                ) / len(qkv_block_layers),
                "permitted_fraction": sum(
                    float(layer.permitted_block_mask.float().mean())
                    for layer in qkv_block_layers
                ) / len(qkv_block_layers),
                "active_weight_values": sum(
                    layer.active_weight_values() for layer in qkv_block_layers
                ),
                "allocation": qkv_block_layers[0].allocation,
                "block_rows": qkv_block_layers[0].block_rows,
                "ticket_selection": qkv_block_layers[0].ticket_selection,
                "head_input_groups": qkv_block_layers[0].head_input_groups,
                "ticket_change_fraction": sum(
                    float(
                        (
                            layer.deployed_mask()
                            != layer.initial_block_mask
                        ).float().mean()
                    )
                    for layer in qkv_block_layers
                ) / len(qkv_block_layers),
                "ticket_gradient_nonzero_fraction": sum(
                    0.0
                    if layer.ticket_logits is None
                    or layer.ticket_logits.grad is None
                    else float((layer.ticket_logits.grad != 0).float().mean())
                    for layer in qkv_block_layers
                ) / len(qkv_block_layers),
                "ticket_gradient_mean_abs": sum(
                    0.0
                    if layer.ticket_logits is None
                    or layer.ticket_logits.grad is None
                    else float(layer.ticket_logits.grad.abs().mean())
                    for layer in qkv_block_layers
                ) / len(qkv_block_layers),
                "q_fraction": float(first_mask[:q_rows].mean()),
                "k_fraction": float(first_mask[q_rows : q_rows + k_rows].mean()),
                "v_fraction": float(first_mask[q_rows + k_rows :].mean()),
            }
        return result

    def isa_manifest(self) -> dict[str, Any]:
        return {
            "graph": "vocabulary_transformer_v0",
            "logical_vocabulary_size": self.logical_vocabulary_size,
            "padded_vocabulary_size": self.padded_vocabulary_size,
            "dimension": self.dimension,
            "context_tokens": self.context,
            "attention_skip_layers": list(self.attention_skip_layers),
            "attention_shared_input_layers": list(
                self.attention_shared_input_layers
            ),
            "attention_shared_qk_layers": list(self.attention_shared_qk_layers),
            "attention_shared_qk_mode": self.attention_shared_qk_mode,
            "attention_shared_qk_program": (
                None
                if not self.attention_shared_qk_layers
                else {
                    "source_layer": self.attention_shared_qk_layers[0],
                    "consumer_layers": list(self.attention_shared_qk_layers[1:]),
                    "reuse": self.attention_shared_qk_mode,
                    "target_instructions": [
                        "register/cache reuse",
                        "omit complete 8x16 SDOT Q/K row blocks",
                    ],
                }
            ),
            "attention_skip_program": (
                None
                if not self.attention_skip_layers
                else {
                    "representation": "checkpoint-static whole attention-block omission",
                    "target_instructions": [
                        "FMUL residual",
                        "omit attention norm/QKV/RoPE/KV/score/output projection",
                    ],
                }
            ),
            "attention_normalizer": self.layers[0].attention.attention_normalizer,
            "rope_fraction": self.layers[0].attention.rope_fraction,
            "partial_key_offset": self.layers[0].attention.partial_key_offset,
            "kv_head_routing": self.layers[0].attention.kv_head_routing,
            "kv_head_sharing": self.layers[0].attention.kv_head_sharing,
            "kv_head_routing_instructions": (
                ["register reuse"]
                if self.layers[0].attention.kv_head_routing == "repeat"
                else ["fixed address offset", "LD1", "ST1"]
            ),
            "qkv_projection": (
                {
                    "representation": "W8A8 dense row-major",
                    "target_instructions": ["FCVTNS", "SQXTN", "SDOT"],
                }
                if self.layers[0].attention.attention_qkv_projection_mode == "dense"
                else {
                    "representation": "checkpoint-static QKV SDOT tickets",
                    "active_fraction": float(
                        self.layers[0].attention.qkv.block_mask.mean()
                    ),
                    "active_permitted_fraction": (
                        float(self.layers[0].attention.qkv.deployed_mask().sum())
                        / int(
                            self.layers[0].attention.qkv.permitted_block_mask.sum()
                        )
                    ),
                    "permitted_fraction": float(
                        self.layers[0]
                        .attention.qkv.permitted_block_mask.float().mean()
                    ),
                    "allocation": self.layers[0].attention.qkv.allocation,
                    "block_rows": self.layers[0].attention.qkv.block_rows,
                    "block_columns": 16,
                    "ticket_selection": (
                        self.layers[0].attention.qkv.ticket_selection
                    ),
                    "head_input_groups": (
                        self.layers[0].attention.qkv.head_input_groups
                    ),
                    "target_instructions": [
                        "FCVTNS", "SQXTN", "SDOT", "fixed block skips"
                    ],
                }
                if self.layers[0].attention.attention_qkv_projection_mode
                == "block_sparse_16x16"
                else {
                    "representation": "shared Q/K W8A8 bases plus head FP16 transforms",
                    "bases": self.layers[0].attention.qkv.bases,
                    "base_rows": self.layers[0].attention.qkv.base_rows,
                    "transform_shape": [
                        self.layers[0].attention.heads,
                        self.layers[0].attention.qk_head_dim,
                        self.layers[0].attention.qk_head_dim,
                    ],
                    "target_instructions": [
                        "FCVTNS", "SQXTN", "SDOT", "FCVTN", "FMLAL", "FMLAL2"
                    ],
                }
                if self.layers[0].attention.attention_qkv_projection_mode
                == "shared_qk_basis"
                else {
                    "representation": (
                        "head-local W8 Q/K plus shared global W8/FP16 basis"
                    ),
                    "local_width": self.layers[0].attention.qkv.local_width,
                    "global_rank": self.layers[0].attention.qkv.global_rank,
                    "target_instructions": [
                        "FCVTNS", "SQXTN", "SDOT", "FCVTN", "FMLAL", "FMLAL2"
                    ],
                }
                if self.layers[0].attention.attention_qkv_projection_mode
                == "local_global_qk"
                else {
                    "representation": (
                        "head-specific W8 residual bases plus shared W8 correction with separate FP16 Q/K transforms"
                        if self.layers[0].attention.qkv.global_rank
                        else "head-specific W8 residual bases with separate FP16 Q/K transforms"
                        if self.layers[0].attention.qkv.transformed
                        else "head-specific W8 residual bases tied across Q/K"
                    ),
                    "base_rows": self.layers[0].attention.qkv.base_weight.shape[0],
                    "global_rank": self.layers[0].attention.qkv.global_rank,
                    "target_instructions": (
                        ["FCVTNS", "SQXTN", "SDOT", "FCVTN", "FMLAL", "FMLAL2"]
                        if self.layers[0].attention.qkv.transformed
                        else ["FCVTNS", "SQXTN", "SDOT", "vector copy"]
                    ),
                }
                if self.layers[0].attention.attention_qkv_projection_mode
                in {"tied_qk", "head_subspace_qk", "head_subspace_global_qk"}
                else {
                    "representation": "FP16 weighted fixed cyclic Q/K routes plus W8 shared V",
                    "routes": self.layers[0].attention.qkv.routes,
                    "offsets": self.layers[0].attention.qkv.route_offsets.tolist(),
                    "target_instructions": [
                        "FCVTN", "duplicated contiguous loads", "FMLAL", "FMLAL2",
                        "FCVTNS", "SQXTN", "SDOT",
                    ],
                }
                if self.layers[0].attention.attention_qkv_projection_mode
                == "cyclic_routes_qk"
                else {
                    "representation": "two-stage low-rank W8A8 row-major",
                    "rank": self.layers[0].attention.qkv.rank,
                    "target_instructions": ["FCVTNS", "SQXTN", "SDOT", "SCVTF", "FMUL"],
                }
                if self.layers[0].attention.attention_qkv_projection_mode == "low_rank_w8a8"
                else {
                    "representation": "packed bitmix trajectory counts plus aligned dense bypass",
                    "bitplanes": self.layers[0].attention.qkv.bitplanes,
                    "rounds": self.layers[0].attention.qkv.rounds,
                    "cell": self.layers[0].attention.qkv.cell,
                    "correction_features": self.layers[0].attention.qkv.correction_features,
                    "bypass_features": self.layers[0].attention.qkv.bypass_features,
                    "input_rounding": self.layers[0].attention.qkv.input_rounding,
                    "target_instructions": [
                        "FCVTN", "FCMGE", "EOR3" if self.layers[0].attention.qkv.cell == "eor3" else "BCAX",
                        "BSL", "EOR", "CNT", "SDOT",
                    ],
                }
                if self.layers[0].attention.attention_qkv_projection_mode
                == "bitmix_trajectory_hybrid"
                else {
                    "representation": "FCMGE signed-popcount squeeze plus dense bypass",
                    "bitplanes": self.layers[0].attention.qkv.bitplanes,
                    "polarities": self.layers[0].attention.qkv.polarities,
                    "group_bits": self.layers[0].attention.qkv.group_bits,
                    "squeeze_features": self.layers[0].attention.qkv.squeeze_features,
                    "bypass_features": self.layers[0].attention.qkv.bypass_features,
                    "input_rounding": self.layers[0].attention.qkv.input_rounding,
                    "target_instructions": (
                        ["FCVTN", "FCMGE", "EOR", "CNT", "SDOT"]
                        if self.layers[0].attention.qkv.input_rounding == "fp16"
                        else ["FMUL", "FRINTN", "FMUL", "FCMGE", "EOR", "CNT", "SDOT"]
                    ),
                }
            ),
            "attention_qk_mode": self.layers[0].attention.attention_qk_mode,
            "attention_value_mode": self.layers[0].attention.attention_value_mode,
            "attention_output_mode": self.layers[0].attention.attention_output_mode,
            "boolean_residual_state": self.boolean_residual_state,
            "residual_normalizer": self.residual_normalizer,
            "vocabulary_path": self.vocabulary_path,
            "boolean_qk": (
                None
                if self.layers[0].attention.boolean_query is None
                else {
                    "width_bits": self.layers[0].attention.boolean_query.width,
                    "rounds": self.layers[0].attention.boolean_query.rounds,
                    "bits_per_head": self.layers[0].attention.boolean_qk_bits_per_head,
                    "correction_features": self.layers[0].attention.boolean_qk_correction_features,
                    "correction_taps": self.layers[0].attention.boolean_query.correction_taps,
                    "correction_dims_per_head": self.layers[0].attention.boolean_qk_correction_dims_per_head,
                    "score": "XNOR plus CNT plus integer reduction",
                    "routes": ["global byte rotate", "lane64 XAR rotation"],
                    "relative_position": (
                        "fixed linear integer distance bias"
                        if self.layers[0].attention.boolean_relative_bias is None
                        else "learned four-bin distance bias"
                    ),
                    "target_instructions": ["EOR", "CNT", "UADDLV", "XAR"],
                }
            ),
            "boolean_value": (
                None
                if self.layers[0].attention.boolean_value is None
                else {
                    "width_bits": self.layers[0].attention.boolean_value.width,
                    "rounds": self.layers[0].attention.boolean_value.rounds,
                    "output_bits": self.layers[0].attention.boolean_value_bits,
                    "correction_features": self.layers[0].attention.boolean_value.correction_features,
                    "correction_dims": self.layers[0].attention.boolean_value_correction_dims,
                    "representation": self.layers[0].attention.attention_value_mode,
                    "routes": ["global byte rotate", "lane64 XAR rotation"],
                    "target_instructions": (
                        ["BSL", "EOR", "XAR", "CNT", "SDOT", "FMLAL", "FMLAL2"]
                        if self.layers[0].attention.boolean_value_correction_dims
                        else ["BSL", "EOR", "XAR"]
                    ),
                }
            ),
            "value_block_tickets": (
                None
                if self.layers[0].attention.attention_value_mode
                != "block_sparse_16x16"
                else {
                    "representation": "dense Q/K plus checkpoint-static 16x16 V blocks",
                    "active_fraction": float(
                        self.layers[0].attention.qkv.value_block_mask.mean()
                    ),
                    "target_instructions": ["SDOT", "fixed block skips"],
                }
            ),
            "boolean_attention_output": (
                None
                if self.layers[0].attention.attention_output_mode == "numerical"
                else (
                    {
                        "representation": "checkpoint-static 16x16 blocks",
                        "active_fraction": float(
                            self.layers[0].attention.output.block_mask.mean()
                        ),
                        "target_instructions": ["SDOT", "fixed block skips"],
                    }
                    if self.layers[0].attention.attention_output_mode == "block_sparse_16x16"
                    else {
                        "representation": "16-lane SDOT product codebooks with hard TBL indices",
                        "codewords": self.layers[0].attention.output.codewords,
                        "response_shift": (
                            self.layers[0].attention.output.response_shift
                        ),
                        "program_bits": (
                            self.layers[0].attention.output.deployed_program_bits()
                        ),
                        "target_instructions": [
                            "FCVTN", "FCVTNS", "SQXTN", "SDOT", "TBL", "SADDW"
                        ],
                    }
                    if self.layers[0].attention.attention_output_mode
                    == "product_codebook_16"
                    else {
                        "representation": (
                            "head-local 32x32 W8A8 blocks plus rank-8 W8/FP16 correction"
                            if self.layers[0].attention.output.rank
                            else "head-local 32x32 W8A8 blocks"
                        ),
                        "heads": self.layers[0].attention.output.heads,
                        "head_dimension": (
                            self.layers[0].attention.output.head_dimension
                        ),
                        "global_rank": self.layers[0].attention.output.rank,
                        "target_instructions": [
                            "FCVTN", "FCVTNS", "SQXTN", "SDOT",
                            "FMLAL", "FMLAL2",
                        ],
                    }
                    if self.layers[0].attention.attention_output_mode
                    == "head_local_hybrid"
                    else {
                        "representation": "FCMGE signed-popcount squeeze plus dense bypass",
                        "bitplanes": self.layers[0].attention.output.bitplanes,
                        "polarities": self.layers[0].attention.output.polarities,
                        "group_bits": self.layers[0].attention.output.group_bits,
                        "squeeze_features": self.layers[0].attention.output.squeeze_features,
                        "bypass_features": self.layers[0].attention.output.bypass_features,
                        "target_instructions": ["FCMGE", "EOR", "CNT", "SDOT"],
                    }
                    if self.layers[0].attention.attention_output_mode == "signed_squeeze_hybrid"
                else {
                    "width_bits": self.layers[0].attention.output.width,
                    "rounds": self.layers[0].attention.output.rounds,
                    "correction_features": self.layers[0].attention.output.correction_features,
                    "correction_taps": self.layers[0].attention.output.correction_taps,
                    "route_schedule": self.layers[0].attention.output.route_schedule,
                    "cell": self.layers[0].attention.output.cell,
                    "input_rounding": self.layers[0].attention.output.input_rounding,
                    "representation": self.layers[0].attention.attention_output_mode,
                    "routes": ["global byte rotate", "lane64 XAR rotation"],
                    "target_instructions": (
                        ["BSL", "EOR", "XAR", "CNT", "SDOT", "FMLA"]
                        if self.layers[0].attention.output.correction_weight is not None
                        else ["BSL", "EOR", "XAR", "FMLA"]
                    ),
                })
            ),
            "attention_window_policy": self.attention_window_policy,
            "attention_head_contexts": [
                list(layer.attention.head_contexts) for layer in self.layers
            ],
            "feedforward_activation": self.layers[0].feedforward_activation,
            "boolean_ffn": (
                None
                if self.layers[0].boolean_ffn is None
                else {
                    "width_bits": self.layers[0].boolean_ffn.width,
                    "rounds": self.layers[0].boolean_ffn.rounds,
                    "correction_features": self.layers[0].boolean_ffn.correction_features,
                    "correction_rank": self.layers[0].boolean_ffn.correction_rank,
                    "correction_mode": self.layers[0].boolean_ffn.correction_mode,
                    "correction_taps": self.layers[0].boolean_ffn.correction_taps,
                    "pool_schedule": self.layers[0].boolean_ffn.pool_schedule,
                    "route_schedule": self.layers[0].boolean_ffn.route_schedule,
                    "threshold_schedule": self.layers[0].boolean_ffn.threshold_schedule,
                    "input_rounding": self.layers[0].boolean_ffn.input_rounding,
                    "cell": self.layers[0].boolean_ffn.cell,
                    "program": "exact hard Boolean cell plus XOR residual",
                    "routes": (
                        ["global byte rotate", "lane64 rotation"]
                        if self.layers[0].boolean_ffn.route_schedule == "byte_lane"
                        else ["global byte rotate", "register rename/rotate"]
                        if self.layers[0].boolean_ffn.route_schedule == "register_mix"
                        else ["global byte rotate", "full-state SHL/SRI bit rotation"]
                    ),
                    "target_instructions": (
                        (
                            ["FCVTN", "FCVTL"]
                            if self.layers[0].boolean_ffn.input_rounding == "fp16"
                            else ["FMUL", "FRINTN"]
                            if self.layers[0].boolean_ffn.input_rounding == "fixed8"
                            else []
                        )
                        +
                        (["BSL", "EOR"] + (
                            ["CNT", "SDOT", "TBL", "SADDW"]
                            if self.layers[0].boolean_ffn.correction_codebook is not None
                            else ["CNT", "SDOT"]
                            if self.layers[0].boolean_ffn.correction_weight is not None
                            else ["CNT", "SDOT", "FCVTN", "FMLAL", "FMLAL2"]
                            if self.layers[0].boolean_ffn.correction_input_weight is not None
                            else []
                        ))
                        if self.layers[0].boolean_ffn.cell == "lut3"
                        else ["EOR3", "AND", "EOR"]
                        if self.layers[0].boolean_ffn.cell == "factorized_minterm"
                        else ["BCAX", "BSL", "EOR3"]
                    ),
                    "hardware_block_bits": self.layers[0].boolean_ffn.register_bits,
                    "gated": self.layers[0].boolean_ffn.block_gate_logits is not None,
                    "persistent_state": self.boolean_residual_state,
                    "carrier_instructions": (
                        []
                        if self.boolean_residual_state == "numerical"
                        else ["EOR"]
                        if self.boolean_residual_state == "persistent_xor"
                        else ["EXT byte route", "BCAX"]
                        if self.boolean_residual_state == "persistent_bcax"
                        else ["fixed byte mask", "BSL"]
                    ),
                }
            ),
            "token_feature": self.token_feature,
            "boolean_vocabulary": (
                None
                if not isinstance(self.readout, BooleanVocabularyReadout)
                else {
                    "mode": self.vocabulary_path,
                    "code_bits_per_token": self.token_embedding.code_width,
                    "token_expansion": self.token_embedding.expansion_mode,
                    "correction_features": self.readout.correction_features,
                    "pool_bits": self.readout.pool_bits,
                    "pool_schedule": self.readout.pool_schedule,
                    "correction_rank": self.readout.correction_rank,
                    "factor_output": self.readout.factor_output,
                    "hierarchy": self.readout.hierarchy,
                    "cluster_count": (
                        32 if self.readout.hierarchy == "balanced_32x64" else 0
                    ),
                    "leaves_per_cluster": (
                        64 if self.readout.hierarchy == "balanced_32x64" else 0
                    ),
                    "layout": (
                        "balanced cluster-major 32x64 packed codes"
                        if self.readout.hierarchy == "balanced_32x64"
                        else "token-major contiguous packed codes"
                    ),
                    "hierarchy_target_instructions": (
                        [
                            "FCMGE", "pack", "EOR", "CNT", "UADDLV",
                            "FMLAL", "FMLAL2", "FMAX", "FMLA exp2 polynomial",
                            "FADDP", "FRECPE", "FRECPS", "FMUL",
                        ]
                        if self.readout.hierarchy == "balanced_32x64"
                        else []
                    ),
                    "target_instructions": (
                        ["FCMGE", "pack", "EOR", "CNT", "UADDLV", "SCVTF", "FADD"]
                        if self.readout.correction_weight is None
                        and self.readout.correction_input_weight is None
                        else (
                            [
                                "FCMGE", "pack", "EOR", "CNT", "UADDLV",
                                "SDOT", "SCVTF", "FCVTN", "FMLAL", "FMLAL2", "FADD",
                            ]
                            if self.readout.factor_output == "fp16"
                            else [
                                "FCMGE", "pack", "EOR", "CNT", "UADDLV",
                                "SDOT", "SCVTF", "FMAX", "FCVTNS", "SQXTN",
                                "SDOT", "SCVTF", "FMUL", "FADD",
                            ]
                        )
                        if self.readout.correction_input_weight is not None
                        else ["FCMGE", "pack", "EOR", "CNT", "UADDLV", "SDOT", "SCVTF", "FMLA"]
                    ),
                }
            ),
            "bigram_sign_mode": self.bigram_sign_mode,
            "bigram_learned_scale": self.bigram_learned_scale,
            "token_smear": self.token_smear,
            "residual_routing": self.residual_routing,
            "subsections": {
                "embedding": ["integer table address", "FP16 load", "FCVTL"],
                "token_feature": (
                    ["none"]
                    if self.token_feature == "plain"
                    else ["integer table address", "FP16 load", "FMUL", "FMLA"]
                    if self.token_feature == "value_embedding"
                    else ["integer MUL", "EOR", "AND", "integer table address", "FP16 load", "FMLA"]
                ),
                "bigram_collision_sign": (
                    ["none"]
                    if self.bigram_sign_mode == "none"
                    else ["integer MUL", "EOR", "LSR", "AND", "sign-bit EOR"]
                    if self.bigram_sign_mode == "scalar"
                    else [
                        "integer EOR",
                        "AND",
                        "fixed sign-table load",
                        "vector sign-bit EOR",
                    ]
                ),
                "token_smear": (
                    ["none"]
                    if self.token_smear == "none"
                    else ["previous FP16 embedding load", "FMUL", "FMLA"]
                    if self.token_smear == "static"
                    else [
                        "previous FP16 embedding load",
                        "16-lane FMLA reduction",
                        "FMAX",
                        "FMIN",
                        "FMUL",
                        "FMLA",
                    ]
                ),
                "residual_normalizer": (
                    ["FMLA reduction", "FRSQRTE", "FRSQRTS", "FMUL"]
                    if self.residual_normalizer == "rms"
                    else [
                        "FADD reduction",
                        "FSUB centering",
                        "FMLA reduction",
                        "FRSQRTE",
                        "FRSQRTS",
                        "FMUL",
                    ]
                    if self.residual_normalizer == "centered_rms"
                    else [
                        "FADD reduction",
                        "FSUB centering",
                        "FABS",
                        "FADD reduction",
                        "FRECPE",
                        "FRECPS",
                        "FMUL",
                    ]
                ),
                "rms_qk_norm": ["FMLA reduction", "FRSQRTE", "FRSQRTS", "FMUL"],
                "rope": [
                    "precomputed FP16 sin/cos load",
                    "FMUL",
                    "FMLA",
                    "FMLS",
                    f"rotated fraction {self.layers[0].attention.rope_fraction}",
                ],
                "partial_key_offset": (
                    ["none"]
                    if self.layers[0].attention.partial_key_offset == "none"
                    else ["EXT/load previous key quarters", "BSL"]
                ),
                "dense_projections": ["FP16 load", "FMLAL", "FMLAL2"],
                "attention": (
                    ["FMLA", "FMAX", "FCVTMS", "SHL/ADD exponent bits", "FMLA quartic", "FRECPE", "FRECPS"]
                    if self.layers[0].attention.attention_normalizer == "isa_exp2"
                    else ["quality reference: exact exponential and reduction"]
                    if self.layers[0].attention.attention_normalizer == "exact"
                    else [
                        "FMAX",
                        "FMLA",
                        *(
                            ["second fixed-slope FMLA/FMAX"]
                            if self.layers[0].attention.attention_normalizer
                            == "clipped_dual_scale"
                            else ["second and third fixed-slope FMLA/FMAX"]
                            if self.layers[0].attention.attention_normalizer
                            == "clipped_triple_scale"
                            else []
                        ),
                        *(
                            ["FMUL"]
                            if self.layers[0].attention.attention_normalizer
                            == "clipped_squared"
                            else ["FMUL", "FMUL"]
                            if self.layers[0].attention.attention_normalizer
                            == "clipped_cubic"
                            else []
                        ),
                        "reduction",
                        "FRECPE",
                        "FRECPS",
                    ]
                ),
                "feedforward_activation": (
                    ["FMAX", "FMUL"]
                    if self.layers[0].feedforward_activation == "relu_squared"
                    else ["FMAX", "FMUL"]
                    if self.layers[0].feedforward_activation == "reglu"
                    else ["FMAX", "FMIN", "FCVTMS", "SHL/ADD exponent bits", "FMLA quartic", "FRECPE", "FRECPS", "FMUL"]
                    if self.layers[0].feedforward_activation == "isa_swiglu"
                    else [
                        "FCMGE threshold pack",
                        "EXT byte routes",
                        "XAR lane64 routes",
                        "BSL LUT3",
                        "EOR residual",
                        "signed bitplane squeeze",
                        "FMLA",
                    ]
                ),
                "residuals": (
                    ["FMUL", "FMLA"]
                    if self.residual_routing == "ordinary"
                    else ["FMUL", "FMLA", "two transient vector snapshots"]
                    if self.residual_routing == "u_skip"
                    else [
                        "FMLA reduction",
                        "FMUL",
                        "FMLA",
                        "FMAX",
                        "FMIN",
                    ]
                ),
                "sampling": ["FMAX", "FCVTMS", "SHL/ADD exponent bits", "FMLA quartic", "FRECPE", "FRECPS"],
            },
        }


def build_model(config, manifest: dict[str, Any]) -> VocabularyTransformerLM:
    multiple = config.vocabulary_padding_multiple
    logical = int(manifest["logical_vocabulary_size"])
    padded = math.ceil(logical / multiple) * multiple
    return VocabularyTransformerLM(
        logical_vocabulary_size=logical,
        padded_vocabulary_size=padded,
        context=config.context_tokens,
        dimension=config.dimension,
        layers=config.layers,
        heads=config.heads,
        kv_heads=config.kv_heads,
        ff_dimension=config.ff_dimension,
        ff_extra_dimension=config.ff_extra_dimension,
        ff_extra_dimension_layers=config.ff_extra_dimension_layers,
        feedforward_activation=config.feedforward_activation,
        boolean_width=config.boolean_width,
        boolean_rounds=config.boolean_rounds,
        boolean_ffn_correction_features=config.boolean_ffn_correction_features,
        boolean_ffn_correction_rank=config.boolean_ffn_correction_rank,
        boolean_ffn_correction_mode=config.boolean_ffn_correction_mode,
        boolean_ffn_correction_block_active_fraction=(
            config.boolean_ffn_correction_block_active_fraction
        ),
        boolean_ffn_correction_taps=config.boolean_ffn_correction_taps,
        boolean_ffn_pool_schedule=config.boolean_ffn_pool_schedule,
        boolean_ffn_routes=config.boolean_ffn_routes,
        boolean_ffn_thresholds=config.boolean_ffn_thresholds,
        boolean_ffn_input_rounding=config.boolean_ffn_input_rounding,
        boolean_backward=config.boolean_backward,
        boolean_temperature=config.boolean_temperature,
        boolean_gradient_leak=config.boolean_gradient_leak,
        boolean_program_init_margin=config.boolean_program_init_margin,
        boolean_register_bits=config.boolean_register_bits,
        boolean_residual_state=config.boolean_residual_state,
        residual_boundary=config.residual_boundary,
        residual_normalizer=config.residual_normalizer,
        token_feature=config.token_feature,
        token_smear=config.token_smear,
        value_embedding_layers=config.value_embedding_layers,
        bigram_hash_size=config.bigram_hash_size,
        bigram_dimension=config.bigram_dimension,
        bigram_sign_mode=config.bigram_sign_mode,
        bigram_learned_scale=config.bigram_learned_scale,
        ngram_orders=config.ngram_orders,
        ngram_heads=config.ngram_heads,
        ngram_table_rows=config.ngram_table_rows,
        ngram_dimension=config.ngram_dimension,
        rope_base=config.rope_base,
        rope_fraction=config.rope_fraction,
        partial_key_offset=config.partial_key_offset,
        qk_norm=config.qk_norm,
        qk_cache_precision=config.qk_cache_precision,
        value_cache_precision=config.value_cache_precision,
        qk_head_dimension=config.qk_head_dimension,
        local_qk_head_dimension=config.local_qk_head_dimension,
        kv_head_routing=config.kv_head_routing,
        kv_head_sharing=config.kv_head_sharing,
        attention_qkv_projection_mode=config.attention_qkv_projection_mode,
        qkv_squeeze_bitplanes=config.qkv_squeeze_bitplanes,
        qkv_squeeze_polarities=config.qkv_squeeze_polarities,
        qkv_squeeze_group_bits=config.qkv_squeeze_group_bits,
        qkv_squeeze_bypass=config.qkv_squeeze_bypass,
        qkv_squeeze_input_rounding=config.qkv_squeeze_input_rounding,
        qkv_low_rank=config.qkv_low_rank,
        qkv_shared_bases=config.qkv_shared_bases,
        qkv_local_width=config.qkv_local_width,
        qkv_global_rank=config.qkv_global_rank,
        qkv_route_count=config.qkv_route_count,
        qkv_bitmix_rounds=config.qkv_bitmix_rounds,
        qkv_bitmix_correction_features=config.qkv_bitmix_correction_features,
        qkv_bitmix_cell=config.qkv_bitmix_cell,
        qkv_block_allocation=config.qkv_block_allocation,
        qkv_block_rows=config.qkv_block_rows,
        qkv_ticket_selection=config.qkv_ticket_selection,
        qkv_ticket_temperature=config.qkv_ticket_temperature,
        qkv_head_input_groups=config.qkv_head_input_groups,
        attention_qk_mode=config.attention_qk_mode,
        boolean_qk_width=config.boolean_qk_width,
        boolean_qk_rounds=config.boolean_qk_rounds,
        boolean_qk_bits_per_head=config.boolean_qk_bits_per_head,
        boolean_qk_correction_features=config.boolean_qk_correction_features,
        boolean_qk_correction_taps=config.boolean_qk_correction_taps,
        attention_value_mode=config.attention_value_mode,
        attention_value_block_active_fraction=config.attention_value_block_active_fraction,
        boolean_value_width=config.boolean_value_width,
        boolean_value_rounds=config.boolean_value_rounds,
        boolean_value_bits=config.boolean_value_bits,
        boolean_value_correction_features=config.boolean_value_correction_features,
        attention_output_mode=config.attention_output_mode,
        boolean_output_width=config.boolean_output_width,
        boolean_output_rounds=config.boolean_output_rounds,
        boolean_output_correction_features=config.boolean_output_correction_features,
        boolean_output_correction_taps=config.boolean_output_correction_taps,
        boolean_output_routes=config.boolean_output_routes,
        boolean_output_cell=config.boolean_output_cell,
        attention_output_block_active_fraction=config.attention_output_block_active_fraction,
        attention_output_codewords=config.attention_output_codewords,
        attention_output_codebook_temperature=(
            config.attention_output_codebook_temperature
        ),
        attention_output_codebook_response_shift=(
            config.attention_output_codebook_response_shift
        ),
        attention_output_head_local_rank=config.attention_output_head_local_rank,
        attention_output_squeeze_bitplanes=config.attention_output_squeeze_bitplanes,
        attention_output_squeeze_polarities=config.attention_output_squeeze_polarities,
        attention_output_squeeze_group_bits=config.attention_output_squeeze_group_bits,
        attention_output_squeeze_bypass=config.attention_output_squeeze_bypass,
        attention_output_input_rounding=config.attention_output_input_rounding,
        attention_normalizer=config.attention_normalizer,
        clipped_linear_initial_slope=config.clipped_linear_initial_slope,
        attention_window_policy=config.attention_window_policy,
        local_attention_window=config.local_attention_window,
        local_attention_heads=config.local_attention_heads,
        local_attention_layers=config.local_attention_layers,
        attention_skip_layers=config.attention_skip_layers,
        attention_shared_input_layers=config.attention_shared_input_layers,
        attention_shared_qk_layers=config.attention_shared_qk_layers,
        attention_shared_qk_mode=config.attention_shared_qk_mode,
        zero_init_residual=config.zero_init_residual,
        tie_embeddings=config.tie_embeddings,
        vocabulary_path=config.vocabulary_path,
        boolean_vocab_code_width=config.boolean_vocab_code_width,
        boolean_vocab_pool_bits=config.boolean_vocab_pool_bits,
        boolean_vocab_pool_schedule=config.boolean_vocab_pool_schedule,
        boolean_vocab_correction_rank=config.boolean_vocab_correction_rank,
        boolean_vocab_factor_output=config.boolean_vocab_factor_output,
        vocabulary_hierarchy=config.vocabulary_hierarchy,
        boolean_token_expansion=config.boolean_token_expansion,
        residual_scalars=config.residual_scalars,
        residual_routing=config.residual_routing,
        u_skip_pairs=config.u_skip_pairs,
        dynamic_route_layers=config.dynamic_route_layers,
        dynamic_route_dimension=config.dynamic_route_dimension,
        dropout=config.dropout,
    )


def capacity_report(model: VocabularyTransformerLM) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    active_attention_layers = [
        layer for layer in model.layers if not layer.skip_attention
    ]
    skipped_attention_training_values = sum(
        sum(parameter.numel() for parameter in layer.attention.parameters())
        + sum(parameter.numel() for parameter in layer.attention_norm.parameters())
        + 1
        for layer in model.layers
        if layer.skip_attention
    )
    embedding = model.token_embedding.weight.numel()
    if isinstance(model.readout, BooleanVocabularyReadout):
        readout = sum(parameter.numel() for parameter in model.readout.parameters())
        hierarchy_code_bits = (
            0
            if model.readout.cluster_code_logits is None
            else model.readout.cluster_code_logits.numel()
        )
        hierarchy_layout_bytes = (
            0
            if model.readout.cluster_token_ids is None
            else model.readout.cluster_token_ids.numel() * 2
        )
        hierarchy_correction_values = (
            0
            if model.readout.cluster_correction_output_weight is None
            else model.readout.cluster_correction_output_weight.numel()
        )
        hierarchy_bias_values = (
            0
            if model.readout.cluster_bias is None
            else model.readout.cluster_bias.numel()
        )
        boolean_token_code_bits = (
            model.token_embedding.weight.numel() + hierarchy_code_bits
        )
        vocabulary_correction_values = (
            model.readout.correction_weight.numel()
            if model.readout.correction_weight is not None
            else (
                model.readout.correction_input_weight.numel()
                + model.readout.correction_output_weight.numel()
            )
            if model.readout.correction_input_weight is not None
            else 0
        ) + hierarchy_correction_values
        vocabulary_bias_values = (
            model.readout.bias.numel() + hierarchy_bias_values
        )
        vocabulary_correction_scale_groups = (
            3
            if model.readout.correction_weight is not None
            and model.token_embedding.code_width == 64
            else 1 if vocabulary_correction_values else 0
        )
        if model.readout.correction_weight is not None:
            correction_deployed_bytes = (
                model.readout.correction_weight.numel()
                + model.padded_vocabulary_size
                * vocabulary_correction_scale_groups * 4
            )
        elif model.readout.correction_input_weight is not None:
            output_scale_groups = (
                0
                if model.readout.factor_output == "fp16"
                else 1
                if model.readout.factor_output == "w8a8_row"
                else model.readout.correction_rank // 16
            )
            correction_deployed_bytes = (
                model.readout.correction_input_weight.numel()
                + model.readout.correction_rank * 4
                + (
                    model.readout.correction_output_weight.numel() * 2
                    if model.readout.factor_output == "fp16"
                    else model.readout.correction_output_weight.numel()
                    + model.padded_vocabulary_size * output_scale_groups * 4
                )
            )
            if model.readout.cluster_correction_output_weight is not None:
                correction_deployed_bytes += (
                    model.readout.cluster_correction_output_weight.numel() * 2
                    if model.readout.factor_output == "fp16"
                    else model.readout.cluster_correction_output_weight.numel()
                    + 32 * output_scale_groups * 4
                )
        else:
            correction_deployed_bytes = 0
        vocabulary_deployed_bytes = (
            math.ceil(boolean_token_code_bits / 8)
            + correction_deployed_bytes
            + vocabulary_bias_values * 4
            + hierarchy_layout_bytes
        )
    else:
        readout = (
            0
            if model.readout.weight is model.token_embedding.weight
            else model.readout.weight.numel()
        )
        boolean_token_code_bits = 0
        vocabulary_correction_values = 0
        vocabulary_bias_values = 0
        vocabulary_correction_scale_groups = 0
        hierarchy_code_bits = 0
        hierarchy_layout_bytes = 0
        hierarchy_correction_values = 0
        hierarchy_bias_values = 0
        vocabulary_deployed_bytes = (embedding + readout) * 2
    feature_table = 0
    if model.value_embedding is not None:
        feature_table += model.value_embedding.weight.numel()
    if model.bigram_embedding is not None:
        feature_table += model.bigram_embedding.weight.numel()
    feature_scalars = sum(
        layer.attention.value_embedding_scale.numel()
        for layer in active_attention_layers
        if layer.attention.value_embedding_scale is not None
    )
    if model.bigram_scale is not None:
        feature_scalars += model.bigram_scale.numel()
    for parameter in (
        model.smear_strength,
        model.smear_scale,
        model.smear_bias,
    ):
        if parameter is not None:
            feature_scalars += parameter.numel()
    feature = feature_table + feature_scalars
    routing = 0
    if model.u_skip_gates is not None:
        routing += model.u_skip_gates.numel()
    if model.dynamic_route_strength is not None:
        routing += model.dynamic_route_strength.numel()
        routing += model.dynamic_route_scale.numel()
        routing += model.dynamic_route_bias.numel()
    boolean_program_bits = sum(
        layer.boolean_ffn.deployed_program_bits()
        for layer in model.layers
        if layer.boolean_ffn is not None
    ) + sum(
        layer.attention.boolean_query.program_logits.numel()
        + layer.attention.boolean_key.program_logits.numel()
        for layer in active_attention_layers
        if layer.attention.boolean_query is not None
    ) + sum(
        layer.attention.boolean_value.program_logits.numel()
        for layer in active_attention_layers
        if layer.attention.boolean_value is not None
    ) + sum(
        layer.attention.qkv.deployed_program_bits()
        for layer in active_attention_layers
        if layer.attention.attention_qkv_projection_mode in {
            "signed_squeeze_hybrid", "bitmix_trajectory_hybrid"
        }
    ) + sum(
        (
            layer.attention.output.deployed_program_bits()
            if layer.attention.attention_output_mode in {
                "boolean_lut3_decoded", "signed_squeeze_hybrid"
            }
            else layer.attention.output.program_logits.numel()
        )
        for layer in active_attention_layers
        if layer.attention.attention_output_mode in {
            "boolean_lut3_direct", "boolean_lut3_decoded", "signed_squeeze_hybrid"
        }
    )
    block_ticket_program_bits = sum(
        layer.attention.output.block_mask.numel()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "block_sparse_16x16"
    ) + sum(
        layer.attention.qkv.value_block_mask.numel()
        for layer in active_attention_layers
        if layer.attention.attention_value_mode == "block_sparse_16x16"
    ) + sum(
        int(layer.attention.qkv.permitted_block_mask.sum())
        for layer in active_attention_layers
        if layer.attention.attention_qkv_projection_mode == "block_sparse_16x16"
    ) + sum(
        0
        if layer.boolean_ffn is None
        or layer.boolean_ffn.correction_block_mask is None
        else layer.boolean_ffn.correction_block_mask.numel()
        for layer in model.layers
    )
    product_codebook_program_bits = sum(
        layer.attention.output.deployed_program_bits()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "product_codebook_16"
    )
    ffn_correction_codebook_program_bits = sum(
        layer.boolean_ffn.correction_codebook.deployed_program_bits()
        for layer in model.layers
        if layer.boolean_ffn is not None
        and layer.boolean_ffn.correction_codebook is not None
    )
    boolean_program_bits += (
        block_ticket_program_bits
        + product_codebook_program_bits
        + ffn_correction_codebook_program_bits
    )
    boolean_training_candidate_values = sum(
        layer.boolean_ffn.training_candidate_program_values()
        for layer in model.layers
        if layer.boolean_ffn is not None
    ) + sum(
        layer.attention.qkv.training_candidate_program_values()
        for layer in active_attention_layers
        if layer.attention.attention_qkv_projection_mode == "block_sparse_16x16"
    ) + sum(
        layer.attention.output.training_candidate_program_values()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "product_codebook_16"
    ) + sum(
        layer.boolean_ffn.correction_codebook.training_candidate_program_values()
        for layer in model.layers
        if layer.boolean_ffn is not None
        and layer.boolean_ffn.correction_codebook is not None
    )
    product_codebook_numeric_values = sum(
        layer.attention.output.codebook.numel()
        + layer.attention.output.row_scale.numel()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "product_codebook_16"
    )
    product_codebook_numeric_bytes = sum(
        layer.attention.output.codebook.numel()
        + layer.attention.output.row_scale.numel() * 4
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "product_codebook_16"
    )
    head_local_numeric_values = sum(
        layer.attention.output.active_weight_values()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "head_local_hybrid"
    )
    head_local_numeric_bytes = sum(
        layer.attention.output.deployed_numeric_bytes()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "head_local_hybrid"
    )
    inactive_block_values = sum(
        layer.attention.output.weight.numel()
        - layer.attention.output.active_weight_values()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "block_sparse_16x16"
    ) + sum(
        layer.attention.qkv.weight.numel()
        - layer.attention.qkv.active_weight_values()
        for layer in active_attention_layers
        if layer.attention.attention_value_mode == "block_sparse_16x16"
    ) + sum(
        layer.attention.qkv.weight.numel()
        - layer.attention.qkv.active_weight_values()
        for layer in active_attention_layers
        if layer.attention.attention_qkv_projection_mode == "block_sparse_16x16"
    ) + sum(
        0
        if layer.boolean_ffn is None
        or layer.boolean_ffn.correction_weight is None
        or layer.boolean_ffn.correction_block_mask is None
        else (
            layer.boolean_ffn.correction_weight.numel()
            - layer.boolean_ffn.correction_active_weight_values()
        )
        for layer in model.layers
    )
    shared_qk_omitted_values = 0
    if model.attention_shared_qk_layers:
        for layer_index in model.attention_shared_qk_layers[1:]:
            attention = model.layers[layer_index].attention
            query_rows = attention.heads * attention.qk_head_dim
            key_rows = attention.key_heads * attention.qk_head_dim
            omitted_rows = key_rows + (
                query_rows if model.attention_shared_qk_mode == "qk" else 0
            )
            if attention.attention_qkv_projection_mode == "dense":
                shared_qk_omitted_values += omitted_rows * attention.dimension
            else:
                projection = attention.qkv
                if not isinstance(projection, QKVBlockSparseLinear16):
                    raise RuntimeError("shared Q/K block projection is invalid")
                omitted_row_blocks = omitted_rows // projection.block_rows
                begin = (
                    0
                    if model.attention_shared_qk_mode == "qk"
                    else projection.region_row_blocks[0]
                )
                end = projection.region_row_blocks[0] + projection.region_row_blocks[1]
                selected = projection.deployed_mask()[begin:end]
                if selected.shape[0] != omitted_row_blocks:
                    raise RuntimeError("shared Q/K rows do not align to deployed blocks")
                shared_qk_omitted_values += int(selected.sum()) * (
                    projection.block_rows * 16
                )
    logical_total = (
        total
        - skipped_attention_training_values
        - boolean_training_candidate_values
        - inactive_block_values
        - shared_qk_omitted_values
        + block_ticket_program_bits
        + product_codebook_program_bits
    )
    block_ticket_metadata_bytes = sum(
        layer.attention.output.prepacked_metadata_bytes()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "block_sparse_16x16"
    ) + sum(
        layer.attention.qkv.prepacked_metadata_bytes()
        for layer in active_attention_layers
        if layer.attention.attention_value_mode == "block_sparse_16x16"
    ) + sum(
        layer.attention.qkv.prepacked_metadata_bytes()
        for layer in active_attention_layers
        if layer.attention.attention_qkv_projection_mode == "block_sparse_16x16"
    ) + sum(
        0
        if layer.boolean_ffn is None
        else layer.boolean_ffn.correction_prepacked_metadata_bytes()
        for layer in model.layers
    )
    boolean_adapter_values = sum(
        layer.boolean_ffn.encoder_offsets.numel()
        + layer.boolean_ffn.decoder_weight.numel()
        + layer.boolean_ffn.decoder_bias.numel()
        for layer in model.layers
        if layer.boolean_ffn is not None
    ) + sum(
        layer.attention.output.encoder_offsets.numel()
        + layer.attention.output.decoder_weight.numel()
        + layer.attention.output.decoder_bias.numel()
        for layer in active_attention_layers
        if layer.attention.attention_output_mode == "boolean_lut3_decoded"
    )
    boolean_ffn_correction_values = sum(
        0
        if layer.boolean_ffn is None
        else layer.boolean_ffn.correction_active_weight_values()
        if layer.boolean_ffn.correction_codebook is not None
        else layer.boolean_ffn.correction_active_weight_values()
        if layer.boolean_ffn.correction_weight is not None
        else (
            layer.boolean_ffn.correction_input_weight.numel()
            + layer.boolean_ffn.correction_output_weight.numel()
        )
        if layer.boolean_ffn.correction_input_weight is not None
        else 0
        for layer in model.layers
    )
    boolean_ffn_correction_scales = sum(
        0
        if layer.boolean_ffn is None
        else layer.boolean_ffn.dimension
        if layer.boolean_ffn.correction_codebook is not None
        else layer.boolean_ffn.dimension
        if layer.boolean_ffn.correction_weight is not None
        else layer.boolean_ffn.correction_rank
        if layer.boolean_ffn.correction_input_weight is not None
        else 0
        for layer in model.layers
    )
    boolean_ffn_correction_deployed_bytes = sum(
        0
        if layer.boolean_ffn is None
        else (
            layer.boolean_ffn.correction_codebook.codebook.numel()
            + layer.boolean_ffn.correction_codebook.row_scale.numel() * 4
        )
        if layer.boolean_ffn.correction_codebook is not None
        else layer.boolean_ffn.correction_active_weight_values()
            + layer.boolean_ffn.dimension * 4
        if layer.boolean_ffn.correction_weight is not None
        else layer.boolean_ffn.correction_input_weight.numel()
            + layer.boolean_ffn.correction_rank * 4
            + layer.boolean_ffn.correction_output_weight.numel() * 2
        if layer.boolean_ffn.correction_input_weight is not None
        else 0
        for layer in model.layers
    )
    boolean_output_correction_values = sum(
        0
        if layer.attention.attention_output_mode != "boolean_lut3_decoded"
        or layer.attention.output.correction_weight is None
        else layer.attention.output.correction_weight.numel()
        for layer in active_attention_layers
    )
    boolean_output_correction_scales = sum(
        0
        if layer.attention.attention_output_mode != "boolean_lut3_decoded"
        or layer.attention.output.correction_weight is None
        else layer.attention.output.dimension
        for layer in active_attention_layers
    )
    boolean_attention_numeric_values = sum(
        (0 if layer.attention.boolean_qk_log_scale is None else layer.attention.boolean_qk_log_scale.numel())
        + (0 if layer.attention.boolean_relative_bias is None else layer.attention.boolean_relative_bias.numel())
        + (0 if layer.attention.boolean_output_strength is None else layer.attention.boolean_output_strength.numel())
        for layer in active_attention_layers
    )
    boolean_qk_correction_values = sum(
        0
        if layer.attention.boolean_query is None
        or layer.attention.boolean_query.correction_weight is None
        else (
            layer.attention.boolean_query.correction_weight.numel()
            + layer.attention.boolean_key.correction_weight.numel()
        )
        for layer in active_attention_layers
    )
    boolean_qk_correction_scales = sum(
        0
        if layer.attention.boolean_query is None
        or layer.attention.boolean_query.correction_weight is None
        else (
            layer.attention.boolean_query.correction_output_bits
            + layer.attention.boolean_key.correction_output_bits
        )
        for layer in active_attention_layers
    )
    compact_kv_cache = 0
    dense_kv_cache = 0
    deployed_compact_kv_cache = 0
    deployed_dense_kv_cache = 0
    shared_qk_cache_consumers = set(model.attention_shared_qk_layers[1:])
    for layer_index, layer in enumerate(model.layers):
        if layer.skip_attention:
            continue
        attention = layer.attention
        def grouped_contexts(groups: int) -> list[int]:
            queries_per_group = attention.heads // groups
            return [
                max(
                    attention.head_contexts[
                        group * queries_per_group : (group + 1) * queries_per_group
                    ]
                )
                for group in range(groups)
            ]

        key_contexts = grouped_contexts(attention.key_heads)
        value_contexts = grouped_contexts(attention.value_heads)
        if attention.attention_value_mode in {"numerical", "block_sparse_16x16"}:
            value_compact_bytes = sum(value_contexts) * attention.head_dim * 2
            value_dense_bytes = (
                attention.value_heads * model.context * attention.head_dim * 2
            )
        else:
            value_bytes_per_head = (
                attention.boolean_value_bits // attention.heads // 8
                + attention.boolean_value_correction_dims // attention.heads * 2
            )
            value_compact_bytes = sum(value_contexts) * value_bytes_per_head
            value_dense_bytes = attention.value_heads * model.context * value_bytes_per_head
        if attention.value_cache_precision.startswith("int8_"):
            deployed_value_compact_bytes = value_compact_bytes // 2
            deployed_value_dense_bytes = value_dense_bytes // 2
        else:
            deployed_value_compact_bytes = value_compact_bytes
            deployed_value_dense_bytes = value_dense_bytes
        if attention.attention_qk_mode == "numerical":
            if attention.key_heads == attention.heads:
                key_compact_bytes = 2 * sum(
                    context * width
                    for context, width in zip(
                        key_contexts, attention.head_qk_dims, strict=True
                    )
                )
                key_dense_bytes = (
                    2 * model.context * sum(attention.head_qk_dims)
                )
            else:
                key_compact_bytes = sum(key_contexts) * attention.qk_head_dim * 2
                key_dense_bytes = (
                    attention.key_heads * model.context * attention.qk_head_dim * 2
                )
            if attention.qk_cache_precision in {
                "int8_fixed4",
                "int8_fixed4_fp16",
            }:
                deployed_key_compact_bytes = key_compact_bytes // 2
                deployed_key_dense_bytes = key_dense_bytes // 2
            elif attention.qk_cache_precision == "int8_dynamic":
                key_vectors_compact = sum(key_contexts)
                key_vectors_dense = attention.key_heads * model.context
                deployed_key_compact_bytes = (
                    key_compact_bytes // 2 + key_vectors_compact * 4
                )
                deployed_key_dense_bytes = (
                    key_dense_bytes // 2 + key_vectors_dense * 4
                )
            else:
                deployed_key_compact_bytes = key_compact_bytes
                deployed_key_dense_bytes = key_dense_bytes
        else:
            key_compact_bytes = sum(key_contexts) * (
                attention.boolean_qk_bits_per_head // 8
                + attention.boolean_qk_correction_dims_per_head * 2
            )
            key_dense_bytes = (
                attention.key_heads
                * model.context
                * attention.boolean_qk_bits_per_head
                // 8
                + attention.key_heads
                * model.context
                * attention.boolean_qk_correction_dims_per_head
                * 2
            )
            deployed_key_compact_bytes = key_compact_bytes
            deployed_key_dense_bytes = key_dense_bytes
        shared_key_cache = layer_index in shared_qk_cache_consumers
        compact_kv_cache += (
            (0 if shared_key_cache else key_compact_bytes) + value_compact_bytes
        )
        dense_kv_cache += (
            (0 if shared_key_cache else key_dense_bytes) + value_dense_bytes
        )
        deployed_compact_kv_cache += (
            (0 if shared_key_cache else deployed_key_compact_bytes)
            + deployed_value_compact_bytes
        )
        deployed_dense_kv_cache += (
            (0 if shared_key_cache else deployed_key_dense_bytes)
            + deployed_value_dense_bytes
        )
    ngram_table_values = (
        0 if model.ngram_table is None else model.ngram_table.weight.numel()
    )
    return {
        "total_scalar_parameters": logical_total,
        "training_scalar_parameters": total,
        "embedding_parameters": embedding,
        "additional_readout_parameters": readout,
        "boolean_token_code_bits": boolean_token_code_bits,
        "vocabulary_correction_values": vocabulary_correction_values,
        "vocabulary_bias_values": vocabulary_bias_values,
        "vocabulary_correction_scale_groups": vocabulary_correction_scale_groups,
        "vocabulary_hierarchy_code_bits": hierarchy_code_bits,
        "vocabulary_hierarchy_layout_bytes": hierarchy_layout_bytes,
        "vocabulary_hierarchy_correction_values": hierarchy_correction_values,
        "vocabulary_hierarchy_bias_values": hierarchy_bias_values,
        "vocabulary_deployed_bytes": vocabulary_deployed_bytes,
        "token_feature_table_parameters": feature_table,
        "token_feature_scalar_parameters": feature_scalars,
        "token_feature_parameters": feature,
        "ngram_table_values": ngram_table_values,
        "ngram_table_deployed_bytes": ngram_table_values,
        "residual_routing_parameters": routing,
        "core_scalar_parameters": logical_total - embedding - readout,
        "transformer_parameters_excluding_token_features": (
            logical_total - embedding - readout - feature
        ),
        "nominal_fp16_parameter_bytes": logical_total * 2,
        "boolean_program_bits": boolean_program_bits,
        "boolean_program_bytes": math.ceil(boolean_program_bits / 8),
        "block_ticket_program_bits": block_ticket_program_bits,
        "product_codebook_program_bits": product_codebook_program_bits,
        "ffn_correction_codebook_program_bits": (
            ffn_correction_codebook_program_bits
        ),
        "product_codebook_numeric_values": product_codebook_numeric_values,
        "product_codebook_numeric_bytes": product_codebook_numeric_bytes,
        "head_local_output_numeric_values": head_local_numeric_values,
        "head_local_output_numeric_bytes": head_local_numeric_bytes,
        "block_ticket_prepacked_metadata_bytes": block_ticket_metadata_bytes,
        "boolean_adapter_values": boolean_adapter_values,
        "boolean_ffn_correction_values": boolean_ffn_correction_values,
        "boolean_output_correction_values": boolean_output_correction_values,
        "boolean_attention_numeric_values": boolean_attention_numeric_values,
        "boolean_qk_correction_values": boolean_qk_correction_values,
        "boolean_training_candidate_values": boolean_training_candidate_values,
        "skipped_attention_training_values": skipped_attention_training_values,
        "inactive_attention_output_training_values": inactive_block_values,
        "shared_qk_omitted_training_values": shared_qk_omitted_values,
        "deployed_parameter_bytes_estimate": (
            (
                logical_total
                - boolean_program_bits
                - embedding
                - readout
                - boolean_ffn_correction_values
                - boolean_output_correction_values
                - boolean_qk_correction_values
                - product_codebook_numeric_values
                - ngram_table_values
            ) * 2
            + ngram_table_values
            + math.ceil(boolean_program_bits / 8)
            + block_ticket_metadata_bytes
            - math.ceil(block_ticket_program_bits / 8)
            + vocabulary_deployed_bytes
            + boolean_ffn_correction_deployed_bytes
            + boolean_output_correction_values
            + boolean_output_correction_scales * 4
            + boolean_qk_correction_values
            + boolean_qk_correction_scales * 4
            + product_codebook_numeric_bytes
            - head_local_numeric_values * 2
            + head_local_numeric_bytes
        ),
        "fp16_kv_cache_bytes_per_sequence": dense_kv_cache,
        "native_compact_fp16_kv_cache_bytes_per_sequence": compact_kv_cache,
        "native_deployed_kv_cache_bytes_per_sequence": deployed_dense_kv_cache,
        "native_compact_deployed_kv_cache_bytes_per_sequence": (
            deployed_compact_kv_cache
        ),
    }
