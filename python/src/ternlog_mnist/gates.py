from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def truth_table(function) -> tuple[int, ...]:
    """Return entries ordered by index ``4*a + 2*b + c``."""
    return tuple(int(bool(function(a, b, c))) for a in (0, 1) for b in (0, 1) for c in (0, 1))


EOR3_TABLE = truth_table(lambda a, b, c: a ^ b ^ c)
BCAX_TABLE = truth_table(lambda a, b, c: a ^ (b & (1 ^ c)))
MUX_TABLE = truth_table(lambda mask, when_true, when_false: when_true if mask else when_false)
XOR_TABLE = truth_table(lambda a, b, _c: a ^ b)
AND_TABLE = truth_table(lambda a, b, _c: a & b)
AND3_TABLE = truth_table(lambda a, b, c: a & b & c)


def _face_derivatives(table: Tensor, a: Tensor, b: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    t = table
    da = (
        (t[4] - t[0]) * (1 - b) * (1 - c)
        + (t[5] - t[1]) * (1 - b) * c
        + (t[6] - t[2]) * b * (1 - c)
        + (t[7] - t[3]) * b * c
    )
    db = (
        (t[2] - t[0]) * (1 - a) * (1 - c)
        + (t[3] - t[1]) * (1 - a) * c
        + (t[6] - t[4]) * a * (1 - c)
        + (t[7] - t[5]) * a * c
    )
    dc = (
        (t[1] - t[0]) * (1 - a) * (1 - b)
        + (t[3] - t[2]) * (1 - a) * b
        + (t[5] - t[4]) * a * (1 - b)
        + (t[7] - t[6]) * a * b
    )
    return da, db, dc


class _HardTruthTable(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        table: Tensor,
        backward_mode: str,
        temperature: float,
        gradient_leak: float,
    ) -> Tensor:
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        ctx.c_shape = c.shape
        ah = a >= 0.5
        bh = b >= 0.5
        ch = c >= 0.5
        index = ah.to(torch.long) * 4 + bh.to(torch.long) * 2 + ch.to(torch.long)
        output = table[index].to(dtype=a.dtype)
        ctx.save_for_backward(ah.to(a.dtype), bh.to(b.dtype), ch.to(c.dtype), table)
        ctx.backward_mode = backward_mode
        ctx.temperature = temperature
        ctx.gradient_leak = gradient_leak
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b, c, table = ctx.saved_tensors
        if ctx.backward_mode == "ste":
            da = db = dc = torch.ones_like(a)
        else:
            if ctx.backward_mode == "softened":
                temperature = max(float(ctx.temperature), 1e-3)
                a = torch.sigmoid((2 * a - 1) / temperature)
                b = torch.sigmoid((2 * b - 1) / temperature)
                c = torch.sigmoid((2 * c - 1) / temperature)
            da, db, dc = _face_derivatives(table, a, b, c)
            if ctx.backward_mode == "hybrid":
                leak = float(ctx.gradient_leak)
                da = torch.where(da == 0, torch.full_like(da, leak), da)
                db = torch.where(db == 0, torch.full_like(db, leak), db)
                dc = torch.where(dc == 0, torch.full_like(dc, leak), dc)
        return (
            (grad_output * da).sum_to_size(ctx.a_shape),
            (grad_output * db).sum_to_size(ctx.b_shape),
            (grad_output * dc).sum_to_size(ctx.c_shape),
            None,
            None,
            None,
            None,
        )


class _HardProgrammableTruthTable(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        program: Tensor,
        backward_mode: str,
        temperature: float,
        gradient_leak: float,
    ) -> Tensor:
        if (
            program.shape[0] != 8
            or torch.broadcast_shapes(program.shape[1:], a.shape) != a.shape
        ):
            raise ValueError(
                "programmable LUT3 expects eight program planes broadcastable "
                "to the input shape"
            )
        ah = (a >= 0.5).to(a.dtype)
        bh = (b >= 0.5).to(b.dtype)
        ch = (c >= 0.5).to(c.dtype)
        table = (program >= 0.5).to(program.dtype)
        index = ah.to(torch.long) * 4 + bh.to(torch.long) * 2 + ch.to(torch.long)
        output = torch.gather(
            table.expand(8, *a.shape), 0, index.unsqueeze(0)
        ).squeeze(0)
        ctx.save_for_backward(ah, bh, ch, table)
        ctx.program_shape = program.shape
        ctx.backward_mode = backward_mode
        ctx.temperature = temperature
        ctx.gradient_leak = gradient_leak
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b, c, table = ctx.saved_tensors
        if ctx.backward_mode == "ste":
            da = db = dc = torch.ones_like(a)
        else:
            if ctx.backward_mode == "softened":
                temperature = max(float(ctx.temperature), 1e-3)
                a = torch.sigmoid((2 * a - 1) / temperature)
                b = torch.sigmoid((2 * b - 1) / temperature)
                c = torch.sigmoid((2 * c - 1) / temperature)
            da, db, dc = _face_derivatives(table, a, b, c)
            if ctx.backward_mode == "hybrid":
                leak = float(ctx.gradient_leak)
                da = torch.where(da == 0, torch.full_like(da, leak), da)
                db = torch.where(db == 0, torch.full_like(db, leak), db)
                dc = torch.where(dc == 0, torch.full_like(dc, leak), dc)
        index = a.to(torch.long) * 4 + b.to(torch.long) * 2 + c.to(torch.long)
        program_plane_shape = tuple(ctx.program_shape[1:])
        # The hard inputs select exactly one truth-table plane.  Reduce each
        # plane directly to the compact program shape instead of materializing
        # an [8, batch, sequence, width] surrogate-gradient tensor.
        grad_program = torch.stack(
            [
                torch.where(
                    index == truth_index,
                    grad_output,
                    torch.zeros((), dtype=grad_output.dtype, device=grad_output.device),
                ).sum_to_size(program_plane_shape)
                for truth_index in range(8)
            ]
        )
        return grad_output * da, grad_output * db, grad_output * dc, grad_program, None, None, None


class _HardBit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, temperature: float) -> Tensor:
        ctx.save_for_backward(logits)
        ctx.temperature = temperature
        return (logits >= 0).to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (logits,) = ctx.saved_tensors
        temperature = max(float(ctx.temperature), 1e-3)
        probability = torch.sigmoid(logits / temperature)
        derivative = probability * (1 - probability) / temperature
        return grad_output * derivative, None


class _ResidualXor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: Tensor, branch: Tensor, identity_backward: bool) -> Tensor:
        ctx.state_shape = state.shape
        ctx.branch_shape = branch.shape
        state_hard = state >= 0.5
        branch_hard = branch >= 0.5
        ctx.save_for_backward(state_hard.to(state.dtype), branch_hard.to(branch.dtype))
        ctx.identity_backward = identity_backward
        return torch.logical_xor(state_hard, branch_hard).to(state.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        state, branch = ctx.saved_tensors
        if ctx.identity_backward:
            return (
                grad_output.sum_to_size(ctx.state_shape),
                grad_output.sum_to_size(ctx.branch_shape),
                None,
            )
        return (
            (grad_output * (1 - 2 * branch)).sum_to_size(ctx.state_shape),
            (grad_output * (1 - 2 * state)).sum_to_size(ctx.branch_shape),
            None,
        )


def hard_bit(logits: Tensor, temperature: float = 1.0) -> Tensor:
    return _HardBit.apply(logits, temperature)


def gate(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    table: Sequence[int],
    backward_mode: str = "multilinear",
    temperature: float = 1.0,
    gradient_leak: float = 0.05,
) -> Tensor:
    table_tensor = torch.as_tensor(table, device=a.device, dtype=a.dtype)
    return _HardTruthTable.apply(a, b, c, table_tensor, backward_mode, temperature, gradient_leak)


def mux(mask: Tensor, when_true: Tensor, when_false: Tensor, backward_mode: str, temperature: float, gradient_leak: float = 0.05) -> Tensor:
    return gate(mask, when_true, when_false, MUX_TABLE, backward_mode, temperature, gradient_leak)


def eor3(a: Tensor, b: Tensor, c: Tensor, backward_mode: str, temperature: float, gradient_leak: float = 0.05) -> Tensor:
    return gate(a, b, c, EOR3_TABLE, backward_mode, temperature, gradient_leak)


def bcax(a: Tensor, b: Tensor, c: Tensor, backward_mode: str, temperature: float, gradient_leak: float = 0.05) -> Tensor:
    return gate(a, b, c, BCAX_TABLE, backward_mode, temperature, gradient_leak)


def and2(
    a: Tensor,
    b: Tensor,
    backward_mode: str,
    temperature: float,
    gradient_leak: float = 0.05,
) -> Tensor:
    return gate(
        a,
        b,
        torch.zeros_like(a),
        AND_TABLE,
        backward_mode,
        temperature,
        gradient_leak,
    )


def minterm(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    program: Tensor,
    backward_mode: str,
    temperature: float,
    gradient_leak: float = 0.05,
) -> Tensor:
    """Enabled, polarity-selectable three-input minterm.

    Program planes are ``enable, polarity_a, polarity_b, polarity_c``. The
    output is one exactly when enabled and every input equals its polarity.
    This covers the sparse one-hot LUTs learned by the unrestricted circuit
    with half as many stored program bits.
    """
    if program.shape[0] != 4:
        raise ValueError("minterm requires four program bitplanes")
    enable, polarity_a, polarity_b, polarity_c = program
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    literal_a = eor3(a, polarity_a, ones, backward_mode, temperature, gradient_leak)
    literal_b = eor3(b, polarity_b, ones, backward_mode, temperature, gradient_leak)
    literal_c = eor3(c, polarity_c, ones, backward_mode, temperature, gradient_leak)
    match = gate(
        literal_a,
        literal_b,
        literal_c,
        AND3_TABLE,
        backward_mode,
        temperature,
        gradient_leak,
    )
    return and2(enable, match, backward_mode, temperature, gradient_leak)


def residual_xor(state: Tensor, branch: Tensor, *, identity_backward: bool = True) -> Tensor:
    return _ResidualXor.apply(state, branch, identity_backward)


def lut2(
    a: Tensor,
    b: Tensor,
    program: Tensor,
    backward_mode: str,
    temperature: float,
    gradient_leak: float = 0.05,
) -> Tensor:
    """Arbitrary per-lane two-input LUT, implemented as a three-MUX tree."""
    if program.shape[0] != 4:
        raise ValueError("LUT2 requires four program bitplanes")
    w00, w01, w10, w11 = program
    low = mux(b, w01, w00, backward_mode, temperature, gradient_leak)
    high = mux(b, w11, w10, backward_mode, temperature, gradient_leak)
    return mux(a, high, low, backward_mode, temperature, gradient_leak)


def lut3(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    program: Tensor,
    backward_mode: str,
    temperature: float,
    gradient_leak: float = 0.05,
) -> Tensor:
    """Arbitrary per-lane ternary truth table with generic lookup backward."""
    return _HardProgrammableTruthTable.apply(
        a,
        b,
        c,
        program,
        backward_mode,
        temperature,
        gradient_leak,
    )


def program_planes_for_cell(cell: str) -> int:
    return {
        "lut2": 4,
        "lut3": 8,
        "minterm": 4,
        "masked_eor3": 1,
        "masked_bcax": 1,
        "xar_lut2": 4,
        "xar_ticket_lut2": 1,
        "xar_bank_lut2": 4,
        "xar_gated_lut2": 4,
        "dual3": 2,
        "bcax": 1,
        "mux3": 1,
        "affine": 1,
        "eor3_xar_lut2": 4,
        "eor3_masked_bcax": 1,
        "sparse_eor3_bcax": 1,
        "sparse_masked_eor3_bcax": 1,
        "eor3_final_lut3": 8,
        "bottleneck_lut3": 8,
        "bottleneck_minterm": 4,
        "bottleneck_categorical_minterm": 4,
        "bottleneck_factorized_minterm": 4,
        "squeeze_gate_lut3": 8,
        "squeeze_mod_lut3": 8,
        "squeeze_first_lut3": 8,
        "squeeze_final_lut3": 8,
        "hamming_final_lut2": 8,
        "bottleneck_final_lut2": 8,
    }[cell]
