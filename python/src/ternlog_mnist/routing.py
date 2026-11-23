from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable

import torch
from torch import Tensor


@dataclasses.dataclass(frozen=True, slots=True)
class RouteSpec:
    kind: str
    argument: int = 0
    estimated_instructions_per_register: int = 0


def _byte_permutation(width: int, byte_mapping: Iterable[int]) -> Tensor:
    byte_mapping = list(byte_mapping)
    if len(byte_mapping) != width // 8:
        raise ValueError("byte mapping has the wrong length")
    indices: list[int] = []
    for source_byte in byte_mapping:
        indices.extend(range(source_byte * 8, source_byte * 8 + 8))
    return torch.tensor(indices, dtype=torch.long)


def route_indices(width: int, spec: RouteSpec, register_bits: int = 128) -> Tensor:
    if width % register_bits:
        raise ValueError("width must be a multiple of the execution register width")
    byte_count = width // 8
    register_count = width // register_bits
    register_bytes = register_bits // 8
    if spec.kind == "identity":
        return torch.arange(width, dtype=torch.long)
    if spec.kind == "byte_rotate":
        shift = spec.argument % byte_count
        mapping = [(i - shift) % byte_count for i in range(byte_count)]
        return _byte_permutation(width, mapping)
    if spec.kind == "bit_rotate":
        shift = spec.argument % width
        return torch.tensor([(i - shift) % width for i in range(width)], dtype=torch.long)
    if spec.kind == "lane64_rotate":
        shift = spec.argument % 64
        return torch.tensor(
            [64 * (i // 64) + ((i % 64) - shift) % 64 for i in range(width)],
            dtype=torch.long,
        )
    if spec.kind.startswith("spatial_x") or spec.kind.startswith("spatial_y"):
        axis = "x" if spec.kind.startswith("spatial_x") else "y"
        side = 32 if spec.kind in {"spatial_x", "spatial_y"} else int(
            spec.kind.rsplit("_", 1)[1]
        )
        plane_bits = side * side
        if width % plane_bits:
            raise ValueError(f"spatial routes require packed {side}x{side} bitplanes")
        shift = spec.argument % side
        indices = []
        for output in range(width):
            plane = output // plane_bits
            position = output % plane_bits
            y, x = divmod(position, side)
            source_y = (y - shift) % side if axis == "y" else y
            source_x = (x - shift) % side if axis == "x" else x
            indices.append(plane * plane_bits + source_y * side + source_x)
        return torch.tensor(indices, dtype=torch.long)
    if spec.kind.startswith("plane_xor_"):
        side = int(spec.kind.rsplit("_", 1)[1])
        plane_bits = side * side
        if width % plane_bits:
            raise ValueError(f"plane butterfly requires packed {side}x{side} planes")
        plane_count = width // plane_bits
        if plane_count & (plane_count - 1):
            raise ValueError("plane XOR butterfly requires a power-of-two plane count")
        distance = spec.argument
        if distance < 1 or distance >= plane_count or distance & (distance - 1):
            raise ValueError("plane_xor distance must be a power of two below plane count")
        return torch.tensor(
            [
                ((output // plane_bits) ^ distance) * plane_bits
                + output % plane_bits
                for output in range(width)
            ],
            dtype=torch.long,
        )
    if spec.kind.startswith("window_"):
        _, axis, side_text, window_text, offset_text = spec.kind.split("_")
        side = int(side_text)
        window = int(window_text)
        offset = int(offset_text)
        if side % window:
            raise ValueError("window size must divide the spatial side")
        plane_bits = side * side
        if width % plane_bits:
            raise ValueError("window route width does not contain whole planes")
        shift = spec.argument % window

        def shifted_source(position: int) -> int:
            coordinate = (position - offset) % side
            block = (coordinate // window) * window
            local = coordinate % window
            return (block + (local - shift) % window + offset) % side

        indices = []
        for output in range(width):
            plane = output // plane_bits
            y, x = divmod(output % plane_bits, side)
            source_x = shifted_source(x) if axis == "x" else x
            source_y = shifted_source(y) if axis == "y" else y
            indices.append(plane * plane_bits + source_y * side + source_x)
        return torch.tensor(indices, dtype=torch.long)
    if spec.kind.startswith("space_to_depth_"):
        source_side = int(spec.kind.rsplit("_", 1)[1])
        if source_side % 2:
            raise ValueError("space-to-depth source side must be even")
        source_plane_bits = source_side * source_side
        target_side = source_side // 2
        target_plane_bits = target_side * target_side
        if width % source_plane_bits:
            raise ValueError("space-to-depth width does not contain whole source planes")
        indices = []
        for output in range(width):
            output_plane, target_pixel = divmod(output, target_plane_bits)
            source_plane, offset = divmod(output_plane, 4)
            offset_y, offset_x = divmod(offset, 2)
            target_y, target_x = divmod(target_pixel, target_side)
            source_y = 2 * target_y + offset_y
            source_x = 2 * target_x + offset_x
            indices.append(
                source_plane * source_plane_bits
                + source_y * source_side
                + source_x
            )
        return torch.tensor(indices, dtype=torch.long)
    if spec.kind == "register_rotate":
        shift = spec.argument % register_count
        mapping = []
        for output_register in range(register_count):
            source_register = (output_register - shift) % register_count
            mapping.extend(
                range(
                    source_register * register_bytes,
                    source_register * register_bytes + register_bytes,
                )
            )
        return _byte_permutation(width, mapping)
    if spec.kind == "register_xor":
        distance = spec.argument
        if distance < 1 or distance >= register_count or distance & (distance - 1):
            raise ValueError("register_xor distance must be a power of two below the register count")
        mapping = []
        for output_register in range(register_count):
            source_register = output_register ^ distance
            mapping.extend(
                range(
                    source_register * register_bytes,
                    source_register * register_bytes + register_bytes,
                )
            )
        return _byte_permutation(width, mapping)
    if spec.kind == "reverse_bytes":
        mapping = []
        for register in range(register_count):
            start = register * register_bytes
            mapping.extend(start + byte for byte in reversed(range(register_bytes)))
        return _byte_permutation(width, mapping)
    if spec.kind == "transpose_bytes":
        mapping = []
        for register in range(register_count):
            start = register * register_bytes
            side = int(math.sqrt(register_bytes))
            if side * side != register_bytes:
                raise ValueError("transpose_bytes requires a square byte register")
            mapping.extend(
                start + ((byte % side) * side + byte // side)
                for byte in range(register_bytes)
            )
        return _byte_permutation(width, mapping)
    raise ValueError(f"unknown route kind: {spec.kind}")


def build_neon_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    register_count = width // register_bits
    power_of_two_registers = register_count & (register_count - 1) == 0
    butterfly_stages = max(1, int(math.log2(register_count)))
    schedule = []
    for round_index in range(rounds):
        first = RouteSpec("identity", estimated_instructions_per_register=0)
        second = RouteSpec(
            "byte_rotate",
            1 << (round_index % 4),
            estimated_instructions_per_register=1,
        )
        if register_count > 1:
            distance = 1 << (round_index % butterfly_stages)
            third = RouteSpec(
                "register_xor" if power_of_two_registers else "register_rotate",
                distance,
                estimated_instructions_per_register=0,
            )
        else:
            third = RouteSpec(
                "reverse_bytes" if round_index % 2 else "transpose_bytes",
                estimated_instructions_per_register=1,
            )
        schedule.append((first, second, third))
    return schedule


def build_neon_bitmix_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """NEON schedule that breaks the otherwise-isolated modulo-8 bit lanes.

    A packed full-state bit rotation can be lowered to shifts/inserts plus a
    register handoff. Byte rotation still supplies cheap long-range diffusion.
    """
    schedule = []
    for round_index in range(rounds):
        schedule.append(
            (
                RouteSpec("identity", estimated_instructions_per_register=0),
                RouteSpec(
                    "byte_rotate",
                    1 << (round_index % 4),
                    estimated_instructions_per_register=1,
                ),
                RouteSpec(
                    "bit_rotate",
                    1 << (round_index % 3),
                    estimated_instructions_per_register=3,
                ),
            )
        )
    return schedule


def build_xar_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Inputs for an Apple SHA3 XAR-fused round.

    The first two operands feed XAR; the third is a register-renamed butterfly
    input to the following LUT2.
    """
    register_count = width // register_bits
    power_of_two_registers = register_count & (register_count - 1) == 0
    butterfly_stages = max(1, int(math.log2(register_count)))
    return [
        (
            RouteSpec("identity", estimated_instructions_per_register=0),
            RouteSpec(
                "byte_rotate",
                1 << (round_index % 4),
                estimated_instructions_per_register=1,
            ),
            RouteSpec(
                "register_xor" if power_of_two_registers else "register_rotate",
                1 << (round_index % butterfly_stages),
                estimated_instructions_per_register=0,
            ),
        )
        for round_index in range(rounds)
    ]


def build_cifar_hierarchical_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Packed 32x32 local routing alternating with butterfly diffusion.

    Every three rounds form a hierarchy stage: local x/y exchange, global
    register butterfly diffusion, then a local/global nonlinear synthesis
    round. Cell selection determines whether the third round uses BCAX or LUT.
    All routes have direct fixed-shape NEON lowerings.
    """
    if width % 1024:
        raise ValueError("CIFAR hierarchical routing requires 32x32 bitplanes")
    register_count = width // register_bits
    if register_count < 16 or register_count & (register_count - 1):
        raise ValueError("CIFAR hierarchical routing requires power-of-two registers")
    schedule = []
    for round_index in range(rounds):
        phase = round_index % 3
        stage = round_index // 3
        identity = RouteSpec("identity", estimated_instructions_per_register=0)
        if phase == 0:
            shift = 1 << (stage % 3)
            second = RouteSpec(
                "spatial_x", shift, estimated_instructions_per_register=3
            )
            third = RouteSpec(
                "spatial_y", shift, estimated_instructions_per_register=1
            )
        elif phase == 1:
            second = RouteSpec(
                "register_xor",
                1 << (stage % int(math.log2(register_count))),
                estimated_instructions_per_register=0,
            )
            third = RouteSpec(
                "byte_rotate",
                1 << (stage % 4),
                estimated_instructions_per_register=1,
            )
        else:
            second = RouteSpec(
                "spatial_x",
                2 << (stage % 3),
                estimated_instructions_per_register=3,
            )
            third = RouteSpec(
                "register_xor",
                8 << (stage % 3),
                estimated_instructions_per_register=0,
            )
        schedule.append((identity, second, third))
    return schedule


def build_cifar_fused_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Mix one spatial neighbor and one butterfly partner every round."""
    if width % 1024:
        raise ValueError("CIFAR fused routing requires 32x32 bitplanes")
    register_count = width // register_bits
    if register_count < 16 or register_count & (register_count - 1):
        raise ValueError("CIFAR fused routing requires power-of-two registers")
    butterfly_stages = int(math.log2(register_count))
    schedule = []
    for round_index in range(rounds):
        scale = 1 << ((round_index // 2) % 3)
        spatial_kind = "spatial_x" if round_index % 2 == 0 else "spatial_y"
        spatial_cost = 3 if spatial_kind == "spatial_x" else 1
        # Alternate within-plane and cross-plane/register butterfly distances.
        butterfly_stage = (round_index // 2) + (3 if round_index % 2 else 0)
        schedule.append(
            (
                RouteSpec("identity", estimated_instructions_per_register=0),
                RouteSpec(
                    spatial_kind,
                    scale,
                    estimated_instructions_per_register=spatial_cost,
                ),
                RouteSpec(
                    "register_xor",
                    1 << (butterfly_stage % butterfly_stages),
                    estimated_instructions_per_register=0,
                ),
            )
        )
    return schedule


def build_cifar_local_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Local-only diagnostic ablation of the hierarchical backbone."""
    if width % 1024:
        raise ValueError("CIFAR local routing requires 32x32 bitplanes")
    return [
        (
            RouteSpec("identity", estimated_instructions_per_register=0),
            RouteSpec(
                "spatial_x",
                1 << (round_index % 3),
                estimated_instructions_per_register=3,
            ),
            RouteSpec(
                "spatial_y",
                1 << (round_index % 3),
                estimated_instructions_per_register=1,
            ),
        )
        for round_index in range(rounds)
    ]


def build_cifar_butterfly_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Global-only diagnostic ablation of the hierarchical backbone."""
    if width % 1024:
        raise ValueError("CIFAR butterfly routing requires 32x32 bitplanes")
    register_count = width // register_bits
    if register_count < 16 or register_count & (register_count - 1):
        raise ValueError("CIFAR butterfly routing requires power-of-two registers")
    stages = int(math.log2(register_count))
    return [
        (
            RouteSpec("identity", estimated_instructions_per_register=0),
            RouteSpec(
                "register_xor",
                1 << (round_index % stages),
                estimated_instructions_per_register=0,
            ),
            RouteSpec(
                "byte_rotate",
                1 << (round_index % 4),
                estimated_instructions_per_register=1,
            ),
        )
        for round_index in range(rounds)
    ]


def build_cifar_sparse_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Butterfly backbone with two spatial injections per eight rounds."""
    if width % 1024:
        raise ValueError("CIFAR sparse routing requires 32x32 bitplanes")
    register_count = width // register_bits
    if register_count < 16 or register_count & (register_count - 1):
        raise ValueError("CIFAR sparse routing requires power-of-two registers")
    stages = int(math.log2(register_count))
    schedule = []
    for round_index in range(rounds):
        cycle_position = round_index % 8
        cycle_scale = 1 << ((round_index // 8) % 3)
        if cycle_position == 0:
            second = RouteSpec(
                "spatial_x", cycle_scale, estimated_instructions_per_register=3
            )
        elif cycle_position == 4:
            second = RouteSpec(
                "spatial_y", cycle_scale, estimated_instructions_per_register=1
            )
        else:
            second = RouteSpec(
                "byte_rotate",
                1 << (round_index % 4),
                estimated_instructions_per_register=1,
            )
        schedule.append(
            (
                RouteSpec("identity", estimated_instructions_per_register=0),
                second,
                RouteSpec(
                    "register_xor",
                    1 << (round_index % stages),
                    estimated_instructions_per_register=0,
                ),
            )
        )
    return schedule


def build_cifar_pyramid_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> tuple[
    list[tuple[RouteSpec, RouteSpec, RouteSpec]],
    list[RouteSpec],
]:
    """Three-stage 32→16→8 constant-width Boolean image hierarchy."""
    if rounds != 9:
        raise ValueError("CIFAR pyramid currently requires exactly nine rounds")
    if width % 1024:
        raise ValueError("CIFAR pyramid requires whole 32x32 input planes")
    sides = (32, 16, 8)
    schedule: list[tuple[RouteSpec, RouteSpec, RouteSpec]] = []
    transitions = [RouteSpec("identity") for _ in range(rounds)]
    transitions[3] = RouteSpec(
        "space_to_depth_32", estimated_instructions_per_register=4
    )
    transitions[6] = RouteSpec(
        "space_to_depth_16", estimated_instructions_per_register=4
    )
    for round_index in range(rounds):
        stage, phase = divmod(round_index, 3)
        side = sides[stage]
        plane_bits = side * side
        plane_count = width // plane_bits
        plane_route_cost = 0 if plane_bits % register_bits == 0 else 1
        identity = RouteSpec("identity", estimated_instructions_per_register=0)
        if phase == 0:
            second = RouteSpec(
                f"spatial_x_{side}", 1, estimated_instructions_per_register=3
            )
            third = RouteSpec(
                f"spatial_y_{side}", 1, estimated_instructions_per_register=1
            )
        elif phase == 1:
            second = RouteSpec(
                f"plane_xor_{side}",
                min(1 << stage, plane_count // 2),
                estimated_instructions_per_register=plane_route_cost,
            )
            third = RouteSpec(
                "byte_rotate",
                1 << stage,
                estimated_instructions_per_register=1,
            )
        else:
            second = RouteSpec(
                f"spatial_x_{side}", 2, estimated_instructions_per_register=3
            )
            third = RouteSpec(
                f"plane_xor_{side}",
                min(4 << stage, plane_count // 2),
                estimated_instructions_per_register=plane_route_cost,
            )
        schedule.append((identity, second, third))
    return schedule, transitions


def build_cifar_prepacked_pyramid_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Transition-free hierarchy over a prepacked 8x8x128 CIFAR state.

    The first four channel-butterfly distances mix the 16 subpixel channels
    produced by two encoder-side space-to-depth transforms.  Later rounds mix
    source bitplanes/repeats and cross 8x8 patch boundaries.  Distances of two
    or more whole 64-bit planes are register renames; only distance one needs
    an in-register half swap.
    """
    if rounds != 9:
        raise ValueError("prepacked CIFAR pyramid requires nine rounds")
    if width % 64:
        raise ValueError("prepacked CIFAR pyramid requires whole 8x8 planes")
    plane_count = width // 64

    def plane(distance: int) -> RouteSpec:
        return RouteSpec(
            "plane_xor_8",
            min(distance, plane_count // 2),
            estimated_instructions_per_register=1 if distance == 1 else 0,
        )

    identity = RouteSpec("identity", estimated_instructions_per_register=0)
    spatial_x_1 = RouteSpec(
        "spatial_x_8", 1, estimated_instructions_per_register=3
    )
    spatial_x_2 = RouteSpec(
        "spatial_x_8", 2, estimated_instructions_per_register=3
    )
    spatial_y_1 = RouteSpec(
        "spatial_y_8", 1, estimated_instructions_per_register=1
    )
    spatial_y_2 = RouteSpec(
        "spatial_y_8", 2, estimated_instructions_per_register=1
    )
    return [
        (identity, plane(1), plane(2)),
        (identity, plane(4), plane(8)),
        (identity, spatial_x_1, spatial_y_1),
        (identity, plane(16), RouteSpec("byte_rotate", 1, 1)),
        (identity, spatial_x_1, plane(32)),
        (identity, spatial_y_1, RouteSpec("byte_rotate", 2, 1)),
        (identity, plane(1), plane(16)),
        (identity, spatial_x_2, spatial_y_2),
        (identity, RouteSpec("byte_rotate", 4, 1), plane(64)),
    ]


def build_cifar_sparse_nonlinear_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> list[tuple[RouteSpec, RouteSpec, RouteSpec]]:
    """Residual-safe prepacked routing for cheap affine/BCAX hybrids.

    None of the three branch inputs is the unpermuted residual state, avoiding
    the algebraic cancellation that occurs in ``state XOR EOR3(state,b,c)``.
    Most channel butterflies are register renames on NEON; spatial diffusion
    is concentrated in the sparse nonlinear rounds.
    """
    if rounds != 9:
        raise ValueError("sparse nonlinear CIFAR pyramid requires nine rounds")
    if width % 64:
        raise ValueError("sparse nonlinear CIFAR pyramid requires 8x8 planes")
    plane_count = width // 64
    if plane_count & (plane_count - 1) or plane_count < 128:
        raise ValueError(
            "sparse nonlinear CIFAR pyramid requires at least 128 power-of-two planes"
        )

    def plane(distance: int) -> RouteSpec:
        if distance >= plane_count:
            raise ValueError("channel butterfly exceeds the packed plane count")
        return RouteSpec(
            "plane_xor_8",
            distance,
            estimated_instructions_per_register=1 if distance == 1 else 0,
        )

    spatial_x_1 = RouteSpec("spatial_x_8", 1, 3)
    spatial_x_2 = RouteSpec("spatial_x_8", 2, 3)
    spatial_y_1 = RouteSpec("spatial_y_8", 1, 1)
    spatial_y_2 = RouteSpec("spatial_y_8", 2, 1)
    return [
        (plane(2), plane(4), plane(8)),
        (plane(8), plane(16), plane(32)),
        (spatial_x_1, spatial_y_1, plane(64)),
        (plane(16), plane(32), plane(64)),
        (plane(4), plane(16), plane(64)),
        (spatial_x_2, spatial_y_2, RouteSpec("byte_rotate", 1, 1)),
        (plane(2), plane(8), plane(32)),
        (plane(4), plane(16), plane(64)),
        (spatial_x_1, spatial_y_1, RouteSpec("byte_rotate", 2, 1)),
    ]


def build_cifar_shifted_pyramid_schedule(
    width: int, rounds: int, register_bits: int = 128
) -> tuple[
    list[tuple[RouteSpec, RouteSpec, RouteSpec]],
    list[RouteSpec],
]:
    """Space-to-depth pyramid with alternating shifted local windows."""
    if rounds != 9:
        raise ValueError("shifted CIFAR pyramid currently requires nine rounds")
    if width % 1024:
        raise ValueError("shifted CIFAR pyramid requires whole 32x32 planes")
    sides = (32, 16, 8)
    windows = (8, 4, 4)
    transitions = [RouteSpec("identity") for _ in range(rounds)]
    transitions[3] = RouteSpec(
        "space_to_depth_32", estimated_instructions_per_register=4
    )
    transitions[6] = RouteSpec(
        "space_to_depth_16", estimated_instructions_per_register=4
    )
    schedule = []
    for round_index in range(rounds):
        stage, phase = divmod(round_index, 3)
        side = sides[stage]
        window = windows[stage]
        plane_bits = side * side
        plane_count = width // plane_bits
        plane_route_cost = 0 if plane_bits % register_bits == 0 else 1
        identity = RouteSpec("identity", estimated_instructions_per_register=0)
        if phase == 0:
            second = RouteSpec(
                f"window_x_{side}_{window}_0",
                1,
                estimated_instructions_per_register=3,
            )
            third = RouteSpec(
                f"window_y_{side}_{window}_0",
                1,
                estimated_instructions_per_register=3,
            )
        elif phase == 1:
            second = RouteSpec(
                f"plane_xor_{side}",
                min(1 << stage, plane_count // 2),
                estimated_instructions_per_register=plane_route_cost,
            )
            third = RouteSpec(
                "byte_rotate",
                1 << stage,
                estimated_instructions_per_register=1,
            )
        else:
            offset = window // 2
            second = RouteSpec(
                f"window_x_{side}_{window}_{offset}",
                1,
                estimated_instructions_per_register=3,
            )
            third = RouteSpec(
                f"window_y_{side}_{window}_{offset}",
                1,
                estimated_instructions_per_register=3,
            )
        schedule.append((identity, second, third))
    return schedule, transitions


def apply_route(state: Tensor, indices: Tensor) -> Tensor:
    return state.index_select(-1, indices.to(state.device))


def schedule_instruction_estimate(
    schedule: list[tuple[RouteSpec, RouteSpec, RouteSpec]],
    width: int,
    register_bits: int = 128,
) -> int:
    registers = width // register_bits
    return registers * sum(
        spec.estimated_instructions_per_register
        for round_specs in schedule
        for spec in round_specs
    )
