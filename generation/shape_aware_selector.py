from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class VectorLaunchConfig:
    """Launch parameters selected from tensor shape metadata."""

    num_elements: int
    block_size: int
    grid: tuple[int]
    reason: str


def fixed_vector_launch_config(num_elements: int, block_size: int = 1024) -> VectorLaunchConfig:
    """Return a fixed launch config used as the baseline policy."""
    if num_elements < 0:
        raise ValueError("num_elements must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    return VectorLaunchConfig(
        num_elements=num_elements,
        block_size=block_size,
        grid=(0,) if num_elements == 0 else (ceil(num_elements / block_size),),
        reason=f"fixed baseline: BLOCK_SIZE={block_size} for every vector shape",
    )


def choose_vector_launch_config(num_elements: int) -> VectorLaunchConfig:
    """Choose Triton vector-kernel launch parameters from the input size.

    The policy is intentionally simple and deterministic. It makes the
    shape-aware decision visible in tests and benchmarks without pretending to
    be a full autotuner.
    """
    if num_elements < 0:
        raise ValueError("num_elements must be non-negative")

    if num_elements == 0:
        return VectorLaunchConfig(
            num_elements=0,
            block_size=1,
            grid=(0,),
            reason="empty input: no Triton programs required",
        )

    if num_elements <= 4_096:
        block_size = 128
        reason = "small vector: lower block size avoids excess inactive lanes"
    elif num_elements <= 1_048_576:
        block_size = 256
        reason = "medium vector: balanced block size for elementwise work"
    else:
        block_size = 512
        reason = "large vector: larger block size reduces launch grid size"

    return VectorLaunchConfig(
        num_elements=num_elements,
        block_size=block_size,
        grid=(ceil(num_elements / block_size),),
        reason=reason,
    )


def select_vector_launch_config(
    num_elements: int,
    policy: str = "shape-aware",
    fixed_block_size: int = 1024,
) -> VectorLaunchConfig:
    """Select a launch config using either the baseline or shape-aware policy."""
    if policy == "shape-aware":
        return choose_vector_launch_config(num_elements)
    if policy == "fixed":
        return fixed_vector_launch_config(num_elements, fixed_block_size)
    raise ValueError(f"unsupported launch policy: {policy}")
