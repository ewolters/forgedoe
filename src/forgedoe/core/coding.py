"""Variable coding — natural ↔ coded transformations."""

from __future__ import annotations


def encode(value: float, low: float, high: float) -> float:
    """Natural → coded (-1 to +1)."""
    mid = (high + low) / 2
    half_range = (high - low) / 2
    if half_range == 0:
        return 0.0
    return (value - mid) / half_range


def decode(coded: float, low: float, high: float) -> float:
    """Coded (-1 to +1) → natural."""
    mid = (high + low) / 2
    half_range = (high - low) / 2
    return mid + coded * half_range


def encode_matrix(matrix: list[list[float]], bounds: list[tuple[float, float]]) -> list[list[float]]:
    """Encode entire design matrix from natural to coded."""
    return [
        [encode(val, bounds[j][0], bounds[j][1]) for j, val in enumerate(row)]
        for row in matrix
    ]


def decode_matrix(matrix: list[list[float]], bounds: list[tuple[float, float]]) -> list[list[float]]:
    """Decode entire design matrix from coded to natural."""
    return [
        [decode(val, bounds[j][0], bounds[j][1]) for j, val in enumerate(row)]
        for row in matrix
    ]
