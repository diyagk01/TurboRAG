"""Common input validation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ensure_float32_vector(
    x: NDArray[np.floating] | list[float], dim: int
) -> NDArray[np.float32]:
    """Validate a single embedding vector and coerce to float32."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D vector, got shape {arr.shape}")
    if arr.shape[0] != dim:
        raise ValueError(f"expected dimension {dim}, got {arr.shape[0]}")
    return arr


def ensure_float32_matrix(
    x: NDArray[np.floating] | list[list[float]], dim: int | None = None
) -> NDArray[np.float32]:
    """Validate a batch embedding matrix and coerce to float32."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"expected dimension {dim}, got {arr.shape[1]}")
    return arr
