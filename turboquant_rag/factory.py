"""High-level factory for TurboQuant SDK."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .turboquant import TurboQuantCompressor

def fit_compressor(
    samples: NDArray[np.floating] | list[list[float]],
    method: str = "turboquant",
    angle_bits: int = 4,
    projections: int = 32,
    seed: int = 42,
) -> TurboQuantCompressor:
    """Create a TurboQuant compressor from sample dimensionality."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {data.shape}")

    if method == "turboquant":
        return TurboQuantCompressor.fit(
            data,
            angle_bits=angle_bits,
            projections=projections,
            seed=seed,
        )

    raise ValueError("unsupported method. only 'turboquant' is available in SDK-only build")
