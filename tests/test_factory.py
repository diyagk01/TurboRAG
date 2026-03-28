from __future__ import annotations

import numpy as np

from turboquant_rag import TurboQuantCompressor, fit_compressor


def test_factory_returns_expected_types() -> None:
    rng = np.random.default_rng(3)
    samples = rng.standard_normal((1000, 192), dtype=np.float32)

    a = fit_compressor(samples, method="turboquant", angle_bits=4, projections=24, seed=7)

    assert isinstance(a, TurboQuantCompressor)
