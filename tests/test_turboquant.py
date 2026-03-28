from __future__ import annotations

import numpy as np

from turboquant_rag import TurboQuantCompressor


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def test_turboquant_encode_decode_and_score() -> None:
    rng = np.random.default_rng(51)
    samples = rng.standard_normal((3000, 128), dtype=np.float32)
    model = TurboQuantCompressor.fit(samples, angle_bits=4, projections=32, seed=7)

    x = rng.standard_normal(128, dtype=np.float32)
    q = rng.standard_normal(128, dtype=np.float32)
    code = model.encode(x)
    x_hat = model.decode(code)

    assert len(code) == model.code_nbytes
    assert x_hat.shape == x.shape
    assert x_hat.dtype == np.float32
    assert _cosine(x, x_hat) > 0.90

    est = model.inner_product_estimate(code, q)
    decoded_score = float(x_hat @ q)
    assert abs(est - decoded_score) < 1e-3


def test_turboquant_score_estimate_has_reasonable_signal() -> None:
    rng = np.random.default_rng(53)
    samples = rng.standard_normal((3000, 128), dtype=np.float32)
    model = TurboQuantCompressor.fit(samples, angle_bits=4, projections=32, seed=19)

    xs = rng.standard_normal((128, 128), dtype=np.float32)
    qs = rng.standard_normal((128, 128), dtype=np.float32)

    estimates = []
    trues = []
    for i in range(xs.shape[0]):
        code = model.encode(xs[i])
        estimates.append(model.inner_product_estimate(code, qs[i]))
        trues.append(float(xs[i] @ qs[i]))

    est = np.asarray(estimates, dtype=np.float32)
    true = np.asarray(trues, dtype=np.float32)
    corr = np.corrcoef(est, true)[0, 1]
    assert corr > 0.9


def test_turboquant_save_load(tmp_path) -> None:
    rng = np.random.default_rng(52)
    samples = rng.standard_normal((2000, 64), dtype=np.float32)
    model = TurboQuantCompressor.fit(samples, angle_bits=3, projections=16, seed=13)

    path = tmp_path / "turboquant_model"
    model.save(str(path))
    loaded = TurboQuantCompressor.load(str(path))

    x = rng.standard_normal(64, dtype=np.float32)
    np.testing.assert_allclose(model.decode(model.encode(x)), loaded.decode(loaded.encode(x)))
