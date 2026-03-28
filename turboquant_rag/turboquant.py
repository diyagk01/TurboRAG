"""TurboQuant-style compressor: rotation + PolarQuant + QJL residual sketch."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .utils import ensure_float32_matrix, ensure_float32_vector


def _stable_matmul(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = a @ b
    out = out.astype(np.float32, copy=False)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite values produced during matmul")
    return out


def _random_orthogonal_matrix(dim: int, seed: int) -> NDArray[np.float32]:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((dim, dim), dtype=np.float64)
    q, r = np.linalg.qr(g)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs[np.newaxis, :]
    return q.astype(np.float32, copy=False)


def _pack_nbit(values: NDArray[np.uint16], bits: int) -> bytes:
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if bits <= 0 or bits > 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    mask = (1 << bits) - 1
    total_bits = int(values.size * bits)
    out = bytearray((total_bits + 7) // 8)
    bit_cursor = 0
    for raw in values.tolist():
        v = int(raw) & mask
        byte_idx = bit_cursor // 8
        bit_offset = bit_cursor % 8
        chunk = v << bit_offset
        out[byte_idx] |= chunk & 0xFF
        if byte_idx + 1 < len(out):
            out[byte_idx + 1] |= (chunk >> 8) & 0xFF
        if byte_idx + 2 < len(out):
            out[byte_idx + 2] |= (chunk >> 16) & 0xFF
        bit_cursor += bits
    return bytes(out)


def _unpack_nbit(data: bytes, count: int, bits: int) -> NDArray[np.uint16]:
    if bits <= 0 or bits > 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    expected = (count * bits + 7) // 8
    if len(data) != expected:
        raise ValueError(f"invalid payload length: expected {expected}, got {len(data)}")
    buf = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(count, dtype=np.uint16)
    mask = (1 << bits) - 1
    bit_cursor = 0
    for i in range(count):
        byte_idx = bit_cursor // 8
        bit_offset = bit_cursor % 8
        w0 = int(buf[byte_idx])
        w1 = int(buf[byte_idx + 1]) if byte_idx + 1 < buf.size else 0
        w2 = int(buf[byte_idx + 2]) if byte_idx + 2 < buf.size else 0
        word = w0 | (w1 << 8) | (w2 << 16)
        out[i] = np.uint16((word >> bit_offset) & mask)
        bit_cursor += bits
    return out


def _pack_signs(signs: NDArray[np.bool_]) -> bytes:
    if signs.ndim != 1:
        raise ValueError("signs must be 1D")
    return np.packbits(signs.astype(np.uint8), bitorder="little").tobytes()


def _unpack_signs(data: bytes, count: int) -> NDArray[np.bool_]:
    expected = (count + 7) // 8
    if len(data) != expected:
        raise ValueError(f"invalid sign payload length: expected {expected}, got {len(data)}")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    return bits[:count].astype(bool, copy=False)


@dataclass(slots=True)
class TurboQuantCompressor:
    """Data-oblivious compressor using fixed rotation, PolarQuant, and QJL signs."""

    dim: int
    angle_bits: int
    projections: int
    seed: int
    rotation: NDArray[np.float32]
    qjl_matrix: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.dim % 2 != 0:
            raise ValueError("TurboQuantCompressor requires an even dim for polar pairing")
        if not (1 <= self.angle_bits <= 8):
            raise ValueError("angle_bits must be in [1, 8]")
        if self.projections < 0:
            raise ValueError("projections must be >= 0")
        if self.rotation.shape != (self.dim, self.dim):
            raise ValueError("rotation shape must be (dim, dim)")
        if self.qjl_matrix.shape != (self.projections, self.dim):
            raise ValueError("qjl_matrix shape must be (projections, dim)")
        self.rotation = self.rotation.astype(np.float32, copy=False)
        self.qjl_matrix = self.qjl_matrix.astype(np.float32, copy=False)

    @classmethod
    def new(
        cls,
        dim: int,
        angle_bits: int = 4,
        projections: int = 32,
        seed: int = 42,
    ) -> "TurboQuantCompressor":
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % 2 != 0:
            raise ValueError("dim must be even for polar coordinate pairing")
        if projections < 0:
            raise ValueError("projections must be >= 0")
        rotation = _random_orthogonal_matrix(dim, seed=seed)
        if projections == 0:
            qjl_matrix = np.empty((0, dim), dtype=np.float32)
        else:
            rng = np.random.default_rng(seed + 1)
            qjl_matrix = rng.standard_normal((projections, dim), dtype=np.float32)
            norms = np.linalg.norm(qjl_matrix, axis=1, keepdims=True).astype(np.float32)
            qjl_matrix = qjl_matrix / np.maximum(norms, 1e-12)
        return cls(
            dim=dim,
            angle_bits=angle_bits,
            projections=projections,
            seed=seed,
            rotation=rotation,
            qjl_matrix=qjl_matrix,
        )

    @classmethod
    def fit(
        cls,
        samples: NDArray[np.floating] | list[list[float]],
        angle_bits: int = 4,
        projections: int = 32,
        seed: int = 42,
    ) -> "TurboQuantCompressor":
        """Create data-oblivious codec from sample dimensionality."""
        data = ensure_float32_matrix(samples)
        return cls.new(
            dim=int(data.shape[1]),
            angle_bits=angle_bits,
            projections=projections,
            seed=seed,
        )

    @property
    def num_pairs(self) -> int:
        return self.dim // 2

    @property
    def angle_levels(self) -> int:
        return 1 << self.angle_bits

    @property
    def _angle_step(self) -> float:
        return (2.0 * math.pi) / float(self.angle_levels)

    @property
    def _radii_nbytes(self) -> int:
        return self.num_pairs * 4

    @property
    def _angles_nbytes(self) -> int:
        return (self.num_pairs * self.angle_bits + 7) // 8

    @property
    def _signs_nbytes(self) -> int:
        return (self.projections + 7) // 8

    @property
    def code_nbytes(self) -> int:
        return self._radii_nbytes + self._angles_nbytes + self._signs_nbytes + 2

    def encode(self, embedding: NDArray[np.floating] | list[float]) -> bytes:
        x = ensure_float32_vector(embedding, self.dim)
        z = _stable_matmul(x[np.newaxis, :], self.rotation)[0]
        payload = self._encode_rotated(z)
        if len(payload) != self.code_nbytes:
            raise RuntimeError("internal encoding error: unexpected payload size")
        return payload

    def decode(self, code: bytes) -> NDArray[np.float32]:
        z_base, signs, residual_norm = self._decode_payload(code)
        z_hat = z_base + self._residual_from_signs(signs, residual_norm)
        return _stable_matmul(z_hat[np.newaxis, :], self.rotation.T)[0]

    def inner_product_estimate(
        self,
        code: bytes,
        query_embedding: NDArray[np.floating] | list[float],
    ) -> float:
        q = ensure_float32_vector(query_embedding, self.dim)
        y = _stable_matmul(q[np.newaxis, :], self.rotation)[0]
        z_base, signs, residual_norm = self._decode_payload(code)
        score = float(z_base @ y)
        if self.projections > 0 and residual_norm > 0.0:
            proj_q = _stable_matmul(self.qjl_matrix, y[:, np.newaxis])[:, 0]
            sign_vals = np.where(signs, 1.0, -1.0).astype(np.float32)
            corr = math.sqrt(math.pi / 2.0) * residual_norm * float(np.mean(sign_vals * proj_q))
            score += corr
        return float(score)

    def inner_product_estimate_batch(
        self,
        codes: Sequence[bytes],
        query_embedding: NDArray[np.floating] | list[float],
    ) -> NDArray[np.float32]:
        if len(codes) == 0:
            return np.empty((0,), dtype=np.float32)
        q = ensure_float32_vector(query_embedding, self.dim)
        y = _stable_matmul(q[np.newaxis, :], self.rotation)[0]
        if self.projections > 0:
            proj_q = _stable_matmul(self.qjl_matrix, y[:, np.newaxis])[:, 0]
        else:
            proj_q = np.empty((0,), dtype=np.float32)

        scores = np.empty((len(codes),), dtype=np.float32)
        for i, code in enumerate(codes):
            z_base, signs, residual_norm = self._decode_payload(code)
            s = float(z_base @ y)
            if self.projections > 0 and residual_norm > 0.0:
                sign_vals = np.where(signs, 1.0, -1.0).astype(np.float32)
                corr = math.sqrt(math.pi / 2.0) * residual_norm * float(np.mean(sign_vals * proj_q))
                s += corr
            scores[i] = np.float32(s)
        return scores

    def encode_batch(self, embeddings: NDArray[np.floating]) -> list[bytes]:
        matrix = ensure_float32_matrix(embeddings, dim=self.dim)
        z = _stable_matmul(matrix, self.rotation)
        return [self._encode_rotated(z[i]) for i in range(z.shape[0])]

    def decode_batch(self, codes: Sequence[bytes]) -> NDArray[np.float32]:
        if len(codes) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        rows = [self.decode(code) for code in codes]
        return np.stack(rows, axis=0).astype(np.float32)

    def save(self, path: str) -> None:
        npz_path, json_path = _resolve_paths(path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, rotation=self.rotation, qjl_matrix=self.qjl_matrix)
        metadata = {
            "format_version": 1,
            "method": "turboquant",
            "dim": self.dim,
            "angle_bits": self.angle_bits,
            "projections": self.projections,
            "seed": self.seed,
            "npz_file": npz_path.name,
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TurboQuantCompressor":
        npz_path, json_path = _resolve_paths(path)
        if not npz_path.exists():
            raise FileNotFoundError(f"array file not found: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"metadata file not found: {json_path}")
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            rotation = data["rotation"].astype(np.float32)
            qjl_matrix = data["qjl_matrix"].astype(np.float32)
        return cls(
            dim=int(meta["dim"]),
            angle_bits=int(meta["angle_bits"]),
            projections=int(meta["projections"]),
            seed=int(meta.get("seed", 42)),
            rotation=rotation,
            qjl_matrix=qjl_matrix,
        )

    def _encode_rotated(self, z: NDArray[np.float32]) -> bytes:
        a = z[0::2]
        b = z[1::2]
        radii = np.hypot(a, b).astype(np.float32)

        theta = np.arctan2(b, a).astype(np.float32)
        theta = np.where(theta < 0.0, theta + np.float32(2.0 * math.pi), theta)
        codes = np.floor(theta / np.float32(self._angle_step) + np.float32(0.5)).astype(np.int32)
        np.mod(codes, self.angle_levels, out=codes)
        angle_payload = _pack_nbit(codes.astype(np.uint16), bits=self.angle_bits)

        z_base = self._polar_decode(radii, codes.astype(np.uint16))
        residual = z - z_base
        residual_norm = float(np.linalg.norm(residual))

        if self.projections > 0:
            proj = _stable_matmul(self.qjl_matrix, residual[:, np.newaxis])[:, 0]
            signs = proj >= 0.0
            sign_payload = _pack_signs(signs)
        else:
            sign_payload = b""

        return (
            radii.tobytes()
            + angle_payload
            + sign_payload
            + np.asarray([residual_norm], dtype=np.float16).tobytes()
        )

    def _decode_payload(self, code: bytes) -> tuple[NDArray[np.float32], NDArray[np.bool_], float]:
        if len(code) != self.code_nbytes:
            raise ValueError(f"invalid code length: expected {self.code_nbytes}, got {len(code)}")
        cursor = 0
        radii = np.frombuffer(code[cursor : cursor + self._radii_nbytes], dtype=np.float32).copy()
        cursor += self._radii_nbytes

        angles_raw = code[cursor : cursor + self._angles_nbytes]
        cursor += self._angles_nbytes
        angle_codes = _unpack_nbit(angles_raw, count=self.num_pairs, bits=self.angle_bits)

        if self.projections > 0:
            signs_raw = code[cursor : cursor + self._signs_nbytes]
            cursor += self._signs_nbytes
            signs = _unpack_signs(signs_raw, count=self.projections)
        else:
            signs = np.empty((0,), dtype=bool)

        residual_norm = float(np.frombuffer(code[cursor : cursor + 2], dtype=np.float16)[0])
        z_base = self._polar_decode(radii, angle_codes)
        return z_base, signs, residual_norm

    def _polar_decode(
        self,
        radii: NDArray[np.float32],
        angle_codes: NDArray[np.uint16],
    ) -> NDArray[np.float32]:
        theta = (angle_codes.astype(np.float32) + 0.5) * np.float32(self._angle_step)
        a = radii * np.cos(theta).astype(np.float32)
        b = radii * np.sin(theta).astype(np.float32)
        z_base = np.empty((self.dim,), dtype=np.float32)
        z_base[0::2] = a
        z_base[1::2] = b
        return z_base

    def _residual_from_signs(
        self,
        signs: NDArray[np.bool_],
        residual_norm: float,
    ) -> NDArray[np.float32]:
        if self.projections == 0 or residual_norm <= 0.0:
            return np.zeros((self.dim,), dtype=np.float32)
        sign_vals = np.where(signs, 1.0, -1.0).astype(np.float32)
        coeff = math.sqrt(math.pi / 2.0) * float(residual_norm) / float(self.projections)
        return coeff * _stable_matmul(sign_vals[np.newaxis, :], self.qjl_matrix)[0]


def _resolve_paths(path: str) -> tuple[Path, Path]:
    raw = Path(path)
    if raw.suffix == ".npz":
        return raw, raw.with_suffix(".json")
    if raw.suffix == ".json":
        return raw.with_suffix(".npz"), raw
    return raw.with_suffix(".npz"), raw.with_suffix(".json")
