<h1 align="center">TurboRAG: An SDK for TurboQuant-style compression for RAG embeddings </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e151f49-eba8-4dc2-ba56-30d1fe945111" alt="TurboRAG banner" width="900" />
</p>

<p align="center">
  <em>“And the compression is totally lossless.”</em><br/>
  <sub>no, it’s not.</sub>
</p>

--- 
## Installation
TurboRAG requires **Python 3.10+** and installs as a lightweight package with `numpy` as its core dependency.
Install the latest version directly from GitHub:
```bash
python -m pip install "git+https://github.com/diyagk01/TurboRAG.git"
```


## Usage (SDK)

Use TurboRAG in three steps: initialize one compressor, encode document embeddings into bytes, then score query embeddings directly against those bytes with `inner_product_estimate(...)`.

```python
import numpy as np
from turboquant_rag import TurboQuantCompressor

# 1) Create deterministic compressor
compressor = TurboQuantCompressor.new(
    dim=1536,
    angle_bits=4,
    projections=32,
    seed=42,
)

# 2) Encode a document embedding
doc_vec = np.random.randn(1536).astype(np.float32)
doc_code = compressor.encode(doc_vec)

# 3) Score a query embedding against compressed code
query_vec = np.random.randn(1536).astype(np.float32)
score = compressor.inner_product_estimate(doc_code, query_vec)

# Optional reconstruction
doc_hat = compressor.decode(doc_code)

print("compressed bytes:", len(doc_code))
print("score:", float(score))
print("decoded shape:", doc_hat.shape)
```

For indexing workloads, use batch APIs and persistence:

```python
codes = compressor.encode_batch(np.random.randn(128, 1536).astype(np.float32))
decoded = compressor.decode_batch(codes)

compressor.save("artifacts/turboquant_model")
reloaded = TurboQuantCompressor.load("artifacts/turboquant_model")
```

## API

- `TurboQuantCompressor.new(dim, angle_bits=4, projections=32, seed=42)`
- `TurboQuantCompressor.fit(samples, angle_bits=4, projections=32, seed=42)`
- `encode(vector) -> bytes`
- `decode(code) -> np.ndarray`
- `inner_product_estimate(code, query_vector) -> float`
- `inner_product_estimate_batch(codes, query_vector) -> np.ndarray`
- `encode_batch(matrix) -> list[bytes]`
- `decode_batch(codes) -> np.ndarray`
- `save(path) -> None`
- `TurboQuantCompressor.load(path) -> TurboQuantCompressor`
 
## Project Overview
TurboRAG is an SDK for compressing embedding vectors in RAG systems. Instead of storing full-precision embeddings, TurboRAG encodes them into a smaller representation and reconstructs them when needed. This reduces memory usage and can speed up retrieval in large vector databases.

This implementation is based on ["TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"](https://arxiv.org/abs/2504.19874), which proposes compressing high-dimensional vectors through random rotation, polar quantization, and residual correction to maintain vector similarity. 

## System Architecture
```bash
turboquant_rag/
├── turboquant_rag/                         # Core SDK package
│   ├── __init__.py                         # Public exports: TurboQuantCompressor, fit_compressor
│   ├── turboquant.py                       # Compression engine (rotation + PolarQuant + QJL residual sketch)
│   ├── factory.py                          # Factory helper for compressor creation
│   └── utils.py                            # Input validation utilities
├── tests/                                  # Unit tests for core behavior
│   ├── test_turboquant.py                  # Encode/decode, scoring signal, save/load
│   └── test_factory.py                     # Factory helper checks
├── rag_with_turboquant.py                  # End-to-end RAG demo/eval pipeline
├── benchmark/
│   ├── export_benchmark_data.py            # Embedding export for offline benchmarking
│   └── rust_turbo_quant_bench/             # Rust benchmarking workspace (experimental)
├── README.md                               # Main SDK documentation
├── RAG.md                                  # Minimal integration snippet
└── pyproject.toml                          # Packaging and dependency config 
```

## Methodology Implementation
## TurboRAG Embedding Compression Overview

TurboRAG compresses high-dimensional embeddings by deterministically mapping each input vector $x \in \mathbb{R}^d$ into a compact byte code. The process combines a reproducible random rotation, a polar quantization step, and a lightweight residual encoding based on sign sketches.

### Step 1: Orthogonal Rotation

The first step applies a fixed orthogonal rotation $R \in \mathbb{R}^{d \times d}$, generated from a shared seed.

The rotated vector is:

$$
z = xR
$$

Because $R$ is orthogonal, it preserves norms and inner-product geometry in expectation. However, the rotation spreads coordinate values more evenly across dimensions, which helps the subsequent quantization stage.

In the SDK, this rotation matrix is created once in `new(...)` and reused for every `encode` and `decode` call. 

### Step 2: Base Quantization (PolarQuant)

The base quantization stage operates on 2D coordinate pairs \((z_{2i}, z_{2i+1})\). For each pair, we convert Cartesian coordinates to polar form, keep the radius in full precision, and quantize only the angle.

For each pair:

$$
r_i = \sqrt{z_{2i}^{2} + z_{2i+1}^{2}}
$$

$$
\theta_i = \mathrm{atan2}(z_{2i+1}, z_{2i})
$$

The radius \(r_i\) is stored losslessly as a 32-bit float.  
The angle \(\theta_i\) is quantized into \(L = 2^b\) bins, where \(b = \text{angle\_bits}\), with angular step size:

$$
\Delta = \frac{2\pi}{L}
$$

#### Encoding

$$
k_i = \mathrm{round}\!\left(\frac{\theta_i}{\Delta}\right) \bmod L
$$

#### Decoding

$$
\hat{\theta}_i \approx (k_i + 0.5)\Delta
$$

#### Reconstruction

$$
\hat{z}_{2i} = r_i \cos(\hat{\theta}_i)
$$

$$
\hat{z}_{2i+1} = r_i \sin(\hat{\theta}_i)
$$

This PolarQuant stage produces a compact base approximation with controlled angular error.

### Step 3: Residual Encoding (Sign-Sketch Compression)

The residual represents what the base quantization missed:

$$
e = z - \hat{z}^{base}, \quad \rho = \|e\|_2
$$

Instead of storing full residual coordinates, the SDK generates $m$ random unit vectors, forming the rows of a Johnson–Lindenstrauss-like matrix $(g_1, \dots, g_m)$, and computes:

$$
s_j = sign(\langle g_j, e \rangle), \quad j = 1, \dots, m
$$

Only these sign bits $(s_j)$ and the residual norm $\rho$ are stored.  
This keeps memory usage extremely low while still retaining useful directional information. 

### Step 4: Decoding

Decoding reconstructs the approximate residual using a sign-sketch estimator:

$$
\hat{e} = \sqrt{\frac{\pi}{2}} \cdot \rho \cdot \frac{1}{m} \sum_{j=1}^{m} s_j g_j
$$

The full reconstruction combines both parts:

$$
\hat{z} = \hat{z}^{base} + \hat{e}, \quad \hat{x} = \hat{z} R^T
$$

Because the SDK reuses the same seeded $R$ and projection vectors $g_j$, encode/decode operations are fully deterministic and reproducible across runs.

### Step 5: Retrieval-Time Scoring

For retrieval tasks, the SDK avoids decoding or storing full vectors.

Given a query vector $q$, it computes $y = qR$ and estimates the inner product as:

$$
\langle x, q \rangle \approx \langle \hat{z}^{base}, y \rangle + \sqrt{\frac{\pi}{2}} \cdot \rho \cdot \frac{1}{m} \sum_{j=1}^{m} s_j \langle g_j, y \rangle
$$

This allows fast similarity scoring using only compact codes and a small correction term, making TurboRAG’s retrieval both efficient and accurate without reconstructing full-precision embeddings. 

## Testing and Evaluation System

We evaluate TurboRAG on the RAGBench dataset, using the `techqa` subset and `train` split in our current pipeline. The evaluation script builds a retrieval corpus directly from RAGBench `documents`, normalizes mixed document formats, chunks text into 220-word windows with 40-word overlap, deduplicates chunks, and caps the index at 4,000 chunks for a cost-controlled benchmark run. Each chunk is embedded with `text-embedding-3-small` at 1536 dimensions, then compressed with TurboQuant using `angle_bits=4`, `projections=32`, and `seed=42`. 

At retrieval time, queries are embedded with the same model and scored against compressed document codes via `inner_product_estimate(...)`, then ranked with top-k search (k=5). For quick validation, the script runs a mini-eval over 5 dataset questions and reports Hit@1, Hit@3, and Hit@5 using a source-row match proxy (a hit is counted when the query’s original row appears in retrieved metadata at cutoff k). This gives a concrete signal of ranking preservation under compression without requiring a full generator in the loop. 

In addition to end-to-end retrieval checks, the SDK has deterministic correctness tests for the compressor itself. The test suite verifies that decoded vectors maintain high semantic similarity (cosine > 0.90), that score estimation is numerically consistent (`|estimated − decoded_dot| < 1e−3`), that estimated vs true dot products have strong correlation (> 0.9), and that serialization is stable (`save/load` reconstruction matches). These checks validate that compression quality remains strong while reducing storage to compact byte codes. 

