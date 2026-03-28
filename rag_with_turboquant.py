import os
import json
import math
from typing import List, Dict, Any, Tuple

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Your package
from turboquant_rag import TurboQuantCompressor


# -----------------------------
# Config
# -----------------------------
DATASET_NAME = "galileo-ai/ragbench"
SUBSET = "techqa"          # try: techqa, hotpotqa, pubmedqa, etc.
SPLIT = "train"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536           # must match model output or chosen dimensions
BATCH_SIZE = 64

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40

ANGLE_BITS = 4
PROJECTIONS = 32
SEED = 42

TOP_K = 5
MAX_CORPUS_CHUNKS = 4000   # keep small first so it is cheap to test
RUN_SAMPLE_EVAL = True
NUM_EVAL_QUESTIONS = 5


# -----------------------------
# Utils
# -----------------------------
def chunk_text(text: str, chunk_words: int = 220, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += max(1, chunk_words - overlap)
    return chunks


def batched(xs: List[Any], batch_size: int) -> List[List[Any]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]


def safe_get_documents(row: Dict[str, Any]) -> List[str]:
    """
    RAGBench examples commonly expose a `documents` field.
    Sometimes entries are strings, sometimes structured text.
    This tries to normalize them into plain strings.
    """
    docs = row.get("documents", [])
    out = []

    for d in docs:
        if isinstance(d, str):
            text = d.strip()
            if text:
                out.append(text)
        elif isinstance(d, dict):
            # best effort: join string-ish values
            pieces = []
            for v in d.values():
                if isinstance(v, str) and v.strip():
                    pieces.append(v.strip())
            text = "\n".join(pieces).strip()
            if text:
                out.append(text)
        else:
            text = str(d).strip()
            if text:
                out.append(text)

    return out


# -----------------------------
# OpenAI Embedding Wrapper
# -----------------------------
class OpenAIEmbedder:
    def __init__(self, model: str, dimensions: int | None = None):
        self.client = OpenAI()
        self.model = model
        self.dimensions = dimensions

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        kwargs = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        resp = self.client.embeddings.create(**kwargs)
        arr = np.array([item.embedding for item in resp.data], dtype=np.float32)
        return arr

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


# -----------------------------
# TurboQuant-backed retriever
# -----------------------------
class TurboQuantRAG:
    def __init__(
        self,
        embedder: OpenAIEmbedder,
        compressor: TurboQuantCompressor,
    ):
        self.embedder = embedder
        self.compressor = compressor
        self.codes: List[bytes] = []
        self.texts: List[str] = []
        self.meta: List[Dict[str, Any]] = []

    def build_index(self, corpus_chunks: List[Dict[str, Any]], batch_size: int = 64) -> None:
        self.codes = []
        self.texts = []
        self.meta = []

        texts = [x["text"] for x in corpus_chunks]

        for batch_idx, text_batch in enumerate(tqdm(list(batched(texts, batch_size)), desc="Embedding+compressing")):
            embs = self.embedder.embed_texts(text_batch)

            for j, emb in enumerate(embs):
                code = self.compressor.encode(emb)
                self.codes.append(code)

                item = corpus_chunks[batch_idx * batch_size + j]
                self.texts.append(item["text"])
                self.meta.append(
                    {
                        "source_row": item.get("source_row"),
                        "doc_id": item.get("doc_id"),
                        "chunk_id": item.get("chunk_id"),
                    }
                )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = self.embedder.embed_text(query)

        scored = []
        for i, code in enumerate(self.codes):
            score = float(self.compressor.inner_product_estimate(code, qvec))
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, idx in top:
            results.append(
                {
                    "score": score,
                    "text": self.texts[idx],
                    "meta": self.meta[idx],
                }
            )
        return results


# -----------------------------
# Corpus builder
# -----------------------------
def build_corpus_from_ragbench(
    dataset_name: str,
    subset: str,
    split: str,
    chunk_words: int,
    chunk_overlap: int,
    max_chunks: int | None = None,
) -> Tuple[List[Dict[str, Any]], Any]:
    ds = load_dataset(dataset_name, subset, split=split)

    corpus_chunks: List[Dict[str, Any]] = []
    seen = set()

    for row_idx, row in enumerate(tqdm(ds, desc="Extracting corpus")):
        docs = safe_get_documents(row)

        for doc_idx, doc_text in enumerate(docs):
            for chunk_idx, chunk in enumerate(chunk_text(doc_text, chunk_words=chunk_words, overlap=chunk_overlap)):
                norm = chunk.strip()
                if not norm:
                    continue

                # cheap dedupe
                key = hash(norm)
                if key in seen:
                    continue
                seen.add(key)

                corpus_chunks.append(
                    {
                        "text": norm,
                        "source_row": row_idx,
                        "doc_id": doc_idx,
                        "chunk_id": chunk_idx,
                    }
                )

                if max_chunks is not None and len(corpus_chunks) >= max_chunks:
                    return corpus_chunks, ds

    return corpus_chunks, ds


# -----------------------------
# Pretty printing
# -----------------------------
def print_retrieval(query: str, results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.6f} meta={r['meta']}")
        print(r["text"][:1000])
        print("-" * 100)


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in your environment or .env file.")

    print(f"Loading dataset: {DATASET_NAME} / {SUBSET} / {SPLIT}")
    corpus_chunks, ds = build_corpus_from_ragbench(
        dataset_name=DATASET_NAME,
        subset=SUBSET,
        split=SPLIT,
        chunk_words=CHUNK_WORDS,
        chunk_overlap=CHUNK_OVERLAP,
        max_chunks=MAX_CORPUS_CHUNKS,
    )

    print(f"Built corpus with {len(corpus_chunks)} chunks")

    embedder = OpenAIEmbedder(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,   # set to None if you want default model dimension
    )

    compressor = TurboQuantCompressor.new(
        dim=EMBED_DIM,
        angle_bits=ANGLE_BITS,
        projections=PROJECTIONS,
        seed=SEED,
    )

    rag = TurboQuantRAG(embedder=embedder, compressor=compressor)
    rag.build_index(corpus_chunks, batch_size=BATCH_SIZE)

    # Interactive query
    demo_query = "How do I troubleshoot a failing hardware component?"
    results = rag.retrieve(demo_query, top_k=TOP_K)
    print_retrieval(demo_query, results)

    # Mini eval using dataset questions
    if RUN_SAMPLE_EVAL:
        print("\nRunning mini retrieval eval on dataset questions...")
        total = min(NUM_EVAL_QUESTIONS, len(ds))
        hit1 = 0
        hit3 = 0
        hit5 = 0

        for i in range(total):
            row = ds[i]
            question = row.get("question", "")
            gold = row.get("response", "")

            if not question:
                continue

            results = rag.retrieve(question, top_k=TOP_K)

            # Weak-but-useful retrieval proxy:
            # the query row's supporting docs were chunked with source_row == i.
            retrieved_rows = [
                r.get("meta", {}).get("source_row")
                for r in results
                if isinstance(r.get("meta"), dict)
            ]
            if len(retrieved_rows) >= 1 and i in retrieved_rows[:1]:
                hit1 += 1
            if len(retrieved_rows) >= 3 and i in retrieved_rows[:3]:
                hit3 += 1
            if len(retrieved_rows) >= 5 and i in retrieved_rows[:5]:
                hit5 += 1

            print("\n" + "#" * 100)
            print(f"Example {i + 1}")
            print(f"QUESTION: {question}")
            print(f"GOLD ANSWER: {str(gold)[:500]}")
            print("#" * 100)

            for j, r in enumerate(results, 1):
                print(f"\nRetrieved {j} | score={r['score']:.6f} | meta={r['meta']}")
                print(r["text"][:700])
                print("-" * 80)

        if total > 0:
            print("\n" + "=" * 100)
            print("Mini Eval Retrieval Metrics (source_row match proxy)")
            print(f"Hit@1: {hit1}/{total} = {hit1 / total:.3f}")
            print(f"Hit@3: {hit3}/{total} = {hit3 / total:.3f}")
            print(f"Hit@5: {hit5}/{total} = {hit5 / total:.3f}")
            print("=" * 100)


if __name__ == "__main__":
    main()