import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_with_turboquant import (
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    DATASET_NAME,
    EMBED_DIM,
    EMBED_MODEL,
    MAX_CORPUS_CHUNKS,
    NUM_EVAL_QUESTIONS,
    SPLIT,
    SUBSET,
    OpenAIEmbedder,
    build_corpus_from_ragbench,
)


def _build_query_rows(ds: Any, limit: int) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i in range(min(limit, len(ds))):
        q = ds[i].get("question", "")
        if isinstance(q, str) and q.strip():
            out.append((i, q))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RAG benchmark vectors to .npy files")
    parser.add_argument("--out-dir", default="./benchmark/data", help="Output directory")
    parser.add_argument("--max-corpus-chunks", type=int, default=MAX_CORPUS_CHUNKS)
    parser.add_argument("--num-eval-questions", type=int, default=NUM_EVAL_QUESTIONS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    load_dotenv()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}/{SUBSET}/{SPLIT}")
    corpus_chunks, ds = build_corpus_from_ragbench(
        dataset_name=DATASET_NAME,
        subset=SUBSET,
        split=SPLIT,
        chunk_words=CHUNK_WORDS,
        chunk_overlap=CHUNK_OVERLAP,
        max_chunks=args.max_corpus_chunks,
    )
    print(f"Corpus chunks: {len(corpus_chunks)}")

    embedder = OpenAIEmbedder(model=EMBED_MODEL, dimensions=EMBED_DIM)

    texts = [c["text"] for c in corpus_chunks]
    source_rows = np.asarray([int(c["source_row"]) for c in corpus_chunks], dtype=np.int32)

    emb_batches = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding corpus"):
        emb_batches.append(embedder.embed_texts(texts[i : i + args.batch_size]))
    corpus_embeddings = np.vstack(emb_batches).astype(np.float32)

    query_rows = _build_query_rows(ds, args.num_eval_questions)
    query_texts = [q for _, q in query_rows]
    query_source_rows = np.asarray([idx for idx, _ in query_rows], dtype=np.int32)
    query_embeddings = embedder.embed_texts(query_texts).astype(np.float32)

    np.save(out_dir / "corpus_embeddings.npy", corpus_embeddings)
    np.save(out_dir / "corpus_source_rows.npy", source_rows)
    np.save(out_dir / "query_embeddings.npy", query_embeddings)
    np.save(out_dir / "query_source_rows.npy", query_source_rows)

    meta = {
        "dataset": DATASET_NAME,
        "subset": SUBSET,
        "split": SPLIT,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "corpus_size": int(corpus_embeddings.shape[0]),
        "num_queries": int(query_embeddings.shape[0]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "meta": meta}, indent=2))


if __name__ == "__main__":
    main()
