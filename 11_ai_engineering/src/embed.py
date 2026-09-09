"""Turn chunks into vectors and store them in Chroma, one collection per strategy.

Two details here are easy to get wrong and expensive to discover later.

First, bge-base is asymmetric: the instruction prefix belongs on the query side
only. Encoding passages with it too is a common copy-paste error that shifts the
whole passage space and costs recall without ever raising an error.

Second, each chunk is encoded with its provenance header attached. A chunk taken
from the middle of a risk-factor section never names its company or year, so
without the header the encoder cannot tell NVIDIA's 2024 supply risk from
AMD's 2021 one -- and the metadata filter can only narrow what the encoder was
able to represent in the first place.

The run takes roughly 45 minutes on CPU, so it is resumable: chunks already in
the collection are skipped, and interrupting with Ctrl+C loses at most one batch.

Usage:
    .venv\\Scripts\\python.exe src/embed.py                    # both strategies
    .venv\\Scripts\\python.exe src/embed.py --strategy structural
    .venv\\Scripts\\python.exe src/embed.py --limit 200        # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    PROCESSED_DIR,
)

# Chroma rejects None in metadata, so keys with no value are dropped instead.
# A filter on a missing key simply matches nothing, which is the right
# behaviour: the fixed-stride chunks genuinely have no Item to filter on.
METADATA_FIELDS = ["ticker", "fiscal_year", "item", "is_core", "chunk_index",
                   "n_chars"]


def load_chunks(strategy: str, limit: int | None) -> list[dict]:
    path = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
    if not path.exists():
        raise SystemExit(f"No existe {path}. Corre antes src/chunk.py.")
    with path.open(encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh]
    return chunks[:limit] if limit else chunks


def to_metadata(chunk: dict) -> dict:
    return {field: chunk[field] for field in METADATA_FIELDS
            if chunk.get(field) is not None}


def embed_strategy(strategy: str, model: SentenceTransformer,
                   client: chromadb.ClientAPI, limit: int | None) -> None:
    chunks = load_chunks(strategy, limit)
    collection = client.get_or_create_collection(
        name=f"tenk_{strategy}",
        # Cosine, because bge vectors are normalised and cosine is what the
        # model was trained against. Chroma defaults to L2 and would silently
        # rank differently.
        metadata={"hnsw:space": "cosine"},
    )

    existing = set(collection.get(include=[])["ids"])
    pending = [chunk for chunk in chunks if chunk["chunk_id"] not in existing]

    print(f"\n{strategy}: {len(chunks):,} fragmentos, "
          f"{len(existing):,} ya vectorizados, {len(pending):,} por hacer")
    if not pending:
        print("  nada que hacer")
        return

    start = time.time()
    for offset in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[offset:offset + EMBED_BATCH_SIZE]
        # Provenance header first, then the chunk body (see module docstring).
        texts = [f"{chunk['context_header']}. {chunk['text']}" for chunk in batch]

        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            embeddings=[vector.tolist() for vector in vectors],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[to_metadata(chunk) for chunk in batch],
        )

        done = offset + len(batch)
        elapsed = time.time() - start
        rate = done / elapsed
        remaining = (len(pending) - done) / rate / 60
        print(f"  {done:>6,}/{len(pending):,}  "
              f"{rate:4.1f} frag/s  faltan {remaining:5.1f} min", flush=True)

    print(f"  listo en {(time.time() - start) / 60:.1f} min  "
          f"-> coleccion tenk_{strategy}: {collection.count():,} vectores")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorizar el corpus troceado.")
    parser.add_argument("--strategy", choices=["structural", "fixed", "both"],
                        default="both")
    parser.add_argument("--limit", type=int,
                        help="solo los primeros N fragmentos (prueba rapida)")
    args = parser.parse_args()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"modelo: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    strategies = (["structural", "fixed"] if args.strategy == "both"
                  else [args.strategy])
    for strategy in strategies:
        embed_strategy(strategy, model, client, args.limit)


if __name__ == "__main__":
    main()
