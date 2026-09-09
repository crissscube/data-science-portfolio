"""Search the vector store. The single entry point every later stage shares.

Keeping retrieval behind one interface is what makes the experiments comparable:
the hybrid retriever added later plugs in here, and the evaluation harness calls
the same `search` regardless of which strategy is under test.

Usage:
    .venv\\Scripts\\python.exe src/retrieve.py "export controls on AI chips" --ticker NVDA --year 2024
"""

from __future__ import annotations

import argparse
import functools
import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BGE_QUERY_PREFIX,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    FUSION_DEPTH,
    PROCESSED_DIR,
    RRF_K,
    TOP_K,
)


def build_where(scope: dict | None) -> dict | None:
    """Translate a scope dict into a Chroma filter.

    Chroma takes a bare mapping for one condition but demands an explicit $and
    once there are two or more, which is easy to trip over.
    """
    if not scope:
        return None
    clauses = [{key: value} for key, value in scope.items() if value is not None]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@functools.lru_cache(maxsize=2)
def _model() -> SentenceTransformer:
    """Loaded once per process; the encoder takes ~20 s to come up."""
    return SentenceTransformer(EMBEDDING_MODEL)


class DenseRetriever:
    """Semantic search over one chunking strategy's collection."""

    def __init__(self, strategy: str = "structural") -> None:
        self.strategy = strategy
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(f"tenk_{strategy}")

    def search(self, query: str, k: int = TOP_K,
               scope: dict | None = None) -> list[dict]:
        # The instruction prefix belongs on the query side only -- bge was
        # trained asymmetrically and applying it to passages costs recall.
        vector = _model().encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True
        )[0].tolist()

        raw = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=build_where(scope),
        )

        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "distance": distance,
                # Cosine distance runs 0 (identical) to 2 (opposite); the
                # complement reads as a similarity score in results tables.
                "score": 1 - distance,
                **metadata,
            }
            for chunk_id, text, distance, metadata in zip(
                raw["ids"][0], raw["documents"][0],
                raw["distances"][0], raw["metadatas"][0],
            )
        ]


TOKEN = re.compile(r"[a-z0-9]+")

# Words so common in a 10-K that matching them says nothing. The point of BM25
# here is rare literal terms -- "TSMC", "Nuvia", "CHIPS" -- and leaving the
# boilerplate in dilutes exactly the signal it is being added for.
STOPWORDS = frozenset("""
a an the and or of to in for on at by with from as is are was were be been being
we our us it its this that these those may might will would could should can
""".split())


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text.lower())
            if token not in STOPWORDS]


class BM25Retriever:
    """Lexical search: rewards the rare literal term dense search blurs away.

    Scores are computed over the whole corpus and filtered afterwards rather
    than building a filtered index per query. BM25's term statistics come from
    the full collection, so scoring a subset would change what counts as a rare
    word and make results depend on the filter -- which is not what is wanted.
    """

    def __init__(self, strategy: str = "structural") -> None:
        self.strategy = strategy
        path = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
        with path.open(encoding="utf-8") as fh:
            self.chunks = [json.loads(line) for line in fh]
        self.index = BM25Okapi([tokenize(chunk["text"]) for chunk in self.chunks])

    def search(self, query: str, k: int = TOP_K,
               scope: dict | None = None) -> list[dict]:
        scores = self.index.get_scores(tokenize(query))

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = []
        for position in ranked:
            chunk = self.chunks[position]
            if scope and any(chunk.get(key) != value
                             for key, value in scope.items() if value is not None):
                continue
            hits.append({**chunk, "score": float(scores[position]),
                         "retriever": "bm25"})
            if len(hits) == k:
                break
        return hits


class HybridRetriever:
    """Dense and lexical results merged by reciprocal rank fusion.

    RRF combines rankings, not scores. That matters: BM25 scores are unbounded
    and corpus-dependent while cosine similarity sits in a fixed range, so any
    weighted sum of the two would need re-tuning per query. Using positions
    sidesteps the problem entirely and needs no calibration.
    """

    def __init__(self, strategy: str = "structural") -> None:
        self.dense = DenseRetriever(strategy)
        self.lexical = BM25Retriever(strategy)

    def search(self, query: str, k: int = TOP_K,
               scope: dict | None = None) -> list[dict]:
        runs = {
            "dense": self.dense.search(query, k=FUSION_DEPTH, scope=scope),
            "bm25": self.lexical.search(query, k=FUSION_DEPTH, scope=scope),
        }

        fused: dict[str, dict] = {}
        for name, hits in runs.items():
            for rank, hit in enumerate(hits, start=1):
                entry = fused.setdefault(hit["chunk_id"], {
                    **hit, "score": 0.0, "ranks": {}, "retriever": ""})
                entry["score"] += 1 / (RRF_K + rank)
                entry["ranks"][name] = rank

        for entry in fused.values():
            # Recorded so results can report which retriever found what -- the
            # whole argument for hybrid rests on them disagreeing.
            entry["retriever"] = "+".join(sorted(entry["ranks"]))

        return sorted(fused.values(), key=lambda hit: -hit["score"])[:k]


RETRIEVERS = {
    "dense": DenseRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
}


def best_window(text: str, query: str, width: int) -> tuple[str, bool]:
    """Return the slice of `text` densest in query terms, and whether it was cut.

    Showing the first N characters of a chunk hides the match whenever the
    relevant sentence sits further in. During labelling that reads as "this
    result is irrelevant" and the chunk gets rejected -- the reviewer is judging
    the preview rather than the chunk.
    """
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat, False

    terms = set(tokenize(query))
    lowered = flat.lower()
    positions = [match.start() for match in TOKEN.finditer(lowered)
                 if match.group() in terms]
    if not positions:
        return flat[:width], True

    # Slide a window over the match positions and keep the one covering most.
    best_start, best_count = 0, 0
    for position in positions:
        start = max(0, position - width // 4)
        count = sum(1 for other in positions if start <= other < start + width)
        if count > best_count:
            best_start, best_count = start, count

    return flat[best_start:best_start + width], True


def format_hit(index: int, hit: dict, width: int = 320,
               query: str | None = None, show_score: bool = True) -> str:
    where = f"{hit['ticker']} FY{hit['fiscal_year']}"
    if hit.get("item"):
        where += f" Item {hit['item']}"
    source = f" [{hit['retriever']}]" if hit.get("retriever") else ""

    if query:
        body, trimmed = best_window(hit["text"], query, width)
        marker = "..." if trimmed else ""
    else:
        body, marker = " ".join(hit["text"].split())[:width], ""

    # Scores from different retrievers sit on different scales -- bounded cosine
    # similarity against unbounded BM25 -- so printing them next to each other in
    # a pooled list invites comparing numbers that mean nothing together.
    head = f"{hit['score']:.3f}  " if show_score else ""
    return f"[{index}] {head}{where}{source}\n     {marker}{body}{marker}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Buscar en el corpus.")
    parser.add_argument("query")
    parser.add_argument("--retriever", choices=list(RETRIEVERS), default="hybrid")
    parser.add_argument("--strategy", choices=["structural", "fixed"],
                        default="structural")
    parser.add_argument("--ticker")
    parser.add_argument("--year", type=int)
    parser.add_argument("--item")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    scope = {"ticker": args.ticker, "fiscal_year": args.year, "item": args.item}
    retriever = RETRIEVERS[args.retriever](args.strategy)
    hits = retriever.search(args.query, k=args.k, scope=scope)

    print(f"\n{args.query}\n" + "-" * 88)
    for index, hit in enumerate(hits, 1):
        print(format_hit(index, hit, query=args.query))
        print()


if __name__ == "__main__":
    main()
