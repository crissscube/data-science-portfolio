"""Pick the reciprocal rank fusion constant by measurement instead of citation.

The constant published with RRF (60) assumes runs a thousand documents deep.
The two retrievers here overlap on about 16 of 100 results, and at that overlap
a large constant rewards agreement so heavily that a mediocre document returned
by both outranks a decisive first-place hit returned by one. This sweep shows
where that stops happening.

The probes are deliberately crude: five questions whose correct answer must
contain a specific literal term, scored by whether that term appears in the top
8 and how high. No LLM judge, no human labels, nothing to argue with. That also
bounds what the result is worth -- five queries drawn from the category that
favours lexical search is a small, biased sample, and the full golden set is
what settles the question.

Usage:
    .venv\\Scripts\\python.exe evals/sweep_rrf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import FUSION_DEPTH  # noqa: E402
from retrieve import BM25Retriever, DenseRetriever  # noqa: E402

PROBES = [
    ("What does Micron say about the CHIPS and Science Act?", {"ticker": "MU"}, "chips act"),
    ("Why do Qualcomm filings discuss Nuvia?", {"ticker": "QCOM"}, "nuvia"),
    ("What does AMD say specifically about TSMC?", {"ticker": "AMD"}, "tsmc"),
    ("What does NVIDIA disclose about Mellanox?", {"ticker": "NVDA"}, "mellanox"),
    ("What does Micron say about HBM high bandwidth memory?", {"ticker": "MU"}, "hbm"),
]

CONSTANTS = [0, 1, 3, 10, 60]
DEPTHS = [20, FUSION_DEPTH]
CUTOFF = 8


def fuse(dense: list[dict], lexical: list[dict], k: int, depth: int) -> list[dict]:
    scored: dict[str, list] = {}
    for run in (dense[:depth], lexical[:depth]):
        for rank, hit in enumerate(run, start=1):
            entry = scored.setdefault(hit["chunk_id"], [0.0, hit])
            entry[0] += 1 / (k + rank)
    return [hit for _, hit in sorted(scored.values(), key=lambda e: -e[0])]


def reciprocal_rank(hits: list[dict], term: str) -> float:
    """1/position of the first hit containing the term, 0 if absent from top-k."""
    for position, hit in enumerate(hits[:CUTOFF], start=1):
        if term in hit["text"].lower():
            return 1 / position
    return 0.0


def main() -> None:
    dense_retriever = DenseRetriever()
    lexical_retriever = BM25Retriever()

    runs = [
        (term,
         dense_retriever.search(question, k=max(DEPTHS), scope=scope),
         lexical_retriever.search(question, k=max(DEPTHS), scope=scope))
        for question, scope, term in PROBES
    ]

    header = "  ".join(f"{term[:9]:<9}" for _, _, term in PROBES)
    print(f"\nMRR@{CUTOFF}  (1.0 = correcto en primer lugar, 0 = ausente)\n")
    print(f"{'config':<22} {header}  MEDIA")
    print("-" * 92)

    def row(label: str, scores: list[float]) -> None:
        cells = "  ".join(f"{score:<9.3f}" for score in scores)
        print(f"{label:<22} {cells}  {sum(scores) / len(scores):.3f}")

    row("solo denso", [reciprocal_rank(dense, term) for term, dense, _ in runs])
    row("solo bm25", [reciprocal_rank(lex, term) for term, _, lex in runs])
    print("-" * 92)

    for depth in DEPTHS:
        for constant in CONSTANTS:
            scores = [
                reciprocal_rank(fuse(dense, lex, constant, depth), term)
                for term, dense, lex in runs
            ]
            row(f"RRF K={constant:<3d} depth={depth}", scores)


if __name__ == "__main__":
    main()
