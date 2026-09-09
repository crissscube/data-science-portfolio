"""Score every retrieval configuration against the golden set.

Six configurations are compared: three retrievers (dense, BM25, hybrid) across
two chunking strategies (structural, fixed). That grid is the point of the
project -- each of the first two differentiators becomes a number here instead
of an argument.

Matching is by content, not by chunk id. The gold labels name chunks from the
structural corpus, so an id comparison would score the fixed strategy at zero
by construction even when it returns the same text. Instead a retrieved chunk
counts as relevant when it contains a literal window taken from a gold chunk.
Those windows are sampled from the middle of the gold chunk, away from the
edges, so the 200-character overlap between neighbouring chunks cannot hand out
credit for merely being adjacent to the right passage.

Questions with no gold chunks are excluded from retrieval scoring: the
abstention set has nothing to retrieve, and q014 was labelled as having no
usable support. Both are reported separately rather than silently averaged in.

Usage:
    .venv\\Scripts\\python.exe evals/run_eval.py
    .venv\\Scripts\\python.exe evals/run_eval.py --k 5 --retriever hybrid
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import EVAL_RESULTS_DIR, PROCESSED_DIR, TOP_K  # noqa: E402
from retrieve import RETRIEVERS  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVALS_DIR / "golden_set.yaml"
LABELS = EVALS_DIR / "labels.jsonl"

# Length of the literal window used to decide whether a retrieved chunk carries
# a gold passage. Long enough to be unique in a 9-million-character corpus,
# short enough to survive the two chunkers cutting at different places.
WINDOW = 160
# Sampled from the middle, never the edges: neighbouring structural chunks share
# 200 characters by design, and matching on those would score adjacency.
WINDOW_POSITIONS = (0.30, 0.50, 0.70)

# Labels written from this date on saw candidates pooled across all four
# configurations. Earlier ones saw only hybrid/structural.
POOL_FIX_DATE = "2026-08-23"


def normalize(text: str) -> str:
    return " ".join(text.split())


def load_questions() -> dict[str, dict]:
    with GOLDEN_SET.open(encoding="utf-8") as fh:
        return {q["id"]: q for q in yaml.safe_load(fh)["questions"]}


def load_labels() -> dict[str, dict]:
    with LABELS.open(encoding="utf-8") as fh:
        return {row["id"]: row for row in map(json.loads, fh)}


def load_chunk_texts() -> dict[str, str]:
    """Gold ids always refer to the structural corpus, whatever is being scored."""
    path = PROCESSED_DIR / "chunks_structural.jsonl"
    with path.open(encoding="utf-8") as fh:
        return {row["chunk_id"]: normalize(row["text"])
                for row in map(json.loads, fh)}


def gold_windows(text: str) -> list[str]:
    if len(text) <= WINDOW:
        return [text]
    windows = []
    for fraction in WINDOW_POSITIONS:
        start = min(int(len(text) * fraction), len(text) - WINDOW)
        windows.append(text[start:start + WINDOW])
    return windows


def build_cases(questions: dict, labels: dict, texts: dict) -> list[dict]:
    """Questions that can be scored on retrieval, with their gold windows."""
    cases = []
    for label in labels.values():
        if label["status"] != "reviewed" or not label.get("gold_chunk_ids"):
            continue
        question = questions[label["id"]]
        golds = [gold_windows(texts[chunk_id])
                 for chunk_id in label["gold_chunk_ids"] if chunk_id in texts]
        if golds:
            cases.append({
                "id": label["id"],
                "category": label["category"],
                "question": question["question"],
                "scope": question.get("scope"),
                "golds": golds,
                # Which pooling regime produced this label. The first block saw
                # only hybrid/structural candidates, so its structural-vs-fixed
                # numbers are biased; the second saw all four configurations.
                # Reporting them apart turns that mistake into a measurement.
                "pool": ("narrow" if label.get("reviewed_at", "") < POOL_FIX_DATE
                         else "full"),
            })
    return sorted(cases, key=lambda case: case["id"])


def score_case(case: dict, hits: list[dict]) -> dict:
    """Which gold passages the run found, and where the first one landed."""
    found = set()
    first_rank = None
    for rank, hit in enumerate(hits, start=1):
        body = normalize(hit["text"])
        for index, windows in enumerate(case["golds"]):
            if index in found:
                continue
            if any(window in body for window in windows):
                found.add(index)
                if first_rank is None:
                    first_rank = rank

    total = len(case["golds"])
    return {
        "hit": bool(found),
        "recall": len(found) / total,
        # Reciprocal rank of the first gold passage; 0 when none was retrieved.
        "mrr": 1 / first_rank if first_rank else 0.0,
        "first_rank": first_rank,
    }


def evaluate(retriever_name: str, strategy: str, cases: list[dict],
             k: int) -> dict:
    retriever = RETRIEVERS[retriever_name](strategy)
    started = time.time()
    per_case = {}
    for case in cases:
        hits = retriever.search(case["question"], k=k, scope=case["scope"])
        per_case[case["id"]] = score_case(case, hits)

    return {
        "retriever": retriever_name,
        "strategy": strategy,
        "k": k,
        "n_cases": len(cases),
        "seconds": round(time.time() - started, 1),
        "per_case": per_case,
    }


def aggregate(result: dict, cases: list[dict]) -> dict:
    by_category = collections.defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(result["per_case"][case["id"]])

    summary = {}
    for category, scores in by_category.items():
        summary[category] = {
            metric: sum(score[metric] for score in scores) / len(scores)
            for metric in ("hit", "recall", "mrr")
        }
    every = list(result["per_case"].values())
    summary["TOTAL"] = {
        metric: sum(score[metric] for score in every) / len(every)
        for metric in ("hit", "recall", "mrr")
    }
    return summary


def print_pool_table(results: list[dict], cases: list[dict],
                     metric: str = "recall") -> None:
    """The same metric split by how the labels were pooled.

    If a configuration's advantage shrinks under full pooling, that gap is the
    size of the bias the narrow pooling introduced.
    """
    blocks = ["narrow", "full"]
    by_block = {block: [case for case in cases if case["pool"] == block]
                for block in blocks}
    counts = {block: len(items) for block, items in by_block.items()}

    print(f"\n{metric}@k por bloque de etiquetado")
    print(f"  narrow = {counts['narrow']} preguntas etiquetadas viendo solo "
          f"hybrid/structural")
    print(f"  full   = {counts['full']} preguntas etiquetadas viendo las "
          f"4 configuraciones")
    print("-" * 62)
    print(f"{'configuracion':<26}{'narrow':>12}{'full':>12}{'dif':>12}")
    for result in results:
        row = []
        for block in blocks:
            items = by_block[block]
            if not items:
                row.append(None)
                continue
            row.append(sum(result["per_case"][case["id"]][metric]
                           for case in items) / len(items))
        name = f"{result['retriever']}/{result['strategy']}"
        cells = "".join(f"{value:>12.3f}" if value is not None else f"{'-':>12}"
                        for value in row)
        gap = (f"{row[1] - row[0]:>+12.3f}"
               if None not in row else f"{'-':>12}")
        print(f"{name:<26}{cells}{gap}")


def print_table(results: list[dict], cases: list[dict], metric: str) -> None:
    categories = sorted({case["category"] for case in cases}) + ["TOTAL"]
    label = {"hit": "acierto@k (encontro al menos uno)",
             "recall": "recall@k (fraccion de fragmentos correctos)",
             "mrr": "MRR@k (que tan arriba sale el primero)"}[metric]

    print(f"\n{label}")
    print("-" * (26 + 12 * len(categories)))
    header = "".join(f"{category[:11]:>12}" for category in categories)
    print(f"{'configuracion':<26}{header}")
    for result in results:
        summary = aggregate(result, cases)
        row = "".join(f"{summary.get(category, {}).get(metric, 0):>12.3f}"
                      for category in categories)
        name = f"{result['retriever']}/{result['strategy']}"
        print(f"{name:<26}{row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar la recuperacion.")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--retriever", choices=list(RETRIEVERS))
    parser.add_argument("--strategy", choices=["structural", "fixed"])
    args = parser.parse_args()

    questions = load_questions()
    labels = load_labels()
    cases = build_cases(questions, labels, load_chunk_texts())

    counts = collections.Counter(case["category"] for case in cases)
    print(f"\n{len(cases)} preguntas puntuables de {len(labels)} etiquetadas")
    print("  " + ", ".join(f"{category}={n}" for category, n in sorted(counts.items())))
    skipped = [row["id"] for row in labels.values()
               if row["status"] == "reviewed" and not row.get("gold_chunk_ids")]
    print(f"  fuera de la metrica (sin fragmentos): {len(skipped)} -> "
          f"{', '.join(sorted(skipped))}")

    retriever_names = [args.retriever] if args.retriever else list(RETRIEVERS)
    strategies = [args.strategy] if args.strategy else ["structural", "fixed"]

    results = []
    for strategy in strategies:
        for name in retriever_names:
            print(f"\ncorriendo {name}/{strategy} ...", end=" ", flush=True)
            result = evaluate(name, strategy, cases, args.k)
            print(f"{result['seconds']} s")
            results.append(result)

    for metric in ("hit", "recall", "mrr"):
        print_table(results, cases, metric)

    if len({case["pool"] for case in cases}) > 1:
        print_pool_table(results, cases, "recall")

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_RESULTS_DIR / f"retrieval_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"k": args.k, "n_cases": len(cases),
                   "results": results,
                   "summaries": {f"{r['retriever']}/{r['strategy']}":
                                 aggregate(r, cases) for r in results}},
                  fh, indent=2)
    print(f"\nguardado en {out_path}")


if __name__ == "__main__":
    main()
