"""Generate an answer for every golden-set question and score what needs no judge.

Three things can be measured by counting, with no model grading another model:

* **Abstention.** Whether the system refuses on the seven questions whose only
  correct response is a refusal -- and, just as important, whether it refuses on
  questions it should have answered. A system that always abstains scores
  perfectly on the first half and is useless, so both directions are reported.
* **Citation compliance.** Whether each answer cites at least one excerpt, and
  whether every citation points at an excerpt that was actually supplied. A
  citation to excerpt [9] when eight were given is a fabricated reference.
* **Citation precision against the gold labels.** Of the excerpts an answer
  cites, how many are ones a human marked as genuinely supporting the answer.

Faithfulness -- whether the sentences follow from the cited text -- is the one
thing left to a judge, and that comes later, validated against human labels.

Answers are written out so the judging stage does not have to regenerate them:
a full pass takes around twenty minutes on CPU.

Usage:
    .venv\\Scripts\\python.exe evals/run_answers.py
    .venv\\Scripts\\python.exe evals/run_answers.py --limit 5 --retriever bm25
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

from config import EVAL_RESULTS_DIR, TOP_K  # noqa: E402
from generate import ABSTAIN, PROMPTS, answer  # noqa: E402
from run_eval import gold_windows, load_chunk_texts, normalize  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVALS_DIR / "golden_set.yaml"
LABELS = EVALS_DIR / "labels.jsonl"


def load_cases() -> list[dict]:
    with GOLDEN_SET.open(encoding="utf-8") as fh:
        questions = {q["id"]: q for q in yaml.safe_load(fh)["questions"]}
    with LABELS.open(encoding="utf-8") as fh:
        labels = {row["id"]: row for row in map(json.loads, fh)}

    texts = load_chunk_texts()
    cases = []
    for label in labels.values():
        if label["status"] != "reviewed":
            continue
        question = questions[label["id"]]
        cases.append({
            # Precomputed once: the literal windows that identify each gold
            # passage regardless of which chunker produced the retrieved text.
            "gold_windows": [gold_windows(texts[chunk_id])
                             for chunk_id in label.get("gold_chunk_ids", [])
                             if chunk_id in texts],
            "id": label["id"],
            "category": label["category"],
            "question": question["question"],
            "scope": question.get("scope"),
            "must_abstain": bool(label.get("must_abstain")),
            "expected_answer": label["expected_answer"],
            "gold_chunk_ids": label.get("gold_chunk_ids", []),
        })
    return sorted(cases, key=lambda case: case["id"])


def score_answer(case: dict, result: dict) -> dict:
    """Everything about this answer that can be decided by counting."""
    supplied = len(result["hits"])
    cited = result["cited"]
    # A citation outside the range of supplied excerpts is invented outright.
    invalid = {number for number in cited if not 1 <= number <= supplied}

    # Matched by content, not by id. Gold labels name structural chunks, so an
    # id comparison would score every fixed-strategy citation as wrong even when
    # it points at the same text -- the same mistake the retrieval harness
    # already avoids.
    windows = case["gold_windows"]
    cited_texts = [normalize(result["hits"][number - 1]["text"])
                   for number in sorted(cited - invalid)]
    on_gold = sum(1 for body in cited_texts
                  if any(any(window in body for window in passage)
                         for passage in windows))
    comparable = bool(windows)

    return {
        "abstained": result["abstained"],
        "should_abstain": case["must_abstain"],
        "correct_abstention": result["abstained"] and case["must_abstain"],
        "false_abstention": result["abstained"] and not case["must_abstain"],
        "missed_abstention": not result["abstained"] and case["must_abstain"],
        "has_citation": bool(cited) and not result["abstained"],
        "invalid_citations": sorted(invalid),
        "n_cited": len(cited - invalid),
        "cited_gold": on_gold if comparable else None,
        "citation_precision": (on_gold / len(cited_texts)
                               if comparable and cited_texts else None),
    }


def report(scored: list[dict]) -> None:
    answerable = [row for row in scored if not row["should_abstain"]]
    unanswerable = [row for row in scored if row["should_abstain"]]

    print("\n" + "=" * 70)
    print("ABSTENCION")
    print("-" * 70)
    correct = sum(row["correct_abstention"] for row in unanswerable)
    print(f"  se abstuvo cuando debia      {correct:>3}/{len(unanswerable)}"
          f"   ({100 * correct / max(1, len(unanswerable)):.0f}%)")
    false = sum(row["false_abstention"] for row in answerable)
    print(f"  se abstuvo sin deber         {false:>3}/{len(answerable)}"
          f"   ({100 * false / max(1, len(answerable)):.0f}%)")
    print("  (el segundo importa igual: abstenerse siempre daria 100% arriba)")

    print("\nCITAS")
    print("-" * 70)
    answered = [row for row in scored if not row["abstained"]]
    with_citation = sum(row["has_citation"] for row in answered)
    print(f"  respuestas con al menos una cita  {with_citation:>3}/{len(answered)}"
          f"   ({100 * with_citation / max(1, len(answered)):.0f}%)")
    invented = [row for row in scored if row["invalid_citations"]]
    print(f"  respuestas con citas inexistentes {len(invented):>3}/{len(scored)}")
    for row in invented:
        print(f"      {row['id']}: cito {row['invalid_citations']}")

    comparable = [row for row in answered
                  if row["citation_precision"] is not None]
    if comparable:
        precision = sum(row["citation_precision"] for row in comparable) / len(comparable)
        print(f"  precision de citas vs etiquetas   {precision:.3f}"
              f"   ({len(comparable)} preguntas comparables)")

    print("\nPOR CATEGORIA")
    print("-" * 70)
    print(f"{'categoria':<24}{'n':>4}{'abstuvo':>10}{'con cita':>10}{'prec.':>8}")
    by_category = collections.defaultdict(list)
    for row in scored:
        by_category[row["category"]].append(row)
    for category, rows in sorted(by_category.items()):
        answered_rows = [row for row in rows if not row["abstained"]]
        precisions = [row["citation_precision"] for row in rows
                      if row["citation_precision"] is not None]
        cited = sum(row["has_citation"] for row in answered_rows)
        print(f"{category:<24}{len(rows):>4}"
              f"{sum(row['abstained'] for row in rows):>10}"
              f"{cited:>10}"
              f"{(sum(precisions) / len(precisions)) if precisions else 0:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar y puntuar respuestas.")
    parser.add_argument("--retriever", default="hybrid")
    parser.add_argument("--strategy", default="structural")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--prompt", choices=list(PROMPTS), default="strict")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]

    print(f"{len(cases)} preguntas | {args.retriever}/{args.strategy} "
          f"via {args.provider} | prompt={args.prompt}")

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    answers_path = EVAL_RESULTS_DIR / f"answers_{args.prompt}_{stamp}.jsonl"

    scored = []
    started = time.time()
    with answers_path.open("w", encoding="utf-8") as fh:
        for index, case in enumerate(cases, start=1):
            result = answer(case["question"], args.retriever, args.strategy,
                            case["scope"], args.k, args.provider,
                            prompt=args.prompt)
            row = score_answer(case, result)
            row.update(id=case["id"], category=case["category"])
            scored.append(row)

            fh.write(json.dumps({
                **{key: value for key, value in case.items()
                   if key != "gold_windows"},
                "answer": result["answer"],
                "cited": sorted(result["cited"]),
                "retrieved_chunk_ids": [hit["chunk_id"] for hit in result["hits"]],
                "score": row,
            }, ensure_ascii=False) + "\n")
            fh.flush()

            mark = "ABST" if result["abstained"] else f"cita {sorted(result['cited'])}"
            elapsed = time.time() - started
            print(f"  {index:>2}/{len(cases)} {case['id']}  {mark:<24}"
                  f"  {elapsed / index:.0f} s/preg", flush=True)

    report(scored)
    print(f"\nrespuestas en {answers_path}")


if __name__ == "__main__":
    main()
