"""Ask whether the differences in the results tables survive a significance test.

A results table invites reading every gap as a finding. With 35 scorable
questions most gaps are not distinguishable from chance, and saying which are
is the difference between a measurement and a ranking of noise.

McNemar's exact test is the right tool here: the configurations answer the same
questions, so the comparison is paired, and only the questions where they
disagree carry information. Two systems that differ on three questions out of
thirty-five are not separated by that evidence, however different their averages
look.

Usage:
    .venv\\Scripts\\python.exe evals/significance.py
"""

from __future__ import annotations

import glob
import json
from math import comb
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
ALPHA = 0.05


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact p-value for b wins against c wins.

    Under the null the discordant pairs are coin flips, so the p-value is the
    binomial tail at the smaller count, doubled.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def latest(pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    return Path(matches[-1]) if matches else None


def compare_retrieval() -> None:
    path = latest("retrieval_*.json")
    if not path:
        print("sin resultados de recuperacion")
        return

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    runs = {f"{r['retriever']}/{r['strategy']}": r["per_case"]
            for r in data["results"]}

    pairs = [
        ("hybrid/structural", "hybrid/fixed"),
        ("hybrid/structural", "dense/structural"),
        ("hybrid/structural", "bm25/structural"),
        ("bm25/structural", "dense/structural"),
        ("dense/structural", "dense/fixed"),
    ]

    print(f"\nRECUPERACION  (acierto@8, {len(next(iter(runs.values())))} preguntas)")
    print(f"{'comparacion':<44}{'gana A':>8}{'gana B':>8}{'p':>9}")
    print("-" * 69)
    for left, right in pairs:
        a, b_run = runs[left], runs[right]
        wins = sum(1 for key in a if a[key]["hit"] and not b_run[key]["hit"])
        losses = sum(1 for key in a if b_run[key]["hit"] and not a[key]["hit"])
        p = mcnemar(wins, losses)
        flag = "  *" if p < ALPHA else ""
        print(f"{left + ' vs ' + right:<44}{wins:>8}{losses:>8}{p:>9.4f}{flag}")


def compare_prompts() -> None:
    strict = latest("answers_2*.jsonl")
    balanced = latest("answers_balanced_*.jsonl")
    if not (strict and balanced):
        print("\nsin dos corridas de prompts que comparar")
        return

    def load(path: Path) -> dict:
        with path.open(encoding="utf-8") as fh:
            return {row["id"]: row for row in map(json.loads, fh)}

    left, right = load(strict), load(balanced)
    shared = sorted(set(left) & set(right))

    print(f"\nPROMPTS  estricto vs equilibrado ({len(shared)} preguntas)")
    print(f"{'medida':<44}{'gana A':>8}{'gana B':>8}{'p':>9}")
    print("-" * 69)

    checks = {
        # Named so that "wins" always means the first run did better.
        "no se abstuvo sin deber": lambda row: not row["score"]["false_abstention"],
        "cito al menos un fragmento": lambda row: row["score"]["has_citation"],
        "se abstuvo cuando debia": lambda row: row["score"]["correct_abstention"],
    }
    for label, good in checks.items():
        wins = sum(1 for key in shared if good(left[key]) and not good(right[key]))
        losses = sum(1 for key in shared if good(right[key]) and not good(left[key]))
        p = mcnemar(wins, losses)
        flag = "  *" if p < ALPHA else ""
        print(f"{label:<44}{wins:>8}{losses:>8}{p:>9.4f}{flag}")


def main() -> None:
    compare_retrieval()
    compare_prompts()
    print(f"\n  * = p < {ALPHA}. Sin asterisco, la diferencia observada es")
    print("  compatible con el azar a este tamano de muestra.")


if __name__ == "__main__":
    main()
