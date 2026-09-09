"""Read the corpus with your own eyes.

Every stage of this pipeline writes JSONL, which is convenient for machines and
unreadable for people. This prints any slice of it in a legible form so that
claims about chunk quality can be checked rather than trusted.

Examples:
    .venv\\Scripts\\python.exe src/peek.py --ticker NVDA --year 2024 --item 1A
    .venv\\Scripts\\python.exe src/peek.py --grep "export restrictions" -n 3
    .venv\\Scripts\\python.exe src/peek.py --strategy fixed --full -n 1
    .venv\\Scripts\\python.exe src/peek.py --sections --ticker INTC
    .venv\\Scripts\\python.exe src/peek.py --stats
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap

from config import PROCESSED_DIR

# How much of a chunk to show by default. Enough to judge where the cuts landed
# without flooding the terminal.
PREVIEW_CHARS = 320
WIDTH = 88


def load(path_name: str) -> list[dict]:
    path = PROCESSED_DIR / path_name
    if not path.exists():
        raise SystemExit(f"No existe {path}. Corre antes src/chunk.py.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def matches(record: dict, args: argparse.Namespace) -> bool:
    if args.ticker and record["ticker"] != args.ticker.upper():
        return False
    if args.year and record["fiscal_year"] != args.year:
        return False
    if args.item and (record.get("item") or "").upper() != args.item.upper():
        return False
    if args.grep and args.grep.lower() not in record["text"].lower():
        return False
    return True


def wrap(text: str) -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent="  ",
                         subsequent_indent="  ")


def show(record: dict, full: bool) -> None:
    label = record.get("chunk_id") or (
        f"{record['ticker']}_FY{record['fiscal_year']}_Item{record['item']}")
    print("=" * WIDTH)
    print(f"{label}   {record['n_chars']:,} caracteres")
    if record.get("context_header"):
        print(f"  [{record['context_header']}]")
    print("-" * WIDTH)

    text = record["text"]
    if full or len(text) <= PREVIEW_CHARS * 2:
        print(wrap(text))
    else:
        # Both ends are what matter: a bad chunker shows up at the seams.
        print(wrap(text[:PREVIEW_CHARS]))
        print(f"\n  [... {len(text) - PREVIEW_CHARS * 2:,} caracteres omitidos ...]\n")
        print(wrap(text[-PREVIEW_CHARS:]))
    print()


def show_stats(records: list[dict]) -> None:
    """Counts per document, so gaps in coverage are visible at a glance."""
    grid: dict[tuple[str, int], int] = {}
    for record in records:
        key = (record["ticker"], record["fiscal_year"])
        grid[key] = grid.get(key, 0) + 1

    years = sorted({year for _, year in grid})
    tickers = sorted({ticker for ticker, _ in grid})

    print("fragmentos por documento\n")
    print(f"{'':<8}" + "".join(f"{year:>8}" for year in years))
    for ticker in tickers:
        row = "".join(f"{grid.get((ticker, year), 0):>8,}" for year in years)
        print(f"{ticker:<8}{row}")
    print(f"\ntotal: {len(records):,} fragmentos")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeccionar el corpus troceado.")
    parser.add_argument("--strategy", choices=["structural", "fixed"],
                        default="structural")
    parser.add_argument("--sections", action="store_true",
                        help="leer sections.jsonl en vez de los fragmentos")
    parser.add_argument("--ticker", help="NVDA, AMD, INTC, QCOM, MU, AVGO")
    parser.add_argument("--year", type=int, help="2021-2024")
    parser.add_argument("--item", help="1, 1A, 7, 7A, ...")
    parser.add_argument("--grep", help="mostrar solo fragmentos que contengan este texto")
    parser.add_argument("-n", type=int, default=5, help="cuantos mostrar")
    parser.add_argument("--skip", type=int, default=0, help="saltar los primeros N")
    parser.add_argument("--random", action="store_true",
                        help="muestra al azar en vez de los primeros")
    parser.add_argument("--full", action="store_true", help="texto completo")
    parser.add_argument("--stats", action="store_true",
                        help="solo el conteo por documento")
    args = parser.parse_args()

    source = "sections.jsonl" if args.sections else f"chunks_{args.strategy}.jsonl"
    records = load(source)
    selected = [record for record in records if matches(record, args)]

    print(f"\nfuente: {source}   {len(selected):,} de {len(records):,} coinciden\n")
    if not selected:
        return

    if args.stats:
        show_stats(selected)
        return

    if args.random:
        chosen = random.sample(selected, min(args.n, len(selected)))
    else:
        chosen = selected[args.skip:args.skip + args.n]

    for record in chosen:
        show(record, full=args.full)


if __name__ == "__main__":
    main()
