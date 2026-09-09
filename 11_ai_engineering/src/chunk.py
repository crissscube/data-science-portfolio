"""Split the 10-K corpus into retrievable chunks, two ways.

The point of building two strategies instead of one is the experiment. The
`structural` strategy honours the Item boundaries parse.py recovered and cuts
on sentence and bullet edges inside them; the `fixed` strategy ignores every
signal and slices the raw document at a fixed stride, which is what a default
pipeline does. Retrieval quality is then measured on both, so "chunk by
structure" becomes a number rather than an opinion.

The fixed strategy also covers the Intel filings, which carry no Item headings
and therefore have no structural variant at all.

Usage:
    .venv\\Scripts\\python.exe src/chunk.py             # both strategies
    .venv\\Scripts\\python.exe src/chunk.py --strategy structural
"""

from __future__ import annotations

import argparse
import json
import re
import statistics

from config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    PROCESSED_DIR,
    RAW_DIR,
)
from parse import ITEM_TITLES, extract_text

# Sentence boundary: terminal punctuation, whitespace, then something that can
# open a sentence. Bullets are treated as boundaries in their own right because
# the risk-factor summaries are bulleted lists with no terminal punctuation.
SENTENCE_SPLIT = re.compile(
    "(?<=[.!?])\\s+(?=[A-Z“‘\"(•])|\\s*(?=•)"
)

# Abbreviations whose trailing period is not a sentence end. Splitting after
# "Inc." or "U.S." would cut a company name in half on almost every page.
ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "llc", "plc", "no", "nos", "u.s", "u.k",
    "approx", "est", "fig", "vs", "etc", "al", "jr", "sr", "mr", "ms", "dr",
    "st", "ave", "dept", "div", "ref", "sec", "fasb", "e.g", "i.e",
}
ABBREV_TAIL = re.compile(r"([A-Za-z.]+)\.$")


def split_sentences(text: str) -> list[str]:
    """Break text into sentences, keeping abbreviations intact."""
    pieces = SENTENCE_SPLIT.split(text)
    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        # Re-attach to the previous sentence when the break followed an
        # abbreviation rather than a real full stop.
        if sentences:
            match = ABBREV_TAIL.search(sentences[-1])
            if match and match.group(1).lower().strip(".") in ABBREVIATIONS:
                sentences[-1] = f"{sentences[-1]} {piece}"
                continue
        sentences.append(piece)
    return sentences


def _hard_split(text: str, size: int) -> list[str]:
    """Last-resort slicer for a single run longer than the cap.

    Financial tables survive text extraction as one unbroken run of thousands
    of characters with no sentence punctuation at all, so there is nothing to
    cut on. Slicing blindly is still better than emitting a chunk the encoder
    would silently truncate.
    """
    return [text[i:i + size] for i in range(0, len(text), size)]


def pack(sentences: list[str]) -> list[str]:
    """Greedily fill chunks up to the target size, with a sentence overlap.

    The overlap repeats whole trailing sentences rather than a raw character
    slice: an idea that straddles a cut then survives complete in the second
    chunk, which is the whole reason for paying the duplication cost.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for sentence in sentences:
        parts = (_hard_split(sentence, CHUNK_MAX_CHARS)
                 if len(sentence) > CHUNK_MAX_CHARS else [sentence])
        for part in parts:
            if current and size + len(part) + 1 > CHUNK_TARGET_CHARS:
                chunks.append(" ".join(current))
                # Carry back the tail sentences that fit in the overlap budget.
                carry: list[str] = []
                carried = 0
                for previous in reversed(current):
                    if carried + len(previous) > CHUNK_OVERLAP_CHARS:
                        break
                    carry.insert(0, previous)
                    carried += len(previous) + 1
                # An oversized part (a table, hard-split at the cap) gets no
                # overlap: prepending the carry would push it past the cap and
                # the encoder would truncate exactly what the split protected.
                if carried + len(part) > CHUNK_MAX_CHARS:
                    carry, carried = [], 0
                current = carry
                size = carried
            current.append(part)
            size += len(part) + 1

    if current:
        tail = " ".join(current)
        # A short tail is a fragment, not an answer: fold it into the previous
        # chunk unless doing so would blow past the encoder's hard limit.
        if (chunks and len(tail) < CHUNK_MIN_CHARS
                and len(chunks[-1]) + len(tail) + 1 <= CHUNK_MAX_CHARS):
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)

    return chunks


def context_header(ticker: str, year: int, item: str | None) -> str:
    """One line of provenance prepended at embedding time.

    Chunks pulled from the middle of a section never name the company or the
    year, so an isolated chunk about "our data center revenue" is unattributable
    on its own. The header restores that, both for the encoder and for the
    citation the answer has to carry.
    """
    if item:
        where = f"Item {item}. {ITEM_TITLES.get(item, '')}".strip(". ")
    else:
        where = "full filing"
    return f"{ticker} fiscal year {year} 10-K, {where}"


def _record(strategy: str, ticker: str, year: int, item: str | None,
            is_core: bool, index: int, text: str) -> dict:
    label = item or "DOC"
    return {
        "chunk_id": f"{strategy[:4]}_{ticker}_FY{year}_{label}_{index:04d}",
        "strategy": strategy,
        "ticker": ticker,
        "fiscal_year": year,
        "item": item,
        "is_core": is_core,
        "chunk_index": index,
        "n_chars": len(text),
        "context_header": context_header(ticker, year, item),
        "text": text,
    }


def build_structural() -> list[dict]:
    """Chunk within Item boundaries, never across them."""
    records = []
    sections_path = PROCESSED_DIR / "sections.jsonl"

    with sections_path.open(encoding="utf-8") as fh:
        for line in fh:
            section = json.loads(line)
            # The size_fallback documents have no structure to respect; they
            # belong to the fixed strategy only.
            if section["parsed_by"] != "item_headings":
                continue

            for index, text in enumerate(pack(split_sentences(section["text"]))):
                records.append(_record(
                    strategy="structural",
                    ticker=section["ticker"],
                    year=section["fiscal_year"],
                    item=section["item"],
                    is_core=section["is_core"],
                    index=index,
                    text=text,
                ))
    return records


def build_fixed() -> list[dict]:
    """Slice the whole filing at a fixed stride, ignoring every heading.

    This is the baseline the structural strategy has to beat. It reads the raw
    HTML rather than sections.jsonl so that nothing parse.py learned leaks in.
    """
    records = []
    stride = CHUNK_TARGET_CHARS - CHUNK_OVERLAP_CHARS

    for path in sorted(RAW_DIR.glob("*.html")):
        ticker, fiscal = path.stem.split("_FY")
        text = extract_text(path.read_text(encoding="utf-8", errors="ignore"))

        for index, start in enumerate(range(0, len(text), stride)):
            piece = text[start:start + CHUNK_TARGET_CHARS]
            if index and len(piece) < CHUNK_MIN_CHARS:
                break
            records.append(_record(
                strategy="fixed",
                ticker=ticker,
                year=int(fiscal),
                item=None,
                is_core=True,
                index=index,
                text=piece,
            ))
    return records


# A chunk that opens mid-word or closes mid-sentence hands the model a mutilated
# passage. Counting them is the cheapest evidence that respecting structure is
# worth its complexity -- it needs no LLM, no labels and no judgement call.
BROKEN_START = re.compile(r"^[a-z]")
CLEAN_END = (".", "!", "?", '"', "’", "”", ")", ";")


def boundary_quality(records: list[dict]) -> tuple[float, float, float]:
    """Share of chunks with a broken start, a broken end, and with neither."""
    starts = ends = clean = 0
    for record in records:
        text = record["text"].strip()
        bad_start = bool(BROKEN_START.match(text))
        bad_end = not text.endswith(CLEAN_END)
        starts += bad_start
        ends += bad_end
        clean += not (bad_start or bad_end)
    total = len(records)
    return 100 * starts / total, 100 * ends / total, 100 * clean / total


def report(name: str, records: list[dict]) -> None:
    sizes = [r["n_chars"] for r in records]
    chars = sum(sizes)
    print(f"\n{name}")
    print(f"  fragmentos      {len(records):>8,}")
    print(f"  caracteres      {chars:>8,}  (~{chars // 4:,} tokens)")
    print(f"  tamano medio    {statistics.mean(sizes):>8,.0f}")
    print(f"  mediana         {statistics.median(sizes):>8,.0f}")
    print(f"  maximo          {max(sizes):>8,}")
    over = sum(1 for size in sizes if size > CHUNK_MAX_CHARS)
    print(f"  sobre el limite {over:>8,}  (se truncarian al vectorizar)")

    bad_start, bad_end, clean = boundary_quality(records)
    print(f"  inicio roto     {bad_start:>7.1f}%")
    print(f"  final roto      {bad_end:>7.1f}%")
    print(f"  limites limpios {clean:>7.1f}%")

    if records[0]["strategy"] == "structural":
        by_item: dict[str, int] = {}
        for record in records:
            by_item[record["item"]] = by_item.get(record["item"], 0) + 1
        top = sorted(by_item.items(), key=lambda kv: -kv[1])[:6]
        print("  Items con mas fragmentos: "
              + ", ".join(f"{item}={n}" for item, n in top))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk the 10-K corpus.")
    parser.add_argument("--strategy", choices=["structural", "fixed", "both"],
                        default="both")
    args = parser.parse_args()

    builders = {"structural": build_structural, "fixed": build_fixed}
    if args.strategy != "both":
        builders = {args.strategy: builders[args.strategy]}

    for name, build in builders.items():
        records = build()
        out_path = PROCESSED_DIR / f"chunks_{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        report(f"{name}  ->  {out_path.name}", records)


if __name__ == "__main__":
    main()
