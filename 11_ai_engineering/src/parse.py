"""Split each 10-K into its numbered Items.

Why split on the outline instead of every N characters: a 10-K follows an
SEC-mandated structure (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A...).
Cutting blindly every N characters straddles those boundaries and mixes
unrelated topics inside one chunk. Splitting on the outline first keeps each
chunk on a single topic and attaches metadata worth filtering on later.

The hard part: every Item number appears several times per document -- in the
table of contents, as the real heading, and inside cross-references ("see Item
1A"). Three filters separate the real headings:

  1. The table of contents is a dense cluster of Item references packed into a
     short span of text. It is located and discarded.
  2. Real sections appear in canonical order, so boundaries are picked greedily
     in increasing Item order; a backwards reference cannot open a section.
  3. Anything shorter than MIN_SECTION_CHARS is treated as a leftover
     cross-reference rather than a real section.

Run:  .venv\Scripts\python.exe src\parse.py
"""

from __future__ import annotations

import json
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from config import PROCESSED_DIR, RAW_DIR

# The filings are inline XBRL (XML), but the HTML parser is more forgiving with
# the malformed markup some filers produce. The warning is expected, not a bug.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# The letter suffix is captured separately: some filers typeset it detached
# ("Item 1 C . Cybersecurity"), which a single \d+[A-C] group would miss.
ITEM_PATTERN = re.compile(
    r"\bItem\s+(\d{1,2})\s*([A-C])?\s*[.:\-\u2013\u2014]", re.IGNORECASE
)

# A heading introduced by one of these is a cross-reference pointing at another
# section, not the section itself: 'Refer to "Item 1A. Risk Factors" for ...'.
CROSS_REF_CUE = re.compile(
    r"\b(refer(?:ence)?s? to|see|described in|discussed in|conjunction with|"
    r"set forth in|included in|contained in|listed in|as defined in)\b"
    r"[^.;]{0,40}$",
    re.IGNORECASE,
)
OPENING_QUOTES = "\"\u201c\u2018'"

# Canonical order of the 10-K outline, used to reject backwards references.
ITEM_ORDER = [
    "1", "1A", "1B", "1C", "2", "3", "4",
    "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16",
]
ITEM_RANK = {item: i for i, item in enumerate(ITEM_ORDER)}

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# The sections worth answering questions about. The rest is boilerplate or
# points at documents filed separately, so it stays out of the corpus.
CORE_ITEMS = ["1", "1A", "1C", "3", "7", "7A"]

# In a table of contents the Item references sit a few dozen characters apart;
# in the body they are thousands apart. That gap is what separates them.
TOC_MAX_GAP = 400
TOC_MIN_ITEMS = 10

# Below this, a match is a leftover cross-reference, not a section heading.
MIN_SECTION_CHARS = 400


def extract_text(html: str) -> str:
    """Strip markup and collapse the whitespace inline XBRL leaves behind."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _is_cross_reference(text: str, start: int) -> bool:
    """True when this Item mention points elsewhere instead of opening a section."""
    before = text[max(0, start - 90):start]
    if CROSS_REF_CUE.search(before):
        return True
    # An unbalanced opening quote right before the heading is the other giveaway.
    return before.rstrip().endswith(tuple(OPENING_QUOTES))


def _find_toc_run(matches: list[tuple[str, int]]) -> tuple[int, int] | None:
    """Return the index range of the table-of-contents run, if present.

    A table of contents is a tightly packed, strictly ascending list of the
    outline. Both properties are required to stay in the run: the gap test
    alone would keep swallowing the real Item 1, which some filers place only a
    few hundred characters after the index ends.
    """
    i = 0
    while i < len(matches):
        j = i
        while (j + 1 < len(matches)
               and matches[j + 1][1] - matches[j][1] <= TOC_MAX_GAP
               and ITEM_RANK[matches[j + 1][0]] > ITEM_RANK[matches[j][0]]):
            j += 1
        if len({item for item, _ in matches[i:j + 1]}) >= TOC_MIN_ITEMS:
            return i, j
        i = j + 1
    return None


def _select_headings(
    matches: list[tuple[str, int]], text_length: int
) -> list[tuple[str, int]]:
    """Choose one real heading per Item.

    Rather than walking forward and taking the first candidate for each Item --
    which loses the whole document to a single stray cross-reference -- this
    searches every ascending chain of candidates and keeps the one yielding the
    most substantial sections. A cross-reference opens a section a few hundred
    characters long, so chains built on one score worse than the real chain and
    lose. New citation phrasings therefore cost accuracy, not a broken parse.
    """
    matches = [m for m in matches if m[0] in ITEM_RANK]

    toc = _find_toc_run(matches)
    if toc:
        lo, hi = toc
        matches = matches[:lo] + matches[hi + 1:]
    if not matches:
        return []

    def is_substantial(start: int, stop: int) -> int:
        return 1 if stop - start >= MIN_SECTION_CHARS else 0

    # Score is (substantial sections, headings). The second term breaks ties:
    # Item 6 is usually just "[Reserved]", so a chain that skips Item 7 scores
    # the same on section count alone -- and the most valuable section in the
    # filing would silently disappear. More headings wins.
    n = len(matches)
    best = [(0, 1)] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if ITEM_RANK[matches[j][0]] >= ITEM_RANK[matches[i][0]]:
                continue
            gain = is_substantial(matches[j][1], matches[i][1])
            score = (best[j][0] + gain, best[j][1] + 1)
            if score > best[i]:
                best[i], prev[i] = score, j

    # The final section of a chain runs to the end of the document.
    tail = max(
        range(n),
        key=lambda i: (
            best[i][0] + is_substantial(matches[i][1], text_length),
            best[i][1],
        ),
    )

    chain = []
    while tail != -1:
        chain.append(matches[tail])
        tail = prev[tail]
    return list(reversed(chain))


def split_items(text: str) -> dict[str, str]:
    """Map each Item to its section text."""
    matches = [
        (m.group(1) + (m.group(2) or "").upper(), m.start())
        for m in ITEM_PATTERN.finditer(text)
        if not _is_cross_reference(text, m.start())
    ]
    headings = _select_headings(matches, len(text))

    sections: dict[str, str] = {}
    for i, (item, start) in enumerate(headings):
        end = headings[i + 1][1] if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        if len(body) >= MIN_SECTION_CHARS:
            sections[item] = body

    return sections


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "sections.jsonl"

    records = []
    print(f"{'documento':<18} {'secciones':>9}  {'secciones clave':<36} faltantes")
    print("-" * 88)

    for path in sorted(RAW_DIR.glob("*.html")):
        ticker, fiscal = path.stem.split("_FY")
        text = extract_text(path.read_text(encoding="utf-8", errors="ignore"))
        sections = split_items(text)

        found = [i for i in CORE_ITEMS if i in sections]
        # Item 1C only exists from fiscal 2023 on, so its absence is expected.
        expected = [i for i in CORE_ITEMS if not (i == "1C" and int(fiscal) < 2023)]
        missing = [i for i in expected if i not in sections]

        # Fallback: a few filers (Intel) drop the Item headings altogether and
        # map the outline to page numbers in an appendix instead. There is
        # nothing to split on, so the document enters the corpus whole and gets
        # chunked by size downstream. It is tagged so the evaluation can compare
        # answers drawn from parsed sections against answers drawn from these.
        fallback = len(found) < 2
        if fallback:
            sections = {"FULL": text}

        for item, body in sections.items():
            records.append({
                "ticker": ticker,
                "fiscal_year": int(fiscal),
                "item": item,
                "title": ITEM_TITLES.get(item, ""),
                "is_core": item in CORE_ITEMS or item == "FULL",
            "parsed_by": "size_fallback" if fallback else "item_headings",
                "n_chars": len(body),
                "text": body,
            })

        if fallback:
            print(f"{path.stem:<18} {'PLAN B':>9}  "
                  f"{'sin titulos Item -- documento entero':<36}")
        else:
            print(f"{path.stem:<18} {len(sections):>9}  {','.join(found):<36} "
                  f"{','.join(missing) or '-'}")

    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    core = [r for r in records if r["is_core"]]
    fb = [r for r in records if r["parsed_by"] == "size_fallback"]
    print("-" * 88)
    print(f"{len(records)} secciones en total, {len(core)} de las principales.")
    print(f"{len(fb)} documentos entraron por plan B (sin titulos Item).")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
