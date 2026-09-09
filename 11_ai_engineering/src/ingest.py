"""Download 10-K annual reports from SEC EDGAR.

Two SEC rules drive the design here:

1. Requests need a descriptive User-Agent containing a contact email,
   otherwise EDGAR returns 403. It is read from .env so no personal data
   is committed to the repository.
2. Traffic is capped at 10 requests/second. Every call goes through
   `_get`, which throttles and reuses one session.

Run:  python src/ingest.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from config import FILING_YEARS, RAW_DIR, SEC_USER_AGENT, TARGET_TICKERS

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EXTRA_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# SEC allows 10 req/s. Staying at ~6 keeps a comfortable margin.
REQUEST_DELAY_SECONDS = 0.17

_session: requests.Session | None = None
_last_request_at = 0.0


@dataclass
class Filing:
    """One 10-K filing, identified by the fiscal year it reports on."""

    ticker: str
    cik: int
    fiscal_year: int
    filing_date: str
    report_date: str
    accession: str
    url: str
    local_path: str = ""


def _get(url: str) -> requests.Response:
    """Throttled GET with the SEC-mandated User-Agent."""
    global _session, _last_request_at

    if not SEC_USER_AGENT:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and fill in "
            "your name and email -- EDGAR rejects anonymous requests."
        )

    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": SEC_USER_AGENT})

    elapsed = time.monotonic() - _last_request_at
    if elapsed < REQUEST_DELAY_SECONDS:
        time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    response = _session.get(url, timeout=30)
    _last_request_at = time.monotonic()
    response.raise_for_status()
    return response


def resolve_ciks(tickers: list[str]) -> dict[str, int]:
    """Map tickers to CIK numbers using SEC's official lookup file.

    Hardcoding CIKs would be shorter but silently breaks on a typo; the
    lookup file is authoritative and costs one request.
    """
    payload = _get(TICKER_MAP_URL).json()
    by_ticker = {entry["ticker"]: entry["cik_str"] for entry in payload.values()}

    missing = [t for t in tickers if t not in by_ticker]
    if missing:
        raise ValueError(f"Tickers not found in SEC lookup: {missing}")

    return {ticker: by_ticker[ticker] for ticker in tickers}


def _iter_filing_records(cik: int):
    """Yield every filing record for a CIK.

    `filings.recent` only holds the latest ~1000 filings. Large issuers file
    hundreds of Form 4s a year, so four years of 10-Ks can fall outside that
    window -- the paginated `filings.files` shards are fetched too.
    """
    payload = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    shards = [payload["filings"]["recent"]]

    for extra in payload["filings"].get("files", []):
        shards.append(_get(EXTRA_SUBMISSIONS_URL.format(name=extra["name"])).json())

    for shard in shards:
        for i in range(len(shard["accessionNumber"])):
            yield {key: shard[key][i] for key in shard}


def find_10k_filings(ticker: str, cik: int, years: list[int]) -> list[Filing]:
    """Collect 10-K filings whose *fiscal* year falls in `years`.

    Fiscal year comes from `reportDate`, not `filingDate`: a 10-K covering
    fiscal 2024 is usually filed in early 2025, and NVIDIA's fiscal year ends
    in January, so filing dates would mislabel the corpus.
    """
    filings: list[Filing] = []

    for record in _iter_filing_records(cik):
        if record["form"] != "10-K" or not record["reportDate"]:
            continue

        fiscal_year = int(record["reportDate"][:4])
        if fiscal_year not in years:
            continue

        accession = record["accessionNumber"].replace("-", "")
        filings.append(
            Filing(
                ticker=ticker,
                cik=cik,
                fiscal_year=fiscal_year,
                filing_date=record["filingDate"],
                report_date=record["reportDate"],
                accession=record["accessionNumber"],
                url=ARCHIVE_URL.format(
                    cik=cik,
                    accession=accession,
                    document=record["primaryDocument"],
                ),
            )
        )

    # Amended or re-filed reports can produce duplicates; keep the latest.
    latest: dict[int, Filing] = {}
    for filing in sorted(filings, key=lambda f: f.filing_date):
        latest[filing.fiscal_year] = filing

    return [latest[year] for year in sorted(latest)]


def download(filing: Filing, destination: Path) -> Filing:
    """Fetch the filing document, skipping files already on disk."""
    path = destination / f"{filing.ticker}_FY{filing.fiscal_year}.html"

    if path.exists():
        print(f"  skip   {path.name} (already downloaded)")
    else:
        path.write_bytes(_get(filing.url).content)
        size_mb = path.stat().st_size / 1_000_000
        print(f"  saved  {path.name}  ({size_mb:.1f} MB)")

    filing.local_path = str(path.relative_to(destination.parent.parent))
    return filing


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Resolving CIKs for {len(TARGET_TICKERS)} tickers...")
    ciks = resolve_ciks(TARGET_TICKERS)

    manifest: list[dict] = []
    for ticker, cik in ciks.items():
        print(f"\n{ticker} (CIK {cik})")
        filings = find_10k_filings(ticker, cik, FILING_YEARS)

        found = [f.fiscal_year for f in filings]
        if missing := sorted(set(FILING_YEARS) - set(found)):
            print(f"  note   no 10-K found for fiscal {missing}")

        for filing in filings:
            manifest.append(asdict(download(filing, RAW_DIR)))

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} filings. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
