"""Collect data job postings from the Get on Board public API.

Why this source: it is a documented public API (no key, no scraping, no ToS
violation), it covers Spain and Latin America in Spanish, and — unlike every
other board tested — it exposes the job's `functions` (the actual tasks) as a
field separate from the marketing blurb. That separation is what makes it
possible to classify a posting by what the job does instead of by its title.

Sources rejected and why: LinkedIn (blocks scraping, and risking the account
that is currently being used to job-hunt is not worth a dataset), Tecnoempleo
(HTTP 403), Computrabajo (bot protection).

Strategy: a census, not a sample. Pull the entire active Data Science /
Analytics category plus targeted searches, then deduplicate by job id. Claiming
"every active posting in this category" is defensible; claiming "the market" is
not.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_ROOT = "https://www.getonbrd.com/api/v0"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Expanding these keeps the human-readable name instead of a bare id.
EXPAND = json.dumps(["seniority", "company", "modality"])

# The category is the population of interest; the searches catch postings filed
# under adjacent categories (engineering, product) that are still data roles.
CATEGORY = "data-science-analytics"
SEARCH_QUERIES = [
    "data scientist",
    "cientifico de datos",
    "machine learning",
    "data analyst",
    "analista de datos",
    "data engineer",
    "ingeniero de datos",
    "business intelligence",
    "analytics",
]

REQUEST_PAUSE_SECONDS = 1.0
MAX_PAGES = 10

# The default urllib user agent gets a 403. Identifying the client honestly is
# both what makes the request work and the right way to use someone's API.
HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "data-job-market-research/1.0 "
        "(portfolio project; contact via github.com/cristiancubero)"
    ),
}


def fetch_page(path, params):
    """Return one decoded API page, or None if the request fails."""
    url = f"{API_ROOT}/{path}?{urlencode(params)}"
    request = Request(url, headers=HEADERS)
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:  # network issues should not abort the whole run
        print(f"  ! failed {path} page {params.get('page')}: {error}")
        return None


def fetch_all_pages(path, params):
    """Follow pagination until the API reports the last page."""
    collected = []
    page = 1
    while page <= MAX_PAGES:
        payload = fetch_page(path, {**params, "page": page, "per_page": 100})
        if not payload or not payload.get("data"):
            break
        collected.extend(payload["data"])
        total_pages = payload.get("meta", {}).get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(REQUEST_PAUSE_SECONDS)
    return collected


def collect():
    """Pull the category census plus targeted searches, deduplicated by id."""
    by_id = {}

    print(f"Category: {CATEGORY}")
    for job in fetch_all_pages(f"categories/{CATEGORY}/jobs", {"expand": EXPAND}):
        by_id[job["id"]] = job
    print(f"  {len(by_id)} postings")

    for query in SEARCH_QUERIES:
        before = len(by_id)
        for job in fetch_all_pages("search/jobs", {"query": query, "expand": EXPAND}):
            by_id[job["id"]] = job
        print(f"Search '{query}': +{len(by_id) - before} new (total {len(by_id)})")
        time.sleep(REQUEST_PAUSE_SECONDS)

    return list(by_id.values())


def main():
    jobs = collect()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(timezone.utc)
    output_path = RAW_DIR / f"getonbrd_{captured_at:%Y%m%d}.json"

    # The capture date is stored with the data: an active-postings census is a
    # snapshot, and a snapshot without its date is not reproducible.
    payload = {
        "source": "getonbrd_api_v0",
        "captured_at": captured_at.isoformat(),
        "category": CATEGORY,
        "search_queries": SEARCH_QUERIES,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n{len(jobs)} unique postings -> {output_path}")


if __name__ == "__main__":
    main()
