"""Collect data postings from three non-LATAM sources, for cross-region comparison.

Why this file exists: the Get on Board capture is Chile-heavy, so on its own it
cannot support any claim beyond Chile. The question a viewer anywhere actually
has is "does this apply to me?" — and that is an empirical question, not a
rhetorical one. Answering it needs postings from more than one region, collected
the same way and measured with the same vocabulary.

Sources, all public APIs with no key required:
  - Jobicy    — remote-first, worldwide, English; carries geo, level and salary.
  - The Muse  — mostly United States, some Europe; full job description.
  - Arbeitnow — Europe, mostly Germany; English and German postings.

All three are normalised here into the same record shape that `extract.py`
consumes, so a posting from Santiago and one from Berlin are measured by
identical rules. Anything source-specific stops at this file.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "data-job-market-research/1.0 "
        "(portfolio project; contact via github.com/cristiancubero)"
    ),
}
REQUEST_PAUSE_SECONDS = 1.0

# Jobicy segments its board by geography, so each region is asked for directly
# rather than hoping a single query happens to be balanced.
JOBICY_GEOS = ["latam", "usa", "canada", "europe", "uk", "anywhere"]
JOBICY_INDUSTRIES = ["data-science", "business"]

MUSE_CATEGORIES = ["Data and Analytics", "Data Science"]
MUSE_PAGES = 25

ARBEITNOW_PAGES = 10

# Region buckets. The raw location strings are inconsistent across sources
# ("USA", "New York, NY", "Aachen; Berlin"), so they are mapped to a small set
# of comparable regions. Order matters: the first pattern that matches wins.
REGION_PATTERNS = [
    ("LATAM", r"latam|latin america|mexico|méxico|brazil|brasil|argentina|chile"
              r"|colombia|peru|perú|uruguay|ecuador|costa rica|panama|panamá"
              r"|dominican|guatemala|bolivia|paraguay|venezuela"),
    ("EE. UU. / Canadá", r"\busa\b|united states|u\.s\.|america|canada|canadá"
                         r"|, (al|az|ca|co|ct|dc|fl|ga|il|in|ma|md|mi|mn|mo|nc|nj|ny"
                         r"|oh|or|pa|tn|tx|ut|va|wa|wi)\b"),
    ("Europa", r"europe|emea|\buk\b|united kingdom|england|scotland|ireland|germany"
               r"|deutschland|berlin|münchen|munich|hamburg|köln|france|spain|españa"
               r"|portugal|italy|netherlands|poland|polska|sweden|norway|denmark"
               r"|austria|switzerland|belgium|czech|romania|greece|hungary"),
    ("Global / remoto", r"anywhere|worldwide|global|remote"),
]


def fetch(url):
    """Return decoded JSON, or None when the request fails."""
    try:
        with urlopen(Request(url, headers=HEADERS), timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        print(f"  ! {error} — {url[:80]}")
        return None


def classify_region(location):
    """Map a free-text location to one of the comparable region buckets."""
    text = (location or "").lower()
    for region, pattern in REGION_PATTERNS:
        if re.search(pattern, text):
            return region
    return "Otra / sin dato"


def record(source, job_id, title, company, location, seniority, description,
           functions="", salary_min=None, salary_max=None, salary_period=None,
           url=None, published_at=None):
    """The common shape every source is flattened into."""
    return {
        "id": f"{source}:{job_id}",
        "fuente": source,
        "titulo": title,
        "empresa": company,
        "ubicacion": location,
        "region": classify_region(location),
        "seniority_raw": seniority,
        "funciones": functions,
        "descripcion": description,
        "salario_min": salary_min,
        "salario_max": salary_max,
        "salario_periodo": salary_period,
        "url": url,
        "published_at": published_at,
    }


def collect_jobicy():
    """Remote-first postings worldwide, segmented by geography."""
    out = {}
    for industry in JOBICY_INDUSTRIES:
        for geo in JOBICY_GEOS:
            params = urlencode({"count": 100, "industry": industry, "geo": geo})
            payload = fetch(f"https://jobicy.com/api/v2/remote-jobs?{params}")
            for job in (payload or {}).get("jobs", []):
                out[job["id"]] = record(
                    "jobicy",
                    job["id"],
                    job.get("jobTitle"),
                    job.get("companyName"),
                    job.get("jobGeo"),
                    job.get("jobLevel"),
                    job.get("jobDescription", ""),
                    salary_min=job.get("salaryMin"),
                    salary_max=job.get("salaryMax"),
                    # Jobicy publishes annual figures; Get on Board publishes
                    # monthly ones. Recording the period keeps the two from
                    # being averaged together by accident.
                    salary_period=job.get("salaryPeriod") or "anual",
                    url=job.get("url"),
                    published_at=job.get("pubDate"),
                )
            print(f"  jobicy {industry}/{geo}: total {len(out)}")
            time.sleep(REQUEST_PAUSE_SECONDS)
    return list(out.values())


def collect_muse():
    """Mostly United States, with some European postings."""
    out = {}
    for category in MUSE_CATEGORIES:
        for page in range(MUSE_PAGES):
            params = urlencode({"category": category, "page": page})
            payload = fetch(f"https://www.themuse.com/api/public/jobs?{params}")
            results = (payload or {}).get("results", [])
            if not results:
                break
            for job in results:
                locations = "; ".join(
                    loc.get("name", "") for loc in job.get("locations", [])
                )
                levels = "; ".join(
                    level.get("name", "") for level in job.get("levels", [])
                )
                out[job["id"]] = record(
                    "themuse",
                    job["id"],
                    job.get("name"),
                    (job.get("company") or {}).get("name"),
                    locations,
                    levels,
                    job.get("contents", ""),
                    url=(job.get("refs") or {}).get("landing_page"),
                    published_at=job.get("publication_date"),
                )
            print(f"  themuse {category} p{page}: total {len(out)}")
            time.sleep(REQUEST_PAUSE_SECONDS)
    return list(out.values())


def collect_arbeitnow():
    """Europe, mostly Germany. Filtered to data roles after fetching.

    This board is general-purpose, so unlike the other two it has no data
    category to request; the filter has to happen here.
    """
    data_title = re.compile(
        r"data|analytics|analyst|machine learning|\bml\b|\bai\b|scientist"
        r"|business intelligence|\bbi\b",
        re.IGNORECASE,
    )
    out = {}
    for page in range(1, ARBEITNOW_PAGES + 1):
        payload = fetch(f"https://www.arbeitnow.com/api/job-board-api?page={page}")
        jobs = (payload or {}).get("data", [])
        if not jobs:
            break
        for job in jobs:
            if not data_title.search(job.get("title") or ""):
                continue
            out[job["slug"]] = record(
                "arbeitnow",
                job["slug"],
                job.get("title"),
                job.get("company_name"),
                job.get("location") or ("Remote" if job.get("remote") else ""),
                "; ".join(job.get("job_types") or []),
                job.get("description", ""),
                url=job.get("url"),
                published_at=job.get("created_at"),
            )
        print(f"  arbeitnow p{page}: {len(out)} data roles so far")
        time.sleep(REQUEST_PAUSE_SECONDS)
    return list(out.values())


def main():
    print("Jobicy:")
    jobs = collect_jobicy()
    print("The Muse:")
    jobs += collect_muse()
    print("Arbeitnow:")
    jobs += collect_arbeitnow()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(timezone.utc)
    output_path = RAW_DIR / f"global_{captured_at:%Y%m%d}.json"
    output_path.write_text(
        json.dumps(
            {
                "source": "multi (jobicy, themuse, arbeitnow)",
                "captured_at": captured_at.isoformat(),
                "n_jobs": len(jobs),
                "jobs": jobs,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    by_region = {}
    for job in jobs:
        by_region[job["region"]] = by_region.get(job["region"], 0) + 1
    print(f"\n{len(jobs)} postings -> {output_path}")
    for region, count in sorted(by_region.items(), key=lambda x: -x[1]):
        print(f"  {region}: {count}")


if __name__ == "__main__":
    main()
