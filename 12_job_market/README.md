# Data Job Market — Do Employers Ask for the Same Things Everywhere?

## Business Problem

Career-change advice for data roles is mostly written by people selling courses,
and it is almost always local: a ranking of tools drawn from one country,
presented as if it described the profession. Anyone choosing what to study — or
which vacancies to apply to — is deciding based on that.

This project asks a question that advice never answers with evidence: **is the
demand profile for data roles the same across regions, or does it depend on where
you live?** If it is the same, one set of study priorities works globally. If it
is not, the advice has to change per market.

## Dataset

- **Captured:** 2026-09-08
- **Postings retrieved:** 1,243 across four public APIs
- **Data roles in the comparable set:** 271
- **Discarded:** 927 postings the broad searches pulled in that are not data roles

| Source | Coverage | Language | Postings in comparable set |
|---|---|---|---|
| [Get on Board](https://www.getonbrd.com/api/v0) | LATAM | Spanish | 133 |
| [The Muse](https://www.themuse.com/developers/api/v2) | United States, some Europe | English | — |
| [Jobicy](https://jobicy.com/jobs-rss-feed) | Remote worldwide | English | — |
| [Arbeitnow](https://www.arbeitnow.com/api) | Europe, mostly Germany | English / German | — |
| **By region** | LATAM 133 · US/Canada 98 · Europe 40 | | **271** |

**Why these sources.** All four are documented public APIs — no scraping, no
terms-of-service violation, no risk to the LinkedIn account being used to
job-hunt. Get on Board is the only one that exposes the job's `functions` (the
listed tasks) as a field separate from the marketing blurb, which is what makes
the task-based classification possible at all.

Rejected: LinkedIn (blocks scraping), Tecnoempleo (HTTP 403), Computrabajo (bot
protection).

**The comparability rule.** Regions are only comparable if the same filter is
applied to each. A posting enters the comparable set when its **title names a
data role** — the same test in Santiago, Austin and Berlin. Get on Board also
publishes its own category taxonomy, but using it would apply a looser filter to
LATAM than to the other regions, and any difference that produced would be an
artefact of the filter rather than a property of the market. The tool vocabulary
and the task-classification patterns are likewise bilingual, for the same reason.

## Key Questions

1. Do the three regions ask for the same tools, in the same order?
2. How is the market split between data scientist, analyst, engineer and governance roles?
3. Does a "Data Scientist" title describe data science work, or analyst work with a better name?
4. How many postings are genuinely open to someone entering the field?
5. What does it cost to apply — how many others apply, and who publishes a salary?

## Main Findings

1. **The core of the job is the same in every region.** Spearman rank
   correlation between regions' tool rankings: **0.80** (LATAM ↔ US/Canada),
   **0.77** (LATAM ↔ Europe), **0.71** (US/Canada ↔ Europe). SQL and Python lead
   everywhere at near-identical rates — SQL 63/61/68%, Python 61/72/68%.

2. **What differs is the layer on top, not the foundation.** The US asks far more
   for deep learning frameworks (PyTorch 14% vs 3% in LATAM) and generative AI
   (28% vs 14%). LATAM leans toward BI and spreadsheets (Power BI 30% vs 15% in
   Europe; Excel 14% vs 2%). Europe leans toward modern data engineering tooling
   (dbt 25% vs 12% elsewhere).

3. **The "data scientist" title is a minority of the market.** Of 271 data
   postings, **38 (14%)** are titled Data Scientist. Data engineering and
   architecture take **79 (29%)**, and analyst-type roles (Data Analyst / BI plus
   Business Analyst) take **91 (34%)**.

4. **Title inflation is real but much smaller than claimed.** The expectation
   going in was that around half of "Data Scientist" postings would describe
   analyst work. Classified by listed tasks: **17 are clearly data science, 17
   are mixed, and 4 are analyst work** — roughly one in ten, not one in two.

5. **Deep learning is close to absent, and this holds in all three regions**
   (4% LATAM, 6% US/Canada, 5% Europe). Generative AI, at 19% overall, is now
   asked for about twice as often as scikit-learn and pandas combined.

6. **The entry door is narrow, but not because of inflated requirements.** Only
   **10 of 271** postings (3.7%) are junior and **none** are open to candidates
   with no experience. Junior postings ask a median of **1 year** — the common
   complaint that junior roles demand five years is not what this sample shows.
   The barrier is volume, not the bar.

7. **Communication outranks every ML library.** Communicating results appears in
   62% of LATAM postings and **80%** of US postings, against 4% for scikit-learn.

8. **LATAM only** (applications and monthly salary exist on Get on Board alone):
   a median of **72 applicants per posting**, one with 2,112. Only 36% publish a
   salary; among those the median range is **3,000–3,500 USD/month**.

## What This Data Does Not Say

These postings say what an employer **asks for when publishing a vacancy**. They
do not say what the work is actually like day to day — a different question this
dataset cannot answer.

The sample is the active postings of four boards on one date. Each board has its
own bias, so the conclusions worth trusting are the ones that **hold across all
three regions**; single-region results are reported as such. **Europe (n=40) is
the thin one** and its percentages move easily.

Costa Rica, and most individual LATAM countries outside Chile, appear close to
zero on these boards. That is not a defect of the analysis — it is the finding:
for a candidate based there the realistic market is remote, LATAM-wide and
global, which is what this dataset covers.

The task-based classification is a judgment call, implemented in
`src/extract.py` and auditable line by line. Postings whose tasks mix profiles
are labelled `mixto` rather than forced into a bucket.

## Project Structure

```
12_job_market/
├── src/
│   ├── collect.py         # Get on Board (LATAM)
│   ├── collect_global.py  # Jobicy, The Muse, Arbeitnow — normalised to one shape
│   ├── extract.py         # Bilingual vocabulary, task classification, scope rules
│   ├── export_excel.py    # Review workbook (+ CSV fallback) for manual auditing
│   ├── apply_review.py    # Fold manual corrections back into the dataset
│   └── analyze.py         # Charts and findings report
├── data/
│   ├── raw/               # API captures, one per source per date (gitignored)
│   └── processed/         # Tidy table (gitignored)
├── outputs/               # Charts, findings report, review files
├── requirements.txt
└── README.md
```

## How to Reproduce

```bash
pip install -r requirements.txt
python src/collect.py         # LATAM capture      (~1 min)
python src/collect_global.py  # US / Europe / remote (~3 min)
python src/extract.py         # -> data/processed/ofertas.csv
python src/analyze.py         # -> charts + outputs/hallazgos.md
python src/export_excel.py    # -> review workbook and CSV
```

To audit the classification: open `outputs/dataset_ofertas.xlsx` (or
`outputs/revision_ofertas.csv`), read `funciones_texto` next to each label, and
fill in `revisado_ok` or `perfil_corregido`. Then:

```bash
python src/apply_review.py    # -> data/processed/ofertas_revisado.csv
python src/analyze.py         # regenerates everything from the reviewed labels
```

`analyze.py` prefers the reviewed file when it exists, so a manual review is
never silently overwritten by a re-run.

## Possible Extensions

- **Track it over time.** The collectors are date-stamped and idempotent; running
  them weekly turns a snapshot into a trend — a far stronger claim than any single
  capture, and the only way to confirm whether the generative-AI share is really
  climbing.
- **Balance the regions.** Europe at n=40 is the weakest leg; adding a European
  board would let the cross-region claim rest on three equally solid samples.
- **Replace the regex classifier with a trained model**, using the manual review
  as labelled data — a clean supervised task on a self-collected dataset.
- **Model salary against required tools** to quantify which skills carry a
  premium, handling the monthly/annual split explicitly rather than pooling it.
