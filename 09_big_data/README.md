# 09 — NYC Taxi Trips: Big Data Analytics with PySpark

**Business Question:** What drives demand and revenue for NYC yellow taxis, and what can the data — and its quality — tell us to support fleet, pricing, and staffing decisions?

---

## Overview

This project processes **~9.5 million NYC yellow taxi trips (2024, Q1)** using **PySpark**, demonstrating a full big-data workflow on a single machine: distributed loading, a data-quality investigation, cleaning, SQL analytics, and business-focused insights. Spark runs inside a **Docker** container, so the environment is fully reproducible with no local Spark/Java setup required.

**Dataset:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — Yellow Taxi records (Jan–Mar 2024) + the taxi zone lookup table.

---

## Key Insights

| # | Insight | Takeaway |
|---|---------|----------|
| 1 | **Hourly demand** | Trips peak 5–7 PM (evening commute), bottom out ~4 AM; strong late-night nightlife signal |
| 2 | **Weekly pattern** | Thursday is busiest, Sunday/Monday quietest; weekend fares run lower |
| 3 | **Revenue by zone** | JFK & LaGuardia airports drive the most revenue (~$51.5M) with 3–4x higher fares; Manhattan wins on volume |
| 4 | **Payment & tips** | The $0 average cash tip is a **data-collection artifact** (only card tips are recorded), not real behavior |

A highlight of the project is the **data-quality mindset**: investigating *why* the cleaning filter removed more rows than expected (NULL handling), and recognizing that cash tips are unmeasured rather than zero — avoiding false conclusions.

---

## Tech Stack

- **PySpark** (Spark SQL + DataFrame API) — distributed processing of ~9.5M rows
- **Docker** — reproducible Spark + Jupyter environment
- **matplotlib** — visualizations
- **Parquet** — columnar storage format

---

## How to Run

This project runs in a Docker container with PySpark preinstalled — no local Spark or Java needed.

**1. Download the data** and place it in `data/raw/`:
- Yellow Taxi trip data (Parquet) for **2024-01, 2024-02, 2024-03** from the [TLC page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- The `taxi_zone_lookup.csv` from the same source

Expected structure:
```
data/raw/
├── yellow_tripdata_2024-01.parquet
├── yellow_tripdata_2024-02.parquet
├── yellow_tripdata_2024-03.parquet
└── taxi_zone_lookup.csv
```

**2. Start the container** (from this folder):
```bash
docker compose up
```

**3. Open Jupyter** at `http://localhost:8888` (token: `bigdata`) and run `nyctaxitrips.ipynb`.

**4. Stop the container** when done:
```bash
docker compose down
```

---

## Skills Demonstrated

- Processing large datasets (~9.5M rows) with distributed computing (Spark)
- Writing analytical queries with **Spark SQL** and the **DataFrame API**
- **Data-quality investigation** — NULL handling, collection bias, questioning results
- Performance awareness (caching reused data)
- Translating technical results into **actionable business recommendations**
- Reproducible environments with **Docker**
