# Retrieval over SEC 10-K filings, measured

A question-answering system over the annual reports of six semiconductor
companies, built to answer one question honestly: **does any of this actually
work better, and by how much?**

Most retrieval-augmented generation projects show a demo. This one ships the
measurement instead: 43 hand-labelled questions, six retrieval configurations
scored against them, and results that contradict what the author expected in
three separate places.

---

## The corpus

24 annual reports (Form 10-K) filed with the SEC by **NVIDIA, AMD, Intel,
Qualcomm, Micron and Broadcom** for fiscal years 2021–2024.

| | |
|---|---|
| Documents | 24 |
| Sections parsed | 284 |
| Characters | 9,165,892 (≈3,500 pages) |
| Retrievable chunks | 13,266 |

The window matters: it spans the chip shortage, the AI demand surge and the
China export restrictions, so the same company's language shifts sharply from
one year to the next. That is what makes year-over-year questions worth asking.

---

## What was actually measured

Six configurations: three retrievers × two chunking strategies.

### recall@8 — the share of relevant passages retrieved

| configuration | acronym | temporal | cross-issuer | single-fact | multi-hop | numeric | **total** |
|---|---|---|---|---|---|---|---|
| dense/structural | 0.800 | 0.500 | 0.583 | 0.733 | 0.417 | 0.583 | 0.626 |
| bm25/structural | 0.767 | 0.600 | 0.600 | 0.942 | 0.625 | 0.917 | 0.779 |
| **hybrid/structural** | **0.933** | **0.633** | **0.719** | **1.000** | **0.750** | **1.000** | **0.869** |
| dense/fixed | 0.733 | 0.367 | 0.350 | 0.150 | 0.542 | 0.667 | 0.426 |
| bm25/fixed | 0.867 | 0.300 | 0.464 | 0.217 | 0.438 | 0.833 | 0.488 |
| hybrid/fixed | 0.867 | 0.200 | 0.507 | 0.250 | 0.688 | 1.000 | 0.546 |

Hybrid retrieval finds at least one correct passage for **100% of questions**.

**Which of these gaps survive a significance test.** The configurations answer the
same questions, so the comparison is paired and McNemar's exact test applies
(`evals/significance.py`). Only the discordant questions carry information:

| comparison | A wins | B wins | p |
|---|---|---|---|
| hybrid/structural vs hybrid/fixed | 12 | 0 | **0.0005** |
| dense/structural vs dense/fixed | 11 | 1 | **0.0063** |
| hybrid/structural vs dense/structural | 6 | 0 | **0.0312** |
| hybrid/structural vs bm25/structural | 3 | 0 | 0.2500 |
| bm25/structural vs dense/structural | 6 | 3 | 0.5078 |

Structural chunking beats fixed-size chunking, and hybrid retrieval beats dense
retrieval; both hold at p < 0.05. **Hybrid's margin over BM25 alone does not.**
The 0.869 against 0.779 rests on three questions out of 35 and is compatible
with chance at this sample size. The honest claim is that hybrid retrieval is
not worse than its stronger component and is clearly better than its weaker one.

---

## Four findings

### 1. The classic beats the modern; only the combination beats both

BM25 — lexical matching from the 1980s — scores **0.779** against dense
embeddings' **0.626**. Fusing them reaches **0.869**.

That ordering is worth stating carefully. Hybrid beats dense at p = 0.031;
BM25's own advantage over dense (p = 0.508) and hybrid's over BM25 (p = 0.250)
are not separable from noise here. What the data support is that the lexical
half is doing real work and the dense half alone is the weakest choice.

The mechanism is visible in the per-category numbers. On single-fact questions
dense retrieval ranks the right passage higher (MRR 0.900 vs 0.670) while BM25
retrieves more of them (recall 0.942 vs 0.733). Each covers the other's blind
spot.

A worked example. Asked *"What does Micron say about the CHIPS and Science
Act?"*, dense retrieval returned the company's marketing boilerplate — *"Micron
is an industry leader in innovative memory and storage solutions"* — in all five
top positions, and **zero** of the 18 chunks that actually discuss the Act
(MRR@8 = 0.000). BM25 ranked the correct passage first. To a dense embedding,
"CHIPS and Science Act" dissolves into a general notion of *semiconductors +
government + United States*, and the boilerplate is saturated with exactly that.

### 2. The fusion constant from the paper made hybrid retrieval worthless

Reciprocal rank fusion is published with a constant of 60. Copying it produced a
hybrid retriever that scored **exactly as well as dense retrieval alone** — the
BM25 half bought nothing.

The published constant assumes runs a thousand documents deep. Here the two
runs overlap on roughly 16 of 100 results, and at that overlap a large constant
flattens every rank difference until the only surviving signal is *"did both
retrievers return this at all"* — so a mediocre document present in both lists
outranks a decisive first-place hit present in one.

Measured over five literal-term probes (`evals/sweep_rrf.py`, MRR@8):

| K | 0 | 1 | 3 | 10 | 60 |
|---|---|---|---|---|---|
| MRR@8 | 0.900 | 0.900 | 0.900 | 0.850 | 0.800 |

*(dense alone: 0.800 · BM25 alone: 0.850)*

`K=3` was chosen for sitting in the middle of the plateau rather than at its
edge. Without measuring, the whole BM25 component would have shipped as dead
weight and looked fine.

### 3. Questions designed to fail turned out easiest

Six questions ask for figures buried in financial tables, included on the
expectation that text extraction would reduce them to unlabelled runs of digits.
Hybrid retrieval scores **1.000** on them — the best category in the set.

The reason: annual reports are written to be read. The figures that matter get
restated in prose — *"we repurchased and retired approximately 67 million shares
of our common stock for $7,176 million"* — not only tabulated.

The hardest category is the one that looked routine: **year-over-year
comparisons**, at 0.633, collapsing to 0.200 under fixed-size chunking.

### 4. A third of the chunking advantage was a measurement artefact

The gold labels were originally produced by showing a human candidates from one
configuration only: hybrid/structural. That makes that configuration's own hits
the definition of relevance and rigs every comparison against it. Once noticed,
the remaining questions were labelled with candidates **pooled across all four
configurations**, the standard fix.

recall@8, split by which labelling regime produced the question:

| configuration | narrow pool (20 q) | full pool (15 q) | Δ |
|---|---|---|---|
| dense/structural | 0.713 | 0.511 | **−0.201** |
| bm25/structural | 0.812 | 0.733 | −0.079 |
| hybrid/structural | 0.913 | 0.811 | −0.102 |
| dense/fixed | 0.346 | 0.533 | **+0.188** |
| bm25/fixed | 0.441 | 0.550 | +0.109 |
| hybrid/fixed | 0.468 | 0.650 | **+0.182** |

Every structural configuration falls, every fixed one rises. Structural chunking
still wins, but its margin over naive chunking drops from **+0.445 to +0.161**.

The first number was the one the project would have reported. It was wrong by a
factor of nearly three, and nothing but the labelling procedure would have
revealed it.

---

## Design decisions worth their complexity

**Chunking follows document structure.** Each filing is split by SEC Item
(Item 1 Business, Item 1A Risk Factors, Item 7 MD&A…), then within each Item at
sentence and bullet boundaries. The naive alternative — cutting every 1,500
characters — is built too, as the baseline to beat. A cheap structural check,
requiring no model and no labels:

| | chunks | broken start | broken end | **clean on both ends** |
|---|---|---|---|---|
| structural | 5,618 | 1.6% | 5.9% | **93.3%** |
| fixed | 7,648 | 85.2% | 98.2% | **0.3%** |

85 of every 100 chunks in the baseline begin mid-word. That is what a default
pipeline produces, and it is rarely measured.

**Chunk size is set by the encoder's real limit, not by habit.** `bge-base`
truncates past 512 tokens silently. Measured against its own tokenizer this
corpus runs 4.75 characters per token, so the 1,500-character target lands at
280 tokens on average and the 2,000-character cap peaks at 346. Nothing is
silently cut.

**Every chunk is embedded with a provenance header.** A chunk from the middle of
a risk-factor section names neither its company nor its year, leaving *"our data
center revenue"* unattributable. The header restores that for the encoder and
for the citation the answer has to carry.

**bge is asymmetric.** Its instruction prefix belongs on the query side only.
Applying it to passages too is a common copy-paste error that shifts the whole
passage space and costs recall without ever raising an error.

**Intel enters through a fallback.** Its filings carry no "Item N" headings in
the body, so they are stored whole and chunked by size, tagged `size_fallback`.
This turned a parsing failure into an experiment — and Intel supplied the
sharpest finding in the corpus: the only one of the six companies that owns
fabs states that *"our most advanced current and future products are or will be
either exclusively manufactured by TSMC or reliant upon critical components…
manufactured by TSMC."*

---

## The golden set

43 questions, written **before** any retrieval improvement existed. A question
set authored afterwards drifts toward whatever the system already answers well.

| category | n | what it tests |
|---|---|---|
| `factual_single` | 10 | one fact, one document |
| `unanswerable` | 7 | the answer is not in the corpus — abstaining is correct |
| `comparative_temporal` | 6 | same issuer, different fiscal years |
| `numeric` | 6 | figures that live in tables |
| `comparative_cross` | 5 | different issuers, same year |
| `acronym_literal` | 5 | rare proper nouns dense retrieval blurs |
| `multi_hop` | 4 | requires combining separate chunks |

135 supporting chunks were marked by hand. Answers were drafted with retrieved
evidence and then reviewed, corrected and approved by a human — the corrections
were not cosmetic. Reviewing caught a question that was really two questions in
one (*"What is Nuvia and why does Qualcomm mention it?"* — the filings never
describe Nuvia's business, so half of it was unanswerable and the item could not
be scored either way; it was split), and seven questions whose scope had been
pinned to the wrong Item, which would have measured the scope rather than the
retriever.

`unanswerable` and `numeric` are the categories most portfolios omit. Including
questions you expect to fail is what separates engineering from marketing — and
for a financial assistant, knowing when to stay silent matters more than
accuracy.

---

## Known limitation: one query, two questions

Retrieval runs a single search. Questions that need two consistently lose one
half, in two distinct modes:

- **Comparative questions** retrieve the topic from one side only. Asked how
  NVIDIA's export-control language changed between 2022 and 2024, retrieval
  returns 2024 chunks and none from 2022, because the recent material is more
  plentiful and more similar to the question.
- **Multi-hop questions** retrieve one topic whole and miss the other entirely.
  Asked which companies discuss both the CHIPS Act and export restrictions,
  all 18 candidates covered export restrictions and none the CHIPS Act.

The fix — **query decomposition**, splitting the question into separate searches
and merging — is not built. The measured cost is a recall of 0.633 on temporal
comparisons against 1.000 on single-fact questions.

A second limitation is inherent to the method: pooling only surfaces passages
that at least one configuration retrieves. Anything all six miss never reaches
the reviewer and stays outside the denominator, so every system is scored
against a slightly generous yardstick.

---

## Pipeline

```
ingest.py   →  parse.py    →  chunk.py     →  embed.py      →  retrieve.py
24 filings     284 sections   13,266 chunks   13,266 vectors   dense|bm25|hybrid
```

| script | what it does |
|---|---|
| `src/ingest.py` | downloads the filings from SEC EDGAR |
| `src/parse.py` | splits each filing into its Items |
| `src/chunk.py` | both chunking strategies, with boundary-quality metrics |
| `src/embed.py` | embeddings into Chroma, resumable |
| `src/retrieve.py` | dense, BM25 and hybrid retrieval behind one interface |
| `src/peek.py` | read any slice of the corpus by hand |
| `src/review.py` | interactive labelling of the golden set |
| `evals/run_eval.py` | scores all six configurations |
| `evals/sweep_rrf.py` | picks the fusion constant by measurement |
| `src/generate.py` | answers with citations, or abstains |
| `evals/run_answers.py` | abstention and citation metrics, no judge involved |
| `evals/significance.py` | McNemar tests on every reported gap |

**Stack:** Python 3.9 · `sentence-transformers` (bge-base-en-v1.5) · ChromaDB ·
`rank-bm25` · CPU only.

Embedding the full corpus takes 58 minutes on CPU (~4 chunks/second). Retrieval
runs in 68–131 ms.

---

## Reproducing

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# SEC requires a contact email in the User-Agent
copy .env.example .env    # then set SEC_USER_AGENT

.venv\Scripts\python.exe src/ingest.py
.venv\Scripts\python.exe src/parse.py
.venv\Scripts\python.exe src/chunk.py
.venv\Scripts\python.exe src/embed.py      # ~58 min on CPU
.venv\Scripts\python.exe evals/run_eval.py
```

Raw filings are not committed. A run is fully described by `src/config.py` plus
the commit hash.

---

## Generation: guardrails, measured

Answers are produced by `qwen2.5:7b-instruct-q4_K_M` running locally, at
temperature 0, from the eight retrieved excerpts. Two guardrails, both checked
by counting rather than by a judge: every claim carries a citation, and when the
excerpts cannot support an answer the model must emit the exact token
`NOT_IN_CORPUS` rather than a plausible sentence.

| | strict prompt | balanced prompt |
|---|---|---|
| abstained when it should | **7/7** | **7/7** |
| abstained when it should not | 7/36 | 5/36 |
| answers carrying a citation | 28/29 | 28/31 |
| **citations to excerpts that were never supplied** | **0/43** | **0/43** |
| citation precision vs human labels | 0.631 | 0.615 |

Zero fabricated references across 86 answers, and no hallucinated answer on any
of the seven unanswerable questions.

The false-abstention rate is the interesting number. The strict prompt refused
on seven answerable questions — and the cause was not retrieval: those questions
had a **higher** mean recall (0.917) than the ones it answered (0.860), and five
of the seven had recall of 1.000. The model was demanding that the filing echo
the question's wording. It refused to describe the memory cycle because the
filing never writes "cyclical", and refused a question about the licensing
segment when the excerpt said "QTL Segment — Licensing revenues $5,572".

The balanced prompt was written to license exactly that reading, changing only
rule 1 and leaving rules 2–5 byte-identical. It rescued both of the questions it
was aimed at — and cost three answers their citations, because a longer rule 1
draws attention away from the rule demanding citations.

**None of the prompt differences are statistically significant** (McNemar,
p = 0.50 for abstentions, p = 1.00 for citations). Two questions out of 36 look
like a 5-point improvement and are not distinguishable from a coin flip. This is
the main limitation of a 43-question set: it separates large effects such as
structural versus fixed chunking (p = 0.0005) and is blind to small ones. Making
the prompt comparison conclusive would need hundreds of questions, not dozens.

## Faithfulness: the judge was measured, and both judges failed

Faithfulness — whether every claim in an answer follows from the excerpts it
cites — is the one metric here that cannot be counted. It needs a judge, and the
usual practice is to appoint an LLM as that judge and publish whatever number it
returns.

28 answers were judged by hand first, blind to any model verdict, under a fixed
rule stated identically to both raters: the unit is the claim, not the citation;
one unsupported claim makes the answer unfaithful; citing a superfluous excerpt
is imprecise rather than unfaithful. The same 28 then went to two judges.

| judge | raw agreement | Cohen's kappa | cost |
|---|---|---|---|
| `qwen2.5:7b` (local) | 0.296 | **−0.075** — worse than chance | free |
| `claude-haiku-4-5` (API) | 0.667 | **0.123** — slight | $0.044 |

**Neither is usable.** Published faithfulness figures from either would be
fiction, and they would be fiction in opposite directions.

**The 7B judge is not lenient — it is wrong.** The expectation going in was that
a small judge would approve everything. It rejected 19 of 28. Three of its
objections were checked against the literal text of the cited excerpts and all
three were false: it said a revenue figure was unsupported when the excerpt
reads *"Revenue for fiscal year 2024 was $60.9 billion"*; it said a buyback
amount was not specified when the excerpt lists three figures with
*"respectively"*; it said NOR memory was not mentioned when the cited excerpt
reads *"technologies, including DRAM, NAND, and NOR"*. Its faithfulness score of
32% describes a system that does not exist.

**The frontier judge is better and still not reliable.** It approved every
numeric answer the small judge had wrongly rejected. Of three objections checked,
one was correct, one was defensible, and one was pedantry — objecting that an
answer said *"reduce deliveries"* where the filing says *"reduce or eliminate
deliveries to us"*.

**The judge that failed validation still earned its cost.** Its one correct
objection caught a human error: an answer claiming Broadcom reports a dependence
on TSMC, citing two excerpts in which the string "TSMC" does not appear. That
answer was re-adjudicated to unfaithful, which is why the human column contains a
negative case at all.

### Why the number is 0.123 and not something respectable

The hand labels came out 26 faithful, 1 unfaithful, 1 unsure. With a distribution
that skewed, kappa is unstable by construction: chance agreement is already 0.62,
so almost all of the observed 0.667 is explained by both raters defaulting to
"faithful". Before adjudication the human column had **no** negative cases at
all, and kappa was exactly 0.000 against *both* judges — not a verdict on either,
but arithmetic. A validation set that cannot fail a judge cannot pass one either.

The honest conclusions:

1. **Faithfulness in this project is reported as human-verified over 28 answers**
   — 26 faithful, 1 unfaithful, 1 unsure — not as a judge-produced score.
2. **The validation set is too small and too imbalanced** to certify a judge.
   Fixing that means deliberately including answers known to be unfaithful, which
   this system rarely produces: the guardrails work.
3. **The methodological result generalises past this corpus.** Two judges
   returned confident faithfulness figures — 32% and 61% — that a human check
   showed to be wrong in both directions. The cost of finding that out was four
   cents and forty minutes of reading.

## Status

Retrieval, generation and judge validation are built and measured. The open work
is a validation set with enough unfaithful answers to certify or reject a judge,
and query decomposition for the comparative questions the single-pass retriever
cannot serve.
