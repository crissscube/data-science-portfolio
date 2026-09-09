"""Central configuration for the 10-K retrieval assistant.

Keeping every tunable in one module makes the evaluation experiments
reproducible: a run is fully described by this file plus the git commit.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# utf-8-sig tolerates the byte-order mark PowerShell prepends to text files;
# without it the first key would be read with an invisible prefix and never found.
load_dotenv(encoding="utf-8-sig")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

# --- Corpus scope ---
# Semiconductor sector, 2021-2024. This window covers the chip shortage,
# the AI demand surge and the China export restrictions, so the risk-factor
# language shifts sharply year over year -- which is what makes the
# comparative questions in the golden set worth asking.
TARGET_TICKERS = ["NVDA", "AMD", "INTC", "QCOM", "MU", "AVGO"]
FILING_YEARS = [2021, 2022, 2023, 2024]

# --- SEC EDGAR access ---
# SEC rejects requests without a contact email in the User-Agent.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

# --- Chunking ---
# bge-base truncates anything past 512 tokens without warning, so the budget is
# set in characters with room to spare. Measured against the model's own
# tokenizer, the corpus runs 4.75 characters per token: the 1500-character
# target lands at 280 tokens on average and the 2000-character cap peaks at 346,
# so nothing reaches the encoder's limit. The budget is deliberately under the
# 300-500 token sweet spot rather than over it -- a chunk that overflows is
# silently cut, while a chunk that undershoots merely costs a little recall.
CHUNK_TARGET_CHARS = 1500
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
# Tails shorter than this are folded back into the previous chunk instead of
# being indexed on their own: a 60-character fragment is never a useful answer.
CHUNK_MIN_CHARS = 300

# --- Embeddings & vector store ---
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
# 16 keeps peak RAM low on CPU; larger batches gave no throughput gain here.
EMBED_BATCH_SIZE = 16
# bge asks for this prefix on queries only -- passages are encoded bare.
# Applying it to both sides is a common mistake that quietly costs recall.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Retrieval ---
# bge-base is the primary encoder; MiniLM is kept as the baseline the
# experiments have to beat, so "we picked an embedding model" becomes a
# measured decision instead of a default copied from a tutorial.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BASELINE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8

# --- Hybrid retrieval ---
# Reciprocal rank fusion: score = sum(1 / (RRF_K + rank)) across retrievers.
# It merges rankings without needing the two score scales to be comparable,
# which matters because BM25 scores are unbounded and cosine similarity is not.
# The original paper uses 60, but it fuses runs 1000 documents deep. Here the
# two runs overlap on roughly 16 of 100 documents, and at that overlap a large
# constant flattens the rank differences until the only surviving signal is
# "did both retrievers return this at all" -- so a mediocre document present in
# both lists outranks a decisive first-place hit present in one.
#
# Measured with evals/sweep_rrf.py over five literal-term probes (MRR@8):
#   dense only 0.800 | bm25 only 0.850
#   K=0 0.900 | K=1 0.900 | K=3 0.900 | K=10 0.850 | K=60 0.800
# At K=60 the fusion performed exactly as well as dense alone, i.e. adding BM25
# bought nothing. 0, 1 and 3 tie; 3 is chosen for sitting in the middle of the
# plateau rather than on its edge.
#
# Five probes is a small sample and they were drawn from the acronym category,
# which favours BM25. Treated as provisional until the full golden set can
# re-run the comparison.
RRF_K = 3
# Depth each retriever contributes to the fusion pool. Wider than TOP_K so a
# document ranked poorly by one retriever can still be rescued by the other.
FUSION_DEPTH = 50

# --- Generation ---
# Local model for unlimited iteration; the paid model is only used for
# the final comparison run and, if needed, the LLM judge.
LOCAL_MODEL = "qwen2.5:7b-instruct-q4_K_M"
FRONTIER_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-haiku-4-5"
