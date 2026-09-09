"""Answer a question from retrieved chunks, with citations or an abstention.

Two guardrails are the whole point of this module, and both are measurable:

1. Every claim carries a citation to the chunk it came from, so an answer can be
   checked against the filing rather than trusted.
2. When the retrieved context does not support an answer, the model must say so
   instead of producing a plausible one. In a financial setting a fabricated
   figure is worse than no answer, and the golden set contains seven questions
   whose only correct response is a refusal.

The provider is abstracted because the plan is deliberately hybrid: iterate for
free against a local model, and spend on a frontier model only for the final
comparison run. Rationing experiments would ruin an evaluation project.

Usage:
    .venv\\Scripts\\python.exe src/generate.py "What does Micron say about the CHIPS Act?"
    .venv\\Scripts\\python.exe src/generate.py "..." --ticker MU --provider ollama
"""

from __future__ import annotations

import argparse
import re
import textwrap

from config import LOCAL_MODEL, TOP_K
from retrieve import RETRIEVERS

# The abstention token. A distinctive string rather than a natural phrase so
# scoring can detect it exactly, without guessing whether "I'm not sure" counts.
ABSTAIN = "NOT_IN_CORPUS"

# Two prompts, identical except for how much interpretation the model is allowed.
# The strict one abstained on 7 of 36 answerable questions whose passages had
# already been retrieved -- it required the filing to echo the question's
# wording, refusing when a 10-K described a cycle without using the word
# "cyclical", or reported "QTL Segment / Licensing revenues" for a question
# about the licensing segment. The balanced one licenses that reading while
# keeping the bans on outside knowledge and unwritten figures. Which trade is
# better is measured, not assumed: loosening should cut false abstentions, and
# the cost is whatever it does to the correct ones.

_SHARED_RULES = f"""2. Every factual claim must cite the excerpt it comes from, as [1], [2], etc.
   A sentence without a citation is not acceptable.
3. Quote figures exactly as written, including units and the fiscal year.
4. Be concise: two to four sentences unless the question asks for a comparison.
5. If the excerpts answer only part of the question, answer that part and state
   plainly which part the excerpts do not cover."""

_HEADER = """You answer questions about SEC 10-K annual reports using ONLY the numbered excerpts provided.

Rules, in order of priority:
"""

PROMPTS = {
    "strict": f"""{_HEADER}
1. If the excerpts do not contain the answer, reply with exactly: {ABSTAIN}
   followed by one sentence saying what is missing. Do not use outside
   knowledge. Do not infer figures that are not written.
{_SHARED_RULES}""",

    "balanced": f"""{_HEADER}
1. Answer whenever the excerpts contain the information, even when they word it
   differently from the question. Filings rarely repeat a question's phrasing:
   they describe a cycle without calling it cyclical, and they name a segment
   "QTL" where the question says "licensing segment". Recognising that the
   excerpt and the question refer to the same thing is part of answering, and
   so is comparing figures that are present across years or companies.
   Only if the information is genuinely absent, reply with exactly: {ABSTAIN}
   followed by one sentence saying what is missing. Never use outside knowledge
   and never state a figure that is not written in an excerpt.
{_SHARED_RULES}""",
}

SYSTEM_PROMPT = PROMPTS["strict"]

USER_TEMPLATE = """Question: {question}

Excerpts:
{context}

Answer using only these excerpts, citing them as [1], [2], etc."""


def format_context(hits: list[dict]) -> str:
    """Number the excerpts and label each with its filing, so citations resolve."""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        item = f", Item {hit['item']}" if hit.get("item") else ""
        header = f"[{index}] {hit['ticker']} fiscal year {hit['fiscal_year']} 10-K{item}"
        blocks.append(f"{header}\n{' '.join(hit['text'].split())}")
    return "\n\n".join(blocks)


def ask_ollama(system: str, user: str, model: str) -> str:
    import ollama

    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        # Near-greedy decoding: the same question must produce the same answer
        # across runs, or the evaluation is measuring sampling noise.
        options={"temperature": 0.0, "num_ctx": 8192},
    )
    return response["message"]["content"].strip()


def ask_claude(system: str, user: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content
                   if block.type == "text").strip()


PROVIDERS = {"ollama": ask_ollama, "claude": ask_claude}


def cited_indices(answer: str) -> set[int]:
    """Excerpt numbers the answer actually cites."""
    return {int(number) for number in re.findall(r"\[(\d+)\]", answer)}


def answer(question: str, retriever_name: str = "hybrid",
           strategy: str = "structural", scope: dict | None = None,
           k: int = TOP_K, provider: str = "ollama",
           model: str | None = None, prompt: str = "strict") -> dict:
    retriever = RETRIEVERS[retriever_name](strategy)
    hits = retriever.search(question, k=k, scope=scope)

    if not hits:
        # Nothing retrieved is itself an abstention, and asking the model to
        # confirm that would only give it room to invent one.
        return {"question": question, "answer": f"{ABSTAIN} No excerpts matched.",
                "abstained": True, "hits": [], "cited": set()}

    model = model or (LOCAL_MODEL if provider == "ollama" else None)
    text = PROVIDERS[provider](
        PROMPTS[prompt], USER_TEMPLATE.format(question=question,
                                              context=format_context(hits)), model)

    return {
        "question": question,
        "answer": text,
        "abstained": text.startswith(ABSTAIN),
        "cited": cited_indices(text),
        "hits": hits,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "retriever": retriever_name,
        "strategy": strategy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Responder con citas o abstenerse.")
    parser.add_argument("question")
    parser.add_argument("--retriever", choices=list(RETRIEVERS), default="hybrid")
    parser.add_argument("--strategy", choices=["structural", "fixed"],
                        default="structural")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="ollama")
    parser.add_argument("--prompt", choices=list(PROMPTS), default="strict")
    parser.add_argument("--ticker")
    parser.add_argument("--year", type=int)
    parser.add_argument("--item")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    scope = {"ticker": args.ticker, "fiscal_year": args.year, "item": args.item}
    result = answer(args.question, args.retriever, args.strategy, scope,
                    args.k, args.provider, prompt=args.prompt)

    print(f"\n{args.question}")
    print("=" * 88)
    print(textwrap.fill(result["answer"], width=88))
    print("-" * 88)

    if result["abstained"]:
        print("se abstuvo")
    else:
        cited = sorted(result["cited"])
        print(f"cito los fragmentos: {cited or 'NINGUNO (viola la regla 2)'}")

    for index, hit in enumerate(result["hits"], start=1):
        mark = "*" if index in result["cited"] else " "
        item = f" Item {hit['item']}" if hit.get("item") else ""
        print(f" {mark}[{index}] {hit['ticker']} FY{hit['fiscal_year']}{item}")


if __name__ == "__main__":
    main()
