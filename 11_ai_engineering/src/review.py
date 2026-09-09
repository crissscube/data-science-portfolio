"""Interactive labelling of the golden set.

Retrieval proposes candidate chunks; the human decides which ones actually
support the answer and writes the answer down. That judgement is the ground
truth every later metric is scored against, so it cannot be automated away --
a model grading chunks it retrieved itself measures nothing.

Labels are written to evals/labels.jsonl rather than back into golden_set.yaml.
Round-tripping YAML would strip the comments that document the design, and
keeping the hand-written questions separate from the generated labels means a
labelling mistake can never corrupt the question set.

Usage:
    .venv\\Scripts\\python.exe src/review.py                 # next unlabelled
    .venv\\Scripts\\python.exe src/review.py --category numeric
    .venv\\Scripts\\python.exe src/review.py --id q017 --redo
    .venv\\Scripts\\python.exe src/review.py --progress
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import textwrap

import yaml

from config import FILING_YEARS, PROJECT_ROOT, TARGET_TICKERS
from retrieve import RETRIEVERS, HybridRetriever, format_hit

EVALS_DIR = PROJECT_ROOT / "evals"
GOLDEN_SET = EVALS_DIR / "golden_set.yaml"
LABELS = EVALS_DIR / "labels.jsonl"

# Candidates shown per question. Wider than the k the system will use at answer
# time, because a chunk the current retriever misses can still be the gold one.
CANDIDATES = 12

# Candidates are pooled across every configuration that appears in the results
# table -- both chunking strategies, dense and lexical. Judging against one
# configuration makes its own hits the definition of relevance and rigs every
# later comparison against it. Pooling the systems under comparison before
# judging is the standard fix; it is what TREC has done for decades.
#
# Questions q001-q043 labelled before 2026-08-23 saw only hybrid/structural
# candidates. Their structural-vs-fixed numbers are not usable, and the gap
# between the two blocks is itself a measurement of how much that bias is worth.
POOL_STRATEGIES = ("structural", "fixed")
PER_RETRIEVER = 4
# Hard ceiling on what a person is asked to read. Pooling four configurations
# across six issuers produced forty candidates for one question, which is not a
# review, it is an endurance test. Selection is round-robin by rank so every
# configuration contributes its best before any contributes its second best.
MAX_CANDIDATES = 18

# Company names as they appear in question text, mapped to their ticker. A
# question that compares two issuers should not spend two thirds of its
# candidate slots on the four it never mentions.
NAMED_ISSUERS = {
    "nvidia": "NVDA", "amd": "AMD", "advanced micro": "AMD",
    "qualcomm": "QCOM", "micron": "MU", "broadcom": "AVGO", "intel": "INTC",
}

MENU = """
  1 3 4     marcar esos fragmentos y seguir a la respuesta
  v 3       ver el fragmento 3 completo
  n         ninguno sirve, pero igual quiero escribir la respuesta
  s         saltar: no guarda nada, la pregunta queda sin tocar
  q         guardar lo hecho y salir
"""


def load_questions() -> list[dict]:
    with GOLDEN_SET.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["questions"]


def load_labels() -> dict[str, dict]:
    if not LABELS.exists():
        return {}
    with LABELS.open(encoding="utf-8") as fh:
        labels = {row["id"]: row for row in map(json.loads, fh)}

    # Normalise rows written before pending_answer existed: an empty answer on a
    # question that needs one was never really reviewed.
    for row in labels.values():
        if not row.get("must_abstain") and not row.get("expected_answer"):
            row["status"] = "pending_answer"
    return labels


def save_labels(labels: dict[str, dict]) -> None:
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", encoding="utf-8") as fh:
        for key in sorted(labels):
            fh.write(json.dumps(labels[key], ensure_ascii=False) + "\n")


def show_progress(questions: list[dict], labels: dict[str, dict]) -> None:
    done = collections.Counter()
    pending = collections.Counter()
    total = collections.Counter()
    for question in questions:
        status = labels.get(question["id"], {}).get("status")
        total[question["category"]] += 1
        done[question["category"]] += status == "reviewed"
        pending[question["category"]] += status == "pending_answer"

    print(f"\n{'categoria':<24}{'listas':>8}{'pend.':>8}{'total':>8}")
    print("-" * 48)
    for category in total:
        print(f"{category:<24}{done[category]:>8}"
              f"{pending[category]:>8}{total[category]:>8}")
    print("-" * 48)
    print(f"{'TOTAL':<24}{sum(done.values()):>8}"
          f"{sum(pending.values()):>8}{sum(total.values()):>8}\n")
    if sum(pending.values()):
        print("  'pend.' = sin respuesta escrita; vuelven a aparecer solas\n")


def issuers_named_in(question: str) -> list[str]:
    """Tickers the question mentions, or all of them when it names none."""
    lowered = question.lower()
    named = {ticker for name, ticker in NAMED_ISSUERS.items()
             if name in lowered}
    return sorted(named) if named else list(TARGET_TICKERS)


def build_pool(strategies: tuple[str, ...] = POOL_STRATEGIES) -> dict:
    """One retriever per configuration under comparison.

    Labelling against a single configuration makes its own hits the definition
    of relevance, which is how the first 27 questions were done and why the
    structural-vs-fixed comparison from them is not usable. Every configuration
    that will appear in the results table has to be able to put a candidate in
    front of the reviewer.
    """
    pool = {}
    for strategy in strategies:
        for name in ("dense", "bm25"):
            pool[f"{name}/{strategy}"] = RETRIEVERS[name](strategy)
    return pool


def gather_candidates(question: dict, pool: dict) -> list[dict]:
    """Pool candidates, stratified when the question spans years or issuers.

    A comparative question needs evidence from both sides of the comparison, but
    a single ranked run gives every slot to whichever year or company wrote most
    about the topic -- asking how NVIDIA's export-control language changed from
    2022 to 2024 returns four chunks from 2024 and none from 2022. Searching each
    side separately and merging guarantees both are on screen to be judged.

    This is a labelling aid, not a retrieval trick: the system under test still
    gets one shot at the question. That gap is the point -- these labels are what
    will measure it.
    """
    scope = dict(question.get("scope") or {})
    category = question["category"]

    if category == "comparative_temporal" and not scope.get("fiscal_year"):
        strata = [{**scope, "fiscal_year": year} for year in FILING_YEARS]
    elif category in ("comparative_cross", "multi_hop") and not scope.get("ticker"):
        strata = [{**scope, "ticker": ticker}
                  for ticker in issuers_named_in(question["question"])]
    else:
        strata = [scope]

    per_stratum = max(2, PER_RETRIEVER // max(1, len(strata) // 2))
    pooled: dict[str, dict] = {}
    for name, engine in pool.items():
        for stratum_index, stratum in enumerate(strata):
            for rank, hit in enumerate(engine.search(question["question"],
                                                     k=per_stratum,
                                                     scope=stratum), start=1):
                # Chunk ids differ between strategies, so the same passage can
                # enter twice. Keep both: the reviewer marks whichever appears,
                # and scoring matches on text, not on id.
                entry = pooled.setdefault(hit["chunk_id"], {
                    **hit, "found_by": [], "order": (rank, stratum_index, name)})
                entry["found_by"].append(name)
                entry["order"] = min(entry["order"], (rank, stratum_index, name))

    for entry in pooled.values():
        entry["retriever"] = "+".join(sorted({
            found.split("/")[0] for found in entry["found_by"]}))
        entry["strategy_tag"] = "+".join(sorted({
            found.split("/")[1] for found in entry["found_by"]}))

    kept = sorted(pooled.values(), key=lambda hit: hit["order"])[:MAX_CANDIDATES]
    # Presented grouped by filing so a comparative question reads in order.
    return sorted(kept, key=lambda hit: (hit["ticker"], hit["fiscal_year"],
                                         hit.get("item") or ""))


def read_choice(hits: list[dict]) -> str:
    """Read the menu input, handling 'v N' inline.

    The preview highlights the window densest in query terms, which is the wrong
    window whenever the answer is a figure the question never names -- asking for
    total revenue surfaces prose about revenue and hides "Total revenue $60,922".
    Rather than guess better, let the reviewer open the chunk.
    """
    while True:
        choice = input("  > ").strip().lower()
        if choice == "q":
            return "quit"
        if choice == "s":
            return "skip"
        if not choice:
            # An empty Enter used to fall through as "mark nothing" and jump
            # straight to the answer prompt, silently. Four questions ended up
            # with an answer and no supporting chunks that way.
            print("  Escribe los numeros a marcar (ej: 1 3 4), o 'n' si ninguno")
            print("  sirve. 'v 3' abre un fragmento completo.")
            continue
        if not choice.startswith("v"):
            return choice

        wanted = [int(token) for token in re.findall(r"\d+", choice)]
        for number in wanted:
            if 1 <= number <= len(hits):
                hit = hits[number - 1]
                print(f"\n  --- [{number}] completo "
                      f"({hit['n_chars']} caracteres) ---")
                print(textwrap.fill(" ".join(hit["text"].split()),
                                    width=88, initial_indent="  ",
                                    subsequent_indent="  "))
                print()
            else:
                print(f"  no existe el fragmento {number}")
        print(MENU)


def review_one(question: dict, pool: dict,
               labels: dict[str, dict]) -> str:
    """Label a single question. Returns 'quit', 'skip' or 'done'."""
    print("\n" + "=" * 88)
    print(f"{question['id']}  [{question['category']}]")
    print(f"\n  {question['question']}\n")
    if question.get("scope"):
        print(f"  ambito: {question['scope']}")

    if question.get("must_abstain"):
        # Nothing to label: the right behaviour is a refusal, and any chunk the
        # retriever returns is by definition a distractor.
        print("\n  Esta pregunta NO tiene respuesta en el corpus.")
        print(f"  Respuesta esperada: {question['expected_answer']}")
        answer = input("\n  Confirmar? [Enter=si / t=corregir texto / s=saltar] ").strip()
        if answer.lower() == "s":
            return "skip"
        expected = question["expected_answer"]
        if answer.lower() == "t":
            expected = input("  Respuesta esperada: ").strip() or expected
        labels[question["id"]] = {
            "id": question["id"],
            "category": question["category"],
            "expected_answer": expected,
            "gold_chunk_ids": [],
            "must_abstain": True,
            "status": "reviewed",
            "reviewed_at": dt.date.today().isoformat(),
        }
        return "done"

    hits = gather_candidates(question, pool)
    print()
    for index, hit in enumerate(hits, 1):
        print(format_hit(index, hit, query=question["question"],
                         show_score=False))
        print()

    print(MENU)
    choice = read_choice(hits)
    if choice in ("quit", "skip"):
        return choice

    gold: list[str] = []
    while choice not in ("n", ""):
        # Accept any separator a person might reasonably type, then report what
        # could not be used. Dropping unparsed input silently is how three
        # questions ended up with an answer and no supporting chunks.
        tokens = re.split(r"[^0-9]+", choice)
        picked, rejected = [], []
        for token in tokens:
            if not token:
                continue
            number = int(token)
            (picked if 1 <= number <= len(hits) else rejected).append(number)

        if rejected:
            print(f"  fuera de rango (hay {len(hits)} candidatos): "
                  f"{' '.join(str(n) for n in rejected)}")
        if picked:
            gold = [hits[number - 1]["chunk_id"] for number in dict.fromkeys(picked)]
            print(f"  marcados: {' '.join(str(n) for n in dict.fromkeys(picked))}")
            break

        print("  No se entendio ningun numero. Escribelos separados por espacio")
        print("  (ej: 1 3 4), 'v 3' para ver uno completo, o 'n' si ninguno sirve.")
        choice = read_choice(hits)
        if choice in ("quit", "skip"):
            return choice

    print("  " + "-" * 60)

    # On a --redo the answer is usually already right and only the chunk marks
    # are missing. Retyping it invites a worse second version.
    previous = labels.get(question["id"], {}).get("expected_answer", "")
    if previous:
        print(f"  Respuesta guardada: {previous}")
        print("  Enter para conservarla, o escribe una nueva.")
    else:
        print("  AHORA EL TEXTO. Enter vacio si no sabes: queda pendiente, no perdida.")

    expected = input("  Respuesta en una o dos frases > ").strip() or previous

    # The two prompts look alike and the numbers get typed into the wrong one.
    # Silently storing "1 2 5" as ground truth would poison every later metric,
    # so the obvious mistake is caught here rather than discovered in results.
    while expected and not any(char.isalpha() for char in expected):
        print("  Eso parecen numeros, no una respuesta. Los fragmentos ya se")
        print("  marcaron; aqui va el texto. Enter vacio para dejarla pendiente.")
        expected = input("  Respuesta en una o dos frases > ").strip()

    # A question with no answer written down cannot score anything, so it stays
    # pending instead of counting as reviewed. These are the ones to revisit
    # once hybrid retrieval can surface what dense search missed.
    status = "reviewed" if expected else "pending_answer"

    labels[question["id"]] = {
        "id": question["id"],
        "category": question["category"],
        "expected_answer": expected,
        "gold_chunk_ids": gold,
        # Dense retrieval found nothing usable. Worth recording separately from
        # the answer: these are the cases BM25 is expected to rescue.
        "no_support_found": not gold,
        "must_abstain": False,
        "status": status,
        "reviewed_at": dt.date.today().isoformat(),
    }
    if status == "pending_answer":
        print("  pendiente: sin respuesta, volvera a aparecer")
    else:
        print(f"  guardado: {len(gold)} fragmentos de apoyo")
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser(description="Etiquetar el golden set.")
    parser.add_argument("--category", help="revisar solo esta categoria")
    parser.add_argument("--id", help="revisar solo esta pregunta")
    parser.add_argument("--redo", action="store_true",
                        help="volver a etiquetar aunque ya este lista")
    parser.add_argument("--progress", action="store_true",
                        help="solo mostrar el avance")
    args = parser.parse_args()

    questions = load_questions()
    labels = load_labels()

    if args.progress:
        show_progress(questions, labels)
        return

    pending = [
        question for question in questions
        if (not args.category or question["category"] == args.category)
        and (not args.id or question["id"] == args.id)
        and (args.redo or labels.get(question["id"], {}).get("status") != "reviewed")
    ]

    if not pending:
        # Saying "nothing pending" when the question exists but is already
        # labelled sends the reviewer looking for a bug instead of adding --redo.
        already = [question for question in questions
                   if (not args.category or question["category"] == args.category)
                   and (not args.id or question["id"] == args.id)]
        if already and not args.redo:
            names = ", ".join(question["id"] for question in already[:5])
            print(f"\n{len(already)} pregunta(s) ya etiquetadas: {names}")
            print("Agrega --redo para volver a etiquetarlas.")
        else:
            print("\nNada pendiente con esos filtros.")
        show_progress(questions, labels)
        return

    print(f"\n{len(pending)} preguntas por etiquetar. 'q' guarda y sale.")
    print(f"cargando indices de {len(POOL_STRATEGIES) * 2} configuraciones...")
    pool = build_pool()

    try:
        for question in pending:
            if review_one(question, pool, labels) == "quit":
                break
    except KeyboardInterrupt:
        print("\n\ninterrumpido")
    finally:
        # Saving in `finally` means Ctrl+C never costs finished labels.
        save_labels(labels)
        print(f"\nguardado en {LABELS}")
        show_progress(questions, labels)


if __name__ == "__main__":
    main()
