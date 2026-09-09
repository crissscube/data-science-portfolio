"""Judge whether answers are faithful to the text they cite, by human and by model.

Faithfulness is not correctness. An answer can be true and unfaithful -- stating
something the model knew rather than something the excerpt says -- and it can be
faithful while failing to answer the question. Only the first distinction is
judged here: does every claim follow from the cited excerpts.

The point of this module is not the score. It is validating the scorer. A 7B
judge approves almost everything, so a faithfulness figure produced by one means
nothing until it is checked against human judgement. Cohen's kappa is the check:
it measures agreement after discounting the agreement two lenient raters would
reach by chance alone.

Order matters. The human judges first, blind to the model's verdict. Reading the
model's answer first would anchor the human to it, and the resulting kappa would
measure a tendency to agree with what was already read.

Usage:
    .venv\\Scripts\\python.exe evals/judge.py --human            # you judge, first
    .venv\\Scripts\\python.exe evals/judge.py --model            # then the judge runs
    .venv\\Scripts\\python.exe evals/judge.py --kappa            # then compare
    .venv\\Scripts\\python.exe evals/judge.py --human --progress
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import EVAL_RESULTS_DIR, JUDGE_MODEL, LOCAL_MODEL  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
HUMAN_PATH = EVALS_DIR / "judgments_human.jsonl"


def model_path(model: str) -> Path:
    """One file per judge. Comparing two judges against the same human labels is
    the point of this stage, so a shared file that each run overwrites would
    destroy the earlier verdicts."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return EVALS_DIR / f"judgments_{slug}.jsonl"

# A binary verdict, not a scale. With around thirty judgements a three-point
# scale leaves cells nearly empty and kappa becomes unstable; "unsure" is kept
# as an explicit escape so borderline cases are excluded rather than forced.
FAITHFUL, UNFAITHFUL, UNSURE = "faithful", "unfaithful", "unsure"

# The unit of judgement is the claim, not the citation. An answer is faithful
# when every claim it makes is supported by the cited excerpts taken together;
# one unsupported claim makes the whole answer unfaithful, however much of the
# rest checks out. Citing an extra excerpt that does not support the claim is
# imprecise rather than unfaithful -- citation precision is measured separately
# in run_answers.py, and penalising it here would count the same flaw twice.
#
# Both raters must apply this same definition. If they do not, kappa measures
# disagreement about the rule instead of disagreement about the answers.
CRITERION = """
  REGLA: la unidad es la AFIRMACION, no la cita.
    fiel     = TODAS las afirmaciones estan sostenidas por el CONJUNTO de
               fragmentos citados
    no fiel  = al menos UNA afirmacion no lo esta
    citar de mas (un fragmento que no aporta) es impreciso, no infiel;
    eso ya se mide aparte como precision de citas
"""

MENU = """
  f    fiel: toda afirmacion se sigue de los fragmentos citados
  n    no fiel: alguna afirmacion no esta sostenida
  d    dudosa: no queda claro; se excluye del calculo
  v    ver los fragmentos citados completos
  s    saltar
  q    guardar y salir
"""

JUDGE_SYSTEM = """You check whether an answer is faithful to the excerpts it cites.

Faithful means EVERY factual claim in the answer is supported by the cited
excerpts taken together. A single unsupported claim makes the answer unfaithful,
however well supported the rest of it is.

Judge support, not usefulness: an answer that is well supported but does not
address the question is still faithful. An answer that states something true but
absent from the cited excerpts is NOT faithful.

Citing an excerpt that does not support the claim is imprecise, not unfaithful,
as long as some cited excerpt does support it.

Reply with exactly one word on the first line: FAITHFUL or UNFAITHFUL.
On a second line give one short sentence of justification."""

JUDGE_USER = """Question: {question}

Answer under review:
{answer}

Cited excerpts:
{excerpts}

Is the answer faithful to these excerpts?"""


def latest_answers(pattern: str = "answers_2*.jsonl") -> Path:
    matches = sorted(glob.glob(str(EVAL_RESULTS_DIR / pattern)))
    if not matches:
        raise SystemExit("No hay respuestas. Corre antes evals/run_answers.py.")
    return Path(matches[-1])


def load_answers(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    # Abstentions have nothing to be faithful to; they are scored separately by
    # run_answers.py and would only pad the agreement statistics here.
    return [row for row in rows if not row["score"]["abstained"] and row["cited"]]


def load_judgments(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["id"]: row for row in map(json.loads, fh)}


def save_judgments(path: Path, judgments: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for key in sorted(judgments):
            fh.write(json.dumps(judgments[key], ensure_ascii=False) + "\n")


def cited_excerpts(row: dict, chunks: dict[str, str]) -> list[tuple[int, str]]:
    """The excerpts the answer actually cited, in citation order."""
    retrieved = row["retrieved_chunk_ids"]
    out = []
    for number in row["cited"]:
        if 1 <= number <= len(retrieved):
            out.append((number, chunks.get(retrieved[number - 1], "")))
    return out


def load_chunk_lookup() -> dict[str, str]:
    lookup = {}
    for name in ("chunks_structural.jsonl", "chunks_fixed.jsonl"):
        path = Path(__file__).resolve().parent.parent / "data" / "processed" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                chunk = json.loads(line)
                lookup[chunk["chunk_id"]] = " ".join(chunk["text"].split())
    return lookup


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=88, initial_indent=indent,
                         subsequent_indent=indent)


def judge_human(rows: list[dict], chunks: dict[str, str], limit: int) -> None:
    judgments = load_judgments(HUMAN_PATH)
    pending = [row for row in rows if row["id"] not in judgments][:limit]

    if not pending:
        print(f"\nNada pendiente. {len(judgments)} juicios guardados.")
        return

    print(f"\n{len(pending)} respuestas por juzgar. 'q' guarda y sale.")
    print(CRITERION)

    try:
        for index, row in enumerate(pending, start=1):
            excerpts = cited_excerpts(row, chunks)
            print("\n" + "=" * 88)
            print(f"{row['id']}  [{row['category']}]   ({index}/{len(pending)})")
            print(f"\nPREGUNTA\n{wrap(row['question'])}")
            print(f"\nRESPUESTA\n{wrap(row['answer'])}")
            print(f"\nCITA los fragmentos: {row['cited']}")
            for number, text in excerpts:
                print(f"\n  --- [{number}] ---")
                print(wrap(text[:700] + ("..." if len(text) > 700 else ""), "    "))

            print(MENU)
            while True:
                choice = input("  > ").strip().lower()
                if choice.startswith("v"):
                    for number, text in excerpts:
                        print(f"\n  --- [{number}] completo ---")
                        print(wrap(text, "    "))
                    print(MENU)
                    continue
                if choice in ("f", "n", "d", "s", "q"):
                    break
                print("  Escribe f, n, d, v, s o q.")

            if choice == "q":
                break
            if choice == "s":
                continue

            verdict = {"f": FAITHFUL, "n": UNFAITHFUL, "d": UNSURE}[choice]
            note = input("  Nota (opcional, Enter para omitir): ").strip()
            judgments[row["id"]] = {
                "id": row["id"],
                "category": row["category"],
                "verdict": verdict,
                "note": note,
                "judged_at": dt.date.today().isoformat(),
                "rater": "human",
            }
            print(f"  guardado: {verdict}")
    except KeyboardInterrupt:
        print("\n\ninterrumpido")
    finally:
        save_judgments(HUMAN_PATH, judgments)
        counts = collections.Counter(row["verdict"] for row in judgments.values())
        print(f"\n{len(judgments)} juicios en {HUMAN_PATH.name}: {dict(counts)}")


def ask_judge(system: str, user: str, model: str, provider: str) -> str:
    if provider == "ollama":
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            options={"temperature": 0.0, "num_ctx": 8192},
        )
        return response["message"]["content"].strip()

    import anthropic
    response = anthropic.Anthropic().messages.create(
        model=model, max_tokens=256, temperature=0.0,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def judge_model(rows: list[dict], chunks: dict[str, str], limit: int,
                model: str, provider: str = "ollama") -> None:
    path = model_path(model)
    judgments = load_judgments(path)
    pending = [row for row in rows if row["id"] not in judgments][:limit]
    print(f"\n{len(pending)} respuestas por juzgar con {model}")

    for index, row in enumerate(pending, start=1):
        excerpts = cited_excerpts(row, chunks)
        rendered = "\n\n".join(f"[{number}] {text}" for number, text in excerpts)
        text = ask_judge(JUDGE_SYSTEM, JUDGE_USER.format(
            question=row["question"], answer=row["answer"],
            excerpts=rendered), model, provider)
        first = text.split("\n")[0].upper()
        verdict = FAITHFUL if "UNFAITHFUL" not in first and "FAITHFUL" in first \
            else UNFAITHFUL

        judgments[row["id"]] = {
            "id": row["id"],
            "category": row["category"],
            "verdict": verdict,
            "note": " ".join(text.split("\n")[1:])[:300],
            "judged_at": dt.date.today().isoformat(),
            "rater": model,
        }
        save_judgments(path, judgments)
        print(f"  {index:>2}/{len(pending)} {row['id']}  {verdict}", flush=True)

    counts = collections.Counter(row["verdict"] for row in judgments.values())
    print(f"\n{len(judgments)} juicios en {path.name}: {dict(counts)}")


def cohens_kappa(pairs: list[tuple[str, str]]) -> dict:
    """Agreement between two raters, discounting chance agreement.

    Two lenient raters who approve almost everything agree often without that
    agreement meaning anything; kappa subtracts the agreement their marginal
    rates alone would produce.
    """
    n = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / n

    labels = {label for pair in pairs for label in pair}
    expected = 0.0
    for label in labels:
        left_rate = sum(1 for left, _ in pairs if left == label) / n
        right_rate = sum(1 for _, right in pairs if right == label) / n
        expected += left_rate * right_rate

    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    return {"n": n, "observed": observed, "expected": expected, "kappa": kappa}


def interpret(kappa: float) -> str:
    for threshold, label in ((0.81, "casi perfecto"), (0.61, "sustancial"),
                             (0.41, "moderado"), (0.21, "aceptable"),
                             (0.0, "escaso")):
        if kappa >= threshold:
            return label
    return "peor que el azar"


def report_kappa(model_name: str) -> None:
    human = load_judgments(HUMAN_PATH)
    model = load_judgments(model_path(model_name))
    shared = sorted(set(human) & set(model))

    # Borderline cases the human could not call are excluded rather than forced
    # into a bucket: guessing on them would inflate or deflate agreement.
    usable = [key for key in shared if human[key]["verdict"] != UNSURE]
    if not usable:
        raise SystemExit(f"Faltan juicios de {model_name}. Corre --model primero.")

    pairs = [(human[key]["verdict"], model[key]["verdict"]) for key in usable]
    stats = cohens_kappa(pairs)

    print(f"\nACUERDO HUMANO vs {model[usable[0]]['rater']}")
    print("-" * 62)
    print(f"  respuestas juzgadas por ambos   {len(shared)}")
    print(f"  excluidas por 'dudosa'          {len(shared) - len(usable)}")
    print(f"  acuerdo observado               {stats['observed']:.3f}")
    print(f"  acuerdo esperado por azar       {stats['expected']:.3f}")
    print(f"  Cohen's kappa                   {stats['kappa']:.3f}"
          f"   ({interpret(stats['kappa'])})")

    print("\n  matriz de confusion")
    print(f"  {'':<14}{'juez: fiel':>14}{'juez: no fiel':>16}")
    for human_label in (FAITHFUL, UNFAITHFUL):
        row = [sum(1 for h, m in pairs if h == human_label and m == model_label)
               for model_label in (FAITHFUL, UNFAITHFUL)]
        print(f"  humano: {human_label:<6}{row[0]:>14}{row[1]:>16}")

    disagreements = [key for key in usable
                     if human[key]["verdict"] != model[key]["verdict"]]
    if disagreements:
        print(f"\n  desacuerdos ({len(disagreements)}):")
        for key in disagreements:
            print(f"    {key}: humano={human[key]['verdict']}, "
                  f"juez={model[key]['verdict']}")
            if human[key].get("note"):
                print(f"        nota: {human[key]['note'][:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Juzgar fidelidad y validar al juez.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--human", action="store_true", help="juzgar tu mismo")
    mode.add_argument("--model", action="store_true", help="que juzgue el modelo")
    mode.add_argument("--kappa", action="store_true", help="comparar ambos")
    parser.add_argument("--answers", help="archivo de respuestas a juzgar")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--provider", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--judge-model")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    # Default judge depends on the provider: the local model, or the cheap
    # frontier model reserved for the paid comparison run.
    judge = args.judge_model or (LOCAL_MODEL if args.provider == "ollama"
                                 else JUDGE_MODEL)

    if args.kappa:
        report_kappa(judge)
        return

    path = Path(args.answers) if args.answers else latest_answers()
    rows = load_answers(path)
    print(f"fuente: {path.name}  ({len(rows)} respuestas no abstenidas)")

    if args.progress:
        done = load_judgments(HUMAN_PATH if args.human else model_path(judge))
        counts = collections.Counter(row["verdict"] for row in done.values())
        print(f"juzgadas: {len(done)}/{len(rows)}  {dict(counts)}")
        return

    if args.human:
        judge_human(rows, load_chunk_lookup(), args.limit)
    else:
        judge_model(rows, load_chunk_lookup(), args.limit, judge, args.provider)


if __name__ == "__main__":
    main()
