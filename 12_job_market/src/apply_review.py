"""Fold the manual review back into the dataset.

The classifier is a starting point, not the verdict. This script reads the
reviewed workbook and lets the human decision win, then writes a separate file
so re-running the pipeline never overwrites the review.

Rules applied, in this order:
  - `perfil_corregido` filled in  -> that value replaces the classifier's label.
  - `revisado_ok` = "no"          -> the row is dropped from the analysis.
  - anything else                 -> the classifier's label stands.

Every override is recorded in `perfil_origen`, so the share of the final numbers
that came from a person rather than a regex can be reported honestly.
"""

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

BASE_DIR = Path(__file__).resolve().parent.parent
WORKBOOK = BASE_DIR / "outputs" / "dataset_ofertas.xlsx"
PROCESSED = BASE_DIR / "data" / "processed" / "ofertas.csv"
REVIEWED = BASE_DIR / "data" / "processed" / "ofertas_revisado.csv"

VALID_PROFILES = {
    "ciencia_datos",
    "analitica",
    "ingenieria",
    "gobierno_datos",
    "mixto",
    "señal_debil",
    "sin_señal",
}


def normalize_cell(value):
    """Empty-ish cells come back from Excel as NaN, '' or whitespace."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def read_sheet(path, sheet_name):
    """Read one sheet with openpyxl directly.

    pandas.read_excel requires openpyxl >= 3.1, and the environment here has
    3.0.9 — which writes fine but fails that check. Reading through openpyxl
    keeps the pipeline working without forcing a dependency upgrade.
    """
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = workbook[sheet_name].iter_rows(values_only=True)
    header = next(rows)
    frame = pd.DataFrame(list(rows), columns=header)
    workbook.close()
    return frame


def load_review():
    """Read whichever review file was filled in.

    The CSV wins when it is the newer of the two: it exists so the review can
    be done without an office suite, and the reviewer should not have to
    remember which format the pipeline prefers.
    """
    csv_path = WORKBOOK.with_name("revision_ofertas.csv")
    have_csv, have_xlsx = csv_path.exists(), WORKBOOK.exists()
    if not (have_csv or have_xlsx):
        raise SystemExit(f"No review file in {WORKBOOK.parent}. Run export_excel.py first.")

    if have_csv and (
        not have_xlsx or csv_path.stat().st_mtime >= WORKBOOK.stat().st_mtime
    ):
        print(f"Reading review from {csv_path.name}")
        return pd.read_csv(csv_path)
    print(f"Reading review from {WORKBOOK.name}")
    return read_sheet(WORKBOOK, "ofertas")


def main():
    review = load_review()
    frame = pd.read_csv(PROCESSED)
    frame = frame[frame["en_alcance"]].copy()

    for column in ("revisado_ok", "perfil_corregido", "notas"):
        if column not in review.columns:
            review[column] = ""
        review[column] = review[column].map(normalize_cell)

    corrections = review.set_index("id")["perfil_corregido"].to_dict()
    verdicts = review.set_index("id")["revisado_ok"].to_dict()
    notes = review.set_index("id")["notas"].to_dict()

    unknown = {
        value for value in corrections.values() if value and value not in VALID_PROFILES
    }
    if unknown:
        raise SystemExit(
            "Unrecognised values in perfil_corregido: "
            + ", ".join(sorted(unknown))
            + f"\nValid values: {', '.join(sorted(VALID_PROFILES))}"
        )

    frame["perfil_origen"] = "clasificador"
    frame["nota_revision"] = frame["id"].map(notes).fillna("")

    corrected = frame["id"].map(corrections).fillna("")
    has_correction = corrected.isin(VALID_PROFILES)
    frame.loc[has_correction, "perfil_por_tareas"] = corrected[has_correction]
    frame.loc[has_correction, "perfil_origen"] = "revision_manual"

    confirmed = frame["id"].map(verdicts).fillna("")
    frame.loc[confirmed.isin({"si", "sí", "ok", "yes"}), "perfil_origen"] = "confirmado"

    rejected = confirmed.isin({"no", "descartar", "fuera"})
    dropped = int(rejected.sum())
    frame = frame[~rejected]

    frame.to_csv(REVIEWED, index=False, encoding="utf-8-sig")

    print(f"{len(frame)} postings -> {REVIEWED}")
    print(f"  corrected by hand: {int(has_correction.sum())}")
    print(f"  confirmed as correct: {int((frame['perfil_origen'] == 'confirmado').sum())}")
    print(f"  dropped from analysis: {dropped}")
    print("\nRun analyze.py again: it prefers the reviewed file when it exists.")


if __name__ == "__main__":
    main()
