"""Export the dataset to a single Excel workbook for manual review.

The point of this file is auditing, not analysis. Every row carries the raw
text the classifier read and a link back to the live posting, so a
disagreement with any label can be settled by looking at the source instead of
by trusting the regexes.

Two columns are left deliberately empty — `revisado_ok` and `perfil_corregido`
— for the reviewer to fill in. `apply_review.py` reads them back, so manual
corrections survive a re-run of the pipeline instead of being overwritten.
"""

import html
import json
import re
from pathlib import Path

import pandas as pd

from extract import load_records

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed" / "ofertas.csv"
OUTPUT = BASE_DIR / "outputs" / "dataset_ofertas.xlsx"

JOB_URL = "https://www.getonbrd.com/jobs/{}"
TAG_RE = re.compile(r"<[^>]+>")

REVIEW_COLUMNS = [
    "id",
    "fuente",
    "region",
    "url",
    "titulo",
    "empresa",
    "familia_titulo",
    "perfil_por_tareas",
    "revisado_ok",
    "perfil_corregido",
    "notas",
    "señales_ciencia_datos",
    "señales_analitica",
    "señales_ingenieria",
    "señales_gobierno_datos",
    "seniority",
    "seniority_raw",
    "anios_experiencia",
    "pais",
    "salario_min",
    "salario_max",
    "salario_periodo",
    "postulaciones",
    "categoria",
    "motivo_alcance",
    "funciones_texto",
    "descripcion_texto",
]


def clean_text(value):
    """Plain readable text: Excel cells cannot show HTML."""
    if not value:
        return ""
    text = TAG_RE.sub(" ", str(value))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    # Excel refuses to store a cell longer than 32767 characters.
    return text[:32000]


def load_texts():
    """Map each job id to its readable tasks and description.

    Reuses `extract.load_records` so the text shown next to a label is exactly
    the text the classifier read — across all four sources, not just one.
    """
    records, captured_at = load_records()
    texts = {
        record["id"]: {
            "funciones_texto": clean_text(record.get("funciones")),
            "descripcion_texto": clean_text(record.get("descripcion")),
        }
        for record in records
    }
    return texts, captured_at


def autosize(worksheet, frame, wide_columns=()):
    """Readable column widths; long text columns get a fixed cap."""
    for index, column in enumerate(frame.columns, start=1):
        letter = worksheet.cell(row=1, column=index).column_letter
        if column in wide_columns:
            worksheet.column_dimensions[letter].width = 70
            continue
        longest = max(
            [len(str(column))] + [len(str(value)) for value in frame[column].head(200)]
        )
        worksheet.column_dimensions[letter].width = min(max(longest + 2, 10), 38)
    worksheet.freeze_panes = "A2"


def build_summaries(scoped):
    """The aggregate tables, so the numbers can be checked without pandas."""
    tools = [c for c in scoped.columns if c.startswith("tool_")]
    herramientas = pd.DataFrame(
        {
            "herramienta": [c.replace("tool_", "") for c in tools],
            "n_ofertas": [int(scoped[c].sum()) for c in tools],
            "pct_ofertas": [round(scoped[c].mean() * 100, 1) for c in tools],
        }
    ).sort_values("pct_ofertas", ascending=False)

    cruce = (
        pd.crosstab(scoped["familia_titulo"], scoped["perfil_por_tareas"])
        .reset_index()
        .rename(columns={"familia_titulo": "familia de título"})
    )

    familias = (
        scoped["familia_titulo"]
        .value_counts()
        .rename_axis("familia de título")
        .reset_index(name="n_ofertas")
    )
    familias["pct"] = (familias["n_ofertas"] / len(scoped) * 100).round(1)

    return herramientas, cruce, familias


def build_dictionary():
    """What each column means, so the file is readable a month from now."""
    rows = [
        ("id", "Identificador único, con el prefijo del portal de origen."),
        ("fuente", "Portal del que salió la oferta: getonbrd, jobicy, themuse, arbeitnow."),
        ("region", "LATAM, EE. UU. / Canadá o Europa, deducida de la ubicación."),
        ("url", "Enlace a la oferta original para verificar."),
        ("familia_titulo", "Familia de puesto inferida SOLO del título."),
        (
            "perfil_por_tareas",
            "Perfil inferido de las TAREAS descritas. Valores: ciencia_datos, "
            "analitica, ingenieria, gobierno_datos, mixto, señal_debil, sin_señal.",
        ),
        ("revisado_ok", "COLUMNA PARA TI. Escribe 'si' si la clasificación es correcta."),
        (
            "perfil_corregido",
            "COLUMNA PARA TI. Si no estás de acuerdo, escribe aquí el perfil correcto.",
        ),
        ("notas", "COLUMNA PARA TI. Comentario libre."),
        (
            "señales_*",
            "Cuántos patrones de cada perfil se encontraron en las tareas. "
            "Es la evidencia detrás de perfil_por_tareas.",
        ),
        ("anios_experiencia", "Menor cifra de años mencionada en el texto. Vacío = no menciona."),
        ("salario_min", "Salario declarado. OJO: lee salario_periodo antes de compararlo."),
        ("salario_periodo", "'mensual' (Get on Board) o 'anual' (Jobicy). NO se pueden mezclar."),
        ("postulaciones", "Cuánta gente postuló. Solo existe en Get on Board."),
        ("motivo_alcance", "Por qué la oferta se consideró de datos: categoría, título o ambos."),
        ("funciones_texto", "Las tareas, en texto plano. Es lo que leyó el clasificador."),
        ("descripcion_texto", "Descripción, proyecto y deseables, en texto plano."),
    ]
    return pd.DataFrame(rows, columns=["columna", "qué significa"])


def main():
    frame = pd.read_csv(PROCESSED)
    texts, captured_at = load_texts()

    frame["funciones_texto"] = frame["id"].map(
        lambda job_id: texts.get(job_id, {}).get("funciones_texto", "")
    )
    frame["descripcion_texto"] = frame["id"].map(
        lambda job_id: texts.get(job_id, {}).get("descripcion_texto", "")
    )
    for column in ("revisado_ok", "perfil_corregido", "notas"):
        frame[column] = ""

    scoped = frame[frame["comparable_regiones"]].copy()
    excluded = frame[~frame["comparable_regiones"]].copy()
    herramientas, cruce, familias = build_summaries(scoped)

    tool_columns = [c for c in scoped.columns if c.startswith("tool_")]
    review = scoped[REVIEW_COLUMNS + tool_columns].sort_values(
        ["familia_titulo", "perfil_por_tareas"]
    )

    notes = pd.DataFrame(
        [
            (
                "Fuentes",
                "APIs públicas: Get on Board (LATAM), Jobicy (remoto global), "
                "The Muse (EE. UU.), Arbeitnow (Europa).",
            ),
            ("Fecha de captura", captured_at),
            ("Ofertas capturadas", len(frame)),
            ("Ofertas de datos comparables entre regiones", len(scoped)),
            ("Descartadas por no ser de datos", len(excluded)),
            (
                "Criterio de comparabilidad",
                "El título nombra un puesto de datos. Se aplica igual en las tres "
                "regiones para que la diferencia no la cree el propio filtro.",
            ),
            (
                "Qué dicen estos datos",
                "Qué PIDE una empresa al publicar una oferta.",
            ),
            (
                "Qué NO dicen",
                "Qué se hace realmente en el puesto. Son cosas distintas.",
            ),
            (
                "Sesgo principal",
                "Cada portal tiene el suyo. Por eso las conclusiones fuertes son "
                "las que coinciden en las tres regiones: "
                + " · ".join(
                    f"{region} {count}"
                    for region, count in scoped["region"].value_counts().items()
                ),
            ),
            (
                "Cómo revisar",
                "En la hoja 'ofertas': lee funciones_texto y marca revisado_ok, "
                "o corrige en perfil_corregido. Luego corre src/apply_review.py.",
            ),
        ],
        columns=["campo", "valor"],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        sheets = [
            ("Leeme", notes, ()),
            ("ofertas", review, ("funciones_texto", "descripcion_texto", "url")),
            ("resumen_herramientas", herramientas, ()),
            ("titulo_vs_tareas", cruce, ()),
            ("familias", familias, ()),
            ("diccionario", build_dictionary(), ("qué significa",)),
            (
                "descartadas",
                excluded[["id", "fuente", "region", "url", "titulo", "categoria"]],
                ("url",),
            ),
        ]
        for name, data, wide in sheets:
            data.to_excel(writer, sheet_name=name, index=False)
            autosize(writer.sheets[name], data, wide)

    # CSV fallback: the workbook needs Google Sheets, LibreOffice or Excel to
    # open, and a CSV opens in anything. Same columns, same review workflow —
    # apply_review.py reads whichever of the two has been filled in.
    csv_path = OUTPUT.with_name("revision_ofertas.csv")
    review.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"{OUTPUT}")
    print(f"{csv_path}")
    print(f"  ofertas (en alcance): {len(review)}")
    print(f"  descartadas: {len(excluded)}")


if __name__ == "__main__":
    main()
