"""Turn raw postings into a tidy table with tools, role type and experience.

Three decisions worth defending, because the whole analysis rests on them:

1. Tools are matched from a controlled vocabulary with word boundaries, not by
   naive substring search. Without boundaries "R" matches every word with an r
   in it and "Go" matches "Google"; the resulting chart would be nonsense.

2. A posting is classified by the tasks it lists, never by its title. That is
   the actual research question: how often does a "Data Scientist" title
   describe analyst work? Classifying by title would assume the answer.

3. Postings whose tasks mix profiles are labelled "mixto" rather than forced
   into a bucket. Forcing them would inflate whichever headline is wanted, and
   a number produced that way cannot be defended.
"""

import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Controlled vocabulary: canonical name -> regex alternatives.
# Accents are stripped and text is lowercased before matching, so patterns here
# are written without accents.
TOOL_PATTERNS = {
    "SQL": r"\bsql\b",
    "Python": r"\bpython\b",
    "Excel": r"\bexcel\b",
    "Power BI": r"\bpower\s?bi\b",
    "Tableau": r"\btableau\b",
    "Looker": r"\blooker\b|\blooker studio\b|\bdata studio\b",
    "R": r"(?<![a-z0-9])r(?![a-z0-9])(?=\s*(?:,|/|\)|\bo\b|\by\b|$))|\blenguaje r\b|\brstudio\b",
    "pandas": r"\bpandas\b",
    "scikit-learn": r"\bscikit[- ]?learn\b|\bsklearn\b",
    "TensorFlow": r"\btensorflow\b",
    "PyTorch": r"\bpytorch\b",
    "Spark": r"\bspark\b|\bpyspark\b|\bdatabricks\b",
    "AWS": r"\baws\b|\bamazon web services\b|\bredshift\b|\bsagemaker\b",
    "Azure": r"\bazure\b|\bmicrosoft fabric\b|\bsynapse\b",
    "GCP": r"\bgcp\b|\bgoogle cloud\b|\bbigquery\b",
    "Snowflake": r"\bsnowflake\b",
    "Airflow": r"\bairflow\b",
    "dbt": r"\bdbt\b",
    "Docker": r"\bdocker\b|\bkubernetes\b",
    "Git": r"\bgit\b|\bgithub\b|\bgitlab\b",
    "NoSQL": r"\bnosql\b|\bmongodb\b|\bcassandra\b|\bdynamodb\b",
    "Estadística": r"\bestadistica\b|\bstatistics\b|\binferencia\b|\bregresion\b",
    "Deep learning": r"\bdeep learning\b|\bredes neuronales\b|\bneural network",
    "LLM / GenAI": r"\bllm\b|\bgenai\b|\bia generativa\b|\bgenerative ai\b|\brag\b",
    "MLOps": r"\bmlops\b|\bmodelos en produccion\b|\bmodel deployment\b",
}

# Task signals per profile. These read the `functions` field, which is what the
# employer says the person will actually do.
#
# Patterns are deliberately written against word stems ("arquitectura(s)?",
# "modelo(s)?") rather than exact phrases. A first pass using exact phrases
# missed obvious matches on plurals alone — "arquitecturas de datos" did not
# match "arquitectura de datos" — and left half the sample unclassified.
#
# The fourth profile, data governance, was added after reading the postings the
# first pass could not classify: master data, metadata, quality and lineage are
# a large and distinct family here, and collapsing them into engineering would
# have hidden a real finding.
# Every pattern carries its English equivalent. Postings from Santiago, Austin
# and Berlin must be scored by identical rules, or a difference between regions
# would just be a difference in how well the vocabulary covers each language —
# which would look exactly like a real finding and would not be one.
ROLE_SIGNALS = {
    "ciencia_datos": [
        r"\bmachine learning\b|\baprendizaje automatico\b",
        r"\bmodelo(s)?\b.{0,25}(predictiv|machine learning|\bml\b|estadistic|analitic)",
        r"\b(predictive|statistical|ml)\s+model",
        r"\bmodelado\b.{0,15}(predictivo|estadistico|de machine)",
        r"\b(entrenar|entrenamiento|desarrollar|construir|implementar)\b.{0,35}\bmodelo(s)?\b",
        r"\b(train|build|develop|deploy)\w*\b.{0,35}\bmodels?\b",
        r"\bpredic(cion|ciones|ir|tiv)|\bpredict(ion|ive|ing)?\b|\bforecast",
        r"\bexperimento(s)?\b|\btest(s)? a/b\b|\ba/b test|\bexperimentacion\b"
        r"|\bexperimentation\b|\bab test",
        r"\binferencia\b|\bestadistica(s)? (avanzada|inferencial|aplicada)\b"
        r"|\bstatistical inference\b|\bhypothesis test",
        r"\balgoritmo(s)?\b.{0,30}(aprendizaje|clasificacion|clustering|recomendacion|ml)"
        r"|\balgorithms?\b.{0,30}(learning|classification|clustering|recommendation)",
        r"\bdeep learning\b|\bredes neuronales\b|\bneural network|\bnlp\b"
        r"|\bvision por computador|\bcomputer vision\b",
        r"\bciencia de datos\b|\bdata science\b",
        r"\bsegmentacion\b|\bclustering\b|\bsegmentation\b",
    ],
    "analitica": [
        r"\bdashboard|\btablero(s)? de\b",
        r"\breporte(s)?\b|\breporteria\b|\breporting\b|\binforme(s)?\b|\breports?\b",
        r"\bkpi|\bindicador(es)?\b|\bmetrica(s)?\b|\bmetrics?\b",
        r"\bpower\s?bi\b|\btableau\b|\blooker\b|\bqlik\b",
        r"\bvisualizacion(es)?\b|\bvisuali[sz]ation",
        r"\banali(sis|zar)\b.{0,30}(datos|informacion|comportamiento|volumenes)"
        r"|\banaly(sis|ze|se|zing|sing)\b.{0,30}(data|information|behaviou?r|trends?)",
        r"\bconsulta(s)?\b.{0,15}sql|\bqueries\b|\bexplotacion de datos\b|\bad hoc\b",
        r"\btoma de decisiones\b|\bdecision[- ]making\b|\bdata[- ]driven decision",
        r"\bhallazgo(s)?\b|\binsight",
    ],
    "ingenieria": [
        r"\bpipeline",
        r"\betl\b|\belt\b",
        r"\bingesta\b|\bintegracion de\b.{0,15}(datos|fuentes)|\bfuentes de datos\b"
        r"|\bingest(ion)?\b|\bdata integration\b|\bdata sources?\b",
        r"\borquestacion\b|\bairflow\b|\bdagster\b|\bflujos de trabajo\b"
        r"|\borchestrat|\bworkflows?\b",
        r"\bdata warehouse\b|\bdata\s?lake\b|\bdatalake\b|\balmacenamiento de datos\b",
        r"\barquitectura(s)? de datos\b|\barquitectura(s)?\b.{0,20}escalable"
        r"|\bdata architecture\b|\bscalable\b.{0,20}architect",
        r"\bmodelado\b.{0,20}(de datos|dimensional)|\bdiseno de datos\b"
        r"|\bdata model(l)?ing\b|\bdimensional model",
        r"\bspark\b|\bkafka\b|\bdatabricks\b",
        r"\bprocesamiento de datos\b|\bmigracion de datos\b|\bdata processing\b"
        r"|\bdata migration\b",
    ],
    "gobierno_datos": [
        r"\bgobierno de datos\b|\bdata governance\b",
        r"\bcalidad de (los )?datos\b|\bdata quality\b",
        r"\bmetadato(s)?\b|\bmetadata\b",
        r"\bdatos maestros\b|\bmaster data\b|\bmdm\b",
        r"\btrazabilidad\b|\blinaje\b|\blineage\b",
        r"\bpolitica(s)?\b.{0,25}dato|\blineamiento(s)?\b.{0,25}dato"
        r"|\bdata polic(y|ies)\b|\bdata standards\b",
        r"\bcatalogo de datos\b|\bdiccionario de datos\b|\bdata catalog",
        r"\bcumplimiento\b|\bnormativa\b|\bgdpr\b|\bcompliance\b|\bregulatory\b",
    ],
}

# Non-technical skills, tracked separately: the claim that these matter more
# than they are advertised is only worth making if it is measured.
SOFT_SIGNALS = {
    "Comunicar resultados": r"\bcomunicar\b|\bcomunicacion\b|\bpresentar\b.{0,25}(resultado|hallazgo)"
                            r"|\bstorytelling\b|\bcommunicat(e|ion|ing)\b"
                            r"|\bpresent\w*\b.{0,25}(results?|findings?)",
    "Stakeholders / negocio": r"\bstakeholder|\bareas de negocio\b|\bequipos de negocio\b"
                              r"|\bnecesidades del negocio\b|\bcliente interno\b"
                              r"|\bbusiness (needs|teams|partners|units)\b|\bcross[- ]functional\b",
    "Trabajo en equipo": r"\btrabajo en equipo\b|\bcolabora|\bcollaborat|\bteamwork\b",
}

# Order matters: the first pattern that matches wins, so the most specific
# titles are listed first. Portuguese ("cientista") appears because the board
# also lists Brazilian postings.
TITLE_BUCKETS = [
    (
        "Data Scientist",
        r"data scientist|cientific[oa] de datos|cientista de dados|cientista de datos"
        r"|scientist.{0,15}(datos|data)",
    ),
    (
        "ML / AI Engineer",
        r"machine learning engineer|\bml engineer\b|\bai engineer\b|mlops"
        r"|ingenier[oa].{0,15}(machine learning|inteligencia artificial|\bia\b)",
    ),
    (
        "Data Engineer / Arquitecto",
        r"data engineer|ingenier[oa] de datos|data architect|arquitect[oa].{0,12}datos"
        r"|databricks engineer|data platform|data layer|big data",
    ),
    (
        "Data Analyst / BI",
        r"data analyst|analista.{0,15}datos|business intelligence|\bbi\b|analytics engineer"
        r"|analista bi|data analytics|analytics (specialist|manager|lead)",
    ),
    (
        "Business Analyst",
        r"business analyst|analista.{0,12}(negocio|funcional)|product analyst",
    ),
    (
        "Data Governance / Calidad",
        r"data quality|data steward|gobierno de datos|data management|metadatos"
        r"|datos maestros|data governance",
    ),
]

# Categories the board itself assigns. Used as the primary scope filter: the
# broad search queries pull in programming, marketing and sales postings that
# merely mention "analytics" somewhere in the text.
DATA_CATEGORIES = {"Data Science / Analytics", "Machine Learning & AI"}

TAG_RE = re.compile(r"<[^>]+>")
YEARS_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|\s*(?:a|to|-)\s*\d{1,2})?\s*(?:anos|years|yrs)", re.IGNORECASE
)

# Seniority labels differ per board; these map them onto one comparable scale.
SENIORITY_MAP = {
    "sin experiencia": "Sin experiencia",
    "internship": "Sin experiencia",
    "entry level": "Junior",
    "entry-level, junior": "Junior",
    "entry-level": "Junior",
    "junior": "Junior",
    "semi senior": "Semi Senior",
    "midweight": "Semi Senior",
    "mid level": "Semi Senior",
    "senior": "Senior",
    "senior level": "Senior",
    "expert": "Experto / Director",
    "director": "Experto / Director",
    "management": "Experto / Director",
}


def map_seniority(raw):
    """Collapse each board's seniority vocabulary onto one scale."""
    if not raw:
        return None
    text = str(raw).strip().lower()
    if text in SENIORITY_MAP:
        return SENIORITY_MAP[text]
    for key, value in SENIORITY_MAP.items():
        if key in text:
            return value
    return None


def normalize(text):
    """Strip HTML, unescape entities, drop accents and lowercase."""
    if not text:
        return ""
    text = TAG_RE.sub(" ", str(text))
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).lower()


def bucket_title(title_norm):
    """Group the free-text title into a comparable role family."""
    for label, pattern in TITLE_BUCKETS:
        if re.search(pattern, title_norm):
            return label
    return "Otro"


def classify_by_tasks(tasks_norm, fallback_norm):
    """Return (profile, per-profile signal counts) based on listed tasks.

    The tasks field is preferred; the description is only used when a posting
    left the tasks empty. Which source was used is recorded in the output so a
    reader can filter those rows out.
    """
    source = "functions"
    text = tasks_norm
    if len(text) < 80:  # effectively empty tasks field
        text = fallback_norm
        source = "description"

    counts = {
        profile: sum(1 for pattern in patterns if re.search(pattern, text))
        for profile, patterns in ROLE_SIGNALS.items()
    }

    ordered = sorted(counts.values(), reverse=True)
    top, runner_up = ordered[0], ordered[1]
    leaders = [profile for profile, count in counts.items() if count == top]

    # Two thresholds, kept separate because they answer different questions.
    # Total evidence decides whether the posting says anything at all about the
    # work; the margin decides whether one profile clearly dominates. Without
    # the first test, a posting with a single vague keyword would be labelled
    # as confidently as one that lists eight.
    if top == 0:
        return "sin_señal", counts, source
    if sum(counts.values()) < 2:
        return "señal_debil", counts, source
    if len(leaders) > 1 or top - runner_up < 2:
        return "mixto", counts, source
    return leaders[0], counts, source


def extract_years(text_norm):
    """Return the smallest experience figure mentioned, if any.

    The smallest is used on purpose: a posting asking for "3 to 5 years" filters
    at 3, and taking the upper bound would overstate the barrier.
    """
    matches = [int(value) for value in YEARS_RE.findall(text_norm)]
    matches = [value for value in matches if 0 < value <= 20]
    return min(matches) if matches else None


def nested_name(attributes, key):
    """Read an expanded relation's name (seniority, modality, company)."""
    node = (attributes.get(key) or {}).get("data") or {}
    return (node.get("attributes") or {}).get("name")


def normalize_getonbrd(job):
    """Flatten a Get on Board posting into the common record shape."""
    attributes = job["attributes"]
    return {
        "id": f"getonbrd:{job['id']}",
        "fuente": "getonbrd",
        "titulo": attributes.get("title"),
        "empresa": nested_name(attributes, "company"),
        "ubicacion": ", ".join(attributes.get("countries") or []) or None,
        # Every Get on Board posting is LATAM, including the ones marked
        # "Remote": it is a LATAM board with LATAM companies hiring for LATAM.
        # Filing those as "global" would move ~58 postings out of the region
        # they belong to and corrupt the comparison this analysis exists for.
        "region": "LATAM",
        "seniority_raw": nested_name(attributes, "seniority"),
        "funciones": attributes.get("functions") or "",
        "descripcion": " ".join(
            str(attributes.get(field) or "")
            for field in ("description", "projects", "desirable")
        ),
        "salario_min": attributes.get("min_salary"),
        "salario_max": attributes.get("max_salary"),
        "salario_periodo": "mensual",
        "url": f"https://www.getonbrd.com/jobs/{job['id']}",
        "published_at": attributes.get("published_at"),
        "categoria": attributes.get("category_name"),
        "modalidad": nested_name(attributes, "modality"),
        "remoto": attributes.get("remote_modality"),
        "postulaciones": attributes.get("applications_count"),
    }


def load_records():
    """Read every capture and return one normalised list plus the capture date."""
    records, dates = [], []

    for path in sorted(RAW_DIR.glob("getonbrd_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records += [normalize_getonbrd(job) for job in payload["jobs"]]
        dates.append(payload["captured_at"][:10])
        print(f"  {path.name}: {payload['n_jobs']} postings")

    for path in sorted(RAW_DIR.glob("global_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records += payload["jobs"]
        dates.append(payload["captured_at"][:10])
        print(f"  {path.name}: {payload['n_jobs']} postings")

    if not records:
        raise SystemExit("No raw captures found. Run collect.py first.")
    return records, max(dates)


def build_row(record):
    title_norm = normalize(record.get("titulo"))
    tasks_norm = normalize(record.get("funciones"))
    description_norm = normalize(record.get("descripcion"))
    full_norm = " ".join([title_norm, tasks_norm, description_norm])

    profile, signal_counts, signal_source = classify_by_tasks(
        tasks_norm, description_norm
    )

    familia = bucket_title(title_norm)
    categoria = record.get("categoria")

    # Scope rule, applied here so every row carries its own justification:
    # a posting is in scope if the board filed it under a data category, or if
    # its title names a data role. The broad search queries were needed to find
    # data roles filed elsewhere, but they also drag in unrelated postings that
    # merely mention "analytics"; those are kept in the CSV and excluded from
    # the analysis, so the discarded count can be reported honestly.
    in_category = categoria in DATA_CATEGORIES
    in_title = familia != "Otro"
    row = {
        "id": record["id"],
        "fuente": record.get("fuente"),
        "region": record.get("region"),
        "titulo": record.get("titulo"),
        "familia_titulo": familia,
        "en_alcance": in_category or in_title,
        # Cross-region comparison uses this stricter rule instead: the title
        # must name a data role. The board-category signal only exists for Get
        # on Board, so including it would apply a looser filter to LATAM than
        # to the other regions — and any difference that produced would be an
        # artefact of the filter, not of the market.
        "comparable_regiones": in_title,
        "motivo_alcance": (
            "categoria+titulo"
            if in_category and in_title
            else "categoria" if in_category
            else "titulo" if in_title
            else "fuera_de_alcance"
        ),
        "empresa": record.get("empresa"),
        "pais": record.get("ubicacion"),
        "seniority": map_seniority(record.get("seniority_raw")),
        "seniority_raw": record.get("seniority_raw"),
        "modalidad": record.get("modalidad"),
        "remoto": record.get("remoto"),
        "categoria": categoria,
        "url": record.get("url"),
        "salario_min": record.get("salario_min"),
        "salario_max": record.get("salario_max"),
        "salario_periodo": record.get("salario_periodo"),
        "postulaciones": record.get("postulaciones"),
        "publicado_ts": record.get("published_at"),
        "perfil_por_tareas": profile,
        "fuente_señal": signal_source,
        "anios_experiencia": extract_years(full_norm),
        "n_chars_funciones": len(tasks_norm),
    }
    for profile_name, count in signal_counts.items():
        row[f"señales_{profile_name}"] = count
    for tool, pattern in TOOL_PATTERNS.items():
        row[f"tool_{tool}"] = bool(re.search(pattern, full_norm))
    for skill, pattern in SOFT_SIGNALS.items():
        row[f"soft_{skill}"] = bool(re.search(pattern, full_norm))
    return row


def main():
    print("Reading captures:")
    records, captured_at = load_records()

    frame = pd.DataFrame([build_row(record) for record in records])
    frame["fecha_captura"] = captured_at

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "ofertas.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    scoped = frame[frame["en_alcance"]]
    comparable = frame[frame["comparable_regiones"]]
    print(f"\n{len(frame)} rows -> {output_path}")
    print(f"  in scope: {len(scoped)} | discarded: {len(frame) - len(scoped)}")
    print(f"  comparable across regions (title names a data role): {len(comparable)}")
    print("\nBy region (comparable set):")
    print(comparable["region"].value_counts().to_string())
    print("\nBy title family (comparable set):")
    print(comparable["familia_titulo"].value_counts().to_string())
    print("\nBy profile inferred from tasks (comparable set):")
    print(comparable["perfil_por_tareas"].value_counts().to_string())


if __name__ == "__main__":
    main()
