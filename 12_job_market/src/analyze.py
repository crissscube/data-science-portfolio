"""Produce the charts and the findings report from the processed table.

Charts are sized and styled for video: few marks, direct labels on every bar,
recessive axes, and type large enough to read on a phone.

Two populations are used, and never mixed:

  - `comparable`  — postings whose title names a data role, across LATAM, the
    United States/Canada and Europe. Same inclusion rule everywhere, so the
    regions can be compared without the filter itself creating a difference.
  - `latam`       — the Get on Board capture only. Applications-per-posting and
    monthly salary exist only there; the other boards publish annual figures or
    nothing at all, so any chart using those is labelled LATAM.

Color follows the job it does. Counts and percentages are magnitude, so they get
a single hue; the categorical charts take the fixed slot order of the reference
palette.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed" / "ofertas.csv"
REVIEWED = BASE_DIR / "data" / "processed" / "ofertas_revisado.csv"
OUTPUTS = BASE_DIR / "outputs"

INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e2e1dc"
ACCENT = "#2a78d6"
ACCENT_MUTED = "#a9c8ea"

REGIONS = ["LATAM", "EE. UU. / Canadá", "Europa"]
# Categorical slots 1-3 of the reference palette, in fixed order. Three series
# is also the documented all-pairs cap for that palette.
REGION_COLORS = {"LATAM": "#2a78d6", "EE. UU. / Canadá": "#eb6834", "Europa": "#1baf7a"}

PROFILE_COLORS = {
    "ciencia_datos": "#2a78d6",
    "analitica": "#eb6834",
    "ingenieria": "#1baf7a",
    "gobierno_datos": "#eda100",
    "mixto": "#b8b6ae",
    "señal_debil": "#d6d4cc",
    "sin_señal": "#eceae4",
}
PROFILE_LABELS = {
    "ciencia_datos": "Ciencia de datos",
    "analitica": "Analítica / BI",
    "ingenieria": "Ingeniería de datos",
    "gobierno_datos": "Gobierno de datos",
    "mixto": "Mixto",
    "señal_debil": "Señal débil",
    "sin_señal": "Sin señal",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK,
        "axes.edgecolor": GRID,
    }
)


def style_axes(ax, xgrid=False):
    """Strip the frame down to what a viewer actually needs."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)


def save(fig, name):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path.name}")


def tool_share_by_region(comparable):
    """Percentage of postings mentioning each tool, one column per region."""
    tools = [c for c in comparable.columns if c.startswith("tool_")]
    table = (comparable.groupby("region")[tools].mean() * 100).T
    table.index = [name.replace("tool_", "") for name in table.index]
    return table.reindex(columns=REGIONS).dropna(axis=1, how="all")


def chart_tools(comparable):
    """Tool frequency overall: the chart viewers will pause and screenshot."""
    tools = [c for c in comparable.columns if c.startswith("tool_")]
    share = (comparable[tools].mean() * 100).sort_values()
    share.index = [name.replace("tool_", "") for name in share.index]
    share = share.tail(16)

    fig, ax = plt.subplots(figsize=(13, 10))
    colors = [ACCENT if value >= 50 else ACCENT_MUTED for value in share.values]
    bars = ax.barh(share.index, share.values, color=colors, height=0.72)
    for bar, value in zip(bars, share.values):
        ax.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            va="center",
            fontsize=16,
            color=INK,
        )
    ax.set_xlim(0, max(share.values) * 1.16)
    ax.set_xticks([])
    ax.set_title(
        f"Qué herramientas piden las ofertas de datos\n"
        f"{len(comparable)} ofertas · LATAM, EE. UU. y Europa · "
        f"{comparable['fecha_captura'].iloc[0]}",
        fontsize=21,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    style_axes(ax)
    save(fig, "01_herramientas.png")


def chart_regions(comparable):
    """The question the viewer actually has: does this apply where I live?"""
    table = tool_share_by_region(comparable)
    counts = comparable["region"].value_counts()
    top = table.loc[
        [
            "SQL",
            "Python",
            "AWS",
            "Azure",
            "GCP",
            "Power BI",
            "Spark",
            "LLM / GenAI",
            "PyTorch",
            "Deep learning",
        ]
    ]

    fig, ax = plt.subplots(figsize=(15, 8.5))
    positions = range(len(top))
    width = 0.27
    for offset, region in enumerate(REGIONS):
        ax.bar(
            [p + (offset - 1) * width for p in positions],
            top[region],
            width=width - 0.015,  # surface gap between adjacent bars
            color=REGION_COLORS[region],
            label=f"{region} (n={counts.get(region, 0)})",
        )
    ax.set_xticks(list(positions))
    # Shortened: the full names ran into each other at the right-hand end.
    short_names = {"LLM / GenAI": "IA gen.", "Deep learning": "Deep\nlearning"}
    ax.set_xticklabels(
        [short_names.get(name, name) for name in top.index], fontsize=15
    )
    ax.set_ylim(0, 85)
    ax.set_yticks([])
    for index, tool in enumerate(top.index):
        for offset, region in enumerate(REGIONS):
            value = top.loc[tool, region]
            ax.text(
                index + (offset - 1) * width,
                value + 1.5,
                f"{value:.0f}",
                ha="center",
                fontsize=12,
                color=INK,
            )
    ax.set_title(
        "¿Piden lo mismo en cada región?\n"
        "% de ofertas que menciona cada herramienta",
        fontsize=21,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=15)
    style_axes(ax)
    save(fig, "07_regiones.png")


def chart_region_agreement(comparable):
    """Rank agreement between regions — the evidence behind the claim.

    A scatter of the same tools ranked in two markets makes the argument
    visually: points on the diagonal mean the two markets want the same things
    in the same order. The correlation number alone would be a claim; this is
    the claim with its evidence attached.
    """
    table = tool_share_by_region(comparable)
    correlations = table.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    pairs = [("LATAM", "EE. UU. / Canadá"), ("LATAM", "Europa")]
    for ax, (left, right) in zip(axes, pairs):
        ax.scatter(
            table[left],
            table[right],
            s=110,
            color=ACCENT,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        limit = max(table[left].max(), table[right].max()) * 1.15
        ax.plot([0, limit], [0, limit], color=GRID, linewidth=2, zorder=1)
        # Per-label offsets: SQL and Python sit almost on top of each other in
        # the Europe panel, so a single shared offset renders them overlapping
        # and unreadable.
        offsets = {
            "SQL": (10, -16),
            "Python": (10, 6),
            "Power BI": (10, 4),
            "PyTorch": (10, -14),
            "LLM / GenAI": (10, 4),
            "Excel": (10, -14),
        }
        for tool, offset in offsets.items():
            ax.annotate(
                tool,
                (table.loc[tool, left], table.loc[tool, right]),
                textcoords="offset points",
                xytext=offset,
                fontsize=13,
                color=INK_SOFT,
            )
        ax.set_xlabel(f"% en {left}", fontsize=15)
        ax.set_ylabel(f"% en {right}", fontsize=15)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_title(
            f"ρ de Spearman = {correlations.loc[left, right]:.2f}",
            fontsize=18,
            fontweight="bold",
            loc="left",
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(length=0, labelsize=13)

    fig.suptitle(
        "Cada punto es una herramienta. Cerca de la diagonal = piden lo mismo",
        fontsize=21,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=1.02,
    )
    save(fig, "08_acuerdo_entre_regiones.png")


def chart_families(comparable):
    """How the market is actually split by role title."""
    counts = comparable["familia_titulo"].value_counts().drop("Otro", errors="ignore")
    counts = counts.sort_values()

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = [
        ACCENT if label == "Data Scientist" else ACCENT_MUTED for label in counts.index
    ]
    bars = ax.barh(counts.index, counts.values, color=colors, height=0.7)
    total = len(comparable)
    for bar, value in zip(bars, counts.values):
        ax.text(
            value + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value}  ({value / total * 100:.0f}%)",
            va="center",
            fontsize=16,
            color=INK,
        )
    ax.set_xlim(0, max(counts.values) * 1.25)
    ax.set_xticks([])
    ax.set_title(
        f"De {total} ofertas de datos, solo "
        f"{counts.get('Data Scientist', 0)} dicen «Data Scientist»",
        fontsize=22,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    style_axes(ax)
    save(fig, "02_familias_de_puesto.png")


def chart_title_vs_tasks(comparable):
    """The core question: does the title match the listed tasks?"""
    families = [
        "Data Scientist",
        "ML / AI Engineer",
        "Data Analyst / BI",
        "Business Analyst",
        "Data Engineer / Arquitecto",
        "Data Governance / Calidad",
    ]
    order = [
        "ciencia_datos",
        "analitica",
        "ingenieria",
        "gobierno_datos",
        "mixto",
        "señal_debil",
        "sin_señal",
    ]
    table = (
        pd.crosstab(comparable["familia_titulo"], comparable["perfil_por_tareas"])
        .reindex(index=families)
        .reindex(columns=order, fill_value=0)
        .fillna(0)
    )
    shares = table.div(table.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    left = pd.Series(0.0, index=shares.index)
    for profile in order:
        values = shares[profile]
        ax.barh(
            shares.index,
            values,
            left=left,
            height=0.66,
            color=PROFILE_COLORS[profile],
            label=PROFILE_LABELS[profile],
            edgecolor=SURFACE,
            linewidth=2,
        )
        for family, value, start in zip(shares.index, values, left):
            if value >= 14:
                ax.text(
                    start + value / 2,
                    list(shares.index).index(family),
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=INK
                    if profile in ("mixto", "señal_debil", "sin_señal")
                    else "#ffffff",
                    fontweight="bold",
                )
        left += values

    labels = [f"{family}  (n={int(table.loc[family].sum())})" for family in shares.index]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_title(
        "El título dice una cosa. ¿Qué dicen las tareas?\n"
        "Cada oferta clasificada por lo que describe que hará la persona",
        fontsize=21,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    style_axes(ax)
    save(fig, "03_titulo_vs_tareas.png")


def chart_hype_gap(comparable):
    """What the marketing sells against what the postings ask for."""
    groups = {
        "SQL": ["tool_SQL"],
        "Python": ["tool_Python"],
        "Cloud\nAWS·Azure·GCP": ["tool_AWS", "tool_Azure", "tool_GCP"],
        "Power BI\nTableau": ["tool_Power BI", "tool_Tableau"],
        "IA\ngenerativa": ["tool_LLM / GenAI"],
        "ML clásico\nsklearn": ["tool_scikit-learn", "tool_pandas"],
        "Deep\nlearning": ["tool_Deep learning", "tool_PyTorch", "tool_TensorFlow"],
    }
    series = pd.Series(
        {
            label: comparable[columns].any(axis=1).mean() * 100
            for label, columns in groups.items()
        }
    )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    colors = [ACCENT if value >= 30 else ACCENT_MUTED for value in series.values]
    bars = ax.bar(series.index, series.values, color=colors, width=0.62)
    for bar, value in zip(bars, series.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.0f}%",
            ha="center",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylim(0, max(series.values) * 1.2)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=15)
    ax.set_title(
        "Lo que te venden vs. lo que piden\n"
        f"% de {len(comparable)} ofertas de datos que menciona cada cosa",
        fontsize=21,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    style_axes(ax)
    save(fig, "04_expectativa_vs_realidad.png")


def chart_seniority(comparable):
    """Where the entry door actually is."""
    order = ["Sin experiencia", "Junior", "Semi Senior", "Senior", "Experto / Director"]
    short = {"Sin experiencia": "Sin exp.", "Experto / Director": "Experto"}
    counts = comparable["seniority"].value_counts().reindex(order).fillna(0)
    years = comparable.groupby("seniority")["anios_experiencia"].median().reindex(order)
    counts.index = [short.get(label, label) for label in counts.index]
    years.index = [short.get(label, label) for label in years.index]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    bars = axes[0].bar(
        counts.index,
        counts.values,
        color=[
            ACCENT if label in ("Sin exp.", "Junior") else ACCENT_MUTED
            for label in counts.index
        ],
        width=0.62,
    )
    for bar, value in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{int(value)}",
            ha="center",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
    axes[0].set_ylim(0, max(counts.values) * 1.2)
    axes[0].set_yticks([])
    axes[0].set_title(
        "Cuántas ofertas hay por nivel", fontsize=19, fontweight="bold", loc="left"
    )

    valid = years.dropna()
    bars = axes[1].bar(valid.index, valid.values, color=ACCENT_MUTED, width=0.62)
    for bar, value in zip(bars, valid.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.0f}",
            ha="center",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
    axes[1].set_ylim(0, max(valid.values) * 1.25)
    axes[1].set_yticks([])
    axes[1].set_title(
        "Años de experiencia pedidos (mediana)",
        fontsize=19,
        fontweight="bold",
        loc="left",
    )

    for ax in axes:
        ax.tick_params(axis="x", labelsize=14)
        style_axes(ax)
    fig.suptitle(
        f"La puerta de entrada: {int(counts.get('Junior', 0))} de {len(comparable)} "
        f"ofertas son junior",
        fontsize=22,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=1.03,
    )
    save(fig, "05_puerta_de_entrada.png")


def chart_competition(latam):
    """What it costs to apply. LATAM only — see the module docstring."""
    buckets = (
        pd.cut(
            latam["postulaciones"],
            bins=[-1, 25, 50, 100, 200, 100000],
            labels=["0–25", "26–50", "51–100", "101–200", "más de 200"],
        )
        .value_counts()
        .sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    bars = axes[0].bar(
        [str(label) for label in buckets.index],
        buckets.values,
        color=ACCENT_MUTED,
        width=0.62,
    )
    for bar, value in zip(bars, buckets.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{int(value)}",
            ha="center",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
    axes[0].set_ylim(0, max(buckets.values) * 1.2)
    axes[0].set_yticks([])
    axes[0].set_title(
        f"Postulantes por oferta (mediana: {latam['postulaciones'].median():.0f})",
        fontsize=18,
        fontweight="bold",
        loc="left",
    )

    with_salary = int(latam["salario_min"].notna().sum())
    shares = [with_salary, len(latam) - with_salary]
    bars = axes[1].bar(
        ["Publica salario", "No lo publica"],
        shares,
        color=[ACCENT, ACCENT_MUTED],
        width=0.5,
    )
    for bar, value in zip(bars, shares):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value}  ({value / len(latam) * 100:.0f}%)",
            ha="center",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
    axes[1].set_ylim(0, max(shares) * 1.2)
    axes[1].set_yticks([])
    axes[1].set_title(
        f"Transparencia salarial · mediana "
        f"{latam['salario_min'].median():.0f}–{latam['salario_max'].median():.0f} USD/mes",
        fontsize=18,
        fontweight="bold",
        loc="left",
    )

    for ax in axes:
        ax.tick_params(axis="x", labelsize=14)
        style_axes(ax)
    fig.suptitle(
        f"Lo que cuesta postular · solo LATAM (n={len(latam)})",
        fontsize=22,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=1.03,
    )
    save(fig, "06_competencia_y_salario.png")


def write_report(frame, comparable, latam):
    """Write the numbers to read on camera, with their caveats attached."""
    tools = [c for c in comparable.columns if c.startswith("tool_")]
    share = (comparable[tools].mean() * 100).sort_values(ascending=False)
    share.index = [name.replace("tool_", "") for name in share.index]

    by_region = tool_share_by_region(comparable)
    correlations = by_region.corr(method="spearman")
    region_counts = comparable["region"].value_counts()

    families = comparable["familia_titulo"].value_counts()
    ds = comparable[comparable["familia_titulo"] == "Data Scientist"]
    junior = comparable[comparable["seniority"] == "Junior"]
    salary = latam.dropna(subset=["salario_min"])
    date = comparable["fecha_captura"].iloc[0]

    lines = [
        "# Hallazgos — mercado laboral de datos",
        "",
        f"**Captura:** {date} · **Ofertas descargadas:** {len(frame)} · "
        f"**Ofertas de datos comparables entre regiones:** {len(comparable)}",
        "",
        "**Fuentes:** Get on Board (LATAM), Jobicy (remoto global), "
        "The Muse (EE. UU.), Arbeitnow (Europa). Todas APIs públicas.",
        "",
        "> Generado por `src/analyze.py`. No editar a mano: se regenera.",
        "",
        "## Lo que estos datos NO dicen",
        "",
        "- Dicen qué **pide** una empresa al publicar. No dicen qué se hace en el puesto.",
        f"- Son ofertas **activas** el {date} en cuatro portales, no el mercado entero.",
        "- Cada portal tiene su propio sesgo. Por eso las conclusiones fuertes son "
        "las que **coinciden en las tres regiones**, no las de una sola.",
        "- La clasificación por tareas es un criterio propio, auditable en `src/extract.py`.",
        "",
        "## 0. ¿Piden lo mismo en todo el mundo?",
        "",
        "Ofertas por región: "
        + " · ".join(f"{region} **{region_counts.get(region, 0)}**" for region in REGIONS),
        "",
        "**Correlación de Spearman entre el ranking de herramientas de cada región:**",
        "",
        f"- LATAM ↔ EE. UU./Canadá: **{correlations.loc['LATAM', 'EE. UU. / Canadá']:.2f}**",
        f"- LATAM ↔ Europa: **{correlations.loc['LATAM', 'Europa']:.2f}**",
        f"- EE. UU./Canadá ↔ Europa: **{correlations.loc['EE. UU. / Canadá', 'Europa']:.2f}**",
        "",
        "**El núcleo coincide** — mismas herramientas arriba, en el mismo orden:",
        "",
        "| Herramienta | " + " | ".join(REGIONS) + " |",
        "|---|" + "---|" * len(REGIONS),
    ]
    for tool in ["SQL", "Python", "AWS", "Azure", "GCP", "Power BI", "Excel",
                 "Spark", "LLM / GenAI", "PyTorch", "TensorFlow", "Deep learning",
                 "scikit-learn", "R", "dbt", "Snowflake"]:
        if tool in by_region.index:
            values = " | ".join(f"{by_region.loc[tool, r]:.0f}%" for r in REGIONS)
            lines.append(f"| {tool} | {values} |")

    lines += [
        "",
        "**Dónde sí difieren** (lo honesto es nombrarlo):",
        "",
        f"- Deep learning y PyTorch pesan más en EE. UU. "
        f"({by_region.loc['PyTorch', 'EE. UU. / Canadá']:.0f}%) que en LATAM "
        f"({by_region.loc['PyTorch', 'LATAM']:.0f}%).",
        f"- IA generativa: EE. UU. {by_region.loc['LLM / GenAI', 'EE. UU. / Canadá']:.0f}% "
        f"vs LATAM {by_region.loc['LLM / GenAI', 'LATAM']:.0f}%.",
        f"- Power BI y Excel pesan más en LATAM "
        f"({by_region.loc['Power BI', 'LATAM']:.0f}% y "
        f"{by_region.loc['Excel', 'LATAM']:.0f}%) que en Europa "
        f"({by_region.loc['Power BI', 'Europa']:.0f}% y "
        f"{by_region.loc['Excel', 'Europa']:.0f}%).",
        "",
        "## 1. El tamaño real de cada puesto",
        "",
        f"- Ofertas de datos comparables: **{len(comparable)}**",
    ]
    for family, count in families.items():
        if family != "Otro":
            lines.append(f"- {family}: **{count}** ({count / len(comparable) * 100:.0f}%)")
    lines += [
        "",
        f"**Titular:** de {len(comparable)} ofertas de datos, solo "
        f"**{families.get('Data Scientist', 0)}** llevan el título de Data Scientist.",
        "",
        "## 2. Herramientas (% del total, las tres regiones juntas)",
        "",
        "| Herramienta | % |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.0f}% |" for name, value in share.items()]

    lines += [
        "",
        "## 3. Título vs. tareas",
        "",
        f"De las {len(ds)} ofertas tituladas «Data Scientist», clasificadas por tareas:",
        "",
    ]
    lines += [
        f"- {PROFILE_LABELS.get(profile, profile)}: **{count}**"
        for profile, count in ds["perfil_por_tareas"].value_counts().items()
    ]

    lines += [
        "",
        "## 4. La puerta de entrada",
        "",
        f"- Ofertas junior: **{len(junior)}** de {len(comparable)} "
        f"({len(junior) / len(comparable) * 100:.1f}%)",
        f"- Sin experiencia: **{(comparable['seniority'] == 'Sin experiencia').sum()}**",
        "- Mediana de años pedidos por nivel:",
    ]
    for level, value in (
        comparable.groupby("seniority")["anios_experiencia"].median().items()
    ):
        if pd.notna(value):
            lines.append(f"  - {level}: {value:.0f} años")

    lines += [
        "",
        "## 5. Competencia y salario — solo LATAM",
        "",
        "> Postulaciones y salario mensual solo existen en Get on Board. Los otros "
        "portales publican cifras anuales o ninguna, y mezclarlas daría un número falso.",
        "",
        f"- Ofertas LATAM: **{len(latam)}**",
        f"- Mediana de postulaciones por oferta: **{latam['postulaciones'].median():.0f}**",
        f"- Máximo observado: **{latam['postulaciones'].max():.0f}**",
        f"- Publican salario: **{len(salary)}** ({len(salary) / len(latam) * 100:.0f}%)",
        f"- Mediana del rango: **{salary['salario_min'].median():.0f}–"
        f"{salary['salario_max'].median():.0f} USD/mes**",
        "",
        "## 6. Habilidades no técnicas",
        "",
        "| Habilidad | " + " | ".join(REGIONS) + " |",
        "|---|" + "---|" * len(REGIONS),
    ]
    for column in [c for c in comparable.columns if c.startswith("soft_")]:
        values = " | ".join(
            f"{comparable[comparable['region'] == region][column].mean() * 100:.0f}%"
            for region in REGIONS
        )
        lines.append(f"| {column.replace('soft_', '')} | {values} |")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "hallazgos.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {path.name}")


def main():
    # The reviewed file wins when it exists: a human verdict outranks the
    # classifier, and re-running must not silently discard it.
    frame = pd.read_csv(PROCESSED)
    if REVIEWED.exists():
        working = pd.read_csv(REVIEWED)
        print(f"Using reviewed labels ({len(working)} rows)")
    else:
        working = frame

    comparable = working[
        working["comparable_regiones"] & working["region"].isin(REGIONS)
    ].copy()
    latam = comparable[comparable["fuente"] == "getonbrd"].copy()
    print(
        f"Analysing {len(comparable)} comparable postings "
        f"({len(latam)} of them LATAM) out of {len(frame)} captured"
    )

    chart_tools(comparable)
    chart_regions(comparable)
    chart_region_agreement(comparable)
    chart_families(comparable)
    chart_title_vs_tasks(comparable)
    chart_hype_gap(comparable)
    chart_seniority(comparable)
    chart_competition(latam)
    write_report(frame, comparable, latam)


if __name__ == "__main__":
    main()
