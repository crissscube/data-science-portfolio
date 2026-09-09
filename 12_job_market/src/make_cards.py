"""Render full-screen number cards for the video.

Some findings are a single figure. A bar chart of one number is worse than the
number itself: it takes longer to read and says less. These cards carry the
figure at a size that survives being watched on a phone.

Same palette and type as the charts, so the video looks like one piece.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "tarjetas"

INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"

# (filename, big number, line above, line below)
CARDS = [
    (
        "cero_sin_experiencia.png",
        "CERO",
        "Ofertas para gente sin experiencia",
        "de 271 ofertas de datos",
    ),
    (
        "2112_postulantes.png",
        "2.112",
        "Personas postularon a UNA sola oferta",
        "Junior Data Engineer · remoto",
    ),
    (
        "4_de_38.png",
        "4 de 38",
        "Ofertas de «Data Scientist» que en realidad",
        "describen trabajo de analista",
    ),
    (
        "10_de_271.png",
        "10",
        "Ofertas junior",
        "de 271 ofertas de datos",
    ),
]


def render(filename, number, line_above, line_below):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURFACE)

    fig.text(
        0.5,
        0.70,
        line_above,
        ha="center",
        va="center",
        fontsize=30,
        color=INK_SOFT,
        family="DejaVu Sans",
    )
    # Scaled down for longer strings so "4 de 38" does not overflow the frame.
    fig.text(
        0.5,
        0.47,
        number,
        ha="center",
        va="center",
        fontsize=200 if len(number) <= 5 else 130,
        fontweight="bold",
        color=ACCENT,
        family="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.24,
        line_below,
        ha="center",
        va="center",
        fontsize=30,
        color=INK_SOFT,
        family="DejaVu Sans",
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / filename
    fig.savefig(path, facecolor=SURFACE, dpi=120)
    plt.close(fig)
    print(f"  {path.name}")


def main():
    for card in CARDS:
        render(*card)


if __name__ == "__main__":
    main()
