"""Genera i grafici "hero" per le slide della presentazione finale.

Legge i CSV dei risultati gia' presenti nel repo (nessun ri-training) e produce
in figures/slides/ tre grafici puliti, a tema con la palette arancione del corso:

  1. data_efficiency_curve.png  - Top-1 vs % dati (Supervised vs MAE+LP)
  2. top1_top5_full.png         - Top-1/Top-5 sul dataset completo
  3. per_class_highlights.png   - classi dove MAE+LP batte il supervised

Uso:
    python scripts/generate_slide_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# --- Palette del corso -------------------------------------------------------
ORANGE = "#2563EB"       # blu cobalto (serie chiave: MAE)
ORANGE_DARK = "#1D4ED8"
GREY = "#9AA0A6"         # serie neutra (baseline supervisionato)
GREY_DARK = "#5F6368"
INK = "#1A1A1A"          # testo
PEACH = "#DBEAFE"        # riempimenti morbidi (blue-100)
GRID = "#E6E6E6"

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"
OUT = REPO / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 15,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "text.color": INK,
            "axes.edgecolor": GREY_DARK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def data_efficiency_curve() -> None:
    """Top-1 accuracy in funzione della frazione di dati etichettati."""
    df = pd.read_csv(RESULTS / "evaluation_results.csv")

    def top1(substr: str) -> float:
        return float(df[df["label"].str.contains(substr, regex=False)].iloc[0]["top1"])

    x = [10, 20, 100]
    sup = [top1("Supervised ViT - 10%"), top1("Supervised ViT - 20%"), top1("Supervised ViT - Full")]
    mae = [top1("MAE + Linear Probe - 10%"), top1("MAE + Linear Probe - 20%"), top1("MAE + Linear Probe - Full")]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    ax.plot(x, mae, "-o", color=ORANGE, linewidth=3.2, markersize=11,
            markeredgecolor="white", markeredgewidth=1.6, label="MAE + Linear Probe", zorder=3)
    ax.plot(x, sup, "--s", color=GREY, linewidth=2.6, markersize=10,
            markeredgecolor="white", markeredgewidth=1.4, label="ViT Supervisionato", zorder=2)

    # etichette dei valori
    for xi, yi in zip(x, mae):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=13, fontweight="bold", color=ORANGE_DARK)
    for xi, yi in zip(x, sup):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, -20),
                    ha="center", fontsize=13, fontweight="bold", color=GREY_DARK)

    # gap al 10%
    gap = mae[0] - sup[0]
    ax.annotate(
        "", xy=(10, mae[0]), xytext=(10, sup[0]),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8),
    )
    ax.text(13.5, (mae[0] + sup[0]) / 2, f"+{gap:.1f} pp", fontsize=15,
            fontweight="bold", color=INK, va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=PEACH, ec=ORANGE, lw=1.2))

    ax.set_xlabel("Dati di training etichettati  (%)", fontsize=15, labelpad=8)
    ax.set_ylabel("Top-1 Accuracy  (%)", fontsize=15, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["10%", "20%", "100%"])
    ax.set_ylim(0, 100)
    ax.legend(loc="center right", frameon=False, fontsize=14)

    fig.tight_layout()
    fig.savefig(OUT / "data_efficiency_curve.png")
    plt.close(fig)
    print(f"  data_efficiency_curve.png  (gap al 10% = +{gap:.1f} pp)")


def top1_top5_full() -> None:
    """Top-1 / Top-5 sul dataset completo: Supervised vs MAE+LP."""
    df = pd.read_csv(RESULTS / "evaluation_results.csv")
    sup = df[df["label"].str.contains("Supervised ViT - Full", regex=False)].iloc[0]
    mae = df[df["label"].str.contains("MAE + Linear Probe - Full", regex=False)].iloc[0]

    metrics = ["Top-1", "Top-5"]
    sup_vals = [sup["top1"], sup["top5"]]
    mae_vals = [mae["top1"], mae["top5"]]

    import numpy as np

    xpos = np.arange(len(metrics))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    b1 = ax.bar(xpos - w / 2, sup_vals, w, color=GREY, label="ViT Supervisionato")
    b2 = ax.bar(xpos + w / 2, mae_vals, w, color=ORANGE, label="MAE + Linear Probe")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=14, fontweight="bold")

    ax.set_xticks(xpos)
    ax.set_xticklabels(metrics, fontsize=16)
    ax.set_ylabel("Accuracy  (%)", fontsize=15, labelpad=8)
    ax.set_ylim(0, 106)
    ax.legend(loc="upper left", frameon=False, fontsize=14, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT / "top1_top5_full.png")
    plt.close(fig)
    print("  top1_top5_full.png")


def per_class_highlights(top_n: int = 5) -> None:
    """Classi con il maggior vantaggio MAE+LP rispetto al supervised (10% dati)."""
    df = pd.read_csv(RESULTS / "per_class_accuracy.csv")
    df["delta"] = df["MAE+LP 10%"] - df["Sup 10%"]
    top = df.sort_values("delta", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ypos = range(len(top))
    bars = ax.barh(list(ypos), top["delta"], color=ORANGE, height=0.62)

    for rect in zip(bars, top["class_name"], top["Sup 10%"], top["MAE+LP 10%"]):
        r, name, dsup, dmae = rect
        w = r.get_width()
        ax.annotate(f"+{w:.0f} pp", (w, r.get_y() + r.get_height() / 2),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=13, fontweight="bold", color=ORANGE_DARK)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(
        [f"{n}\n({int(s)}% → {int(m)}%)" for n, s, m in
         zip(top["class_name"], top["Sup 10%"], top["MAE+LP 10%"])],
        fontsize=12.5,
    )
    ax.set_xlabel("Vantaggio MAE+LP  vs  Supervisionato  (10% dati, punti %)", fontsize=14, labelpad=8)
    ax.set_xlim(0, max(top["delta"]) + 8)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(OUT / "per_class_highlights.png")
    plt.close(fig)
    print(f"  per_class_highlights.png  (top {top_n} al 10%: {', '.join(top['class_name'].iloc[::-1])})")


def failure_case(bottom_n: int = 5) -> None:
    """Classi con accuracy più bassa per entrambi i modelli (100% dati)."""
    import numpy as np
    df = pd.read_csv(RESULTS / "per_class_accuracy.csv")
    df["avg"] = (df["MAE+LP 100%"] + df["Sup 100%"]) / 2
    worst = df.sort_values("avg").head(bottom_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ypos = np.arange(len(worst))
    h = 0.35

    ax.barh(ypos + h / 2, worst["Sup 100%"], h, color=GREY, label="ViT Supervisionato")
    ax.barh(ypos - h / 2, worst["MAE+LP 100%"], h, color=ORANGE, label="MAE + Linear Probe")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(worst["class_name"], fontsize=13)
    ax.set_xlabel("Top-1 Accuracy  (%, dataset completo)", fontsize=14, labelpad=8)
    ax.set_xlim(0, 60)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    ax.grid(axis="y", visible=False)
    ax.axvline(x=50, color=GREY, linestyle="--", linewidth=1.2, alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "failure_case.png")
    plt.close(fig)
    print(f"  failure_case.png  (bottom {bottom_n}: {', '.join(worst['class_name'].iloc[::-1])})")


def reconstruction_compact(rows: int = 4) -> None:
    """Ritaglia la griglia di ricostruzione (8 esempi, verticale) ai primi `rows`
    esempi: ogni esempio risulta circa 2x piu' grande a parita' di larghezza."""
    from PIL import Image

    src = REPO / "experiments" / "visualizations" / "reconstruction_grid.png"
    if not src.exists():
        print("  reconstruction_grid.png assente, salto il crop")
        return
    img = Image.open(src)
    w, h = img.size
    # header (etichette colonne) + 8 righe esempi: tengo header + primi `rows`
    header = int(h * 0.045)
    body = h - header
    crop_h = header + int(body * rows / 8)
    img.crop((0, 0, w, crop_h)).save(OUT / "reconstruction_compact.png")
    print(f"  reconstruction_compact.png  ({rows} esempi, {w}x{crop_h})")


def main() -> None:
    set_style()
    print(f"Genero i grafici in {OUT} ...")
    data_efficiency_curve()
    top1_top5_full()
    per_class_highlights()
    failure_case()
    reconstruction_compact()
    print("Fatto.")


if __name__ == "__main__":
    main()
