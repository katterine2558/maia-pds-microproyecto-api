"""Exploracion de los datos y figuras de la Entrega 1.

Produce las graficas del punto 6 del reporte. Cada figura sostiene una
afirmacion concreta; las que no cambiaban ninguna conclusion se dejaron por
fuera a proposito, porque el reporte tiene tope de 10 paginas.

    python -m src.data.exploracion
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CRUDOS = RAIZ / "data" / "raw" / "diabetic_data.csv"
FIGURAS = RAIZ / "docs" / "entregas" / "figuras"

# Egresos que no pueden o no deben entrar al analisis (ver diccionario de variables).
EXCLUIR = {"11", "19", "20", "21", "13", "14"}

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
SERIE = "#2a78d6"
REFERENCIA = "#9a9992"

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": TINTA,
    "axes.labelcolor": TINTA_2,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "axes.edgecolor": "#dcdbd4",
    "axes.linewidth": 0.8,
})


def cargar() -> pd.DataFrame:
    """Lectura honesta: '?' es el centinela, y 'None' NO es un faltante."""
    c = pd.read_csv(CRUDOS, dtype=str, keep_default_na=False, na_values=[])
    d = c[~c["discharge_disposition_id"].isin(EXCLUIR)].copy()
    d["y"] = (d["readmitted"] == "<30").astype(int)
    return d


def _limpiar(ax, horizontal: bool = False) -> None:
    """Rejilla y ejes recesivos: la unica linea con peso es el dato."""
    for lado in ("top", "right", "left" if not horizontal else "bottom"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)
    if horizontal:
        ax.xaxis.set_visible(False)
    else:
        ax.yaxis.set_visible(False)


def _referencia(ax, base: float, horizontal: bool = False) -> None:
    linea = ax.axvline if horizontal else ax.axhline
    linea(base, color=REFERENCIA, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    rotulo = f"tasa general {base:.1f} %".replace(".", ",")
    if horizontal:
        ax.annotate(rotulo, xy=(base, 1.005), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5, color=TINTA_2)
    else:
        ax.annotate(rotulo, xy=(-0.004, base), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.5, color=TINTA_2)


def fig_hospitalizaciones(d: pd.DataFrame, base: float) -> Path:
    ni = pd.to_numeric(d["number_inpatient"])
    grupos = pd.cut(ni, [-1, 0, 1, 2, 4, 1000], labels=["0", "1", "2", "3-4", "5 o más"])
    t = d.groupby(grupos, observed=True)["y"].agg(["size", "mean"])

    etiquetas = [f"{k}\nn = {int(v):,}".replace(",", " ") for k, v in zip(t.index.astype(str), t["size"])]

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    barras = ax.bar(etiquetas, t["mean"] * 100, color=SERIE, width=0.62, zorder=2)
    for b, valor in zip(barras, t["mean"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9,
                f"{valor:.1f} %".replace(".", ","), ha="center", fontsize=9, color=TINTA)
    _referencia(ax, base)
    ax.set_ylim(0, 42)
    ax.set_xlabel("Hospitalizaciones en el año previo al egreso", labelpad=14)
    _limpiar(ax)
    return _guardar(fig, "01-reingreso-por-hospitalizaciones-previas.png")


def fig_especialidad(d: pd.DataFrame, base: float, minimo: int = 500) -> Path:
    t = d[d["medical_specialty"] != "?"].groupby("medical_specialty", observed=True)["y"].agg(["size", "mean"])
    t = t[t["size"] >= minimo].sort_values("mean")

    nombres = {
        "Nephrology": "Nefrología", "Surgery-Vascular": "Cirugía vascular",
        "Psychiatry": "Psiquiatría", "Family/GeneralPractice": "Medicina familiar",
        "InternalMedicine": "Medicina interna", "Emergency/Trauma": "Urgencias",
        "Gastroenterology": "Gastroenterología", "Cardiology": "Cardiología",
        "Surgery-General": "Cirugía general", "Orthopedics": "Ortopedia",
        "Orthopedics-Reconstructive": "Ortopedia reconstructiva",
        "Radiologist": "Radiología", "Pulmonology": "Neumología",
        "Surgery-Cardiovascular/Thoracic": "Cirugía cardiotorácica",
        "ObstetricsandGynecology": "Ginecobstetricia", "Oncology": "Oncología",
        "Nephrology-Pediatric": "Nefrología pediátrica", "Hematology/Oncology": "Hematooncología",
        "Urology": "Urología", "Endocrinology": "Endocrinología",
        "Neurology": "Neurología", "Podiatry": "Podología",
        "Surgery-Neuro": "Neurocirugía", "Otolaryngology": "Otorrinolaringología",
        "Pediatrics": "Pediatría", "Hospitalist": "Hospitalista",
    }
    etiquetas = [nombres.get(k, k) for k in t.index]

    fig, ax = plt.subplots(figsize=(6.2, 0.29 * len(t) + 0.8))
    barras = ax.barh(etiquetas, t["mean"] * 100, color=SERIE, height=0.6, zorder=2)
    for b, valor in zip(barras, t["mean"] * 100):
        ax.text(19.6, b.get_y() + b.get_height() / 2,
                f"{valor:.1f} %".replace(".", ","), va="center", ha="right",
                fontsize=8.5, color=TINTA)
    _referencia(ax, base, horizontal=True)
    ax.set_xlim(0, 20)
    _limpiar(ax, horizontal=True)
    return _guardar(fig, "02-reingreso-por-especialidad.png")


def fig_edad(d: pd.DataFrame, base: float) -> Path:
    t = d.groupby("age", observed=True)["y"].agg(["size", "mean"]).sort_index()
    etiquetas = [k.strip("[)").replace("-", "–") for k in t.index]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    barras = ax.bar(etiquetas, t["mean"] * 100, color=SERIE, width=0.62, zorder=2)
    for b, valor in zip(barras, t["mean"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{valor:.1f} %".replace(".", ","), ha="center", fontsize=8, color=TINTA)
    # Solo el pico de 20-30 lleva su n: es la barra sobre la que el texto advierte.
    pico = list(t.index).index("[20-30)")
    ax.annotate(f"n = {int(t['size'].iloc[pico]):,}".replace(",", " "),
                xy=(pico, t["mean"].iloc[pico] * 100 + 1.6), ha="center",
                fontsize=7.5, color=TINTA_2)
    _referencia(ax, base)
    ax.set_ylim(0, 18.5)
    ax.set_xlabel("Grupo de edad (años)", labelpad=6)
    _limpiar(ax)
    return _guardar(fig, "03-reingreso-por-edad.png")


def _guardar(fig, nombre: str) -> Path:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    return ruta


def main() -> int:
    d = cargar()
    base = d["y"].mean() * 100
    print(f"egresos tras exclusiones: {len(d):,}".replace(",", " "))
    print(f"tasa base de reingreso <30 dias: {base:.2f} %\n")
    for ruta in (fig_hospitalizaciones(d, base), fig_especialidad(d, base), fig_edad(d, base)):
        print("figura:", ruta.relative_to(RAIZ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
