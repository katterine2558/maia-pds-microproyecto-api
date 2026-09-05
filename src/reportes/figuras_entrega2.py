"""Figuras del reporte de la Entrega 2, construidas desde MLflow.

Las cifras no se transcriben a mano: se leen del servidor de seguimiento, de
modo que la figura del reporte y la evidencia registrada no puedan divergir.

Uso:

    export MLFLOW_TRACKING_URI=http://<ip>:5000
    export MLFLOW_TRACKING_USERNAME=<usuario>
    export MLFLOW_TRACKING_PASSWORD=<clave>

    python -m src.reportes.figuras_entrega2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow

from src.seguimiento.mlflow_config import EXPERIMENTO, iniciar


RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "docs" / "entregas" / "figuras"

UMBRAL_ELEGIDO = 0.30

# Paleta categorica validada para daltonismo: el par tiene una separacion de
# 24,7 en protanopia y 33,6 en vision normal, muy por encima del piso de 8.
AZUL = "#2a78d6"
NARANJA = "#eb6834"

TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRIS_TENUE = "#d4d3ce"
SUPERFICIE = "#fcfcfb"


def leer_barrido_umbral() -> list[dict]:
    """Recupera las corridas V5, una por umbral evaluado."""

    cliente = mlflow.MlflowClient()
    experimento = cliente.get_experiment_by_name(EXPERIMENTO)

    corridas = cliente.search_runs(
        [experimento.experiment_id],
        max_results=500,
    )

    puntos = []

    for corrida in corridas:
        parametros = corrida.data.params
        metricas = corrida.data.metrics

        if parametros.get("version") != "V5" or "umbral" not in parametros:
            continue

        puntos.append(
            {
                "umbral": float(parametros["umbral"]),
                "recall": metricas["recall"],
                "precision": metricas["precision"],
            }
        )

    return sorted(puntos, key=lambda p: p["umbral"])


def figura_umbral(puntos: list[dict], destino: Path) -> None:
    """Recall y precision contra el umbral de decision.

    Ambas series son proporciones, asi que comparten un solo eje: un segundo
    eje con otra escala haria que el punto de cruce dependiera del encuadre y
    no de los datos.
    """

    umbrales = [p["umbral"] for p in puntos]
    recall = [p["recall"] * 100 for p in puntos]
    precision = [p["precision"] * 100 for p in puntos]

    figura, ejes = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    figura.patch.set_facecolor(SUPERFICIE)
    ejes.set_facecolor(SUPERFICIE)

    ejes.axvline(
        UMBRAL_ELEGIDO,
        color=GRIS_TENUE,
        linewidth=8,
        zorder=0,
    )

    ejes.plot(umbrales, recall, color=AZUL, linewidth=2, marker="o", markersize=5.5,
              markeredgecolor=SUPERFICIE, markeredgewidth=1.2, label="Recall", zorder=3)
    ejes.plot(umbrales, precision, color=NARANJA, linewidth=2, marker="o", markersize=5.5,
              markeredgecolor=SUPERFICIE, markeredgewidth=1.2, label="Precision", zorder=3)

    # Etiqueta directa al final de cada linea: la identidad de la serie no
    # depende solo del color.
    ejes.annotate(
        "Recall",
        (umbrales[-1], recall[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=TINTA,
        fontsize=9,
        va="center",
    )
    ejes.annotate(
        "Precision",
        (umbrales[-1], precision[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=TINTA,
        fontsize=9,
        va="center",
    )

    elegido = next(p for p in puntos if abs(p["umbral"] - UMBRAL_ELEGIDO) < 1e-9)

    ejes.annotate(
        f"Umbral seleccionado: {UMBRAL_ELEGIDO:.2f}\n"
        f"recall {elegido['recall'] * 100:.1f} %  ·  "
        f"precision {elegido['precision'] * 100:.1f} %",
        xy=(UMBRAL_ELEGIDO, elegido["recall"] * 100),
        xytext=(0.335, 88),
        fontsize=8.5,
        color=TINTA_SECUNDARIA,
    )

    ejes.set_xlabel("Umbral de decision", fontsize=9.5, color=TINTA_SECUNDARIA)
    ejes.set_ylabel("Porcentaje", fontsize=9.5, color=TINTA_SECUNDARIA)
    ejes.set_ylim(0, 105)
    ejes.set_xticks(umbrales)

    # Holgura a la derecha para que quepan las etiquetas directas.
    ancho = umbrales[-1] - umbrales[0]
    ejes.set_xlim(umbrales[0] - ancho * 0.04, umbrales[-1] + ancho * 0.16)

    ejes.grid(axis="y", color=GRIS_TENUE, linewidth=0.6, alpha=0.7)
    ejes.set_axisbelow(True)

    for lado in ("top", "right"):
        ejes.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ejes.spines[lado].set_color(GRIS_TENUE)

    ejes.tick_params(colors=TINTA_SECUNDARIA, labelsize=8.5)

    ejes.legend(
        frameon=False,
        loc="center left",
        fontsize=9,
        labelcolor=TINTA,
    )

    figura.tight_layout()

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(figura)


def main() -> None:
    iniciar()

    puntos = leer_barrido_umbral()

    if not puntos:
        raise SystemExit(
            "No hay corridas V5 en MLflow. Correr antes:\n"
            "    python -m src.models.experimentos_mlflow --familias umbral"
        )

    destino = SALIDA / "entrega-2-umbral-recall-precision.png"
    figura_umbral(puntos, destino)

    print(f"Puntos leidos de MLflow: {len(puntos)}")
    for p in puntos:
        print(
            f"  umbral {p['umbral']:.2f}  "
            f"recall {p['recall'] * 100:6.2f} %  "
            f"precision {p['precision'] * 100:6.2f} %"
        )
    print(f"\nFigura: {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
