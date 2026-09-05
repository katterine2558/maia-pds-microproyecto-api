"""Comparacion de tecnicas de balanceo de clases.

Con 11,4 % de reingresos, un bosque entrenado sobre la distribucion original
aprende a responder casi siempre "no reingresa". El problema es que ese es
justamente el error caro: un falso negativo deja salir sin control a un
paciente que vuelve, y el reingreso significa cama ocupada, costo hospitalario
alto y posible deterioro. Un falso positivo, en cambio, solo consume un cupo
de seguimiento.

Por eso la comparacion se hace con F2, que pesa el doble la sensibilidad, y
por eso interesa mirar el conteo de falsos negativos junto a cada tecnica.

Las tecnicas contrastadas son las que Nunes et al. (2025) evaluan sobre datos
clinicos desbalanceados:

    ninguno        entrena sobre la distribucion original
    class_weight   pondera las clases dentro del bosque
    sobremuestreo  replica observaciones de la clase minoritaria
    smote          interpola ejemplos sinteticos entre vecinos
    submuestreo    descarta observaciones de la clase mayoritaria al azar
    nearmiss       descarta las mayoritarias mas alejadas de la frontera

Todas se aplican DENTRO del pipeline. Remuestrear antes de particionar
copiaria informacion de los pacientes de evaluacion hacia el entrenamiento y
el desempeno medido dejaria de ser real.

Uso:
    python -m src.models.balanceo --uri http://IP:8050
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.features import construccion as cons
from src.features import esquema as esq
from src.models import entrenamiento as ent
from src.models import metricas as met
from src.models import particion as part

EXPERIMENTO = "reingreso-30d-balanceo"


def comparar(criterio: str = esq.CRITERIO_UMBRAL) -> pd.DataFrame:
    """Entrena un bosque por tecnica y registra cada uno en MLflow."""
    trabajo = cons.conjunto_de_trabajo(cons.cargar())
    X, y, grupos = cons.matriz(trabajo)
    X_ent, X_eva, y_ent, y_eva, g_ent, g_eva = part.particionar(X, y, grupos)

    fugas = part.verificar_particion(g_ent, g_eva)
    if fugas:
        raise RuntimeError("la particion no aisla a los pacientes: " + "; ".join(fugas))

    filas = []
    for tecnica in ent.DESBALANCES:
        inicio = time.time()
        resultados = ent.correr(
            "bosque", desbalance=tecnica, criterio_umbral=criterio,
            subconjunto=None, experimento=EXPERIMENTO,
        )
        filas.append({
            "tecnica": tecnica,
            "umbral": resultados["umbral"],
            "f2": resultados["f2"],
            "sensibilidad": resultados["sensibilidad"],
            "precision": resultados["precision"],
            "f1": resultados["f1"],
            "falsos_negativos": resultados["falsos_negativos"],
            "falsos_positivos": resultados["falsos_positivos"],
            "roc_auc": resultados["roc_auc"],
            "segundos": round(time.time() - inicio, 1),
        })
        print(f"  {tecnica:<15} F2 {resultados['f2']:.4f}   "
              f"sensibilidad {resultados['sensibilidad']:.4f}   "
              f"FN {resultados['falsos_negativos']:>5}", flush=True)

    return pd.DataFrame(filas).sort_values("f2", ascending=False)


def figura_comparacion(tabla: pd.DataFrame, destino: Path) -> None:
    """F2 por tecnica, con el conteo de falsos negativos al lado."""
    d = tabla.sort_values("f2")
    fig, ejes = plt.subplots(1, 2, figsize=(9.2, 0.45 * len(d) + 1.6), sharey=True)

    ejes[0].barh(range(len(d)), d["f2"], color="#2a78d6", height=0.6)
    for i, v in enumerate(d["f2"]):
        ejes[0].text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8)
    ejes[0].set_xlim(0, d["f2"].max() * 1.25)
    ejes[0].set_title("F2 (pesa el doble la sensibilidad)", fontsize=10, loc="left", pad=10)

    ejes[1].barh(range(len(d)), d["falsos_negativos"], color="#c8532b", height=0.6)
    for i, v in enumerate(d["falsos_negativos"]):
        ejes[1].text(v + 20, i, f"{int(v):,}".replace(",", " "), va="center", fontsize=8)
    ejes[1].set_xlim(0, d["falsos_negativos"].max() * 1.3)
    ejes[1].set_title("Falsos negativos (el error caro)", fontsize=10, loc="left", pad=10)

    ejes[0].set_yticks(range(len(d)), d["tecnica"], fontsize=9)
    for ax in ejes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def main() -> None:
    lector = argparse.ArgumentParser(description=__doc__)
    lector.add_argument("--criterio", default=esq.CRITERIO_UMBRAL,
                        choices=["f2", "f1", "f05", "sensibilidad", "exactitud_balanceada"])
    lector.add_argument("--experimento", default=EXPERIMENTO)
    lector.add_argument("--uri", default=None)
    args = lector.parse_args()

    if args.uri:
        mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment(args.experimento)

    print(f"comparando {len(ent.DESBALANCES)} tecnicas, criterio de umbral: {args.criterio}",
          flush=True)
    tabla = comparar(args.criterio)

    with mlflow.start_run(run_name="comparacion-balanceo"):
        mlflow.log_params({"criterio_umbral": args.criterio,
                           "tecnicas": ", ".join(ent.DESBALANCES)})
        mlflow.set_tags({"autor": "lealUniandes", "entrega": "2",
                         "etapa": "balanceo", "problema": "clasificacion binaria"})
        mejor = tabla.iloc[0]
        mlflow.log_metrics({"mejor_f2": mejor["f2"],
                            "mejor_sensibilidad": mejor["sensibilidad"],
                            "mejor_falsos_negativos": mejor["falsos_negativos"]})
        mlflow.set_tag("tecnica_elegida", mejor["tecnica"])

        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            tabla.to_csv(carpeta / "comparacion-balanceo.csv", index=False)
            figura_comparacion(tabla, carpeta / "comparacion-balanceo.png")
            mlflow.log_artifacts(str(carpeta), artifact_path="balanceo")

    print()
    print(tabla.to_string(index=False))
    print(f"\ntecnica elegida por F2: {tabla.iloc[0]['tecnica']}")


if __name__ == "__main__":
    main()
