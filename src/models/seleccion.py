"""Seleccion de caracteristicas para el clasificador.

El EDA descarto columnas por calidad del dato —vacias, constantes, casi
constantes— pero dejo pendiente verificar cuales aportan de verdad: "su aporte
definitivo debera confirmarse durante el entrenamiento y la validacion del
modelo" (reporte E1, seccion 6.3).

Este modulo hace esa verificacion en dos pasos:

  1. Importancia por permutacion: cuanto se degrada el F1 del clasificador al
     desordenar cada variable. Se mide sobre entrenamiento, no sobre el
     conjunto reservado: el ranking define el subconjunto "utiles", y elegir
     esas variables mirando el mismo conjunto con que despues se las mide
     produciria una cifra optimista que no seria comparable con la de los
     demas subconjuntos, definidos a priori.
  2. Comparacion de subconjuntos: un modelo por familia de variables, para
     poner a prueba la afirmacion del EDA 7.5 de que el historial previo del
     paciente pesa mas que las caracteristicas del episodio actual.

Uso:
    python -m src.models.seleccion --importancia
    python -m src.models.seleccion --subconjuntos
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features import construccion as cons
from src.seguimiento import mlflow_config as seguimiento
from src.features import esquema as esq
from src.models import entrenamiento as ent
from src.models import particion as part

# Variables del historial del ano previo. El EDA 7.5 sostiene que dominan
# sobre las del episodio actual; la ablacion pone esa afirmacion a prueba.
HISTORIAL = ["number_inpatient", "number_emergency", "number_outpatient"]

EPISODIO = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "diag_1", "diag_2", "diag_3",
]


def importancia_por_permutacion(modelo, X_medicion, y_medicion,
                                n_repeticiones: int = 5) -> pd.DataFrame:
    """Mide cuanto se degrada el clasificador al desordenar cada variable.

    `X_medicion` debe ser de entrenamiento. El ranking que produce esta funcion
    selecciona variables, y seleccionar mirando el conjunto reservado dejaria
    su desempeno sin valor como estimacion.

    Se mide sobre F1 y no sobre exactitud: con 11,4 % de positivos, desordenar
    una variable util puede dejar la exactitud casi intacta mientras destruye
    la capacidad de identificar a la clase que interesa.

    Se prefiere a la importancia por impureza que trae el bosque, que esta
    sesgada hacia las variables de alta cardinalidad: ordenaria
    medical_specialty por tener 30 categorias y no por aportar mas.

    La permutacion opera sobre la columna cruda, antes de la codificacion, de
    modo que desordenar una variable desordena todas sus indicadoras a la vez.
    """
    resultado = permutation_importance(
        modelo, X_medicion, y_medicion,
        scoring=["f1", "balanced_accuracy"],
        n_repeats=n_repeticiones,
        random_state=esq.SEMILLA,
        n_jobs=-1,
    )

    filas = [
        {
            "variable": variable,
            "caida_f1": resultado["f1"].importances_mean[i],
            "desviacion_f1": resultado["f1"].importances_std[i],
            "caida_balanceada": resultado["balanced_accuracy"].importances_mean[i],
        }
        for i, variable in enumerate(X_medicion.columns)
    ]

    return (pd.DataFrame(filas)
            .sort_values("caida_f1", ascending=False)
            .reset_index(drop=True))


def definir_subconjuntos(todas: list[str], ranking: pd.DataFrame | None = None) -> dict:
    """Subconjuntos a comparar. Cada uno responde una pregunta distinta."""
    subconjuntos = {
        "completo": todas,
        "solo-historial": [c for c in HISTORIAL if c in todas],
        "sin-historial": [c for c in todas if c not in HISTORIAL],
        "solo-episodio": [c for c in EPISODIO if c in todas],
        "formulario": [c for c in esq.FORMULARIO if c in todas],
    }
    if ranking is not None:
        # Una variable con caida nula o negativa no aporta: el clasificador
        # va igual o mejor sin ella.
        utiles = set(ranking.loc[ranking["caida_f1"] > 0, "variable"])
        subconjuntos["utiles"] = [c for c in todas if c in utiles]
    return subconjuntos


def figura_importancia(ranking: pd.DataFrame, destino: Path, n: int = 20) -> None:
    d = ranking.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(d) + 1.2))
    colores = ["#2a78d6" if v > 0 else "#c8532b" for v in d["caida_f1"]]
    ax.barh(range(len(d)), d["caida_f1"], xerr=d["desviacion_f1"], color=colores,
            height=0.62, error_kw={"linewidth": 0.8, "ecolor": "#9a9992"})
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_yticks(range(len(d)), d["variable"], fontsize=8)
    ax.set_xlabel("Caida del F1 al desordenar la variable", labelpad=8)
    ax.set_title("Importancia por permutacion", fontsize=10, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(destino, dpi=150); plt.close(fig)


def figura_subconjuntos(tabla: pd.DataFrame, destino: Path) -> None:
    d = tabla.sort_values("f1")
    fig, ax = plt.subplots(figsize=(6.2, 0.4 * len(d) + 1.4))
    barras = ax.barh(range(len(d)), d["f1"], color="#2a78d6", height=0.6)
    for b, v, n in zip(barras, d["f1"], d["n_variables"]):
        ax.text(b.get_width() + 0.004, b.get_y() + b.get_height() / 2,
                f"{v:.3f}   ({n} variables)", va="center", fontsize=8)
    ax.set_yticks(range(len(d)), d["subconjunto"], fontsize=8)
    ax.set_xlim(0, max(d["f1"]) * 1.4)
    ax.set_xlabel("F1 sobre la clase positiva", labelpad=8)
    ax.set_title("Desempeno del clasificador segun el subconjunto de variables",
                 fontsize=10, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(destino, dpi=150); plt.close(fig)


def _modelo_entrenado():
    """Entrena el clasificador sobre la particion estandar, sin registrar nada."""
    trabajo = cons.conjunto_de_trabajo(cons.cargar())
    X, y, grupos = cons.matriz(trabajo)
    X_ent, X_eva, y_ent, y_eva, g_ent, _ = part.particionar(X, y, grupos)

    ficha = ent.CATALOGO["bosque"]
    modelo = ent.armar_estimador(
        X, ficha["constructor"](class_weight="balanced_subsample"), "class_weight"
    )
    modelo.fit(X_ent, y_ent)

    from sklearn.model_selection import FixedThresholdClassifier
    ajustado = FixedThresholdClassifier(modelo, threshold=esq.UMBRAL_FIJO).fit(X_ent, y_ent)
    # Se devuelve entrenamiento para medir la importancia; el conjunto
    # reservado no interviene en la seleccion.
    return ajustado, X, X_ent, y_ent


def correr_importancia(n_repeticiones: int = 5) -> pd.DataFrame:
    modelo, X, X_medicion, y_medicion = _modelo_entrenado()
    ranking = importancia_por_permutacion(modelo, X_medicion, y_medicion, n_repeticiones)

    with seguimiento.corrida("importancia-permutacion", familia="bosque-aleatorio"):
        mlflow.log_params({"modelo": "bosque", "n_repeticiones": n_repeticiones,
                           "n_variables": X.shape[1], "medida": "caida del F1"})
        aportan = int((ranking["caida_f1"] > 0).sum())
        mlflow.log_metrics({"variables_que_aportan": aportan,
                            "variables_sin_aporte": len(ranking) - aportan})
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            ranking.to_csv(carpeta / "importancia-permutacion.csv", index=False)
            figura_importancia(ranking, carpeta / "importancia-permutacion.png")
            mlflow.log_artifacts(str(carpeta), artifact_path="seleccion")
    return ranking


def correr_subconjuntos(ranking: pd.DataFrame | None = None) -> pd.DataFrame:
    trabajo = cons.conjunto_de_trabajo(cons.cargar())
    X, _, _ = cons.matriz(trabajo)
    subconjuntos = definir_subconjuntos(list(X.columns), ranking)

    filas = []
    for nombre, columnas in subconjuntos.items():
        if not columnas:
            continue
        r = ent.correr("bosque", columnas=columnas, subconjunto=nombre)
        filas.append({"subconjunto": nombre, "n_variables": len(columnas),
                      "f1": r["f1"], "precision": r["precision"],
                      "sensibilidad": r["sensibilidad"],
                      "exactitud_balanceada": r["exactitud_balanceada"],
                      "roc_auc": r["roc_auc"]})
        print(f"{nombre:<16} {len(columnas):>2} vars   F1 {r['f1']:.3f}   "
              f"sens {r['sensibilidad']:.3f}   bal {r['exactitud_balanceada']:.3f}")

    tabla = pd.DataFrame(filas)
    with seguimiento.corrida("comparacion-subconjuntos", familia="bosque-aleatorio"):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            tabla.to_csv(carpeta / "comparacion-subconjuntos.csv", index=False)
            figura_subconjuntos(tabla, carpeta / "comparacion-subconjuntos.png")
            mlflow.log_artifacts(str(carpeta), artifact_path="seleccion")
    return tabla


def main() -> None:
    lector = argparse.ArgumentParser(description=__doc__)
    lector.add_argument("--importancia", action="store_true")
    lector.add_argument("--subconjuntos", action="store_true")
    lector.add_argument("--repeticiones", type=int, default=5)
    lector.add_argument("--experimento", default=ent.EXPERIMENTO)
    lector.add_argument("--uri", default=None)
    args = lector.parse_args()

    if not (args.importancia or args.subconjuntos):
        lector.error("indique --importancia o --subconjuntos")
    if args.uri:
        mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment(args.experimento)

    ranking = None
    if args.importancia:
        ranking = correr_importancia(args.repeticiones)
        print(ranking.head(15).to_string(index=False))
    if args.subconjuntos:
        print()
        print(correr_subconjuntos(ranking).to_string(index=False))


if __name__ == "__main__":
    main()
