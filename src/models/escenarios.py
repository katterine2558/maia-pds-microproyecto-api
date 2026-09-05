"""Escenarios del bosque aleatorio.

Cuatro configuraciones que recorren el efecto de los hiperparametros del
arbol, dejando fijos el manejo del desbalance y el umbral para que la
comparacion aisle lo que cambia.

La progresion va de un bosque sin restricciones a uno cada vez mas regulado:

    V1  sin restricciones     el arbol crece hasta agotar los datos
    V2  profundidad acotada   se limita cuanto puede crecer
    V3  hojas minimas         se exige un minimo de casos por hoja
    V4  criterio de entropia  cambia la medida de impureza de V3

Los resultados se registran en MLflow y se exportan a un CSV con las mismas
columnas que el resto de los modelos del proyecto, para que las familias sean
comparables sin reprocesar nada.

Uso:
    python -m src.models.escenarios --uri http://IP:8050
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src.features import construccion as cons
from src.features import esquema as esq
from src.models import entrenamiento as ent
from src.models import metricas as met
from src.models import particion as part

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "docs" / "soportes" / "modelos" / "bosque_aleatorio"

EXPERIMENTO = "reingreso-30d-escenarios"

# Manejo del desbalance y umbral, iguales en los cuatro escenarios.
DESBALANCE = "peso"
PESO_POSITIVO = 5

ESCENARIOS = {
    "V1_sin_restricciones": {
        "n_estimators": 400, "max_depth": None,
        "min_samples_leaf": 1, "min_samples_split": 2, "criterion": "gini",
    },
    "V2_profundidad_acotada": {
        "n_estimators": 400, "max_depth": 12,
        "min_samples_leaf": 1, "min_samples_split": 2, "criterion": "gini",
    },
    "V3_hojas_minimas": {
        "n_estimators": 400, "max_depth": 12,
        "min_samples_leaf": 50, "min_samples_split": 2, "criterion": "gini",
    },
    "V4_criterio_entropia": {
        "n_estimators": 400, "max_depth": 12,
        "min_samples_leaf": 50, "min_samples_split": 2, "criterion": "entropy",
    },
}


def metricas_completas(nombre: str, y_real, probabilidades, umbral: float) -> dict:
    """Fila de resultados con las columnas que comparten los modelos del proyecto."""
    y_predicho = (probabilidades >= umbral).astype(int)
    conteos = met.matriz_confusion(y_real, y_predicho)
    return {
        "modelo": nombre,
        "roc_auc": roc_auc_score(y_real, probabilidades),
        "pr_auc": average_precision_score(y_real, probabilidades),
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "recall": recall_score(y_real, y_predicho, zero_division=0),
        "f1": f1_score(y_real, y_predicho, zero_division=0),
        "accuracy": accuracy_score(y_real, y_predicho),
        **conteos,
    }


def correr(n_folds: int = 5) -> pd.DataFrame:
    """Entrena los cuatro escenarios y registra cada uno en MLflow."""
    trabajo = cons.conjunto_de_trabajo(cons.cargar())
    X, y, grupos = cons.matriz(trabajo)
    X_ent, X_eva, y_ent, y_eva, g_ent, g_eva = part.particionar(X, y, grupos)

    fugas = part.verificar_particion(g_ent, g_eva)
    if fugas:
        raise RuntimeError("la particion no aisla a los pacientes: " + "; ".join(fugas))

    division = part.validacion_cruzada(n_folds)
    filas, filas_cv = [], []

    for nombre, hiperparametros in ESCENARIOS.items():
        parametros = {**hiperparametros, "class_weight": {0: 1, 1: PESO_POSITIVO}}

        # Validacion cruzada sobre entrenamiento: es la estimacion con que se
        # comparan los escenarios entre si.
        recalls = []
        for i_ajuste, i_valida in division.split(X_ent, y_ent, groups=g_ent):
            modelo = ent.armar_estimador(
                X, ent.CATALOGO["bosque"]["constructor"](**parametros), DESBALANCE)
            modelo.fit(X_ent.iloc[i_ajuste], y_ent.iloc[i_ajuste])
            p = modelo.predict_proba(X_ent.iloc[i_valida])[:, 1]
            recalls.append(recall_score(y_ent.iloc[i_valida],
                                        (p >= esq.UMBRAL_FIJO).astype(int)))

        # El conjunto reservado se usa una sola vez, para reportar.
        modelo = ent.armar_estimador(
            X, ent.CATALOGO["bosque"]["constructor"](**parametros), DESBALANCE)
        modelo.fit(X_ent, y_ent)
        probabilidades = modelo.predict_proba(X_eva)[:, 1]
        fila = metricas_completas(nombre, y_eva, probabilidades, esq.UMBRAL_FIJO)

        with mlflow.start_run(run_name=nombre):
            mlflow.log_params({
                "escenario": nombre, "desbalance": DESBALANCE,
                "peso_positivo": PESO_POSITIVO, "umbral": esq.UMBRAL_FIJO,
                "n_folds": n_folds, "n_predictoras": X.shape[1],
                **{f"hp_{k}": v for k, v in hiperparametros.items()},
            })
            mlflow.set_tags({"autor": "lealUniandes", "entrega": "2",
                             "etapa": "escenarios", "problema": "clasificacion binaria"})
            mlflow.log_metrics({
                **{k: v for k, v in fila.items() if k != "modelo"},
                "cv_recall": float(pd.Series(recalls).mean()),
                "cv_recall_desv": float(pd.Series(recalls).std()),
            })

        fila["cv_recall"] = round(float(pd.Series(recalls).mean()), 4)
        filas.append(fila)
        print(f"  {nombre:<24} recall {fila['recall']:.4f}   "
              f"cv_recall {fila['cv_recall']:.4f}   FN {fila['falsos_negativos']:>4}",
              flush=True)

    return pd.DataFrame(filas)


def main() -> None:
    lector = argparse.ArgumentParser(description=__doc__)
    lector.add_argument("--folds", type=int, default=5)
    lector.add_argument("--experimento", default=EXPERIMENTO)
    lector.add_argument("--uri", default=None)
    args = lector.parse_args()

    if args.uri:
        mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment(args.experimento)

    print(f"{len(ESCENARIOS)} escenarios | desbalance {DESBALANCE} "
          f"(peso {PESO_POSITIVO}) | umbral {esq.UMBRAL_FIJO}\n", flush=True)
    tabla = correr(args.folds)

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "metricas_bosque_aleatorio.csv"
    tabla.to_csv(destino, index=False)

    with mlflow.start_run(run_name="resumen-escenarios"):
        mlflow.set_tags({"autor": "lealUniandes", "entrega": "2", "etapa": "escenarios"})
        mlflow.log_artifact(str(destino), artifact_path="escenarios")
        mejor = tabla.loc[tabla["cv_recall"].idxmax()]
        mlflow.set_tag("escenario_elegido", mejor["modelo"])

    print(f"\n{tabla.to_string(index=False)}")
    print(f"\nescenario elegido por recall de validacion cruzada: {mejor['modelo']}")
    print(f"resultados en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
