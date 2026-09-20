"""Registra en MLflow el modelo que sirve la API de la Entrega 3.

`entrenar_formulario_e3.py` entrena la variante de diez variables y deja el
artefacto en `api/artifacts/`. Este modulo hace lo mismo pero dejando rastro en
el servidor de MLflow: parametros, metricas, el artefacto empaquetado y las
etiquetas que el equipo usa para comparar corridas.

Son dos archivos y no uno porque el entrenamiento tiene que poder correrse sin
credenciales ni red — al empaquetar la API, por ejemplo — mientras que el
experimento exige el servidor arriba.

    export MLFLOW_TRACKING_URI=http://<ip>:5000
    export MLFLOW_TRACKING_USERNAME=<usuario>
    export MLFLOW_TRACKING_PASSWORD=<clave>
    export MLFLOW_AUTOR=camilo          # el modelo es suyo
    uv run python -m src.models.experimento_formulario_e3
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from src.models.entrenar_formulario_e3 import (
    CATEGORICAS,
    NUMERICAS,
    SALIDA,
    UMBRAL,
    ejecutar,
)
from src.seguimiento import mlflow_config

# Hiperparametros del bosque, replicados aqui para registrarlos. Viven en el
# Pipeline de `entrenar_formulario_e3`; si cambian alla, cambian aca.
PARAMETROS = {
    "algoritmo": "RandomForestClassifier",
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": "{0: 1, 1: 5}",
    "umbral_decision": UMBRAL,
    "particion": "GroupShuffleSplit por patient_nbr, test_size=0.20, seed=42",
    "variables": len(NUMERICAS) + len(CATEGORICAS),
    "variables_numericas": ", ".join(NUMERICAS),
    "variables_categoricas": ", ".join(CATEGORICAS),
    "codificacion": "OneHotEncoder(handle_unknown='ignore')",
}

NOMBRE_CORRIDA = "bosque-formulario-e3-v1"
FAMILIA = "bosque-formulario"


def main() -> dict:
    mlflow_config.iniciar()

    metricas = ejecutar()

    with mlflow_config.corrida(NOMBRE_CORRIDA, FAMILIA):
        mlflow_config.registrar(PARAMETROS, metricas)

        # El nombre del modelo es texto: `registrar` lo descarta como metrica,
        # asi que queda como etiqueta para poder filtrar por el.
        mlflow.set_tags(
            {
                "modelo": metricas["modelo"],
                "sirve_api": "si",
                "entrega": "3",
            }
        )

        artefacto = Path(SALIDA) / "modelo.joblib"
        mlflow.log_artifact(str(artefacto), artifact_path="modelo")
        mlflow.log_artifact(str(Path(SALIDA) / "metricas.json"), artifact_path="modelo")

        print(f"corrida: {mlflow.active_run().info.run_id}")

    return metricas


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
