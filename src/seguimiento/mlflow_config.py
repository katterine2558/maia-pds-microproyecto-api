"""Configuracion del seguimiento de experimentos con MLflow.

El servidor vive en una EC2 (ver docs/soportes/mlflow-ec2.md). Las credenciales
NO van en el codigo: se leen del entorno, porque este repositorio es publico.

Antes de correr cualquier experimento:

    export MLFLOW_TRACKING_URI=http://<ip>:5000
    export MLFLOW_TRACKING_USERNAME=<usuario>
    export MLFLOW_TRACKING_PASSWORD=<clave>

Opcional, cuando quien lanza la corrida no es quien escribio el modelo:

    export MLFLOW_AUTOR=camilo
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow


RAIZ = Path(__file__).resolve().parents[2]

EXPERIMENTO = "readmision-diabetes"

VARIABLES_REQUERIDAS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


class ConfiguracionIncompleta(RuntimeError):
    """Faltan credenciales del servidor de MLflow."""


def verificar_entorno() -> None:
    """Falla temprano y con un mensaje util, no a mitad del entrenamiento."""

    faltantes = [v for v in VARIABLES_REQUERIDAS if not os.environ.get(v)]

    if faltantes:
        raise ConfiguracionIncompleta(
            "Faltan variables de entorno: "
            + ", ".join(faltantes)
            + "\n\nAntes de correr:\n"
            "    export MLFLOW_TRACKING_URI=http://<ip>:5000\n"
            "    export MLFLOW_TRACKING_USERNAME=<usuario>\n"
            "    export MLFLOW_TRACKING_PASSWORD=<clave>\n\n"
            "Las credenciales las reparte quien administra la EC2. "
            "Ver docs/soportes/mlflow-ec2.md"
        )


def _commit_actual() -> str:
    """Deja rastro de con que version del codigo se corrio el experimento."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def iniciar(experimento: str = EXPERIMENTO) -> str:
    """Deja MLflow listo para registrar. Devuelve el id del experimento."""

    verificar_entorno()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    exp = mlflow.set_experiment(experimento)

    return exp.experiment_id


@contextmanager
def corrida(nombre: str, familia: str, anidada: bool = False):
    """Una corrida con las etiquetas que el equipo necesita para comparar.

    `familia` agrupa las versiones de un mismo enfoque (por ejemplo
    "regresion-logistica"), para poder filtrarlas en la interfaz.
    """

    # El autor del modelo no siempre es quien lanza la corrida. Con
    # MLFLOW_AUTOR la corrida queda a nombre de quien desarrollo el modelo,
    # que es la contribucion que evalua el curso.
    autor = os.environ.get(
        "MLFLOW_AUTOR",
        os.environ.get("MLFLOW_TRACKING_USERNAME", "desconocido"),
    )

    with mlflow.start_run(run_name=nombre, nested=anidada) as run:
        mlflow.set_tags(
            {
                "familia": familia,
                "commit": _commit_actual(),
                "autor": autor,
                # La interfaz de MLflow muestra este como "Created by".
                "mlflow.user": autor,
            }
        )
        yield run


def registrar(parametros: dict, metricas: dict) -> None:
    """Registra parametros y metricas, descartando lo que no sea numerico.

    `calcular_metricas` devuelve tambien el nombre del modelo, que es texto y
    MLflow rechazaria como metrica.
    """

    mlflow.log_params(parametros)

    numericas = {
        clave: float(valor)
        for clave, valor in metricas.items()
        if isinstance(valor, (int, float)) and not isinstance(valor, bool)
    }

    mlflow.log_metrics(numericas)
