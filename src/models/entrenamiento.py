"""Entrenamiento del clasificador binario y registro en MLflow.

El producto es un clasificador: para cada egreso responde si el paciente
reingresara antes de 30 dias, y entrega junto a esa respuesta la probabilidad
estimada, que el tablero usa para que el personal de enfermeria filtre el
listado por el valor que digite.

El artefacto registrado expone las dos salidas:

    modelo.predict(X)        -> 0 o 1
    modelo.predict_proba(X)  -> probabilidad de la clase positiva

El umbral de decision no se hereda de `predict`. Con 11,4 % de positivos casi
ningun paciente supera 0,5 y el modelo termina marcando a muy pocos, de modo
que el umbral se elige por validacion cruzada sobre entrenamiento y se fija en
el artefacto con FixedThresholdClassifier.

Uso:
    python -m src.models.entrenamiento --modelo bosque
    python -m src.models.entrenamiento --todos --desbalance smote
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as PipelineDesbalance
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import FixedThresholdClassifier, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.features import construccion as cons
from src.features import esquema as esq
from src.models import metricas as met
from src.models import particion as part

EXPERIMENTO = "reingreso-30d"

# MLflow 3 serializa con skops, que exige declarar los tipos ajenos a su lista
# base. El codificador guarda una funcion de validacion parcialmente aplicada
# para tratar las categorias no vistas.
TIPOS_CONFIABLES = [
    "functools.partial",
    "sklearn.utils.validation.check_array",
    "imblearn.pipeline.Pipeline",
    "imblearn.over_sampling._smote.base.SMOTE",
    "imblearn.over_sampling._random_over_sampler.RandomOverSampler",
    "imblearn.under_sampling._prototype_selection._random_under_sampler.RandomUnderSampler",
    "imblearn.under_sampling._prototype_selection._nearmiss.NearMiss",
]


# --------------------------------------------------------------------------
# Catalogo
# --------------------------------------------------------------------------

CATALOGO = {
    "trivial": {
        "constructor": lambda **k: DummyClassifier(strategy="prior"),
        "descripcion": "Referencia sin capacidad predictiva: responde siempre la clase mayoritaria",
    },
    "bosque": {
        "constructor": lambda **k: RandomForestClassifier(
            n_estimators=k.get("n_estimators", 400),
            max_depth=k.get("max_depth", 20),
            min_samples_leaf=k.get("min_samples_leaf", 10),
            min_samples_split=k.get("min_samples_split", 10),
            max_features=k.get("max_features", "sqrt"),
            # Medida de impureza con que cada nodo elige su particion: Gini
            # usa 1 - sum(p^2) y la entropia -sum(p log2 p).
            criterion=k.get("criterion", "gini"),
            class_weight=k.get("class_weight", "balanced_subsample"),
            n_jobs=-1,
            random_state=esq.SEMILLA,
        ),
        "descripcion": "Bosque aleatorio: arboles de decision independientes, votados por mayoria",
    },
}


# --------------------------------------------------------------------------
# Manejo del desbalance
# --------------------------------------------------------------------------

# Tecnicas de remuestreo comparadas. Las cuatro ultimas siguen el repertorio
# que Nunes et al. (2025) contrastan sobre datos clinicos desbalanceados.
REMUESTREADORES = {
    "smote": lambda: SMOTE(random_state=esq.SEMILLA, k_neighbors=5),
    "sobremuestreo": lambda: RandomOverSampler(random_state=esq.SEMILLA),
    "submuestreo": lambda: RandomUnderSampler(random_state=esq.SEMILLA),
    "nearmiss": lambda: NearMiss(version=1),
}

DESBALANCES = ["class_weight", "peso", "ninguno", *REMUESTREADORES]


def armar_estimador(X, estimador, desbalance: str = "class_weight"):
    """Encadena preparacion, remuestreo opcional y modelo en un solo objeto.

    Sobre `desbalance`:

    - "class_weight" pondera las clases dentro del propio bosque. No inventa
      observaciones ni descarta ninguna.
    - "ninguno" entrena sobre la distribucion original y deja que el umbral
      haga todo el trabajo.
    - "smote" genera ejemplos sinteticos interpolando entre vecinos de la
      clase minoritaria; "sobremuestreo" replica observaciones existentes;
      "submuestreo" descarta observaciones de la clase mayoritaria al azar y
      "nearmiss" descarta las que estan lejos de la frontera.

    Los cuatro remuestreadores van DENTRO del pipeline, nunca antes de
    particionar. Aplicarlos sobre el conjunto completo copiaria informacion de
    los pacientes de evaluacion hacia el entrenamiento y el desempeno medido
    dejaria de ser real: es la fuga que produce esos F1 cercanos a 1,00 que
    aparecen en la literatura y que no se sostienen en operacion.

    `imblearn.Pipeline` los aplica solo en `fit` y los omite en `predict`, de
    modo que en validacion cruzada cada fold se remuestrea por separado con
    los datos de ese fold y el conjunto de validacion nunca se toca.
    """
    preparacion = cons.construir_preprocesador(X)

    if desbalance in REMUESTREADORES:
        return PipelineDesbalance([
            ("preparacion", preparacion),
            ("remuestreo", REMUESTREADORES[desbalance]()),
            ("estimador", estimador),
        ])

    return Pipeline([("preparacion", preparacion), ("estimador", estimador)])


def elegir_umbral(modelo, X_ent, y_ent, g_ent, criterio: str = "f1") -> float:
    """Elige el umbral de decision sobre una porcion reservada del entrenamiento.

    El umbral es parte del clasificador y hay que estimarlo con datos que el
    modelo no vio al ajustarse, pero que tampoco pertenezcan al conjunto de
    evaluacion. Se separa por paciente, por la misma razon que la particion
    principal (EDA 8.1).
    """
    division = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=esq.SEMILLA)
    i_ajuste, i_umbral = next(division.split(X_ent, y_ent, groups=g_ent))

    modelo.fit(X_ent.iloc[i_ajuste], y_ent.iloc[i_ajuste])
    probabilidades = modelo.predict_proba(X_ent.iloc[i_umbral])[:, 1]

    return met.mejor_umbral(y_ent.iloc[i_umbral], probabilidades, criterio)


# --------------------------------------------------------------------------
# Artefactos
# --------------------------------------------------------------------------

def _figura_matriz(conteos: dict, destino: Path) -> None:
    """Matriz de confusion con los cuatro conteos."""
    m = np.array([[conteos["verdaderos_negativos"], conteos["falsos_positivos"]],
                  [conteos["falsos_negativos"], conteos["verdaderos_positivos"]]])

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(m, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}".replace(",", " "), ha="center", va="center",
                    fontsize=13, color="white" if m[i, j] > m.max() / 2 else "#0b0b0b")

    ax.set_xticks([0, 1], ["No reingresa", "Reingresa"])
    ax.set_yticks([0, 1], ["No reingresa", "Reingresa"])
    ax.set_xlabel("Prediccion", labelpad=8)
    ax.set_ylabel("Realidad", labelpad=8)
    ax.set_title("Matriz de confusion", fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def _figura_umbral(barrido: pd.DataFrame, umbral: float, destino: Path) -> None:
    """Precision, sensibilidad y F1 a lo largo de los umbrales posibles."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    for columna, color, etiqueta in [
        ("precision", "#2a78d6", "Precision"),
        ("sensibilidad", "#c8532b", "Sensibilidad"),
        ("f1", "#0b0b0b", "F1"),
    ]:
        ax.plot(barrido["umbral"], barrido[columna], color=color, linewidth=1.6, label=etiqueta)

    ax.axvline(umbral, color="#9a9992", linestyle=(0, (4, 3)), linewidth=1)
    ax.text(umbral, 1.02, f" umbral elegido {umbral:.2f}", fontsize=7.5, color="#52514e")

    ax.set_xlabel("Umbral de decision", labelpad=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Comportamiento del clasificador segun el umbral", fontsize=10, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def _figura_calibracion(curva: pd.DataFrame, destino: Path) -> None:
    """Probabilidad estimada frente a frecuencia observada, por decil."""
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    tope = max(curva["estimada"].max(), curva["observada"].max()) * 1.1

    ax.plot([0, tope], [0, tope], linestyle=(0, (4, 3)), color="#9a9992",
            linewidth=1, label="Calibracion perfecta")
    ax.plot(curva["estimada"], curva["observada"], marker="o", color="#2a78d6",
            linewidth=1.4, markersize=5, label="Modelo")

    ax.set_xlabel("Probabilidad estimada")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibracion de la probabilidad", fontsize=10, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def registrar_artefactos(y_eva, probabilidades, umbral: float, carpeta: Path) -> None:
    """Genera y guarda las tablas y figuras de una corrida."""
    resultados = met.evaluar(y_eva, probabilidades, umbral)

    pd.DataFrame([resultados]).to_csv(carpeta / "metricas.csv", index=False)

    barrido = met.barrido_umbral(y_eva, probabilidades)
    barrido.to_csv(carpeta / "barrido-umbral.csv", index=False)

    calibracion = met.curva_calibracion(y_eva, probabilidades)
    calibracion.to_csv(carpeta / "calibracion.csv", index=False)

    _figura_matriz(resultados, carpeta / "matriz-confusion.png")
    _figura_umbral(barrido, umbral, carpeta / "umbral.png")
    _figura_calibracion(calibracion, carpeta / "calibracion.png")


# --------------------------------------------------------------------------
# Corrida
# --------------------------------------------------------------------------

def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "desconocido"


def correr(
    nombre_modelo: str = "bosque",
    desbalance: str = "class_weight",
    ajustar_umbral: bool = False,
    criterio_umbral: str = "f1",
    solo_formulario: bool = False,
    columnas: list[str] | None = None,
    subconjunto: str | None = None,
    hiperparametros: dict | None = None,
    experimento: str = EXPERIMENTO,
    anidada: bool = False,
) -> dict:
    """Entrena el clasificador, lo evalua y registra la corrida en MLflow."""
    if nombre_modelo not in CATALOGO:
        raise ValueError(f"modelo desconocido: {nombre_modelo}. Opciones: {', '.join(CATALOGO)}")

    hiperparametros = hiperparametros or {}
    ficha = CATALOGO[nombre_modelo]

    # --- datos ---------------------------------------------------------
    crudos = cons.cargar()
    fallas = cons.verificar_carga(crudos)
    if fallas:
        raise RuntimeError("la carga no paso las comprobaciones: " + "; ".join(fallas))

    trabajo = cons.conjunto_de_trabajo(crudos)
    X, y, grupos = cons.matriz(trabajo, solo_formulario=solo_formulario, columnas=columnas)
    X_ent, X_eva, y_ent, y_eva, g_ent, g_eva = part.particionar(X, y, grupos)

    fugas = part.verificar_particion(g_ent, g_eva)
    if fugas:
        raise RuntimeError("la particion no aisla a los pacientes: " + "; ".join(fugas))

    # --- modelo --------------------------------------------------------
    if nombre_modelo == "trivial":
        desbalance, ajustar_umbral = "ninguno", False

    # El modo de desbalance decide el class_weight del bosque. Sin esto,
    # "ninguno" y "smote" heredaban el ponderado por defecto del constructor
    # y las alternativas entrenaban el mismo modelo.
    if nombre_modelo == "bosque" and "class_weight" not in hiperparametros:
        peso = hiperparametros.pop("peso_positivo", None)
        if desbalance == "peso" or peso is not None:
            # Peso explicito sobre la clase positiva. Se separa de
            # "balanced_subsample", que lo deriva de la razon entre clases y
            # equivale aqui a unos 7,8: como parametro barrible permite medir
            # el efecto del ponderado en vez de heredarlo.
            hiperparametros["class_weight"] = {0: 1, 1: float(peso or 5)}
        else:
            hiperparametros["class_weight"] = (
                "balanced_subsample" if desbalance == "class_weight" else None
            )

    estimador = ficha["constructor"](**hiperparametros)
    modelo = armar_estimador(X, estimador, desbalance)

    etiqueta = nombre_modelo
    if desbalance != "class_weight":
        etiqueta += f"-{desbalance}"
    if subconjunto:
        etiqueta += f"-{subconjunto}"
    elif solo_formulario:
        etiqueta += "-formulario"

    with mlflow.start_run(run_name=etiqueta, nested=anidada):
        # El umbral se estima con datos de entrenamiento, nunca de evaluacion.
        # Por defecto se usa el umbral fijo del proyecto. Ajustarlo por
        # validacion queda disponible, pero rompe la comparabilidad entre
        # configuraciones cuando se barren hiperparametros.
        umbral = (elegir_umbral(modelo, X_ent, y_ent, g_ent, criterio_umbral)
                  if ajustar_umbral else esq.UMBRAL_FIJO)

        modelo.fit(X_ent, y_ent)
        clasificador = FixedThresholdClassifier(modelo, threshold=umbral).fit(X_ent, y_ent)

        probabilidades = clasificador.predict_proba(X_eva)[:, 1]
        resultados = met.evaluar(y_eva, probabilidades, umbral)

        mlflow.log_params({
            "modelo": nombre_modelo,
            "desbalance": desbalance,
            "umbral_ajustado": ajustar_umbral,
            "criterio_umbral": criterio_umbral if ajustar_umbral else "",
            "n_predictoras": X.shape[1],
            "n_entrenamiento": len(X_ent),
            "n_evaluacion": len(X_eva),
            "semilla": esq.SEMILLA,
            "particion": "GroupShuffleSplit por patient_nbr",
            "subconjunto": subconjunto or ("formulario" if solo_formulario else "completo"),
            **{f"hp_{k}": v for k, v in estimador.get_params().items()
               if k in ("n_estimators", "max_depth", "min_samples_leaf",
                        "min_samples_split", "max_features", "criterion",
                        "class_weight", "strategy")},
        })

        mlflow.set_tags({
            "autor": "lealUniandes",
            "entrega": "2",
            "commit": _git("rev-parse", "--short", "HEAD"),
            "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "descripcion": ficha["descripcion"],
            "problema": "clasificacion binaria",
        })

        mlflow.log_metrics(resultados)

        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            registrar_artefactos(y_eva, probabilidades, umbral, carpeta)
            mlflow.log_artifacts(str(carpeta), artifact_path="evaluacion")

        # El artefacto recibe el DataFrame sin transformar: la preparacion
        # viaja dentro y la API no tiene que reproducirla.
        ejemplo = X_eva.head(5)
        firma = infer_signature(ejemplo, clasificador.predict(ejemplo))
        mlflow.sklearn.log_model(
            clasificador, name="modelo", signature=firma, input_example=ejemplo,
            skops_trusted_types=TIPOS_CONFIABLES,
        )

        resultados["run_id"] = mlflow.active_run().info.run_id

    return resultados


def main() -> None:
    lector = argparse.ArgumentParser(description=__doc__)
    lector.add_argument("--modelo", choices=list(CATALOGO), default="bosque")
    lector.add_argument("--todos", action="store_true")
    lector.add_argument("--desbalance", choices=DESBALANCES, default="class_weight")
    lector.add_argument("--ajustar-umbral", action="store_true",
                        help="elige el umbral por validacion en vez de usar el fijo")
    lector.add_argument("--criterio-umbral", default=esq.CRITERIO_UMBRAL,
                        choices=["f1", "f2", "f05", "exactitud_balanceada",
                                 "sensibilidad", "precision"])
    lector.add_argument("--solo-formulario", action="store_true")
    lector.add_argument("--experimento", default=EXPERIMENTO)
    lector.add_argument("--uri", default=None)
    args = lector.parse_args()

    if args.uri:
        mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment(args.experimento)

    for nombre in (list(CATALOGO) if args.todos else [args.modelo]):
        r = correr(
            nombre,
            desbalance=args.desbalance,
            ajustar_umbral=args.ajustar_umbral,
            criterio_umbral=args.criterio_umbral,
            solo_formulario=args.solo_formulario,
            experimento=args.experimento,
        )
        print(f"{nombre:<9} umbral {r['umbral']:.2f}  "
              f"exactitud {r['exactitud']:.3f}  precision {r['precision']:.3f}  "
              f"sensibilidad {r['sensibilidad']:.3f}  F1 {r['f1']:.3f}  "
              f"bal {r['exactitud_balanceada']:.3f}  roc {r['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
