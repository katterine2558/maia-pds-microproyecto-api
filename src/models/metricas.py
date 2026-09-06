"""Evaluacion del clasificador binario.

El modelo resuelve un problema de clasificacion binaria: para cada egreso
responde si el paciente reingresara antes de 30 dias. Junto a esa clase
entrega la probabilidad estimada, que el tablero usa como dato de referencia
para que el personal de enfermeria filtre el listado por el valor que digite.

De ahi se desprende que la evaluacion tenga dos partes:

  - La clase predicha se evalua con la matriz de confusion y las medidas que
    se derivan de ella.
  - La probabilidad se evalua aparte, porque una clase correcta acompanada de
    una probabilidad mal calibrada haria que el filtro del tablero seleccione
    pacientes que no corresponden al riesgo que la enfermera creyo pedir.

Con 11,4 % de positivos, la exactitud por si sola no informa: un clasificador
que responda siempre "no reingresa" alcanza 88,6 % sin identificar a ningun
paciente en riesgo (EDA 4.3). Por eso se reportan siempre juntas la
sensibilidad, la precision y la exactitud balanceada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from src.features import esquema as esq

UMBRAL_POR_DEFECTO = 0.5


# --------------------------------------------------------------------------
# Clase predicha
# --------------------------------------------------------------------------

def matriz_confusion(y_real, y_predicho) -> dict:
    """Los cuatro conteos, con nombres explicitos.

    Verdadero positivo es el paciente que reingresa y el modelo marca; falso
    negativo es el que reingresa y el modelo deja pasar, que es el error caro
    en este problema porque significa un egreso sin control programado.
    """
    vn, fp, fn, vp = confusion_matrix(y_real, y_predicho, labels=[0, 1]).ravel()
    return {
        "verdaderos_positivos": int(vp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_negativos": int(vn),
    }


def evaluar_clase(y_real, y_predicho) -> dict:
    """Medidas derivadas de la matriz de confusion.

    La especificidad y la exactitud balanceada acompanan a las habituales
    porque con clases desbalanceadas la exactitud sola es enganosa: promedia
    un acierto masivo sobre la clase mayoritaria con un fracaso sobre la que
    interesa.
    """
    conteos = matriz_confusion(y_real, y_predicho)
    vn = conteos["verdaderos_negativos"]
    fp = conteos["falsos_positivos"]

    return {
        **conteos,
        "exactitud": round(accuracy_score(y_real, y_predicho), 4),
        "precision": round(precision_score(y_real, y_predicho, zero_division=0), 4),
        # Se registra como "recall" para que todas las familias de modelos del
        # proyecto compartan el nombre de la metrica y sean ordenables juntas.
        "recall": round(recall_score(y_real, y_predicho, zero_division=0), 4),
        "especificidad": round(vn / (vn + fp), 4) if (vn + fp) else 0.0,
        "f1": round(f1_score(y_real, y_predicho, zero_division=0), 4),
        # F2 pesa el doble la sensibilidad: sirve cuando el error caro es el
        # falso negativo. F0.5 pesa el doble la precision: sirve cuando el
        # error caro es el falso positivo, que en este proyecto es el
        # seguimiento programado a un paciente que no iba a reingresar.
        "f2": round(fbeta_score(y_real, y_predicho, beta=2, zero_division=0), 4),
        "f05": round(fbeta_score(y_real, y_predicho, beta=0.5, zero_division=0), 4),
        "exactitud_balanceada": round(balanced_accuracy_score(y_real, y_predicho), 4),
        # Cuantos seguimientos hay que programar por cada reingreso evitado.
        # Es la lectura operativa de la precision: con precision 0,20 se
        # invierten cinco seguimientos por cada acierto.
        "seguimientos_por_acierto": (round((vp + fp) / vp, 2)
                                     if (vp := conteos["verdaderos_positivos"]) else None),
    }


# --------------------------------------------------------------------------
# Probabilidad
# --------------------------------------------------------------------------

def evaluar_probabilidad(y_real, probabilidades) -> dict:
    """Calidad de la probabilidad que acompana a cada clasificacion.

    `roc_auc` y `pr_auc` miden si la probabilidad separa las clases con
    independencia del umbral. `brier` mide si su valor es interpretable: es el
    error cuadratico medio entre la probabilidad y el desenlace, y es lo que
    determina si el filtro del tablero se comporta como el usuario espera.
    Un modelo puede clasificar bien y estar mal calibrado.
    """
    return {
        "roc_auc": round(roc_auc_score(y_real, probabilidades), 4),
        "pr_auc": round(average_precision_score(y_real, probabilidades), 4),
        "brier": round(brier_score_loss(y_real, probabilidades), 5),
    }


def curva_calibracion(y_real, probabilidades, n_grupos: int = 10) -> pd.DataFrame:
    """Compara la probabilidad estimada con la frecuencia observada.

    Los grupos se arman por cuantiles: con probabilidades concentradas en
    valores bajos, los cortes uniformes dejan los grupos altos casi vacios.
    """
    df = pd.DataFrame({"estimada": np.asarray(probabilidades),
                       "real": np.asarray(y_real)})
    df["grupo"] = pd.qcut(df["estimada"], q=n_grupos, duplicates="drop")

    resumen = df.groupby("grupo", observed=True).agg(
        n=("real", "size"),
        estimada=("estimada", "mean"),
        observada=("real", "mean"),
    ).reset_index(drop=True)

    resumen["diferencia"] = resumen["observada"] - resumen["estimada"]
    return resumen.round(4)


# --------------------------------------------------------------------------
# Evaluacion completa
# --------------------------------------------------------------------------

def evaluar(y_real, probabilidades, umbral: float = UMBRAL_POR_DEFECTO) -> dict:
    """Diccionario plano con todas las metricas, apto para `mlflow.log_metrics`."""
    y_real = np.asarray(y_real)
    probabilidades = np.asarray(probabilidades)
    y_predicho = (probabilidades >= umbral).astype(int)

    return {
        **evaluar_clase(y_real, y_predicho),
        **evaluar_probabilidad(y_real, probabilidades),
        "umbral": round(float(umbral), 4),
        "tasa_base": round(float(y_real.mean()), 4),
        "n": int(len(y_real)),
    }


# --------------------------------------------------------------------------
# Umbral de decision
# --------------------------------------------------------------------------

def barrido_umbral(y_real, probabilidades, umbrales=None) -> pd.DataFrame:
    """Metricas de clasificacion a lo largo de una grid de umbrales.

    El umbral forma parte del clasificador y no se hereda: 0,5 es el valor por
    defecto de `predict`, pero con 11,4 % de positivos casi ningun paciente lo
    supera y el modelo termina marcando a muy pocos. Elegirlo es una decision
    que debe tomarse sobre entrenamiento, nunca sobre el conjunto reservado.
    """
    y_real = np.asarray(y_real)
    probabilidades = np.asarray(probabilidades)
    umbrales = umbrales if umbrales is not None else np.arange(0.05, 0.95, 0.01)

    filas = []
    for u in umbrales:
        y_predicho = (probabilidades >= u).astype(int)
        filas.append({"umbral": round(float(u), 3), **evaluar_clase(y_real, y_predicho)})

    return pd.DataFrame(filas)


def costo_de_los_errores(
    y_real, y_predicho, costo_falso_positivo: float = 1.0, costo_falso_negativo: float = 5.0
) -> dict:
    """Traduce la matriz de confusion a la unidad en que se decide.

    Los dos errores no cuestan lo mismo y no son intercambiables:

    - Un falso positivo consume un seguimiento —una llamada, un control
      agendado— en un paciente que no iba a reingresar. El costo es el recurso
      desperdiciado y es acotado.
    - Un falso negativo deja salir sin control a un paciente que reingresa. El
      costo es el reingreso mismo, que la literatura sobre readmision estima
      en un orden de magnitud mayor que el de una llamada de seguimiento.

    Los valores por defecto expresan esa razon de cinco a uno, pero son un
    parametro explicito y no una constante escondida: cada institucion tiene
    su propia relacion entre el costo de un control y el de una readmision, y
    el umbral optimo se mueve con ella.
    """
    conteos = matriz_confusion(y_real, y_predicho)
    fp = conteos["falsos_positivos"]
    fn = conteos["falsos_negativos"]

    return {
        **conteos,
        "costo_falsos_positivos": round(fp * costo_falso_positivo, 1),
        "costo_falsos_negativos": round(fn * costo_falso_negativo, 1),
        "costo_total": round(fp * costo_falso_positivo + fn * costo_falso_negativo, 1),
    }


def umbral_de_minimo_costo(
    y_real, probabilidades,
    costo_falso_positivo: float = 1.0, costo_falso_negativo: float = 5.0,
) -> pd.DataFrame:
    """Costo total a lo largo de los umbrales, para localizar el minimo.

    Es la alternativa a elegir el umbral por F1 cuando la institucion puede
    poner cifras a los dos errores.
    """
    filas = []
    for u in np.arange(0.05, 0.95, 0.01):
        y_predicho = (np.asarray(probabilidades) >= u).astype(int)
        filas.append({"umbral": round(float(u), 3),
                      **costo_de_los_errores(y_real, y_predicho,
                                             costo_falso_positivo, costo_falso_negativo)})
    return pd.DataFrame(filas)


def mejor_umbral(y_real, probabilidades, criterio: str = "f1") -> float:
    """Umbral que maximiza el criterio indicado.

    Se calcula siempre sobre datos de entrenamiento o validacion. Ajustarlo
    mirando el conjunto reservado convertiria su desempeno en el maximo de
    todos los umbrales probados y no en una estimacion del comportamiento
    futuro.
    """
    barrido = barrido_umbral(y_real, probabilidades)
    return float(barrido.loc[barrido[criterio].idxmax(), "umbral"])


def curva_pr(y_real, probabilidades) -> pd.DataFrame:
    """Puntos de la curva de precision y sensibilidad, para graficar."""
    precision, sensibilidad, umbrales = precision_recall_curve(y_real, probabilidades)
    return pd.DataFrame({
        "precision": precision[:-1],
        "sensibilidad": sensibilidad[:-1],
        "umbral": umbrales,
    })
