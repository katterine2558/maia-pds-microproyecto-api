"""Particion de entrenamiento y evaluacion.

El conjunto tiene 16 341 pacientes con mas de un encuentro, y el 46 % de las
filas les pertenece. Una particion aleatoria por fila deja encuentros del
mismo paciente a ambos lados y compromete el 41,5 % de las filas de
evaluacion (EDA 8.1).

El efecto no es menor: los pacientes con un solo encuentro reingresan al
4,26 % y los que aparecen varias veces al 19,76 %. La razon es circular —si
un paciente reingresa antes de 30 dias, ese reingreso genera un segundo
registro—, de modo que reconocer al paciente equivale a conocer el desenlace.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.features import esquema as esq


def particionar(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: pd.Series,
    fraccion_prueba: float = esq.FRACCION_PRUEBA,
    semilla: int = esq.SEMILLA,
):
    """Divide en entrenamiento y evaluacion agrupando por paciente.

    Devuelve (X_ent, X_eva, y_ent, y_eva, grupos_ent, grupos_eva).
    """
    division = GroupShuffleSplit(
        n_splits=1, test_size=fraccion_prueba, random_state=semilla
    )
    i_ent, i_eva = next(division.split(X, y, groups=grupos))

    return (
        X.iloc[i_ent],
        X.iloc[i_eva],
        y.iloc[i_ent],
        y.iloc[i_eva],
        grupos.iloc[i_ent],
        grupos.iloc[i_eva],
    )


def validacion_cruzada(n_folds: int = 5, semilla: int = esq.SEMILLA):
    """Validacion cruzada que respeta grupos y preserva la proporcion de positivos.

    `StratifiedGroupKFold` es la unica de las dos que hace ambas cosas: sin
    estratificar, con 11,4 % de positivos, un fold puede quedar con una
    proporcion sensiblemente distinta y la comparacion entre modelos pierde
    sentido.
    """
    return StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=semilla
    )


def verificar_particion(grupos_ent: pd.Series, grupos_eva: pd.Series) -> list[str]:
    """Comprueba que ningun paciente quedo en los dos conjuntos."""
    compartidos = set(grupos_ent) & set(grupos_eva)
    if compartidos:
        return [f"{len(compartidos):,} pacientes aparecen en entrenamiento y evaluacion"]
    return []
