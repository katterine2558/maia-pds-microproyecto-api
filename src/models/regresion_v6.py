"""Version 6 de regresion logistica con variables derivadas y Elastic Net."""

from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.preparacion_regresion import (
    VARIABLES_CATEGORICAS,
    preparar_datos,
)
from src.models.regresion_logistica import (
    SALIDA,
    dividir_por_paciente,
    verificar_separacion,
)


SEMILLA = 42

VALORES_C = [0.05, 0.1, 0.5, 1.0, 2.0]
VALORES_L1 = [0.0, 0.25, 0.5, 0.75, 1.0]
PESOS = [3, 5, 7]

VARIABLES_NUMERICAS_V6 = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "utilizacion_previa_total",
    "tuvo_hospitalizacion_previa",
    "tuvo_urgencia_previa",
    "tuvo_consulta_ambulatoria",
]


def agregar_variables(X: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables derivadas de utilizacion previa."""
    X = X.copy()

    for col in [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ]:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    X["utilizacion_previa_total"] = (
        X["number_outpatient"].fillna(0)
        + X["number_emergency"].fillna(0)
        + X["number_inpatient"].fillna(0)
    )

    X["tuvo_hospitalizacion_previa"] = (
        X["number_inpatient"].fillna(0) > 0
    ).astype(int)

    X["tuvo_urgencia_previa"] = (
        X["number_emergency"].fillna(0) > 0
    ).astype(int)

    X["tuvo_consulta_ambulatoria"] = (
        X["number_outpatient"].fillna(0) > 0
    ).astype(int)

    return X


def construir_modelo(
    c: float,
    l1_ratio: float,
    peso: int,
) -> Pipeline:
    """Construye la regresion logistica V6 con Elastic Net."""

    numericas = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
        ]
    )

    categoricas = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocesamiento = ColumnTransformer(
        transformers=[
            ("num", numericas, VARIABLES_NUMERICAS_V6),
            ("cat", categoricas, VARIABLES_CATEGORICAS),
        ],
        remainder="drop",
    )

    modelo = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=c,
        l1_ratio=l1_ratio,
        class_weight={0: 1, 1: peso},
        max_iter=4000,
        random_state=SEMILLA,
    )

    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", modelo),
        ]
    )


def metricas(
    y,
    probabilidad,
    umbral: float = 0.5,
) -> dict:
    """Calcula metricas de clasificacion para un umbral dado."""

    prediccion = (
        probabilidad >= umbral
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediccion,
    ).ravel()

    return {
        "roc_auc": roc_auc_score(
            y,
            probabilidad,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilidad,
        ),
        "precision": precision_score(
            y,
            prediccion,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            prediccion,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y,
            prediccion,
            beta=2,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y,
            prediccion,
        ),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def ejecutar() -> None:
    """Busca la mejor configuracion V6 y la evalua en prueba."""

    X, y, grupos, _ = preparar_datos()

    X = agregar_variables(X)

    conjuntos = dividir_por_paciente(
        X,
        y,
        grupos,
    )

    verificar_separacion(conjuntos)

    resultados = []

    print("=== V6: BUSQUEDA ELASTIC NET ===")

    for c in VALORES_C:
        for l1_ratio in VALORES_L1:
            for peso in PESOS:

                modelo = construir_modelo(
                    c=c,
                    l1_ratio=l1_ratio,
                    peso=peso,
                )

                modelo.fit(
                    conjuntos["X_entrenamiento"],
                    conjuntos["y_entrenamiento"],
                )

                probabilidad_validacion = modelo.predict_proba(
                    conjuntos["X_validacion"]
                )[:, 1]

                resultado = metricas(
                    conjuntos["y_validacion"],
                    probabilidad_validacion,
                    umbral=0.5,
                )

                resultados.append(
                    {
                        "C": c,
                        "l1_ratio": l1_ratio,
                        "peso": peso,
                        **resultado,
                    }
                )

    tabla = pd.DataFrame(
        resultados
    )

    mejores = tabla.sort_values(
        [
            "pr_auc",
            "f2",
            "roc_auc",
        ],
        ascending=False,
    )

    print(
        mejores.head(10)
        .round(4)
        .to_string(index=False)
    )

    mejor = mejores.iloc[0]

    mejor_c = float(
        mejor["C"]
    )
    mejor_l1_ratio = float(
        mejor["l1_ratio"]
    )
    mejor_peso = int(
        mejor["peso"]
    )

    print("\nMejor configuracion:")
    print(f"C = {mejor_c}")
    print(
        f"l1_ratio = "
        f"{mejor_l1_ratio}"
    )
    print(
        f"peso positivo = "
        f"{mejor_peso}"
    )

    modelo_final = construir_modelo(
        c=mejor_c,
        l1_ratio=mejor_l1_ratio,
        peso=mejor_peso,
    )

    modelo_final.fit(
        conjuntos["X_desarrollo"],
        conjuntos["y_desarrollo"],
    )

    probabilidad_prueba = modelo_final.predict_proba(
        conjuntos["X_prueba"]
    )[:, 1]

    final_05 = metricas(
        conjuntos["y_prueba"],
        probabilidad_prueba,
        umbral=0.5,
    )

    print(
        "\n=== V6 EN PRUEBA - "
        "UMBRAL 0.50 ==="
    )

    for clave, valor in final_05.items():
        if isinstance(
            valor,
            float,
        ):
            print(
                f"{clave}: "
                f"{valor:.4f}"
            )
        else:
            print(
                f"{clave}: "
                f"{valor}"
            )

    SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla.to_csv(
        SALIDA
        / "busqueda_v6_elasticnet.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "C": mejor_c,
                "l1_ratio": mejor_l1_ratio,
                "peso": mejor_peso,
                **final_05,
            }
        ]
    ).to_csv(
        SALIDA / "metricas_v6.csv",
        index=False,
    )


if __name__ == "__main__":
    ejecutar()