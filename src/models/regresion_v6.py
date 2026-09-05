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


def construir_modelo(c, l1_ratio, peso):
    numericas = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
        ]
    )

    categoricas = Pipeline(
        steps=[
            (
                "imputacion",
                SimpleImputer(strategy="most_frequent"),
            ),
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
        ]
    )

    modelo = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=c,
        l1_ratio=l1_ratio,
        class_weight={0: 1, 1: peso},
        max_iter=4000,
        random_state=SEMILLA,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", modelo),
        ]
    )


def metricas(y, prob, umbral=0.5):
    pred = (prob >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
    ).ravel()

    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "precision": precision_score(
            y,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            pred,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y,
            pred,
            beta=2,
            zero_division=0,
        ),
        "accuracy": accuracy_score(y, pred),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def ejecutar():
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
        for l1 in VALORES_L1:
            for peso in PESOS:

                modelo = construir_modelo(
                    c=c,
                    l1_ratio=l1,
                    peso=peso,
                )

                modelo.fit(
                    conjuntos["X_entrenamiento"],
                    conjuntos["y_entrenamiento"],
                )

                prob = modelo.predict_proba(
                    conjuntos["X_validacion"]
                )[:, 1]

                m = metricas(
                    conjuntos["y_validacion"],
                    prob,
                    umbral=0.5,
                )

                resultados.append(
                    {
                        "C": c,
                        "l1_ratio": l1,
                        "peso": peso,
                        **m,
                    }
                )

    tabla = pd.DataFrame(resultados)

    mejores = tabla.sort_values(
        ["pr_auc", "f2", "roc_auc"],
        ascending=False,
    )

    print(
        mejores.head(10)
        .round(4)
        .to_string(index=False)
    )

    mejor = mejores.iloc[0]

    mejor_c = float(mejor["C"])
    mejor_l1 = float(mejor["l1_ratio"])
    mejor_peso = int(mejor["peso"])

    print("\nMejor configuracion:")
    print(f"C = {mejor_c}")
    print(f"l1_ratio = {mejor_l1}")
    print(f"peso positivo = {mejor_peso}")

    modelo_final = construir_modelo(
        c=mejor_c,
        l1_ratio=mejor_l1,
        peso=mejor_peso,
    )

    modelo_final.fit(
        conjuntos["X_desarrollo"],
        conjuntos["y_desarrollo"],
    )

    prob_test = modelo_final.predict_proba(
        conjuntos["X_prueba"]
    )[:, 1]

    # Primero medimos con umbral 0.50.
    final_05 = metricas(
        conjuntos["y_prueba"],
        prob_test,
        umbral=0.5,
    )

    print("\n=== V6 EN PRUEBA - UMBRAL 0.50 ===")

    for clave, valor in final_05.items():
        if isinstance(valor, float):
            print(f"{clave}: {valor:.4f}")
        else:
            print(f"{clave}: {valor}")

    tabla.to_csv(
        SALIDA / "busqueda_v6_elasticnet.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "C": mejor_c,
                "l1_ratio": mejor_l1,
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

## Ajustes orientados a sensibilidad

Después de las primeras tres versiones se evaluaron configuraciones adicionales para mejorar la detección de reingresos.

### V4 - Ajuste del peso de la clase positiva

Se evaluaron pesos de 2, 3, 4 y 5 para la clase positiva. El mejor resultado de validación según F2 se obtuvo con peso positivo igual a 5.

### V5 - Peso de clase y ajuste del umbral

Sobre la configuración anterior se evaluaron umbrales entre 0,20 y 0,50. El mejor F2 de validación se obtuvo con umbral 0,30.

Resultados en prueba:

- Peso positivo: 5
- C: 0,5
- Umbral: 0,30
- ROC-AUC: 0,6594
- PR-AUC: 0,2143
- Precision: 0,1387
- Recall: 0,8201
- F2: 0,4137
- Accuracy: 0,4130
- Verdaderos positivos: 1.810
- Falsos negativos: 397
- Falsos positivos: 11.237
- Verdaderos negativos: 6.377

El aumento del recall implica un costo importante en falsos positivos. Por esta razón esta configuración se interpreta como una alternativa orientada a sensibilidad y no como una mejora general de todas las métricas.

### V6 - Elastic Net y variables derivadas

Se evaluó una versión con regularización Elastic Net y variables derivadas de utilización previa.

Resultados en prueba con umbral 0,50:

- ROC-AUC: 0,6603
- PR-AUC: 0,2139
- Precision: 0,3211
- Recall: 0,1382
- F2: 0,1560
- Accuracy: 0,8715

Aunque V6 obtuvo el ROC-AUC más alto, la diferencia fue marginal y el PR-AUC no mejoró. Además, su recall con el umbral estándar fue considerablemente menor que el de V5. Por lo tanto, no se seleccionó como configuración final.

## Selección de la regresión

La configuración seleccionada para el uso operativo es V5, con peso positivo 5, C igual a 0,5 y umbral 0,30. La selección responde al objetivo de aumentar la detección de pacientes con reingreso temprano.

Sin embargo, para la priorización diaria se recomienda utilizar principalmente la probabilidad estimada como puntaje de riesgo y ordenar los egresos según la capacidad disponible, en lugar de depender exclusivamente de una clasificación binaria fija.

La evaluación por capacidad mostró que el 10 % de mayor riesgo concentra el 22,84 % de los reingresos y el 30 % concentra aproximadamente el 49,4 %.