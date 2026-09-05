"""Entrenamiento y evaluacion de modelos de regresion logistica."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.preparacion_regresion import (
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
    preparar_datos,
)


RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "docs" / "soportes" / "modelos" / "regresion_logistica"

SEMILLA = 42
TAMANO_PRUEBA = 0.20
TAMANO_VALIDACION = 0.20

VALORES_C = [0.1, 0.5, 1.0, 2.0, 10.0]


def construir_preprocesador() -> ColumnTransformer:
    """Define transformaciones numericas y categoricas."""

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

    return ColumnTransformer(
        transformers=[
            ("num", numericas, VARIABLES_NUMERICAS),
            ("cat", categoricas, VARIABLES_CATEGORICAS),
        ],
        remainder="drop",
    )


def construir_modelo(
    class_weight=None,
    c: float = 1.0,
) -> Pipeline:
    """Construye el pipeline completo de preprocesamiento y regresion."""

    return Pipeline(
        steps=[
            ("preprocesamiento", construir_preprocesador()),
            (
                "modelo",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="liblinear",
                    random_state=SEMILLA,
                ),
            ),
        ]
    )


def dividir_por_paciente(X, y, grupos):
    """Separa desarrollo y prueba sin compartir pacientes."""

    separador_prueba = GroupShuffleSplit(
        n_splits=1,
        test_size=TAMANO_PRUEBA,
        random_state=SEMILLA,
    )

    idx_desarrollo, idx_prueba = next(
        separador_prueba.split(X, y, groups=grupos)
    )

    X_desarrollo = X.iloc[idx_desarrollo].copy()
    y_desarrollo = y.iloc[idx_desarrollo].copy()
    grupos_desarrollo = grupos.iloc[idx_desarrollo].copy()

    X_prueba = X.iloc[idx_prueba].copy()
    y_prueba = y.iloc[idx_prueba].copy()
    grupos_prueba = grupos.iloc[idx_prueba].copy()

    separador_validacion = GroupShuffleSplit(
        n_splits=1,
        test_size=TAMANO_VALIDACION,
        random_state=SEMILLA,
    )

    idx_entrenamiento, idx_validacion = next(
        separador_validacion.split(
            X_desarrollo,
            y_desarrollo,
            groups=grupos_desarrollo,
        )
    )

    X_entrenamiento = X_desarrollo.iloc[idx_entrenamiento].copy()
    y_entrenamiento = y_desarrollo.iloc[idx_entrenamiento].copy()
    grupos_entrenamiento = grupos_desarrollo.iloc[idx_entrenamiento].copy()

    X_validacion = X_desarrollo.iloc[idx_validacion].copy()
    y_validacion = y_desarrollo.iloc[idx_validacion].copy()
    grupos_validacion = grupos_desarrollo.iloc[idx_validacion].copy()

    conjuntos = {
        "X_entrenamiento": X_entrenamiento,
        "y_entrenamiento": y_entrenamiento,
        "grupos_entrenamiento": grupos_entrenamiento,
        "X_validacion": X_validacion,
        "y_validacion": y_validacion,
        "grupos_validacion": grupos_validacion,
        "X_desarrollo": X_desarrollo,
        "y_desarrollo": y_desarrollo,
        "grupos_desarrollo": grupos_desarrollo,
        "X_prueba": X_prueba,
        "y_prueba": y_prueba,
        "grupos_prueba": grupos_prueba,
    }

    return conjuntos


def calcular_metricas(nombre, modelo, X, y):
    """Calcula metricas de clasificacion y matriz de confusion."""

    probabilidad = modelo.predict_proba(X)[:, 1]
    prediccion = modelo.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, prediccion).ravel()

    return {
        "modelo": nombre,
        "roc_auc": roc_auc_score(y, probabilidad),
        "pr_auc": average_precision_score(y, probabilidad),
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
        "f1": f1_score(
            y,
            prediccion,
            zero_division=0,
        ),
        "accuracy": accuracy_score(y, prediccion),
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    }


def verificar_separacion(conjuntos) -> None:
    """Comprueba que ningun paciente se comparta entre particiones."""

    pacientes_train = set(conjuntos["grupos_entrenamiento"])
    pacientes_val = set(conjuntos["grupos_validacion"])
    pacientes_test = set(conjuntos["grupos_prueba"])

    solapamiento_train_val = pacientes_train & pacientes_val
    solapamiento_train_test = pacientes_train & pacientes_test
    solapamiento_val_test = pacientes_val & pacientes_test

    if (
        solapamiento_train_val
        or solapamiento_train_test
        or solapamiento_val_test
    ):
        raise RuntimeError(
            "Se detectaron pacientes compartidos entre particiones."
        )


def imprimir_distribucion(nombre, y, grupos) -> None:
    """Resume tamano, pacientes y frecuencia de la clase positiva."""

    print(
        f"{nombre}: "
        f"{len(y):,} encuentros | "
        f"{grupos.nunique():,} pacientes | "
        f"{int(y.sum()):,} positivos | "
        f"{y.mean() * 100:.2f} % positivo"
    )


def entrenar() -> None:
    """Entrena las versiones de regresion y guarda sus resultados."""

    X, y, grupos, _ = preparar_datos()

    conjuntos = dividir_por_paciente(X, y, grupos)
    verificar_separacion(conjuntos)

    print("=== PARTICION POR PACIENTE ===")
    imprimir_distribucion(
        "Entrenamiento",
        conjuntos["y_entrenamiento"],
        conjuntos["grupos_entrenamiento"],
    )
    imprimir_distribucion(
        "Validacion",
        conjuntos["y_validacion"],
        conjuntos["grupos_validacion"],
    )
    imprimir_distribucion(
        "Prueba",
        conjuntos["y_prueba"],
        conjuntos["grupos_prueba"],
    )

    print("\nPacientes compartidos entre particiones: 0")

    resultados = []

    print("\n=== V1: REGRESION LOGISTICA BASE ===")

    modelo_v1 = construir_modelo(
        class_weight=None,
        c=1.0,
    )

    modelo_v1.fit(
        conjuntos["X_desarrollo"],
        conjuntos["y_desarrollo"],
    )

    resultado_v1 = calcular_metricas(
        "V1_base",
        modelo_v1,
        conjuntos["X_prueba"],
        conjuntos["y_prueba"],
    )

    resultados.append(resultado_v1)

    print(
        f"ROC-AUC: {resultado_v1['roc_auc']:.4f} | "
        f"PR-AUC: {resultado_v1['pr_auc']:.4f} | "
        f"Recall: {resultado_v1['recall']:.4f} | "
        f"F1: {resultado_v1['f1']:.4f}"
    )

    print("\n=== V2: REGRESION LOGISTICA BALANCEADA ===")

    modelo_v2 = construir_modelo(
        class_weight="balanced",
        c=1.0,
    )

    modelo_v2.fit(
        conjuntos["X_desarrollo"],
        conjuntos["y_desarrollo"],
    )

    resultado_v2 = calcular_metricas(
        "V2_balanceada",
        modelo_v2,
        conjuntos["X_prueba"],
        conjuntos["y_prueba"],
    )

    resultados.append(resultado_v2)

    print(
        f"ROC-AUC: {resultado_v2['roc_auc']:.4f} | "
        f"PR-AUC: {resultado_v2['pr_auc']:.4f} | "
        f"Recall: {resultado_v2['recall']:.4f} | "
        f"F1: {resultado_v2['f1']:.4f}"
    )

    print("\n=== V3: AJUSTE DE REGULARIZACION ===")

    ajuste = []

    for c in VALORES_C:
        modelo = construir_modelo(
            class_weight="balanced",
            c=c,
        )

        modelo.fit(
            conjuntos["X_entrenamiento"],
            conjuntos["y_entrenamiento"],
        )

        metricas = calcular_metricas(
            f"C={c}",
            modelo,
            conjuntos["X_validacion"],
            conjuntos["y_validacion"],
        )

        ajuste.append(
            {
                "C": c,
                "roc_auc_validacion": metricas["roc_auc"],
                "pr_auc_validacion": metricas["pr_auc"],
                "recall_validacion": metricas["recall"],
                "f1_validacion": metricas["f1"],
            }
        )

        print(
            f"C={c:<4} | "
            f"ROC-AUC={metricas['roc_auc']:.4f} | "
            f"PR-AUC={metricas['pr_auc']:.4f} | "
            f"Recall={metricas['recall']:.4f} | "
            f"F1={metricas['f1']:.4f}"
        )

    tabla_ajuste = pd.DataFrame(ajuste)

    mejor_fila = tabla_ajuste.sort_values(
        ["pr_auc_validacion", "roc_auc_validacion"],
        ascending=False,
    ).iloc[0]

    mejor_c = float(mejor_fila["C"])

    print(f"\nMejor C segun PR-AUC de validacion: {mejor_c}")

    modelo_v3 = construir_modelo(
        class_weight="balanced",
        c=mejor_c,
    )

    modelo_v3.fit(
        conjuntos["X_desarrollo"],
        conjuntos["y_desarrollo"],
    )

    resultado_v3 = calcular_metricas(
        f"V3_balanceada_C_{mejor_c}",
        modelo_v3,
        conjuntos["X_prueba"],
        conjuntos["y_prueba"],
    )

    resultados.append(resultado_v3)

    SALIDA.mkdir(parents=True, exist_ok=True)

    tabla_resultados = pd.DataFrame(resultados)

    tabla_resultados.to_csv(
        SALIDA / "metricas_regresion_logistica.csv",
        index=False,
    )

    tabla_ajuste.to_csv(
        SALIDA / "ajuste_regularizacion.csv",
        index=False,
    )

    print("\n=== RESULTADOS EN PRUEBA ===")

    columnas = [
        "modelo",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "verdaderos_negativos",
        "falsos_positivos",
        "falsos_negativos",
        "verdaderos_positivos",
    ]

    print(
        tabla_resultados[columnas]
        .round(4)
        .to_string(index=False)
    )

    print("\nResultados guardados en:")
    print(SALIDA.relative_to(RAIZ))


if __name__ == "__main__":
    entrenar()