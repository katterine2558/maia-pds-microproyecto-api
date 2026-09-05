"""Optimiza peso de clase y umbral para mejorar la sensibilidad de la regresion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.preparacion_regresion import preparar_datos
from src.models.regresion_logistica import (
    RAIZ,
    SALIDA,
    construir_modelo,
    dividir_por_paciente,
    verificar_separacion,
)


PESOS_POSITIVOS = [2, 3, 4, 5]

UMBRALES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

ARCHIVO_AJUSTE_C = (
    SALIDA
    / "ajuste_regularizacion.csv"
)


def obtener_mejor_c() -> float:
    """Recupera el C seleccionado previamente sobre validacion."""

    if not ARCHIVO_AJUSTE_C.exists():
        raise FileNotFoundError(
            "No existe ajuste_regularizacion.csv. "
            "Ejecute primero: "
            "python -m src.models.regresion_logistica"
        )

    ajuste = pd.read_csv(
        ARCHIVO_AJUSTE_C
    )

    mejor = ajuste.sort_values(
        [
            "pr_auc_validacion",
            "roc_auc_validacion",
        ],
        ascending=False,
    ).iloc[0]

    return float(
        mejor["C"]
    )


def metricas_umbral(
    y_real,
    probabilidad,
    umbral: float,
) -> dict:
    """Calcula metricas para un umbral de clasificacion."""

    prediccion = (
        probabilidad >= umbral
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        prediccion,
    ).ravel()

    return {
        "umbral": umbral,
        "roc_auc": roc_auc_score(
            y_real,
            probabilidad,
        ),
        "pr_auc": average_precision_score(
            y_real,
            probabilidad,
        ),
        "precision": precision_score(
            y_real,
            prediccion,
            zero_division=0,
        ),
        "recall": recall_score(
            y_real,
            prediccion,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_real,
            prediccion,
            beta=2,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_real,
            prediccion,
        ),
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    }


def evaluar_priorizacion_final(
    modelo,
    X_prueba,
    y_prueba,
    mejor_peso: int,
    mejor_c: float,
    mejor_umbral: float,
) -> pd.DataFrame:
    """Evalua el modelo final como ranking de riesgo."""

    probabilidad = modelo.predict_proba(
        X_prueba
    )[:, 1]

    evaluacion = pd.DataFrame(
        {
            "y_real": y_prueba.to_numpy(),
            "probabilidad": probabilidad,
        }
    ).sort_values(
        "probabilidad",
        ascending=False,
    )

    total_reingresos = int(
        evaluacion["y_real"].sum()
    )

    tasa_base = float(
        evaluacion["y_real"].mean()
    )

    resultados = []

    for proporcion in [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]:
        cantidad = round(
            len(evaluacion)
            * proporcion
        )

        priorizados = evaluacion.head(
            cantidad
        )

        reingresos_capturados = int(
            priorizados["y_real"].sum()
        )

        cobertura = (
            reingresos_capturados
            / total_reingresos
        )

        precision = float(
            priorizados["y_real"].mean()
        )

        lift = (
            precision
            / tasa_base
        )

        resultados.append(
            {
                "C": mejor_c,
                "peso_positivo": mejor_peso,
                "umbral_seleccionado": mejor_umbral,
                "porcentaje_priorizado": (
                    proporcion * 100
                ),
                "pacientes_priorizados": cantidad,
                "reingresos_capturados": (
                    reingresos_capturados
                ),
                "cobertura_reingresos": cobertura,
                "precision_en_priorizados": precision,
                "lift_vs_tasa_base": lift,
            }
        )

    tabla = pd.DataFrame(
        resultados
    )

    SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla.to_csv(
        SALIDA
        / "priorizacion_version_ajustada.csv",
        index=False,
    )

    print(
        "\n=== PRIORIZACION VERSION AJUSTADA ==="
    )

    for _, fila in tabla.iterrows():
        print(
            f"{fila['porcentaje_priorizado']:.0f} % | "
            f"captura="
            f"{fila['cobertura_reingresos'] * 100:.2f} % | "
            f"precision="
            f"{fila['precision_en_priorizados'] * 100:.2f} % | "
            f"lift="
            f"{fila['lift_vs_tasa_base']:.2f}x"
        )

    return tabla


def evaluar() -> None:
    """Selecciona peso y umbral en validacion y confirma una vez en prueba."""

    X, y, grupos, _ = preparar_datos()

    conjuntos = dividir_por_paciente(
        X,
        y,
        grupos,
    )

    verificar_separacion(
        conjuntos
    )

    mejor_c = obtener_mejor_c()

    print(
        "=== AJUSTE DE PESO DE CLASE ==="
    )

    print(
        f"C seleccionado previamente "
        f"en validacion: {mejor_c}"
    )

    resultados_pesos = []

    for peso in PESOS_POSITIVOS:

        modelo = construir_modelo(
            class_weight={
                0: 1,
                1: peso,
            },
            c=mejor_c,
        )

        modelo.fit(
            conjuntos["X_entrenamiento"],
            conjuntos["y_entrenamiento"],
        )

        probabilidad_validacion = (
            modelo.predict_proba(
                conjuntos["X_validacion"]
            )[:, 1]
        )

        resultado = metricas_umbral(
            conjuntos["y_validacion"],
            probabilidad_validacion,
            umbral=0.50,
        )

        resultados_pesos.append(
            {
                "C": mejor_c,
                "peso_positivo": peso,
                "roc_auc": resultado[
                    "roc_auc"
                ],
                "pr_auc": resultado[
                    "pr_auc"
                ],
                "precision": resultado[
                    "precision"
                ],
                "recall": resultado[
                    "recall"
                ],
                "f2": resultado[
                    "f2"
                ],
            }
        )

        print(
            f"peso={peso} | "
            f"ROC-AUC="
            f"{resultado['roc_auc']:.4f} | "
            f"PR-AUC="
            f"{resultado['pr_auc']:.4f} | "
            f"Precision="
            f"{resultado['precision']:.4f} | "
            f"Recall="
            f"{resultado['recall']:.4f} | "
            f"F2="
            f"{resultado['f2']:.4f}"
        )

    tabla_pesos = pd.DataFrame(
        resultados_pesos
    )

    mejor_fila_peso = (
        tabla_pesos.sort_values(
            [
                "f2",
                "pr_auc",
                "roc_auc",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    mejor_peso = int(
        mejor_fila_peso[
            "peso_positivo"
        ]
    )

    print(
        "\nMejor peso segun "
        f"F2 de validacion: {mejor_peso}"
    )

    # Este es el modelo usado tanto para seleccionar
    # el umbral como para la evaluacion final en prueba.
    # No se reentrena con X_desarrollo despues de calibrar
    # el umbral, evitando cambiar la calibracion.
    modelo_seleccionado = construir_modelo(
        class_weight={
            0: 1,
            1: mejor_peso,
        },
        c=mejor_c,
    )

    modelo_seleccionado.fit(
        conjuntos["X_entrenamiento"],
        conjuntos["y_entrenamiento"],
    )

    probabilidad_validacion = (
        modelo_seleccionado.predict_proba(
            conjuntos["X_validacion"]
        )[:, 1]
    )

    print(
        "\n=== AJUSTE DE UMBRAL ==="
    )

    resultados_umbrales = []

    for umbral in UMBRALES:

        resultado = metricas_umbral(
            conjuntos["y_validacion"],
            probabilidad_validacion,
            umbral=umbral,
        )

        resultados_umbrales.append(
            {
                "C": mejor_c,
                "peso_positivo": mejor_peso,
                **resultado,
            }
        )

        print(
            f"umbral={umbral:.2f} | "
            f"Precision="
            f"{resultado['precision']:.4f} | "
            f"Recall="
            f"{resultado['recall']:.4f} | "
            f"F2="
            f"{resultado['f2']:.4f} | "
            f"Accuracy="
            f"{resultado['accuracy']:.4f}"
        )

    tabla_umbrales = pd.DataFrame(
        resultados_umbrales
    )

    mejor_fila_umbral = (
        tabla_umbrales.sort_values(
            [
                "f2",
                "precision",
                "pr_auc",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    mejor_umbral = float(
        mejor_fila_umbral[
            "umbral"
        ]
    )

    print(
        "\nMejor umbral segun "
        f"F2 de validacion: {mejor_umbral}"
    )

    # Evaluacion final:
    # se utiliza exactamente el mismo modelo que fue
    # entrenado con X_entrenamiento y calibrado usando
    # X_validacion. X_prueba no participa en seleccion.
    probabilidad_prueba = (
        modelo_seleccionado.predict_proba(
            conjuntos["X_prueba"]
        )[:, 1]
    )

    resultado_final = metricas_umbral(
        conjuntos["y_prueba"],
        probabilidad_prueba,
        umbral=mejor_umbral,
    )

    resultado_final = {
        "C": mejor_c,
        "peso_positivo": mejor_peso,
        **resultado_final,
    }

    print(
        "\n=== VERSION FINAL AJUSTADA EN PRUEBA ==="
    )

    print(
        f"C: {mejor_c}"
    )
    print(
        f"Peso positivo: {mejor_peso}"
    )
    print(
        f"Umbral: {mejor_umbral}"
    )
    print(
        f"ROC-AUC: "
        f"{resultado_final['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC: "
        f"{resultado_final['pr_auc']:.4f}"
    )
    print(
        f"Precision: "
        f"{resultado_final['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{resultado_final['recall']:.4f}"
    )
    print(
        f"F2: "
        f"{resultado_final['f2']:.4f}"
    )
    print(
        f"Accuracy: "
        f"{resultado_final['accuracy']:.4f}"
    )
    print(
        f"TP: "
        f"{resultado_final['verdaderos_positivos']}"
    )
    print(
        f"FN: "
        f"{resultado_final['falsos_negativos']}"
    )
    print(
        f"FP: "
        f"{resultado_final['falsos_positivos']}"
    )
    print(
        f"TN: "
        f"{resultado_final['verdaderos_negativos']}"
    )

    SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla_pesos.to_csv(
        SALIDA
        / "ajuste_peso_clase.csv",
        index=False,
    )

    tabla_umbrales.to_csv(
        SALIDA
        / "ajuste_umbral.csv",
        index=False,
    )

    pd.DataFrame(
        [resultado_final]
    ).to_csv(
        SALIDA
        / "metricas_version_final.csv",
        index=False,
    )

    evaluar_priorizacion_final(
        modelo=modelo_seleccionado,
        X_prueba=conjuntos["X_prueba"],
        y_prueba=conjuntos["y_prueba"],
        mejor_peso=mejor_peso,
        mejor_c=mejor_c,
        mejor_umbral=mejor_umbral,
    )


if __name__ == "__main__":
    evaluar()