"""Ajustes de umbral y peso de clase para mejorar la deteccion de reingresos."""

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


UMBRALES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

PESOS_POSITIVOS = [2, 3, 4, 5]


def metricas_umbral(y_real, probabilidad, umbral):
    prediccion = (probabilidad >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        prediccion,
    ).ravel()

    return {
        "umbral": umbral,
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


def evaluar():
    X, y, grupos, _ = preparar_datos()

    conjuntos = dividir_por_paciente(
        X,
        y,
        grupos,
    )

    verificar_separacion(conjuntos)

    resultados_pesos = []

    print("=== AJUSTE DE PESO DE CLASE ===")

    for peso in PESOS_POSITIVOS:
        modelo = construir_modelo(
            class_weight={0: 1, 1: peso},
            c=0.5,
        )

        modelo.fit(
            conjuntos["X_entrenamiento"],
            conjuntos["y_entrenamiento"],
        )

        prob_val = modelo.predict_proba(
            conjuntos["X_validacion"]
        )[:, 1]

        pred_val = (
            prob_val >= 0.5
        ).astype(int)

        resultado = {
            "peso_positivo": peso,
            "roc_auc": roc_auc_score(
                conjuntos["y_validacion"],
                prob_val,
            ),
            "pr_auc": average_precision_score(
                conjuntos["y_validacion"],
                prob_val,
            ),
            "precision": precision_score(
                conjuntos["y_validacion"],
                pred_val,
                zero_division=0,
            ),
            "recall": recall_score(
                conjuntos["y_validacion"],
                pred_val,
                zero_division=0,
            ),
            "f2": fbeta_score(
                conjuntos["y_validacion"],
                pred_val,
                beta=2,
                zero_division=0,
            ),
        }

        resultados_pesos.append(resultado)

        print(
            f"peso={peso} | "
            f"ROC-AUC={resultado['roc_auc']:.4f} | "
            f"PR-AUC={resultado['pr_auc']:.4f} | "
            f"Precision={resultado['precision']:.4f} | "
            f"Recall={resultado['recall']:.4f} | "
            f"F2={resultado['f2']:.4f}"
        )

    tabla_pesos = pd.DataFrame(
        resultados_pesos
    )

    mejor_peso = int(
        tabla_pesos.sort_values(
            ["f2", "pr_auc"],
            ascending=False,
        ).iloc[0]["peso_positivo"]
    )

    print(
        f"\nMejor peso segun F2 de validacion: "
        f"{mejor_peso}"
    )

    modelo = construir_modelo(
        class_weight={0: 1, 1: mejor_peso},
        c=0.5,
    )

    modelo.fit(
        conjuntos["X_entrenamiento"],
        conjuntos["y_entrenamiento"],
    )

    prob_val = modelo.predict_proba(
        conjuntos["X_validacion"]
    )[:, 1]

    resultados_umbrales = []

    print("\n=== AJUSTE DE UMBRAL ===")

    for umbral in UMBRALES:
        resultado = metricas_umbral(
            conjuntos["y_validacion"],
            prob_val,
            umbral,
        )

        resultados_umbrales.append(
            resultado
        )

        print(
            f"umbral={umbral:.2f} | "
            f"Precision={resultado['precision']:.4f} | "
            f"Recall={resultado['recall']:.4f} | "
            f"F2={resultado['f2']:.4f} | "
            f"Accuracy={resultado['accuracy']:.4f}"
        )

    tabla_umbrales = pd.DataFrame(
        resultados_umbrales
    )

    mejor_umbral = float(
        tabla_umbrales.sort_values(
            ["f2", "precision"],
            ascending=False,
        ).iloc[0]["umbral"]
    )

    print(
        f"\nMejor umbral segun F2 de validacion: "
        f"{mejor_umbral}"
    )

    modelo_final = construir_modelo(
        class_weight={0: 1, 1: mejor_peso},
        c=0.5,
    )

    modelo_final.fit(
        conjuntos["X_desarrollo"],
        conjuntos["y_desarrollo"],
    )

    prob_prueba = modelo_final.predict_proba(
        conjuntos["X_prueba"]
    )[:, 1]

    final = metricas_umbral(
        conjuntos["y_prueba"],
        prob_prueba,
        mejor_umbral,
    )

    final["roc_auc"] = roc_auc_score(
        conjuntos["y_prueba"],
        prob_prueba,
    )

    final["pr_auc"] = average_precision_score(
        conjuntos["y_prueba"],
        prob_prueba,
    )

    print("\n=== VERSION FINAL AJUSTADA EN PRUEBA ===")

    print(
        f"Peso positivo: {mejor_peso}\n"
        f"Umbral: {mejor_umbral}\n"
        f"ROC-AUC: {final['roc_auc']:.4f}\n"
        f"PR-AUC: {final['pr_auc']:.4f}\n"
        f"Precision: {final['precision']:.4f}\n"
        f"Recall: {final['recall']:.4f}\n"
        f"F2: {final['f2']:.4f}\n"
        f"Accuracy: {final['accuracy']:.4f}\n"
        f"TP: {final['verdaderos_positivos']}\n"
        f"FN: {final['falsos_negativos']}\n"
        f"FP: {final['falsos_positivos']}\n"
        f"TN: {final['verdaderos_negativos']}"
    )

    SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla_pesos.to_csv(
        SALIDA / "ajuste_peso_clase.csv",
        index=False,
    )

    tabla_umbrales.to_csv(
        SALIDA / "ajuste_umbral.csv",
        index=False,
    )

    pd.DataFrame(
        [final]
    ).to_csv(
        SALIDA / "metricas_version_final.csv",
        index=False,
    )

def evaluar_priorizacion_final():
    """Evalua la version ajustada como ranking de riesgo."""

    X, y, grupos, _ = preparar_datos()

    conjuntos = dividir_por_paciente(
        X,
        y,
        grupos,
    )

    verificar_separacion(conjuntos)

    modelo = construir_modelo(
        class_weight={0: 1, 1: 5},
        c=0.5,
    )

    modelo.fit(
        conjuntos["X_desarrollo"],
        conjuntos["y_desarrollo"],
    )

    probabilidad = modelo.predict_proba(
        conjuntos["X_prueba"]
    )[:, 1]

    evaluacion = pd.DataFrame(
        {
            "y_real": conjuntos["y_prueba"].to_numpy(),
            "probabilidad": probabilidad,
        }
    ).sort_values(
        "probabilidad",
        ascending=False,
    )

    total_reingresos = int(
        evaluacion["y_real"].sum()
    )

    tasa_base = evaluacion["y_real"].mean()

    resultados = []

    for proporcion in [0.10, 0.20, 0.30, 0.40, 0.50]:
        cantidad = round(
            len(evaluacion) * proporcion
        )

        grupo = evaluacion.head(cantidad)

        capturados = int(
            grupo["y_real"].sum()
        )

        cobertura = (
            capturados / total_reingresos
        )

        precision = grupo["y_real"].mean()

        lift = precision / tasa_base

        resultados.append(
            {
                "porcentaje_priorizado": proporcion * 100,
                "reingresos_capturados": capturados,
                "cobertura": cobertura,
                "precision": precision,
                "lift": lift,
            }
        )

    tabla = pd.DataFrame(resultados)

    print(
        "\n=== PRIORIZACION VERSION AJUSTADA ==="
    )

    for _, fila in tabla.iterrows():
        print(
            f"{fila['porcentaje_priorizado']:.0f} % | "
            f"captura={fila['cobertura'] * 100:.2f} % | "
            f"precision={fila['precision'] * 100:.2f} % | "
            f"lift={fila['lift']:.2f}x"
        )

    tabla.to_csv(
        SALIDA / "priorizacion_version_ajustada.csv",
        index=False,
    )

if __name__ == "__main__":
    evaluar()
    evaluar_priorizacion_final()