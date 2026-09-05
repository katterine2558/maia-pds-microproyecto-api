"""Evalua el modelo de regresion como herramienta de priorizacion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.preparacion_regresion import preparar_datos
from src.models.regresion_logistica import (
    RAIZ,
    SALIDA,
    construir_modelo,
    dividir_por_paciente,
    verificar_separacion,
)


ARCHIVO_AJUSTE = SALIDA / "ajuste_regularizacion.csv"

PROPORCIONES = [0.10, 0.20, 0.30, 0.40, 0.50]


def obtener_mejor_c() -> float:
    """Recupera el valor de C elegido con el conjunto de validacion."""

    ajuste = pd.read_csv(ARCHIVO_AJUSTE)

    mejor = ajuste.sort_values(
        ["pr_auc_validacion", "roc_auc_validacion"],
        ascending=False,
    ).iloc[0]

    return float(mejor["C"])


def evaluar_priorizacion() -> pd.DataFrame:
    """Calcula cuanto riesgo se captura al priorizar por probabilidad."""

    X, y, grupos, _ = preparar_datos()

    conjuntos = dividir_por_paciente(X, y, grupos)
    verificar_separacion(conjuntos)

    mejor_c = obtener_mejor_c()

    modelo = construir_modelo(
        class_weight="balanced",
        c=mejor_c,
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

    positivos_totales = int(evaluacion["y_real"].sum())
    tasa_base = evaluacion["y_real"].mean()

    resultados = []

    for proporcion in PROPORCIONES:
        cantidad = round(len(evaluacion) * proporcion)

        priorizados = evaluacion.head(cantidad)

        positivos_capturados = int(priorizados["y_real"].sum())

        cobertura = positivos_capturados / positivos_totales
        precision = priorizados["y_real"].mean()
        lift = precision / tasa_base

        resultados.append(
            {
                "porcentaje_priorizado": proporcion * 100,
                "pacientes_priorizados": cantidad,
                "reingresos_capturados": positivos_capturados,
                "cobertura_reingresos": cobertura,
                "precision_en_priorizados": precision,
                "lift_vs_tasa_base": lift,
            }
        )

    tabla = pd.DataFrame(resultados)

    SALIDA.mkdir(parents=True, exist_ok=True)

    tabla.to_csv(
        SALIDA / "evaluacion_priorizacion.csv",
        index=False,
    )

    print("=== EVALUACION DE PRIORIZACION ===")
    print(f"Mejor C utilizado: {mejor_c}")
    print(f"Encuentros en prueba: {len(evaluacion):,}")
    print(f"Reingresos reales en prueba: {positivos_totales:,}")
    print(f"Tasa base: {tasa_base * 100:.2f} %\n")

    salida = tabla.copy()

    salida["porcentaje_priorizado"] = (
        salida["porcentaje_priorizado"].map(
            lambda x: f"{x:.0f} %"
        )
    )

    salida["cobertura_reingresos"] = (
        salida["cobertura_reingresos"].map(
            lambda x: f"{x * 100:.2f} %"
        )
    )

    salida["precision_en_priorizados"] = (
        salida["precision_en_priorizados"].map(
            lambda x: f"{x * 100:.2f} %"
        )
    )

    salida["lift_vs_tasa_base"] = (
        salida["lift_vs_tasa_base"].map(
            lambda x: f"{x:.2f}x"
        )
    )

    print(salida.to_string(index=False))

    print("\nResultados guardados en:")
    print(
        (
            SALIDA / "evaluacion_priorizacion.csv"
        ).relative_to(RAIZ)
    )

    return tabla


if __name__ == "__main__":
    evaluar_priorizacion()