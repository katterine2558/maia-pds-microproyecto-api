"""Evalua la regresion logistica como herramienta de priorizacion.

Orden de ejecucion requerido:

1. Ejecutar:
   python -m src.models.regresion_logistica

   Ese proceso genera:
   docs/soportes/modelos/regresion_logistica/ajuste_regularizacion.csv

2. Ejecutar:
   python -m src.models.evaluacion_priorizacion

El archivo ajuste_regularizacion.csv se utiliza para recuperar el valor
de C seleccionado previamente sobre el conjunto de validacion.
"""

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


ARCHIVO_AJUSTE = (
    SALIDA
    / "ajuste_regularizacion.csv"
)

PROPORCIONES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


def obtener_mejor_c() -> float:
    """Recupera el C seleccionado previamente en validacion."""

    if not ARCHIVO_AJUSTE.exists():
        raise FileNotFoundError(
            "No existe ajuste_regularizacion.csv. "
            "Ejecute primero: "
            "python -m src.models.regresion_logistica"
        )

    ajuste = pd.read_csv(
        ARCHIVO_AJUSTE
    )

    columnas_requeridas = {
        "C",
        "pr_auc_validacion",
        "roc_auc_validacion",
    }

    faltantes = (
        columnas_requeridas
        - set(ajuste.columns)
    )

    if faltantes:
        raise ValueError(
            "ajuste_regularizacion.csv no contiene "
            "las columnas requeridas: "
            + ", ".join(
                sorted(faltantes)
            )
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


def evaluar_priorizacion() -> pd.DataFrame:
    """Evalua cobertura y lift al priorizar por probabilidad estimada."""

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
            "y_real": conjuntos[
                "y_prueba"
            ].to_numpy(),
            "probabilidad": probabilidad,
        }
    ).sort_values(
        "probabilidad",
        ascending=False,
    )

    positivos_totales = int(
        evaluacion["y_real"].sum()
    )

    tasa_base = float(
        evaluacion["y_real"].mean()
    )

    if positivos_totales == 0:
        raise ValueError(
            "El conjunto de prueba no contiene "
            "casos positivos."
        )

    if tasa_base == 0:
        raise ValueError(
            "La tasa base del conjunto de prueba "
            "es igual a cero."
        )

    resultados = []

    for proporcion in PROPORCIONES:

        cantidad = round(
            len(evaluacion)
            * proporcion
        )

        priorizados = evaluacion.head(
            cantidad
        )

        positivos_capturados = int(
            priorizados[
                "y_real"
            ].sum()
        )

        cobertura = (
            positivos_capturados
            / positivos_totales
        )

        precision = float(
            priorizados[
                "y_real"
            ].mean()
        )

        lift = (
            precision
            / tasa_base
        )

        resultados.append(
            {
                "C": mejor_c,
                "porcentaje_priorizado": (
                    proporcion * 100
                ),
                "pacientes_priorizados": cantidad,
                "reingresos_capturados": (
                    positivos_capturados
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
        / "evaluacion_priorizacion.csv",
        index=False,
    )

    print(
        "=== EVALUACION DE PRIORIZACION ==="
    )
    print(
        f"C seleccionado en validacion: "
        f"{mejor_c}"
    )
    print(
        f"Encuentros en prueba: "
        f"{len(evaluacion):,}"
    )
    print(
        f"Reingresos reales en prueba: "
        f"{positivos_totales:,}"
    )
    print(
        f"Tasa base: "
        f"{tasa_base * 100:.2f} %"
    )

    print()

    salida = tabla.copy()

    salida[
        "porcentaje_priorizado"
    ] = salida[
        "porcentaje_priorizado"
    ].map(
        lambda valor: (
            f"{valor:.0f} %"
        )
    )

    salida[
        "cobertura_reingresos"
    ] = salida[
        "cobertura_reingresos"
    ].map(
        lambda valor: (
            f"{valor * 100:.2f} %"
        )
    )

    salida[
        "precision_en_priorizados"
    ] = salida[
        "precision_en_priorizados"
    ].map(
        lambda valor: (
            f"{valor * 100:.2f} %"
        )
    )

    salida[
        "lift_vs_tasa_base"
    ] = salida[
        "lift_vs_tasa_base"
    ].map(
        lambda valor: (
            f"{valor:.2f}x"
        )
    )

    print(
        salida.to_string(
            index=False
        )
    )

    print(
        "\nResultados guardados en:"
    )
    print(
        (
            SALIDA
            / "evaluacion_priorizacion.csv"
        ).relative_to(
            RAIZ
        )
    )

    return tabla


if __name__ == "__main__":
    evaluar_priorizacion()