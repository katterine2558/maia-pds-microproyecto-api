"""Registra en MLflow las versiones de regresion logistica del equipo.

No reimplementa nada: importa las funciones de los modulos de modelado, de
modo que lo que queda registrado en MLflow es exactamente lo que producen los
scripts. Si Camilo cambia el preprocesamiento, este archivo lo hereda.

Uso:

    export MLFLOW_TRACKING_URI=http://<ip>:5000
    export MLFLOW_TRACKING_USERNAME=<usuario>
    export MLFLOW_TRACKING_PASSWORD=<clave>

    python -m src.models.experimentos_mlflow            # todo
    python -m src.models.experimentos_mlflow --rapido   # rejillas recortadas
    python -m src.models.experimentos_mlflow --familias base umbral

La rejilla de V6 son 75 ajustes con solver saga: en una maquina modesta puede
tardar bastante. Para una pasada de prueba, usar --rapido.
"""

from __future__ import annotations

import argparse

import mlflow

from src.features.preparacion_regresion import preparar_datos
from src.models import regresion_v6
from src.models.optimizar_recall import metricas_umbral
from src.models.regresion_logistica import (
    SEMILLA,
    VALORES_C,
    calcular_metricas,
    construir_modelo,
    dividir_por_paciente,
    verificar_separacion,
)
from src.seguimiento.mlflow_config import corrida, iniciar, registrar


PESOS_POSITIVOS = [2, 3, 4, 5]

UMBRALES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

FAMILIAS = ("base", "regularizacion", "peso", "umbral", "elasticnet")


def _ajustar(modelo, conjuntos):
    """Entrena en entrenamiento y evalua en validacion.

    La seleccion entre alternativas se hace SIEMPRE contra validacion. El
    conjunto de prueba se reserva para la version final, para que la
    comparacion no lo contamine.
    """

    modelo.fit(
        conjuntos["X_entrenamiento"],
        conjuntos["y_entrenamiento"],
    )
    return modelo


def familia_base(conjuntos) -> None:
    """V1 sin balanceo y V2 con class_weight balanceado."""

    for nombre, peso in (("V1_base", None), ("V2_balanceada", "balanced")):
        with corrida(nombre, "regresion-logistica"):
            modelo = _ajustar(construir_modelo(class_weight=peso, c=1.0), conjuntos)

            resultado = calcular_metricas(
                nombre,
                modelo,
                conjuntos["X_validacion"],
                conjuntos["y_validacion"],
            )

            registrar(
                {
                    "version": nombre,
                    "C": 1.0,
                    "class_weight": str(peso),
                    "solver": "liblinear",
                    "semilla": SEMILLA,
                    "conjunto_evaluacion": "validacion",
                },
                resultado,
            )

            print(f"  {nombre}: PR-AUC={resultado['pr_auc']:.4f}")


def familia_regularizacion(conjuntos, rapido: bool) -> None:
    """V3: barrido de C. Cada valor queda como corrida anidada."""

    valores = VALORES_C[:2] if rapido else VALORES_C

    with corrida("V3_barrido_C", "regresion-logistica"):
        mlflow.log_param("valores_C", str(valores))

        for c in valores:
            with corrida(f"V3_C_{c}", "regresion-logistica", anidada=True):
                modelo = _ajustar(
                    construir_modelo(class_weight="balanced", c=c),
                    conjuntos,
                )

                resultado = calcular_metricas(
                    f"V3_C_{c}",
                    modelo,
                    conjuntos["X_validacion"],
                    conjuntos["y_validacion"],
                )

                registrar(
                    {
                        "version": "V3",
                        "C": c,
                        "class_weight": "balanced",
                        "solver": "liblinear",
                        "conjunto_evaluacion": "validacion",
                    },
                    resultado,
                )

                print(f"  V3 C={c}: PR-AUC={resultado['pr_auc']:.4f}")


def familia_peso(conjuntos, rapido: bool) -> None:
    """V4: peso de la clase positiva, buscando sensibilidad."""

    pesos = PESOS_POSITIVOS[:2] if rapido else PESOS_POSITIVOS

    with corrida("V4_barrido_peso", "regresion-logistica"):
        mlflow.log_param("pesos_evaluados", str(pesos))

        for peso in pesos:
            with corrida(f"V4_peso_{peso}", "regresion-logistica", anidada=True):
                modelo = _ajustar(
                    construir_modelo(class_weight={0: 1, 1: peso}, c=0.5),
                    conjuntos,
                )

                resultado = calcular_metricas(
                    f"V4_peso_{peso}",
                    modelo,
                    conjuntos["X_validacion"],
                    conjuntos["y_validacion"],
                )

                registrar(
                    {
                        "version": "V4",
                        "C": 0.5,
                        "peso_positivo": peso,
                        "conjunto_evaluacion": "validacion",
                    },
                    resultado,
                )

                print(f"  V4 peso={peso}: recall={resultado['recall']:.4f}")


def familia_umbral(conjuntos) -> None:
    """V5: con el peso fijo en 5, barrido del umbral de clasificacion.

    El modelo se entrena una sola vez; lo que cambia es el punto de corte
    sobre la misma probabilidad estimada.
    """

    with corrida("V5_barrido_umbral", "regresion-logistica"):
        modelo = _ajustar(
            construir_modelo(class_weight={0: 1, 1: 5}, c=0.5),
            conjuntos,
        )

        probabilidad = modelo.predict_proba(conjuntos["X_validacion"])[:, 1]

        mlflow.log_params(
            {"version": "V5", "C": 0.5, "peso_positivo": 5, "umbrales": str(UMBRALES)}
        )

        for umbral in UMBRALES:
            with corrida(f"V5_umbral_{umbral}", "regresion-logistica", anidada=True):
                resultado = metricas_umbral(
                    conjuntos["y_validacion"],
                    probabilidad,
                    umbral,
                )

                registrar(
                    {
                        "version": "V5",
                        "C": 0.5,
                        "peso_positivo": 5,
                        "umbral": umbral,
                        "conjunto_evaluacion": "validacion",
                    },
                    resultado,
                )

                print(
                    f"  V5 umbral={umbral}: "
                    f"recall={resultado['recall']:.4f} "
                    f"precision={resultado['precision']:.4f}"
                )


def familia_elasticnet(X, y, grupos, rapido: bool) -> None:
    """V6: Elastic Net con variables derivadas de utilizacion previa.

    Usa su propia particion porque agrega columnas a X antes de dividir.
    """

    X_v6 = regresion_v6.agregar_variables(X)
    conjuntos = dividir_por_paciente(X_v6, y, grupos)
    verificar_separacion(conjuntos)

    if rapido:
        valores_c = regresion_v6.VALORES_C[:1]
        valores_l1 = regresion_v6.VALORES_L1[:2]
        pesos = regresion_v6.PESOS[:1]
    else:
        valores_c = regresion_v6.VALORES_C
        valores_l1 = regresion_v6.VALORES_L1
        pesos = regresion_v6.PESOS

    total = len(valores_c) * len(valores_l1) * len(pesos)

    with corrida("V6_rejilla_elasticnet", "elasticnet"):
        mlflow.log_params(
            {
                "version": "V6",
                "solver": "saga",
                "combinaciones": total,
                "variables_derivadas": "utilizacion_previa",
            }
        )

        hecho = 0

        for c in valores_c:
            for l1 in valores_l1:
                for peso in pesos:
                    hecho += 1
                    nombre = f"V6_C{c}_l1{l1}_peso{peso}"

                    with corrida(nombre, "elasticnet", anidada=True):
                        modelo = _ajustar(
                            regresion_v6.construir_modelo(
                                c=c,
                                l1_ratio=l1,
                                peso=peso,
                            ),
                            conjuntos,
                        )

                        probabilidad = modelo.predict_proba(
                            conjuntos["X_validacion"]
                        )[:, 1]

                        resultado = regresion_v6.metricas(
                            conjuntos["y_validacion"],
                            probabilidad,
                        )

                        registrar(
                            {
                                "version": "V6",
                                "C": c,
                                "l1_ratio": l1,
                                "peso_positivo": peso,
                                "solver": "saga",
                                "conjunto_evaluacion": "validacion",
                            },
                            resultado,
                        )

                    print(
                        f"  V6 [{hecho}/{total}] C={c} l1={l1} peso={peso}: "
                        f"PR-AUC={resultado['pr_auc']:.4f}"
                    )


def main() -> None:
    analizador = argparse.ArgumentParser(
        description="Registra en MLflow las versiones de regresion logistica."
    )
    analizador.add_argument(
        "--familias",
        nargs="+",
        choices=FAMILIAS,
        default=list(FAMILIAS),
        help="Familias de experimentos a registrar.",
    )
    analizador.add_argument(
        "--rapido",
        action="store_true",
        help="Recorta las rejillas, para verificar la conexion sin esperar.",
    )
    argumentos = analizador.parse_args()

    iniciar()

    print("=== PREPARACION ===")
    X, y, grupos, _ = preparar_datos()

    conjuntos = dividir_por_paciente(X, y, grupos)
    verificar_separacion(conjuntos)

    print(f"Encuentros: {len(X):,} | Pacientes: {grupos.nunique():,}")
    print(f"Tasa positiva: {y.mean() * 100:.2f} %\n")

    if "base" in argumentos.familias:
        print("=== V1-V2: BASE ===")
        familia_base(conjuntos)

    if "regularizacion" in argumentos.familias:
        print("\n=== V3: REGULARIZACION ===")
        familia_regularizacion(conjuntos, argumentos.rapido)

    if "peso" in argumentos.familias:
        print("\n=== V4: PESO DE CLASE ===")
        familia_peso(conjuntos, argumentos.rapido)

    if "umbral" in argumentos.familias:
        print("\n=== V5: UMBRAL ===")
        familia_umbral(conjuntos)

    if "elasticnet" in argumentos.familias:
        print("\n=== V6: ELASTIC NET ===")
        familia_elasticnet(X, y, grupos, argumentos.rapido)

    print(f"\nListo. Ver en {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
