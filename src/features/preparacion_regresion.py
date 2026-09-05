"""Preparacion de datos para el modelo de regresion logistica."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[2]
ARCHIVO_DATOS = RAIZ / "data" / "raw" / "diabetic_data.csv"

# Destinos de egreso excluidos desde la Entrega 1:
# fallecimiento y hospicio.
EXCLUIR_EGRESO = {"11", "19", "20", "21", "13", "14"}

# Medicamentos con menos de 100 registros diferentes de "No"
# en la poblacion analitica.
MEDICAMENTOS_BAJA_FRECUENCIA = {
    "examide",
    "citoglipton",
    "glimepiride-pioglitazone",
    "acetohexamide",
    "metformin-pioglitazone",
    "metformin-rosiglitazone",
    "troglitazone",
    "glipizide-metformin",
    "tolbutamide",
    "miglitol",
    "tolazamide",
    "chlorpropamide",
}

MEDICAMENTOS_CONSERVADOS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "insulin",
    "glyburide-metformin",
]

VARIABLES_NUMERICAS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

VARIABLES_CATEGORICAS = [
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "diag_1_grupo",
    "diag_2_grupo",
    "diag_3_grupo",
    "max_glu_serum",
    "A1Cresult",
    *MEDICAMENTOS_CONSERVADOS,
    "change",
    "diabetesMed",
]


def agrupar_icd9(valor: str) -> str:
    """Agrupa codigos ICD-9 en familias clinicas de menor cardinalidad."""
    valor = str(valor).strip()

    if valor in {"", "?"}:
        return "Desconocido"

    if valor.startswith(("V", "E")):
        return "Otros"

    try:
        codigo = int(float(valor))
    except ValueError:
        return "Otros"

    if codigo == 250:
        return "Diabetes"
    if 390 <= codigo <= 459 or codigo == 785:
        return "Circulatorio"
    if 460 <= codigo <= 519 or codigo == 786:
        return "Respiratorio"
    if 520 <= codigo <= 579 or codigo == 787:
        return "Digestivo"
    if 800 <= codigo <= 999:
        return "Lesiones"
    if 710 <= codigo <= 739:
        return "Musculoesqueletico"
    if 580 <= codigo <= 629 or codigo == 788:
        return "Genitourinario"
    if 140 <= codigo <= 239:
        return "Neoplasias"

    return "Otros"


def cargar_base_analitica() -> pd.DataFrame:
    """Carga los datos y aplica las exclusiones definidas en la Entrega 1."""
    datos = pd.read_csv(
        ARCHIVO_DATOS,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )

    datos = datos[
        ~datos["discharge_disposition_id"].isin(EXCLUIR_EGRESO)
    ].copy()

    datos["y"] = (datos["readmitted"] == "<30").astype(int)

    return datos


def preparar_datos():
    """Retorna predictores, objetivo, paciente y variable para auditoria."""
    datos = cargar_base_analitica()

    for columna in ("diag_1", "diag_2", "diag_3"):
        datos[f"{columna}_grupo"] = datos[columna].map(agrupar_icd9)

    for columna in VARIABLES_NUMERICAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    columnas_modelo = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS

    X = datos[columnas_modelo].copy()
    y = datos["y"].copy()

    # patient_nbr no entra al modelo. Se conserva para impedir que un mismo
    # paciente quede simultaneamente en entrenamiento y prueba.
    grupos = datos["patient_nbr"].copy()

    # race no se usa como predictor. Se conserva para evaluar posteriormente
    # el desempeño del modelo entre grupos.
    auditoria_race = datos["race"].copy()

    return X, y, grupos, auditoria_race


def resumen_preparacion() -> None:
    """Muestra controles basicos antes del entrenamiento."""
    datos = cargar_base_analitica()
    X, y, grupos, _ = preparar_datos()

    print("=== PREPARACION REGRESION LOGISTICA ===")
    print(f"Encuentros: {len(X):,}")
    print(f"Pacientes unicos: {grupos.nunique():,}")
    print(f"Predictores antes de codificacion: {X.shape[1]}")
    print(f"Reingresos <30 dias: {int(y.sum()):,}")
    print(f"Tasa positiva: {y.mean() * 100:.4f} %")

    print("\nVariables numericas:")
    for variable in VARIABLES_NUMERICAS:
        print(f" - {variable}")

    print("\nVariables categoricas:")
    for variable in VARIABLES_CATEGORICAS:
        print(f" - {variable}")

    print("\nGrupos de diagnostico:")
    for columna in ("diag_1", "diag_2", "diag_3"):
        grupos_diag = datos[columna].map(agrupar_icd9)
        print(f"\n{columna}:")
        print(grupos_diag.value_counts().to_string())


if __name__ == "__main__":
    resumen_preparacion()