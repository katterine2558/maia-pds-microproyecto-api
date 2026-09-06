"""Ingenieria de caracteristicas para aumentar los verdaderos positivos.

El objetivo es uno solo: que el clasificador detecte mas reingresos reales sin
cambiar el umbral ni los hiperparametros del arbol. Cada variable derivada
responde a una hipotesis concreta sobre por que un paciente vuelve, y todas se
construyen con informacion disponible el dia del alta.

Las hipotesis salen del EDA de la Entrega 1:

  - El historial previo domina sobre el episodio actual (EDA 7.5). Si tres
    columnas separadas ya son la senal mas fuerte, su agregado deberia
    concentrarla.
  - Que se ordene la hemoglobina glicosilada se asocia con menor reingreso
    (EDA 3.3), y esa es la hipotesis del articulo original del conjunto.
  - Los codigos de diagnostico se agrupan en nueve categorias (EDA 6.3); que
    las tres coincidan indica un cuadro concentrado y no multiples problemas.

Ninguna variable usa informacion posterior al alta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import esquema as esq

# Farmacos que sobrevivieron al descarte por baja frecuencia (EDA 6.2).
FARMACOS = [
    "metformin", "repaglinide", "nateglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "insulin",
    "glyburide-metformin",
]

# Destinos al egreso que implican continuidad institucional del cuidado.
INSTITUCION = {"2", "3", "4", "5", "22", "23", "24", "27", "28", "29", "30"}


def agregar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables derivadas al conjunto de trabajo.

    Recibe el DataFrame antes de `transformar()`, con las columnas crudas, y
    devuelve una copia con las nuevas columnas.
    """
    d = df.copy()

    # --- Historial previo -------------------------------------------------
    # Las tres columnas de utilizacion previa son la senal mas fuerte del
    # conjunto por separado; el agregado las concentra en una sola.
    d["utilizacion_previa"] = (
        d["number_inpatient"] + d["number_emergency"] + d["number_outpatient"]
    )
    # Haber estado hospitalizado el ano anterior es cualitativamente distinto
    # de cuantas veces: separa el 8,6 % de reingreso del 37,1 % (EDA 7.1).
    d["hospitalizado_antes"] = (d["number_inpatient"] > 0).astype("int64")
    # Proporcion del historial que fue urgencia: distingue al paciente que
    # llega descompensado del que tiene seguimiento programado.
    total = d["utilizacion_previa"].replace(0, np.nan)
    d["fraccion_urgencias"] = (d["number_emergency"] / total).fillna(0.0)

    # --- Complejidad del tratamiento --------------------------------------
    # Polifarmacia: cuantos de los once farmacos conservados se recetaron.
    recetados = sum((d[f] != "No").astype("int64") for f in FARMACOS)
    d["farmacos_recetados"] = recetados
    # Inestabilidad: cuantos cambiaron de dosis durante la hospitalizacion.
    cambios = sum(d[f].isin(["Up", "Down"]).astype("int64") for f in FARMACOS)
    d["farmacos_ajustados"] = cambios

    # --- Manejo clinico ---------------------------------------------------
    # Que se ordene el examen es informacion, con independencia del resultado:
    # es la hipotesis del articulo original sobre este conjunto (EDA 3.3).
    d["midieron_a1c"] = (d["A1Cresult"] != "None").astype("int64")
    d["midieron_glucosa"] = (d["max_glu_serum"] != "None").astype("int64")

    # --- Intensidad del episodio ------------------------------------------
    # Normalizar por estancia distingue al paciente que concentro muchos
    # procedimientos en pocos dias del que estuvo internado sin actividad.
    estancia = d["time_in_hospital"].clip(lower=1)
    d["procedimientos_por_dia"] = d["num_procedures"] / estancia
    d["laboratorios_por_dia"] = d["num_lab_procedures"] / estancia
    d["medicamentos_por_diagnostico"] = (
        d["num_medications"] / d["number_diagnoses"].clip(lower=1)
    )

    # --- Destino al egreso ------------------------------------------------
    # El EDA solo uso esta columna para excluir filas, pero la importancia por
    # permutacion la puso primera entre las predictoras.
    d["egreso_a_institucion"] = (
        d["discharge_disposition_id"].isin(INSTITUCION).astype("int64")
    )
    d["egreso_a_casa"] = (d["discharge_disposition_id"] == "1").astype("int64")

    return d


def derivadas() -> list[str]:
    """Nombres de las variables que agrega este modulo."""
    return [
        "utilizacion_previa", "hospitalizado_antes", "fraccion_urgencias",
        "farmacos_recetados", "farmacos_ajustados",
        "midieron_a1c", "midieron_glucosa",
        "procedimientos_por_dia", "laboratorios_por_dia",
        "medicamentos_por_diagnostico",
        "egreso_a_institucion", "egreso_a_casa",
    ]


def diagnosticos_coinciden(d: pd.DataFrame) -> pd.Series:
    """Cuantas de las tres categorias de diagnostico se repiten.

    Se calcula despues de `transformar()`, cuando los codigos ICD-9 ya estan
    agrupados en las nueve categorias clinicas.
    """
    return (
        (d["diag_1"] == d["diag_2"]).astype("int64")
        + (d["diag_2"] == d["diag_3"]).astype("int64")
        + (d["diag_1"] == d["diag_3"]).astype("int64")
    )
