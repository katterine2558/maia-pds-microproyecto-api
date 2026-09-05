"""Esquema de datos del proyecto.

Traduce a constantes las decisiones consolidadas en la seccion 9 del EDA
(`notebooks/eda.ipynb`). Cada bloque indica la subseccion que lo sustenta,
de modo que una discrepancia entre el codigo y el reporte sea localizable.

Ninguna decision se toma aqui: este modulo solo las declara.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Lectura del archivo crudo  (EDA 1.2 y 1.3)
# --------------------------------------------------------------------------

# El archivo marca el dato ausente con "?", no con campo vacio.
AUSENTE = "?"

FILAS_ESPERADAS = 101_766
COLUMNAS_ESPERADAS = 50

# Unicas columnas sobre las que tiene sentido operar aritmeticamente.
ENTERAS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# --------------------------------------------------------------------------
# Exclusion de filas  (EDA 5.2, 5.3 y 5.4)
# --------------------------------------------------------------------------

# Destinos al egreso que no admiten reingreso: el paciente fallecio.
FALLECIDOS = {"11", "19", "20", "21"}

# Cuidado de fin de vida: el reingreso ocurre, pero programar un control
# de seguimiento no es la intervencion que el tablero decide.
HOSPICIO = {"13", "14"}

EXCLUIDOS = FALLECIDOS | HOSPICIO

# Resultado esperado tras aplicar ambas exclusiones.
FILAS_TRABAJO = 99_343
PACIENTES_TRABAJO = 69_990
TASA_BASE = 0.1139  # 11,39 % de reingreso temprano

# --------------------------------------------------------------------------
# Variable objetivo  (EDA 4.2)
# --------------------------------------------------------------------------

OBJETIVO_ORIGEN = "readmitted"
OBJETIVO = "objetivo"
VALOR_POSITIVO = "<30"

# --------------------------------------------------------------------------
# Columnas descartadas  (EDA 9.2)
# --------------------------------------------------------------------------

# 96,9 % sin valor y la ausencia no discrimina (EDA 3.4).
POR_AUSENCIA = ["weight"]

# Un unico valor en las 101 766 filas (EDA 2.4 y 6.2).
CONSTANTES = ["examide", "citoglipton"]

# Menos de 100 registros con prescripcion sobre el conjunto de trabajo.
# La lista reproduce la salida ejecutada de la celda de EDA 6.4; el texto
# de 6.2 y la tabla 9.2 la resumen con un conteo que no coincide.
CASI_CONSTANTES = [
    "acetohexamide",            #  1 registro
    "glimepiride-pioglitazone", #  1
    "metformin-pioglitazone",   #  1
    "metformin-rosiglitazone",  #  2
    "troglitazone",             #  3
    "glipizide-metformin",      # 13
    "tolbutamide",              # 21
    "miglitol",                 # 38
    "tolazamide",               # 39
    "chlorpropamide",           # 85
]

# Variable administrativa, no clinica. Decision de modelamiento de la E1.
ADMINISTRATIVAS = ["payer_code"]

# Identificadores. `patient_nbr` no se descarta del DataFrame porque agrupa
# la particion (EDA 8.1), pero nunca entra como predictora.
IDENTIFICADORES = ["encounter_id"]
AGRUPADOR = "patient_nbr"

# Reservada para revisar el desempeno entre grupos, no como predictora.
# Si entrara, el tablero podria priorizar por grupo y no por riesgo clinico.
RESERVADAS_SESGO = ["race"]

DESCARTADAS = (
    POR_AUSENCIA
    + CONSTANTES
    + CASI_CONSTANTES
    + ADMINISTRATIVAS
    + IDENTIFICADORES
)

# Fuera de las predictoras, pero conservadas en el DataFrame de trabajo.
NO_PREDICTORAS = [AGRUPADOR, OBJETIVO_ORIGEN, OBJETIVO] + RESERVADAS_SESGO

# --------------------------------------------------------------------------
# Transformaciones  (EDA 9.3)
# --------------------------------------------------------------------------

# Historial del ano previo. Mediana cero y asimetria de hasta 22,85: la
# agrupacion por tramos es la forma en que el EDA las llevo al modelo.
TRAMOS = {
    "number_inpatient":  ([-1, 0, 1, 2, 4, 10**6], ["0", "1", "2", "3-4", "5 o mas"]),
    "number_emergency":  ([-1, 0, 1, 2, 10**6], ["0", "1", "2", "3 o mas"]),
    "number_outpatient": ([-1, 0, 1, 2, 10**6], ["0", "1", "2", "3 o mas"]),
}

# `age` viene agrupada en decadas por el propio conjunto. Es ordinal y el
# orden se preserva al codificarla.
ORDEN_EDAD = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]

# Los codigos ICD-9 se agrupan a nueve categorias clinicas (EDA 6.3).
DIAGNOSTICOS = ["diag_1", "diag_2", "diag_3"]

# El orden numerico de estos identificadores no significa nada: son nominales.
NOMINALES_CODIFICADAS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

# Marca explicita para el dato ausente. Ninguna columna se imputa: la
# ausencia entra al modelo como una categoria mas (EDA 3.4).
SIN_REGISTRO = "Sin registro"

# Categorias con muy pocos registros. Agruparlas reduce la cardinalidad sin
# perder informacion: en medical_specialty, 44 de las 73 categorias reunen el
# 0,9 % de las filas y ninguna sostiene una tasa estimable.
UMBRAL_CATEGORIA_RARA = 100
CATEGORIA_RESTO = "Otras"

# --------------------------------------------------------------------------
# Predictoras del episodio, sin transformar  (EDA 6.1)
# --------------------------------------------------------------------------

NUMERICAS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
]

# --------------------------------------------------------------------------
# Formulario de la Pantalla 2 de la maqueta  (memoria de la maqueta, E6)
# --------------------------------------------------------------------------

# Diez datos que la enfermera puede consultar en la historia clinica sin
# salir de la sala. El EDA 9.5 deja encargado comparar en MLflow el modelo
# completo contra uno restringido a estos campos.
FORMULARIO = [
    "age",
    "admission_type_id",
    "medical_specialty",
    "time_in_hospital",
    "number_diagnoses",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "A1Cresult",
    "change",
]

# --------------------------------------------------------------------------
# Particion y evaluacion  (EDA 9.4)
# --------------------------------------------------------------------------

SEMILLA = 42
FRACCION_PRUEBA = 0.2

# Criterio con que se elige el umbral de decision y se compara todo lo demas.
#
# F2 pesa el doble la sensibilidad sobre la precision. La eleccion responde a
# que los dos errores no cuestan lo mismo:
#
#   Falso positivo  se programa un seguimiento a quien no iba a reingresar.
#                   Consume un cupo de una capacidad limitada; el costo es de
#                   oportunidad y es acotado.
#   Falso negativo  el paciente sale sin control y reingresa. Se pierde la
#                   oportunidad de intervenir: cama ocupada, costo hospitalario
#                   alto, posible deterioro del paciente y mas carga al sistema.
#
# Es la misma razon por la que Nunes et al. (2025) adoptan F2 como metrica
# principal sobre datos clinicos desbalanceados.
#
# La exactitud queda descartada: el clasificador trivial llega a 88,6 % sin
# identificar a ningun paciente en riesgo (EDA 4.3).
CRITERIO_UMBRAL = "f2"

# Umbral de decision fijo. Se mantiene constante entre configuraciones para que
# la comparacion de hiperparametros sea legitima: la sensibilidad por si sola
# no se puede optimizar —marcar a todos los pacientes la lleva a 1,00— de modo
# que necesita una restriccion. Fijar el umbral es esa restriccion, y permite
# usar la sensibilidad como criterio de seleccion sin que colapse.
#
# El valor replica el de la version V5 de la regresion logistica del equipo,
# para que las dos familias de modelos sean comparables entre si.
UMBRAL_FIJO = 0.30

# Metrica con que se comparan las configuraciones de hiperparametros, medida
# siempre al umbral fijo.
METRICA_SELECCION = "sensibilidad"

# Razon entre el costo de un falso negativo y el de un falso positivo. Se usa
# para el analisis de costo del notebook; cada institucion tiene la suya.
COSTO_FALSO_NEGATIVO = 5.0
COSTO_FALSO_POSITIVO = 1.0
