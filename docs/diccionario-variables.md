# Diccionario de variables

Referencia de trabajo del equipo. **No es un entregable del enunciado**: existe para
no adivinar entre 50 columnas mientras se explora y se modela.

Dataset: *Diabetes 130-US Hospitals for Years 1999-2008* (UCI 296) — 101 766 filas x 50 columnas.

Generado por `src/data/diccionario.py` el 2026-08-19. Para regenerarlo:

```bash
python -m src.data.diccionario
```

## Como leer el archivo

Con las opciones por defecto pandas se equivoca **en las dos direcciones**:
esconde faltantes reales e inventa faltantes que no existen.

| | Por defecto | La realidad |
|---|---|---|
| `race` | 0 nulos | 2 273 valores `?` |
| `A1Cresult` | 84 748 nulos | 0 faltantes: `None` = no se ordeno el examen |

El centinela del archivo es `?`, no `NaN` — asi que `df.isna().sum()` da cero y
parece que no faltara nada. Y `None`, que en `A1Cresult` y `max_glu_serum` es una
categoria clinica legitima, esta en la lista de nulos por defecto de pandas y se
convierte en `NaN` sin avisar.

Lectura correcta:

```python
df = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_values=[])
df = df.replace("?", pd.NA)   # ahora si, y solo donde corresponde
```

## Variables

`Falta` cuenta unicamente el centinela `?`. Los `None` de `A1Cresult` y
`max_glu_serum` no se cuentan porque no son faltantes.

| # | Columna | Tipo | Distintos | Falta | Descripcion | Notas |
|--:|---|---|--:|--:|---|---|
| 1 | `encounter_id` | Identificador | 101766 | — | Identificador unico del encuentro hospitalario. Es la unidad de analisis: una fila = un encuentro. | — |
| 2 | `patient_nbr` | Identificador | 71518 | — | Identificador unico del paciente. Se repite entre filas: 101 766 encuentros sobre 71 518 pacientes. | Obliga a partir train/test por paciente y no al azar, o hay fuga. |
| 3 | `race` | Categorica nominal | 6 | 2,2 % | Raza declarada. | — |
| 4 | `gender` | Categorica nominal | 3 | — | Sexo. Incluye 3 filas `Unknown/Invalid`. | — |
| 5 | `age` | Categorica ordinal | 10 | — | Edad agrupada en decadas, de `[0-10)` a `[90-100)`. | Ordinal: conserva el orden al codificar, no la vuelvas one-hot sin pensarlo. |
| 6 | `weight` | Categorica ordinal | 10 | 96,9 % | Peso en bandas de 25 libras, mas la categoria abierta `>200`. | UCI dice 'peso en libras' como si fuera continua: es categorica. Vacia al 96,9 %. |
| 7 | `admission_type_id` | Categorica nominal | 8 | — | Tipo de admision (urgencia, electiva, recien nacido...). Codigo entero; ver tabla abajo. | Entero por como se almacena, pero NO es ordinal: 3 no es mayor que 1. |
| 8 | `discharge_disposition_id` | Categorica nominal | 26 | — | Destino al egreso (casa, otra institucion, fallecido, hospicio...). Codigo entero; ver tabla abajo. | De aqui salen las exclusiones del analisis; ver el detalle en las tablas de codigos. |
| 9 | `admission_source_id` | Categorica nominal | 17 | — | Via de ingreso (remision medica, urgencias, traslado...). Codigo entero; ver tabla abajo. | — |
| 10 | `time_in_hospital` | Entero de conteo | 14 | — | Dias entre la admision y el egreso. | — |
| 11 | `payer_code` | Categorica nominal | 18 | 39,6 % | Pagador o asegurador (`MC` Medicare, `BC` Blue Cross...). | UCI dice 'integer identifier': son textos. Falta el 39,6 %. |
| 12 | `medical_specialty` | Categorica nominal | 73 | 49,1 % | Especialidad del medico que admite. | UCI dice 'integer identifier': son textos. Falta el 49,1 %. |
| 13 | `num_lab_procedures` | Entero de conteo | 118 | — | Examenes de laboratorio hechos durante el encuentro. | — |
| 14 | `num_procedures` | Entero de conteo | 7 | — | Procedimientos distintos de laboratorio durante el encuentro. | — |
| 15 | `num_medications` | Entero de conteo | 75 | — | Medicamentos genericos distintos administrados durante el encuentro. | — |
| 16 | `number_outpatient` | Entero de conteo | 39 | — | Consultas ambulatorias del paciente en el ano previo al encuentro. | Historia previa, no del encuentro actual. |
| 17 | `number_emergency` | Entero de conteo | 33 | — | Visitas a urgencias del paciente en el ano previo al encuentro. | Historia previa, no del encuentro actual. |
| 18 | `number_inpatient` | Entero de conteo | 21 | — | Hospitalizaciones del paciente en el ano previo al encuentro. | Historia previa. Suele ser el predictor mas fuerte de reingreso. |
| 19 | `diag_1` | Categorica nominal | 717 | 0,02 % | Diagnostico principal, codigo ICD-9. | UCI dice 'primeros tres digitos': hay 8 522 con decimal (`250.83`) y 1 645 con letra (`V57`, `E878`). Cardinalidad alta: hay que agrupar. |
| 20 | `diag_2` | Categorica nominal | 749 | 0,4 % | Diagnostico secundario, codigo ICD-9. | Mismo formato que diag_1. |
| 21 | `diag_3` | Categorica nominal | 790 | 1,4 % | Diagnostico secundario adicional, codigo ICD-9. | Mismo formato que diag_1. |
| 22 | `number_diagnoses` | Entero de conteo | 16 | — | Cantidad de diagnosticos registrados en el sistema. | — |
| 23 | `max_glu_serum` | Categorica ordinal | 4 | — | Resultado de glucosa serica: `>200`, `>300`, `Norm`, o `None` si no se ordeno el examen. | `None` NO es faltante: significa que no se pidio el examen. Son 96 420 filas (94,7 %). |
| 24 | `A1Cresult` | Categorica ordinal | 4 | — | Hemoglobina glicosilada: `>8`, `>7`, `Norm`, o `None` si no se ordeno el examen. | `None` NO es faltante: 84 748 filas (83,3 %). Que se haya medido o no es la hipotesis del paper original. |
| 25 | `metformin` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 26 | `repaglinide` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 27 | `nateglinide` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 28 | `chlorpropamide` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 29 | `glimepiride` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 30 | `acetohexamide` | Categorica ordinal | 2 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 31 | `glipizide` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 32 | `glyburide` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 33 | `tolbutamide` | Categorica ordinal | 2 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 34 | `pioglitazone` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 35 | `rosiglitazone` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 36 | `acarbose` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 37 | `miglitol` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 38 | `troglitazone` | Categorica ordinal | 2 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 39 | `tolazamide` | Categorica ordinal | 3 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 40 | `examide` | Categorica ordinal | 1 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | **Varianza cero**: un solo valor (`No`). Descartable. |
| 41 | `citoglipton` | Categorica ordinal | 1 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | **Varianza cero**: un solo valor (`No`). Descartable. |
| 42 | `insulin` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 43 | `glyburide-metformin` | Categorica ordinal | 4 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 44 | `glipizide-metformin` | Categorica ordinal | 2 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 45 | `glimepiride-pioglitazone` | Categorica ordinal | 2 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 46 | `metformin-rosiglitazone` | Categorica ordinal | 2 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 47 | `metformin-pioglitazone` | Categorica ordinal | 2 | — | Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla). | — |
| 48 | `change` | Binaria | 2 | — | Hubo cambio en la medicacion para diabetes: `Ch` o `No`. | — |
| 49 | `diabetesMed` | Binaria | 2 | — | Se receto algun medicamento para diabetes: `Yes` o `No`. | — |
| 50 | `readmitted` | Objetivo | 3 | — | Dias hasta el reingreso: `<30`, `>30`, o `NO` si no hubo registro de reingreso. | Se binariza a `<30` contra el resto: un reingreso a los ocho meses no es un fallo de la transicion de cuidado. |

Las 23 columnas de farmaco comparten codificacion: `Up` subieron la dosis, `Down` la bajaron, `Steady` la mantuvieron, `No` no se receto durante el encuentro.

## Columnas casi constantes

Aparte de las de varianza cero, varias columnas de farmaco tienen tan pocos casos
distintos de `No` que no aportan senal:

| Columna | Filas distintas de `No` |
|---|--:|
| `acetohexamide` | 1 |
| `glimepiride-pioglitazone` | 1 |
| `metformin-pioglitazone` | 1 |
| `metformin-rosiglitazone` | 2 |
| `troglitazone` | 3 |
| `glipizide-metformin` | 13 |
| `tolbutamide` | 23 |
| `miglitol` | 38 |
| `tolazamide` | 39 |

## Tablas de codigos

De `IDs_mapping.csv`. Sin esto las tres columnas de arriba son numeros sueltos.

La marca ⛔ senala los destinos que se excluyen, por dos razones **distintas**:

- **Fallecidos** (11, 19, 20, 21): 1 652 filas. No pueden reingresar, y los datos lo
  confirman: 0 reingresos registrados. Excluirlos es corregir un imposible.
- **Hospicio** (13, 14): 771 filas. Estos si reingresan — hay 86 casos.
  Se excluyen por alcance, no por imposibilidad: son pacientes en cuidado de fin de
  vida, donde agendar un control para evitar el reingreso no es la intervencion que
  decide el tablero.

En total salen 2 423 encuentros (2,38 %).

### `admission_type_id` — Tipo de admision

| Codigo | Descripcion | En los datos |
|--:|---|--:|
| 1 | Emergency | 53 990 |
| 2 | Urgent | 18 480 |
| 3 | Elective | 18 869 |
| 4 | Newborn | 10 |
| 5 | Not Available | 4 785 |
| 6 | NULL | 5 291 |
| 7 | Trauma Center | 21 |
| 8 | Not Mapped | 320 |

### `discharge_disposition_id` — Destino al egreso

| Codigo | Descripcion | En los datos |
|--:|---|--:|
| 1 | Discharged to home | 60 234 |
| 2 | Discharged/transferred to another short term hospital | 2 128 |
| 3 | Discharged/transferred to SNF | 13 954 |
| 4 | Discharged/transferred to ICF | 815 |
| 5 | Discharged/transferred to another type of inpatient care institution | 1 184 |
| 6 | Discharged/transferred to home with home health service | 12 902 |
| 7 | Left AMA | 623 |
| 8 | Discharged/transferred to home under care of Home IV provider | 108 |
| 9 | Admitted as an inpatient to this hospital | 21 |
| 10 | Neonate discharged to another hospital for neonatal aftercare | 6 |
| 11 | Expired ⛔ | 1 642 |
| 12 | Still patient or expected to return for outpatient services | 3 |
| 13 | Hospice / home ⛔ | 399 |
| 14 | Hospice / medical facility ⛔ | 372 |
| 15 | Discharged/transferred within this institution to Medicare approved swing bed | 63 |
| 16 | Discharged/transferred/referred another institution for outpatient services | 11 |
| 17 | Discharged/transferred/referred to this institution for outpatient services | 14 |
| 18 | NULL | 3 691 |
| 19 | Expired at home. Medicaid only, hospice. ⛔ | 8 |
| 20 | Expired in a medical facility. Medicaid only, hospice. ⛔ | 2 |
| 21 | Expired, place unknown. Medicaid only, hospice. ⛔ | — |
| 22 | Discharged/transferred to another rehab fac including rehab units of a hospital . | 1 993 |
| 23 | Discharged/transferred to a long term care hospital. | 412 |
| 24 | Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare. | 48 |
| 25 | Not Mapped | 989 |
| 26 | Unknown/Invalid | — |
| 27 | Discharged/transferred to a federal health care facility. | 5 |
| 28 | Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital | 139 |
| 29 | Discharged/transferred to a Critical Access Hospital (CAH). | — |
| 30 | Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere | — |

### `admission_source_id` — Via de ingreso

| Codigo | Descripcion | En los datos |
|--:|---|--:|
| 1 | Physician Referral | 29 565 |
| 2 | Clinic Referral | 1 104 |
| 3 | HMO Referral | 187 |
| 4 | Transfer from a hospital | 3 187 |
| 5 | Transfer from a Skilled Nursing Facility (SNF) | 855 |
| 6 | Transfer from another health care facility | 2 264 |
| 7 | Emergency Room | 57 494 |
| 8 | Court/Law Enforcement | 16 |
| 9 | Not Available | 125 |
| 10 | Transfer from critial access hospital | 8 |
| 11 | Normal Delivery | 2 |
| 12 | Premature Delivery | — |
| 13 | Sick Baby | 1 |
| 14 | Extramural Birth | 2 |
| 15 | Not Available | — |
| 17 | NULL | 6 781 |
| 18 | Transfer From Another Home Health Agency | — |
| 19 | Readmission to Same Home Health Agency | — |
| 20 | Not Mapped | 161 |
| 21 | Unknown/Invalid | — |
| 22 | Transfer from hospital inpt/same fac reslt in a sep claim | 12 |
| 23 | Born inside this hospital | — |
| 24 | Born outside this hospital | — |
| 25 | Transfer from Ambulatory Surgery Center | 2 |
| 26 | Transfer from Hospice | — |

## Donde la ficha de UCI se equivoca

Verificado contra el archivo, no copiado de la pagina:

| Campo | UCI dice | El archivo dice |
|---|---|---|
| Simbolo de faltante | `NaN` | `?` — no hay un solo literal `NaN` |
| `weight` | Peso en libras | Bandas categoricas, vacio al 96,9 % |
| `payer_code` | Integer identifier | Texto (`MC`, `BC`, `HM`) |
| `medical_specialty` | Integer identifier | Texto (`Cardiology`, `InternalMedicine`) |
| `diag_*` | Primeros tres digitos de ICD-9 | 8 522 con decimal, 1 645 con letra `V`/`E` |
| Cardinalidades | 29 destinos, 848 diagnosticos | 26 y 717 en los datos reales |

