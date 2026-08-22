"""Diccionario de variables del dataset de reingreso.

Referencia de trabajo del equipo: que significa cada columna, de que tipo es y
como hay que leerla. No es un entregable del enunciado; existe para no tener
que adivinar entre 50 columnas ni volver a la pagina de UCI cada vez.

Las descripciones salen de la ficha de UCI (dataset 296), traducidas y
CORREGIDAS donde no concuerdan con el archivo: la ficha tiene varios errores
que se verificaron uno por uno contra los datos. Cada correccion queda marcada
en la columna "Notas" del documento generado.

    python -m src.data.diccionario
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CRUDOS = RAIZ / "data" / "raw" / "diabetic_data.csv"
MAPA_IDS = RAIZ / "data" / "raw" / "IDs_mapping.csv"
SALIDA = RAIZ / "docs" / "diccionario-variables.md"

FALTANTE = "?"          # el centinela real del archivo (UCI declara "NaN": es falso)
NO_MEDIDO = "None"      # en A1Cresult y max_glu_serum NO es un faltante

# Tipos semanticos. Determinan como se codifica cada columna al modelar, que es
# la razon de tenerlos separados del dtype que infiere pandas.
ID = "Identificador"
NOMINAL = "Categorica nominal"
ORDINAL = "Categorica ordinal"
CONTEO = "Entero de conteo"
BINARIA = "Binaria"
OBJETIVO = "Objetivo"

FARMACOS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

DESC_FARMACO = "Manejo del farmaco durante el encuentro (ver leyenda bajo la tabla)."

LEYENDA_FARMACOS = (
    "Las 23 columnas de farmaco comparten codificacion: `Up` subieron la dosis, "
    "`Down` la bajaron, `Steady` la mantuvieron, `No` no se receto durante el encuentro."
)

# columna -> (tipo semantico, descripcion, nota)
VARIABLES: dict[str, tuple[str, str, str]] = {
    "encounter_id": (ID, "Identificador unico del encuentro hospitalario. Es la unidad de analisis: una fila = un encuentro.", ""),
    "patient_nbr": (ID, "Identificador unico del paciente. Se repite entre filas: 101 766 encuentros sobre 71 518 pacientes.", "Obliga a partir train/test por paciente y no al azar, o hay fuga."),
    "race": (NOMINAL, "Raza declarada.", ""),
    "gender": (NOMINAL, "Sexo. Incluye 3 filas `Unknown/Invalid`.", ""),
    "age": (ORDINAL, "Edad agrupada en decadas, de `[0-10)` a `[90-100)`.", "Ordinal: conserva el orden al codificar, no la vuelvas one-hot sin pensarlo."),
    "weight": (ORDINAL, "Peso en bandas de 25 libras, mas la categoria abierta `>200`.", "UCI dice 'peso en libras' como si fuera continua: es categorica. Vacia al 96,9 %."),
    "admission_type_id": (NOMINAL, "Tipo de admision (urgencia, electiva, recien nacido...). Codigo entero; ver tabla abajo.", "Entero por como se almacena, pero NO es ordinal: 3 no es mayor que 1."),
    "discharge_disposition_id": (NOMINAL, "Destino al egreso (casa, otra institucion, fallecido, hospicio...). Codigo entero; ver tabla abajo.", "De aqui salen las exclusiones del analisis; ver el detalle en las tablas de codigos."),
    "admission_source_id": (NOMINAL, "Via de ingreso (remision medica, urgencias, traslado...). Codigo entero; ver tabla abajo.", ""),
    "time_in_hospital": (CONTEO, "Dias entre la admision y el egreso.", ""),
    "payer_code": (NOMINAL, "Pagador o asegurador (`MC` Medicare, `BC` Blue Cross...).", "UCI dice 'integer identifier': son textos. Falta el 39,6 %."),
    "medical_specialty": (NOMINAL, "Especialidad del medico que admite.", "UCI dice 'integer identifier': son textos. Falta el 49,1 %."),
    "num_lab_procedures": (CONTEO, "Examenes de laboratorio hechos durante el encuentro.", ""),
    "num_procedures": (CONTEO, "Procedimientos distintos de laboratorio durante el encuentro.", ""),
    "num_medications": (CONTEO, "Medicamentos genericos distintos administrados durante el encuentro.", ""),
    "number_outpatient": (CONTEO, "Consultas ambulatorias del paciente en el ano previo al encuentro.", "Historia previa, no del encuentro actual."),
    "number_emergency": (CONTEO, "Visitas a urgencias del paciente en el ano previo al encuentro.", "Historia previa, no del encuentro actual."),
    "number_inpatient": (CONTEO, "Hospitalizaciones del paciente en el ano previo al encuentro.", "Historia previa. Suele ser el predictor mas fuerte de reingreso."),
    "diag_1": (NOMINAL, "Diagnostico principal, codigo ICD-9.", "UCI dice 'primeros tres digitos': hay 8 522 con decimal (`250.83`) y 1 645 con letra (`V57`, `E878`). Cardinalidad alta: hay que agrupar."),
    "diag_2": (NOMINAL, "Diagnostico secundario, codigo ICD-9.", "Mismo formato que diag_1."),
    "diag_3": (NOMINAL, "Diagnostico secundario adicional, codigo ICD-9.", "Mismo formato que diag_1."),
    "number_diagnoses": (CONTEO, "Cantidad de diagnosticos registrados en el sistema.", ""),
    "max_glu_serum": (ORDINAL, "Resultado de glucosa serica: `>200`, `>300`, `Norm`, o `None` si no se ordeno el examen.", "`None` NO es faltante: significa que no se pidio el examen. Son 96 420 filas (94,7 %)."),
    "A1Cresult": (ORDINAL, "Hemoglobina glicosilada: `>8`, `>7`, `Norm`, o `None` si no se ordeno el examen.", "`None` NO es faltante: 84 748 filas (83,3 %). Que se haya medido o no es la hipotesis del paper original."),
    "change": (BINARIA, "Hubo cambio en la medicacion para diabetes: `Ch` o `No`.", ""),
    "diabetesMed": (BINARIA, "Se receto algun medicamento para diabetes: `Yes` o `No`.", ""),
    "readmitted": (OBJETIVO, "Dias hasta el reingreso: `<30`, `>30`, o `NO` si no hubo registro de reingreso.", "Se binariza a `<30` contra el resto: un reingreso a los ocho meses no es un fallo de la transicion de cuidado."),
}
for _f in FARMACOS:
    VARIABLES[_f] = (ORDINAL, DESC_FARMACO, "")


def mil(n: int) -> str:
    """1652 -> '1 652'. Separador de miles con espacio, como se escribe en espanol."""
    return f"{n:,}".replace(",", " ")


def pct(x: float, decimales: int = 1) -> str:
    """2.38 -> '2,38 %'. Coma decimal."""
    return f"{x:.{decimales}f}".replace(".", ",") + " %"


def leer_crudo() -> pd.DataFrame:
    """Lee sin que pandas interprete centinelas. Es la unica lectura honesta."""
    return pd.read_csv(CRUDOS, dtype=str, keep_default_na=False, na_values=[])


def leer_mapa_ids() -> dict[str, list[tuple[str, str]]]:
    """IDs_mapping.csv trae tres tablas pegadas, separadas por una linea ','."""
    bloques: dict[str, list[tuple[str, str]]] = {}
    actual = None
    with MAPA_IDS.open(newline="", encoding="utf-8") as f:
        for fila in csv.reader(f):
            if len(fila) < 2 or not fila[0].strip():
                continue
            codigo, descripcion = fila[0].strip(), fila[1].strip()
            if descripcion == "description":      # cabecera: abre un bloque
                actual = codigo
                bloques[actual] = []
            elif actual:
                bloques[actual].append((codigo, descripcion))
    return bloques


def perfilar(c: pd.DataFrame) -> dict[str, dict]:
    n = len(c)
    perfil = {}
    for col in c.columns:
        s = c[col]
        faltantes = int((s == FALTANTE).sum())
        perfil[col] = {
            "distintos": int(s.nunique()),
            "faltantes": faltantes,
            "pct": faltantes / n * 100,
            "no_medido": int((s == NO_MEDIDO).sum()) if col in ("A1Cresult", "max_glu_serum") else 0,
            "valores": list(s.value_counts().index),
            "constante": s.nunique() == 1,
        }
    return perfil


def render(c: pd.DataFrame, perfil: dict, mapas: dict) -> str:
    n = len(c)
    L: list[str] = []
    A = L.append

    A("# Diccionario de variables")
    A("")
    A("Referencia de trabajo del equipo. **No es un entregable del enunciado**: existe para")
    A("no adivinar entre 50 columnas mientras se explora y se modela.")
    A("")
    A(f"Dataset: *Diabetes 130-US Hospitals for Years 1999-2008* (UCI 296) — {mil(n)} filas x {len(c.columns)} columnas.")
    A("")
    A(f"Generado por `src/data/diccionario.py` el {date.today().isoformat()}. Para regenerarlo:")
    A("")
    A("```bash")
    A("python -m src.data.diccionario")
    A("```")
    A("")
    A("## Como leer el archivo")
    A("")
    A("Con las opciones por defecto pandas se equivoca **en las dos direcciones**:")
    A("esconde faltantes reales e inventa faltantes que no existen.")
    A("")
    A("| | Por defecto | La realidad |")
    A("|---|---|---|")
    A(f"| `race` | 0 nulos | {mil(perfil['race']['faltantes'])} valores `?` |")
    A(f"| `A1Cresult` | {mil(perfil['A1Cresult']['no_medido'])} nulos | 0 faltantes: `None` = no se ordeno el examen |")
    A("")
    A("El centinela del archivo es `?`, no `NaN` — asi que `df.isna().sum()` da cero y")
    A("parece que no faltara nada. Y `None`, que en `A1Cresult` y `max_glu_serum` es una")
    A("categoria clinica legitima, esta en la lista de nulos por defecto de pandas y se")
    A("convierte en `NaN` sin avisar.")
    A("")
    A("Lectura correcta:")
    A("")
    A("```python")
    A("df = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_values=[])")
    A('df = df.replace("?", pd.NA)   # ahora si, y solo donde corresponde')
    A("```")
    A("")
    A("## Variables")
    A("")
    A("`Falta` cuenta unicamente el centinela `?`. Los `None` de `A1Cresult` y")
    A("`max_glu_serum` no se cuentan porque no son faltantes.")
    A("")
    A("| # | Columna | Tipo | Distintos | Falta | Descripcion | Notas |")
    A("|--:|---|---|--:|--:|---|---|")
    for i, col in enumerate(c.columns, 1):
        tipo, desc, nota = VARIABLES[col]
        p = perfil[col]
        if p["faltantes"] == 0:
            falta = "—"
        else:
            # Con un decimal, diag_1 (21 filas) se mostraba como "0,0 %" y parecia vacio.
            falta = pct(p["pct"], 2 if p["pct"] < 0.1 else 1)
        if p["constante"]:
            nota = (nota + " " if nota else "") + f"**Varianza cero**: un solo valor (`{p['valores'][0]}`). Descartable."
        A(f"| {i} | `{col}` | {tipo} | {p['distintos']} | {falta} | {desc} | {nota or '—'} |")
    A("")
    A(LEYENDA_FARMACOS)
    A("")

    A("## Columnas casi constantes")
    A("")
    A("Aparte de las de varianza cero, varias columnas de farmaco tienen tan pocos casos")
    A("distintos de `No` que no aportan senal:")
    A("")
    A("| Columna | Filas distintas de `No` |")
    A("|---|--:|")
    casi = []
    for col in FARMACOS:
        s = c[col]
        distintas = int((s != "No").sum())
        if 0 < distintas <= 50:
            casi.append((col, distintas))
    for col, k in sorted(casi, key=lambda x: x[1]):
        A(f"| `{col}` | {k} |")
    A("")

    etiquetas = {
        "admission_type_id": "Tipo de admision",
        "discharge_disposition_id": "Destino al egreso",
        "admission_source_id": "Via de ingreso",
    }
    A("## Tablas de codigos")
    A("")
    A("De `IDs_mapping.csv`. Sin esto las tres columnas de arriba son numeros sueltos.")
    A("")
    A("La marca ⛔ senala los destinos que se excluyen, por dos razones **distintas**:")
    A("")
    fallecido = {"11", "19", "20", "21"}
    hospicio = {"13", "14"}
    excluir = fallecido | hospicio
    col = c["discharge_disposition_id"]
    obj = c["readmitted"]

    n_fall = int(col.isin(fallecido).sum())
    # Para fallecidos se cuenta cualquier reingreso: la afirmacion es mas fuerte asi.
    r_fall = int(((col.isin(fallecido)) & (obj != "NO")).sum())
    n_hosp = int(col.isin(hospicio).sum())
    # Para hospicio se cuenta solo el reingreso temprano, que es la variable objetivo.
    r_hosp = int(((col.isin(hospicio)) & (obj == "<30")).sum())

    A(f"- **Fallecidos** (11, 19, 20, 21): {mil(n_fall)} filas. No pueden reingresar, y los datos lo")
    A(f"  confirman: {r_fall} reingresos registrados, ni antes ni despues de los 30 dias.")
    A(f"  Excluirlos es corregir un imposible.")
    A(f"- **Hospicio** (13, 14): {mil(n_hosp)} filas. Estos si reingresan: {r_hosp} lo hacen antes")
    A(f"  de los 30 dias.")
    A("  Se excluyen por alcance, no por imposibilidad: son pacientes en cuidado de fin de")
    A("  vida, donde agendar un control para evitar el reingreso no es la intervencion que")
    A("  decide el tablero.")
    A("")
    A(f"En total salen {mil(int(col.isin(excluir).sum()))} encuentros "
      f"({pct(col.isin(excluir).mean() * 100, 2)}).")
    A("")
    for columna, filas in mapas.items():
        presentes = set(c[columna].unique()) if columna in c.columns else set()
        A(f"### `{columna}` — {etiquetas.get(columna, '')}")
        A("")
        A("| Codigo | Descripcion | En los datos |")
        A("|--:|---|--:|")
        for codigo, descripcion in sorted(filas, key=lambda x: int(x[0])):
            marca = " ⛔" if columna == "discharge_disposition_id" and codigo in excluir else ""
            cuenta = int((c[columna] == codigo).sum()) if columna in c.columns else 0
            visto = mil(cuenta) if codigo in presentes else "—"
            A(f"| {codigo} | {descripcion}{marca} | {visto} |")
        A("")

    A("## Donde la ficha de UCI se equivoca")
    A("")
    A("Verificado contra el archivo, no copiado de la pagina:")
    A("")
    A("| Campo | UCI dice | El archivo dice |")
    A("|---|---|---|")
    A("| Simbolo de faltante | `NaN` | `?` — no hay un solo literal `NaN` |")
    A("| `weight` | Peso en libras | Bandas categoricas, vacio al 96,9 % |")
    A("| `payer_code` | Integer identifier | Texto (`MC`, `BC`, `HM`) |")
    A("| `medical_specialty` | Integer identifier | Texto (`Cardiology`, `InternalMedicine`) |")
    A("| `diag_*` | Primeros tres digitos de ICD-9 | 8 522 con decimal, 1 645 con letra `V`/`E` |")
    A("| Cardinalidades | 29 destinos, 848 diagnosticos | 26 y 717 en los datos reales |")
    A("")
    return "\n".join(L) + "\n"


def main() -> int:
    c = leer_crudo()
    texto = render(c, perfilar(c), leer_mapa_ids())
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(texto, encoding="utf-8")
    print(f"{SALIDA.relative_to(RAIZ)}: {len(c.columns)} variables, {len(texto.splitlines())} lineas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
