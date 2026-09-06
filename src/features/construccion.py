"""Construccion del conjunto de modelamiento.

Va del archivo crudo al par (X, y) listo para entrenar, aplicando las
decisiones declaradas en `esquema.py`. El modulo es deliberadamente
determinista y sin estado: la misma entrada produce siempre la misma salida,
para que un experimento registrado en MLflow sea reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.features import esquema as esq

RAIZ = Path(__file__).resolve().parents[2]
CRUDOS = RAIZ / "data" / "raw" / "diabetic_data.csv"


# --------------------------------------------------------------------------
# 1. Carga
# --------------------------------------------------------------------------

def cargar(ruta: Path | str = CRUDOS) -> pd.DataFrame:
    """Lee el archivo crudo sin que pandas reinterprete los valores.

    Tres argumentos hacen el trabajo. `keep_default_na=False` y `na_values=[]`
    impiden que el texto "None" de A1Cresult y max_glu_serum se convierta en
    nulo: alli "None" significa que el examen no se ordeno, y eso es
    informacion clinica, no un dato perdido (EDA 1.2). `dtype=str` evita que
    los identificadores numericos se lean como enteros.

    El centinela "?" se convierte despues, de forma explicita.
    """
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_values=[])
    df = df.replace(esq.AUSENTE, pd.NA)

    for col in esq.ENTERAS:
        df[col] = pd.to_numeric(df[col]).astype("int64")

    return df


def verificar_carga(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de comprobaciones fallidas. Vacia si todas pasan."""
    fallas = []

    if len(df) != esq.FILAS_ESPERADAS:
        fallas.append(f"filas: {len(df):,} (se esperaban {esq.FILAS_ESPERADAS:,})")
    if df.shape[1] != esq.COLUMNAS_ESPERADAS:
        fallas.append(f"columnas: {df.shape[1]} (se esperaban {esq.COLUMNAS_ESPERADAS})")

    for col in ("A1Cresult", "max_glu_serum"):
        if df[col].isna().any():
            fallas.append(f"{col}: tiene nulos; 'None' debio conservarse como texto")

    if (df.astype(str) == esq.AUSENTE).any().any():
        fallas.append("quedan valores '?' sin convertir")

    return fallas


# --------------------------------------------------------------------------
# 2. Conjunto de trabajo
# --------------------------------------------------------------------------

def conjunto_de_trabajo(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las exclusiones por destino al egreso y crea la variable objetivo.

    Salen los egresos por fallecimiento, que no admiten reingreso, y los
    egresos a hospicio, donde programar un control no es la intervencion
    pertinente (EDA 5.2 y 5.3).
    """
    trabajo = df[~df["discharge_disposition_id"].isin(esq.EXCLUIDOS)].copy()
    trabajo[esq.OBJETIVO] = (
        trabajo[esq.OBJETIVO_ORIGEN] == esq.VALOR_POSITIVO
    ).astype("int64")
    return trabajo


# --------------------------------------------------------------------------
# 3. Transformaciones
# --------------------------------------------------------------------------

def grupo_icd9(codigo) -> str:
    """Agrupa un codigo ICD-9 en su categoria clinica.

    Reduce los 716 codigos de diag_1 a nueve categorias. La agrupacion es la
    del articulo original (Strack et al., 2014), con una salvedad: los codigos
    que empiezan por V o E son factores externos y de contacto con servicios
    de salud, y van a la categoria restante (EDA 6.3).
    """
    if pd.isna(codigo):
        return esq.SIN_REGISTRO

    codigo = str(codigo)
    if codigo.startswith("250"):
        return "Diabetes"
    if codigo.startswith(("V", "E")):
        return "Otras"

    n = float(codigo)
    if 390 <= n <= 459 or n == 785:
        return "Circulatorio"
    if 460 <= n <= 519 or n == 786:
        return "Respiratorio"
    if 520 <= n <= 579 or n == 787:
        return "Digestivo"
    if 800 <= n <= 999:
        return "Lesiones"
    if 710 <= n <= 739:
        return "Musculoesqueletico"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinario"
    if 140 <= n <= 239:
        return "Neoplasias"
    return "Otras"


def _tramo_ordinal(serie: pd.Series, cortes: list, etiquetas: list) -> pd.Series:
    """Agrupa una variable numerica en tramos y los codifica por su posicion.

    El tramo es ordinal por construccion —"3-4" esta entre "2" y "5 o mas"—
    de modo que se codifica con un entero y no con indicadoras.
    """
    tramos = pd.cut(serie, bins=cortes, labels=etiquetas)
    posicion = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
    return tramos.map(posicion).astype("int64")


def agrupar_raras(serie: pd.Series, umbral: int = esq.UMBRAL_CATEGORIA_RARA) -> pd.Series:
    """Reune bajo una sola categoria los valores con menos de `umbral` registros.

    Reduce la cardinalidad sin costo informativo apreciable: una categoria con
    treinta casos no sostiene una tasa estimable, y en cambio obliga al arbol a
    considerar una particion mas en cada nodo.
    """
    frecuencias = serie.value_counts()
    raras = set(frecuencias[frecuencias < umbral].index)
    return serie.where(~serie.isin(raras), esq.CATEGORIA_RESTO)


def transformar(
    trabajo: pd.DataFrame,
    usar_tramos: bool = True,
    umbral_raras: int | None = esq.UMBRAL_CATEGORIA_RARA,
) -> pd.DataFrame:
    """Aplica las transformaciones de EDA 9.3 sobre el conjunto de trabajo.

    `usar_tramos` controla la decision del EDA 7.1 de llevar el historial del
    ano previo agrupado en tramos. `umbral_raras` agrupa las categorias poco
    frecuentes. Ambos se dejan como parametros para poder contrastar las
    alternativas como experimentos separados en MLflow.
    """
    df = trabajo.copy()

    # Los tres diagnosticos comparten la misma agrupacion.
    for col in esq.DIAGNOSTICOS:
        df[col] = df[col].map(grupo_icd9)

    # Ordinal: se preserva el orden de las decadas.
    posicion_edad = {tramo: i for i, tramo in enumerate(esq.ORDEN_EDAD)}
    df["age"] = df["age"].map(posicion_edad).astype("int64")

    if usar_tramos:
        for col, (cortes, etiquetas) in esq.TRAMOS.items():
            df[col] = _tramo_ordinal(df[col], cortes, etiquetas)

    # Nominales cuyo codigo numerico no ordena nada.
    for col in esq.NOMINALES_CODIFICADAS:
        df[col] = df[col].astype(str)

    # Ninguna columna se imputa: la ausencia es una categoria explicita.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna(esq.SIN_REGISTRO)

    if umbral_raras:
        # Solo sobre predictoras. Agrupar `patient_nbr` fundiria a los 69 990
        # pacientes en una sola categoria y dejaria la particion sin grupos;
        # `race` esta reservada para el analisis de sesgo y no se altera.
        intocables = set(esq.NO_PREDICTORAS) | set(esq.IDENTIFICADORES)
        for col in df.columns:
            if df[col].dtype == object and col not in intocables:
                df[col] = agrupar_raras(df[col], umbral_raras)

    return df


# --------------------------------------------------------------------------
# 4. Matriz de modelamiento
# --------------------------------------------------------------------------

def predictoras(df: pd.DataFrame, solo_formulario: bool = False) -> list[str]:
    """Columnas que entran al modelo.

    `solo_formulario` restringe la matriz a los diez campos de la Pantalla 2
    de la maqueta, para la comparacion que el EDA 9.5 dejo encargada.
    """
    if solo_formulario:
        return list(esq.FORMULARIO)

    fuera = set(esq.DESCARTADAS) | set(esq.NO_PREDICTORAS)
    return [c for c in df.columns if c not in fuera]


def matriz(
    trabajo: pd.DataFrame,
    usar_tramos: bool = True,
    solo_formulario: bool = False,
    umbral_raras: int | None = esq.UMBRAL_CATEGORIA_RARA,
    columnas: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Devuelve (X, y, grupos) listos para particionar y entrenar.

    `grupos` es `patient_nbr`: no entra como variable, agrupa la particion
    para que ningun paciente quede a ambos lados (EDA 8.1).

    `columnas` permite fijar un subconjunto explicito de predictoras, para las
    comparaciones de seleccion de caracteristicas.
    """
    df = transformar(trabajo, usar_tramos=usar_tramos, umbral_raras=umbral_raras)
    if columnas is None:
        columnas = predictoras(df, solo_formulario=solo_formulario)

    X = df[columnas]
    y = df[esq.OBJETIVO]
    grupos = df[esq.AGRUPADOR]

    return X, y, grupos


def columnas_por_tipo(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa las columnas numericas de las categoricas."""
    numericas = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categoricas = [c for c in X.columns if c not in numericas]
    return numericas, categoricas


def indices_categoricas(X: pd.DataFrame) -> list[int]:
    """Posicion de las categoricas en la matriz que produce el preprocesador nativo.

    Los modelos de arboles que manejan categoricas de forma nativa reciben esas
    posiciones, no los nombres, porque el ColumnTransformer entrega un arreglo.
    """
    numericas, categoricas = columnas_por_tipo(X)
    return list(range(len(numericas), len(numericas) + len(categoricas)))


def construir_preprocesador_nativo(X: pd.DataFrame) -> ColumnTransformer:
    """Preprocesador para los modelos que manejan categoricas de forma nativa.

    Codifica cada categoria con un entero y deja que el modelo decida como
    agruparlas. Frente a las indicadoras, sobre este conjunto reduce la matriz
    de 215 a 32 columnas y de 137 a 20 MB, entrena en algo mas de un tercio del
    tiempo y mejora ligeramente el desempeno.

    La mejora no es casual: con indicadoras, un arbol solo puede preguntar
    "es esta categoria, si o no", de modo que separar un grupo de veinte
    especialidades exige veinte particiones sucesivas. Con soporte nativo lo
    resuelve en una.

    Las categorias no vistas y los ausentes van a NaN, que estos modelos
    manejan sin imputar.
    """
    numericas, categoricas = columnas_por_tipo(X)

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numericas),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            ), categoricas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """Arma el preprocesador a partir del tipo de cada columna de X.

    Las numericas —incluidas edad y los tramos, que ya son ordinales— pasan
    directo o escaladas segun el modelo. Las categoricas se codifican con
    indicadoras; `handle_unknown="ignore"` evita que una categoria vista solo
    en evaluacion rompa la inferencia.

    No se escala: los modelos del proyecto son de arboles, invariantes a las
    transformaciones monotonas. Se verifico con StandardScaler, MinMaxScaler y
    RobustScaler, y las metricas resultaron identicas.
    """
    numericas, categoricas = columnas_por_tipo(X)

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def armar_pipeline(
    X: pd.DataFrame,
    estimador,
    representacion: str = "indicadoras",
) -> Pipeline:
    """Encadena preprocesamiento y estimador en un solo objeto.

    Que sean un solo objeto importa por dos razones: el preprocesamiento se
    ajusta unicamente con los datos de entrenamiento, y el artefacto que se
    registra en MLflow recibe el DataFrame crudo y no exige que la API
    reproduzca la transformacion por su cuenta.

    `representacion` elige entre indicadoras y codificacion nativa. La segunda
    solo aplica a los modelos que declaran que columnas son categoricas; el
    llamador debe haberselas pasado al estimador.
    """
    if representacion == "nativa":
        preparacion = construir_preprocesador_nativo(X)
    else:
        preparacion = construir_preprocesador(X)

    return Pipeline([("preparacion", preparacion), ("estimador", estimador)])
