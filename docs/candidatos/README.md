# Candidatos de problema y dataset

Espacio de trabajo de las semanas 1 y 2: explorar problemas posibles, y para cada uno identificar la pregunta de negocio y el conjunto de datos asociado **garantizando su disponibilidad**.

Un archivo por candidato, copiando `PLANTILLA.md`. La decisión final y su justificación van al reporte de la Entrega 1.

## Criterios de descarte

Un candidato debe pasar los cinco. Si falla uno, se descarta y se documenta por qué — esa justificación es material del reporte.

| # | Criterio | Por qué |
|---|---|---|
| 1 | **Target supervisado claro** | El enunciado exige modelos supervisados. Debe existir una columna a predecir, o construible sin ambigüedad. |
| 2 | **Datos descargables ya** | El criterio de priorización del enunciado es la disponibilidad *inmediata*. Un dataset "que vamos a pedir" no clasifica. |
| 3 | **Pregunta de negocio real** | Alguien debe poder tomar una decisión distinta según la predicción. Si no cambia ninguna decisión, no hay producto. |
| 4 | **Columnas interpretables para el tablero** | El tablero debe visualizar *otros datos relevantes*, no solo predecir. Un dataset de features anónimas (`V1..V28`) no da material descriptivo. |
| 5 | **Tamaño manejable** | Debe entrenar en minutos en un portátil y moverse por DVC sin fricción. |

El criterio 4 es el que más se pasa por alto: se elige un dataset excelente para modelar y luego el tablero no tiene nada que mostrar más allá del score. Conviene revisarlo antes de comprometerse.

## Fuentes

- [Kaggle Datasets](https://www.kaggle.com/datasets) — CLI: `kaggle datasets list -s "<tema>" --sort-by votes`
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)
- [Datos Abiertos Colombia](https://www.datos.gov.co)
- [DANE — microdatos](https://microdatos.dane.gov.co)
- [Banco Mundial](https://datos.bancomundial.org)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Google Dataset Search](https://datasetsearch.research.google.com)

## Estado

| Candidato | Tipo | Target | Fuente | Veredicto |
|---|---|---|---|---|
| _(pendiente)_ | | | | |
