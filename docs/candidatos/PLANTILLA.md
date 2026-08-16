# Candidato: <nombre corto>

## Problema y contexto

<Qué pasa, a quién le pasa, por qué importa. Dos o tres frases.>

## Pregunta de negocio

<Una sola pregunta, en lenguaje de negocio, no de modelo.
Ej: "¿Qué clientes van a cancelar el servicio el próximo mes?"
No: "¿Podemos hacer clasificación binaria sobre churn?">

## Quién la usa y qué decide

<Usuario del tablero y la decisión concreta que cambia con la predicción.
Si nadie cambia ninguna decisión, el candidato no sirve.>

## Datos

| | |
|---|---|
| Fuente | <URL> |
| Licencia | <permite uso académico?> |
| Formato / tamaño | <CSV, N filas × M columnas, MB> |
| Descargable ya | <sí / no> |
| Periodo cubierto | |

### Variable objetivo

| | |
|---|---|
| Columna | |
| Tipo de tarea | <clasificación binaria / multiclase / regresión> |
| Distribución | <balance de clases, o rango y media> |
| Faltantes en el target | |

### Variables predictoras

<Cuántas, de qué tipo (numéricas, categóricas, fechas, texto), y calidad general.>

### Columnas para el componente descriptivo

<Cuáles sirven para las visualizaciones del tablero: geografía, tiempo,
segmentos, categorías. Sin esto el tablero solo muestra el score.>

## Modelo

<Qué se predice, con qué familia de modelos se empezaría y qué métrica
decide si sirve — ligada a la decisión de negocio, no solo accuracy.>

## Tablero (esbozo)

<Qué vería el usuario: qué entra como input de la predicción y qué
gráficas descriptivas acompañan.>

## Riesgos

<Fugas de información (leakage), desbalance severo, sesgos, columnas
que no existirían en el momento real de la predicción.>

## Evaluación

| # | Criterio | ¿Pasa? | Nota |
|---|---|---|---|
| 1 | Target supervisado claro | | |
| 2 | Datos descargables ya | | |
| 3 | Pregunta de negocio real | | |
| 4 | Columnas interpretables para el tablero | | |
| 5 | Tamaño manejable | | |

**Veredicto:** <viable / descartado — y por qué>
