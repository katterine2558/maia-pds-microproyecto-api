# Guía de los reportes de entrega

Qué debe contener cada punto que pide el enunciado, para no escribir de más ni dejar huecos.
Fuente: `maia_pds_proy.pdf`.

## Reglas que aplican a los tres reportes

| Regla | Consecuencia práctica |
|---|---|
| **Máximo 10 páginas** | Si entregan 12, califican las 10 primeras. Lo importante va al frente. |
| **Los soportes son parte fundamental** | *"Su no entrega lleva a una alta penalización."* Van junto al reporte, no dentro de las 10 páginas. |
| **La nota es individual** | Se demuestra por commits + reporte de trabajo en equipo + sustentación. |
| **Cada gráfica debe ganarse su lugar** | Recomendación 2 del enunciado. Una figura que no sustenta una afirmación, sobra. |

Regla de oro para no desperdiciar espacio: **cada figura o tabla debe estar amarrada a una
afirmación del texto.** Si una gráfica no cambia ninguna conclusión, va a soportes.

---

# Entrega 1 — domingo 23 de agosto

Ocho puntos exigidos. Presupuesto sugerido sobre 10 páginas:

| # | Punto | Págs. |
|---|---|---|
| 1 | Problema y su contexto | 1 |
| 2 | Pregunta de negocio y alcance | 1 |
| 3 | Conjuntos de datos a emplear | 1,5 |
| 4 | Repositorio Git en uso | 0,5 |
| 5 | Repositorio DVC en uso | 0,5 |
| 6 | Exploración de los datos | 2,5 |
| 7 | Maqueta del prototipo | 1,5 |
| 8 | Reporte de trabajo en equipo | 1 |

Los puntos 4 y 5 son cortos: una captura o bloque de comando bien elegido y una línea de texto.
No conviene gastarles más espacio.

---

## 3. Conjuntos de datos a emplear

Este punto responde *qué datos vamos a usar y por qué son creíbles*. Once ítems:

### Procedencia y disponibilidad

1. **Identificación**: nombre, fuente institucional, URL, año.
2. **Licencia**: que permita uso académico. La nuestra es CC BY 4.0 → citar a Strack et al. (2014).
3. **Cómo se obtuvo**: URL exacta de descarga y `sha256` de los archivos. Esto es lo que hace
   reproducible la entrega y casi nadie lo pone.
4. **Por qué este dataset**: el enunciado fija el criterio de priorización —
   *disponibilidad inmediata de los datos*. Decir explícitamente que estaba descargable sin
   registro ni solicitud. Si descartaron alternativas, una frase por cada una.

### Qué hay adentro

5. **Unidad de análisis.** El punto que más se equivoca. Una fila es **un encuentro
   hospitalario**, no un paciente: 101 766 encuentros sobre 71 518 pacientes únicos. De aquí
   sale el riesgo de fuga del punto 10.
6. **Volumen y estructura**: filas × columnas, tipos, periodo cubierto.
7. **Variable objetivo**: definición, valores originales (`<30` / `>30` / `NO`), distribución
   y **la decisión de binarización con su justificación**. Nuestro caso: `<30` contra todo lo
   demás, porque un reingreso a los ocho meses no es un fallo de la transición de cuidado y
   no cambia a quién se agenda seguimiento. Es una decisión que hay que defender, no informar.
8. **Familias de variables**: demográficas, administrativas, utilización previa, clínicas,
   farmacológicas. Con una tabla resumida basta; no listar las 50 columnas.

### Qué le pasa a los datos

9. **Calidad y faltantes**: porcentajes reales y qué se hace con cada caso. Incluir la trampa
   de `None` en `A1Cresult` y `max_glu_serum` — no son faltantes sino la categoría "no se
   ordenó el examen", y pandas los convierte en nulos en silencio. Es exactamente el tipo de
   observación que el enunciado premia.
10. **Exclusiones y riesgos**, con su porqué:
    - Encuentros con `discharge_disposition_id` de fallecido u hospicio: no pueden reingresar.
    - Partición por `patient_nbr`, no aleatoria, por los pacientes repetidos.
    - Columnas de varianza cero (`examide`, `citoglipton`).
11. **Limitaciones y alcance**: 1999–2008, sistema de salud de EE. UU. El producto es un
    prototipo metodológico. Declararlo evita que el reporte prometa de más.

### El amarre que no puede faltar

Cerrar conectando los datos con la pregunta: **qué columnas alimentan el modelo y cuáles el
componente descriptivo del tablero.** Sin eso el punto queda como una ficha técnica suelta.

**Soportes de este punto:** `sha256` de los archivos, salida de `dvc status -c`, y el
diccionario de variables completo (que no cabe en 10 páginas).

---

## 6. Exploración de los datos

Distinto del punto 3. Allí se **describe** el dataset; aquí se **encuentra** algo.

Cada gráfica debe cerrar con una frase de tipo *"por lo tanto…"*. Cinco o seis hallazgos bien
argumentados valen más que quince gráficas.

Candidatos a hallazgo en nuestro caso:

- Desbalance de la clase objetivo → decide la métrica, no es un dato decorativo.
- Tasa de reingreso `<30` **por segmento** (edad, especialidad, tipo de admisión): dónde se
  concentra el riesgo. Es lo que después alimenta el tablero.
- Utilización previa (`number_inpatient`, `number_emergency`) contra la tasa de reingreso:
  suele ser el predictor más fuerte, y sostiene la elección de variables.
- Que se haya medido A1C contra la tasa de reingreso: es la hipótesis del paper original.
- Pacientes con encuentros repetidos: evidencia del riesgo de fuga y justificación del split
  por paciente.

**Soportes:** el notebook completo con todas las gráficas, incluidas las que no entraron.

---

## 4 y 5. Repositorios Git y DVC

La palabra que importa es **"en uso"**. No basta con que existan.

**Git** — en orden de peso:

| Evidencia | Comando |
|---|---|
| Aportes por miembro (sustenta la nota individual) | `git shortlog -sne --all` |
| Actividad distribuida en el tiempo | `git log --pretty='%ad %an %s' --date=short` |
| Ramas y git-flow funcionando | `git log --oneline --graph --all` |

Antes que nada: **la URL del repositorio y que los tutores tengan acceso.** Si está privado y
no los agregaron, el punto no se puede verificar.

**DVC:**

| Evidencia | Comando |
|---|---|
| Repo DVC inicializado | `ls .dvc/` |
| Remoto configurado | `dvc remote list` |
| Datos sincronizados | `dvc status -c` |
| El puntero en Git | `cat data/raw.dvc` |

Lo más elegante: mostrar el `.dvc` versionado en Git al lado del CSV ausente. Prueba las dos
cosas de un golpe — datos versionados por DVC, y fuera de Git.

En el reporte, un bloque de texto con la salida del comando ocupa cinco líneas y se lee mejor
que una captura de pantalla de media página. Las capturas van a soportes.

---

## 1, 2 y 7. Problema, alcance y maqueta

**Problema y contexto**: qué pasa, a quién, por qué importa, y qué se hace hoy sin el modelo.
Sin ese "hoy", no se puede argumentar que la solución aporte algo.

**Pregunta de negocio y alcance**: una sola pregunta, en lenguaje de negocio. El alcance debe
decir explícitamente qué **no** incluye el prototipo.

**Maqueta**: dos piezas, más una tabla.

1. Wireframe de las pantallas del tablero.
2. Diagrama de arquitectura `tablero → API → modelo`, cada uno en su contenedor — es la
   frontera que califican.
3. **Tabla de trazabilidad**: qué elemento del tablero responde qué parte de la pregunta de
   negocio. El enunciado pide la *relación* con la pregunta, no solo el dibujo; es la mitad
   del punto y la que más se olvida.

Las semanas 4, 5 y 6 exigen el tablero "de acuerdo con la maqueta": es un contrato. El momento
de cambiarla es ahora, con los datos ya explorados.

---

## 8. Reporte de trabajo en equipo

Qué hizo cada quien, cómo se repartió, cómo se coordinaron, fechas de revisión interna.
Debe ser **consistente con `git shortlog`**: si el reporte dice que alguien lideró el EDA y no
tiene commits del notebook, la inconsistencia se nota.

---

# Checklist antes de entregar

- [ ] ¿Diez páginas o menos?
- [ ] ¿Cada gráfica sostiene una afirmación del texto?
- [ ] ¿Los soportes están completos y adjuntos?
- [ ] ¿La URL del repo aparece y los tutores tienen acceso?
- [ ] ¿`git shortlog` muestra a los cuatro integrantes?
- [ ] ¿El reporte de equipo concuerda con el historial de commits?
- [ ] ¿Las decisiones están justificadas, no solo enunciadas?
