# Micro-proyecto — Desarrollo de Soluciones

Producto de datos desplegado: modelo supervisado empaquetado, API de inferencia y tablero, todo sobre contenedores Docker.

Maestría en Inteligencia Artificial (MAIA) — Universidad de los Andes.

## Problema

**¿Qué pacientes diabéticos van a reingresar al hospital dentro de los 30 días siguientes al alta?**

El reingreso temprano es un indicador de calidad asistencial y una fuente de costo evitable. El equipo de gestión del alta puede intervenir — agendar control, ajustar medicación, activar enfermería domiciliaria — pero solo alcanza a hacerlo con una fracción de los pacientes.

La decisión que cambia con la predicción: **a quién se le agenda seguimiento antes de que salga del hospital**, cuando la capacidad de seguimiento es limitada.

## Datos

[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI, CC BY 4.0). 101 766 encuentros hospitalarios × 50 columnas.

| | |
|---|---|
| Unidad de análisis | un encuentro hospitalario (no un paciente: hay 71 518 pacientes únicos) |
| Variable objetivo | `readmitted`, binarizada a `<30` vs. resto — 11,16 % positivos |
| Versionado | DVC, puntero en `data/raw.dvc` |
| Remoto | `s3://maia-pds-diabetes-dvc-982005835034` (us-east-1) |

### Traerse los datos

```bash
uv tool install "dvc[s3]"     # o: pip install "dvc[s3]"
dvc pull
```

Eso es todo: **no hacen falta credenciales**. El bucket tiene lectura pública, así que cualquiera que clone el repositorio se trae los datos.

Para **escribir** (`dvc push`) sí se necesitan credenciales. El equipo usa un usuario
de AWS dedicado, `maia-pds-dvc`, con permisos acotados a este único bucket: puede leer,
escribir y listar, y nada más. No puede borrar objetos ni tocar ningún otro recurso de
la cuenta.

Las llaves se comparten **por fuera del repositorio** — gestor de contraseñas o mensaje
directo. Una vez que las tengas:

```bash
aws configure --profile maia-pds        # pega las llaves aquí
dvc remote modify --local storage profile maia-pds
```

`.dvc/config.local` está en `.gitignore`. Las credenciales **nunca** van al repositorio:
ni en un `.env`, ni como secreto de GitHub, ni en el historial. Un secreto commiteado
queda en el historial para siempre y limpiarlo obliga a reescribir la historia, que es
justo lo que borraría la autoría que se califica.

## Arquitectura

```
tablero  →  API  →  modelo empaquetado
```

El tablero consume las predicciones **a través de la API**, nunca cargando el artefacto del modelo directamente. El tablero además visualiza datos descriptivos relevantes para el usuario.

Los tres componentes se despliegan en contenedores Docker.

### Repositorios

El proyecto vive en dos repositorios, uno por unidad desplegable:

| Repositorio | Contiene |
|---|---|
| [`maia-pds-microproyecto-api`](https://github.com/katterine2558/maia-pds-microproyecto-api) (este) | Datos (DVC), pipelines de procesamiento y entrenamiento, experimentos MLflow, modelos empaquetados y la API que los sirve |
| [`maia-pds-microproyecto-ui`](https://github.com/katterine2558/maia-pds-microproyecto-ui) | Tablero: fuentes, artefactos de despliegue y su Dockerfile |

Separarlos hace estructural la frontera que evalúa el enunciado: el tablero no
comparte proceso ni sistema de archivos con el modelo, así que solo puede llegar
a las predicciones por HTTP.

El costo de la separación es que la evidencia de commits queda repartida. La nota
es individual y se sustenta en los commits, así que **cada integrante commitea en
ambos repositorios** y los reportes de entrega enlazan los dos.

### Stack

| Componente | Herramienta |
|---|---|
| API de inferencia | FastAPI |
| Tablero | Streamlit |
| Modelos | scikit-learn |
| Experimentos | MLflow |
| Datos | DVC |
| Despliegue | Docker + Docker Compose |

## Versionado

| Sistema | Versiona |
|---|---|
| Git | Código: pipelines de procesamiento y entrenamiento en este repo, fuentes del tablero en `-ui`, artefactos de despliegue en ambos |
| DVC | Datos (`data/`) y artefactos de modelo (`models/`) |
| MLflow | Experimentos, versiones de modelos y sus resultados |

Los datos no van en Git. Los punteros `.dvc` sí.

## Flujo de trabajo con Git

Repositorio con git-flow adaptado al proyecto:

| Rama | Rol |
|---|---|
| `main` | Solo estados entregados. Cada entrega queda taggeada aquí. |
| `develop` | Integración. Rama de trabajo diario. |
| `feature/*` | Una por ítem de trabajo. Sale de `develop` y vuelve a `develop`. |
| `release/*` | Preparación de cada entrega: `develop` → `release/entrega-N` → `main`. |
| `hotfix/*`, `bugfix/*` | Configuradas, sin uso previsto. |

Tags sin prefijo de versión: `release/entrega-1` produce el tag `entrega-1`.

```bash
git flow feature start exploracion-datos    # nueva rama de trabajo
git flow feature finish exploracion-datos   # mergea a develop

git flow release start entrega-1            # congela para la entrega
git flow release finish entrega-1           # mergea a main + develop, taggea
```

### Reglas de commits

La nota del curso es **individual** y se evalúa sobre los aportes reflejados en los commits. Por lo tanto:

- Cada integrante commitea **con su propia identidad**. Nadie sube el trabajo de otro.
- **No hacer squash ni rebase que colapse la autoría** al integrar ramas. Preferir merges que preserven quién hizo qué.
- Nombrar las ramas `feature/*` por ítem de trabajo, no por persona: el autor ya queda registrado en los commits.

## Estructura

```
maia-pds-microproyecto-api/
├── README.md
├── maia_pds_proy.pdf          # enunciado del curso
├── pyproject.toml             # dependencias y config del paquete src/   [pendiente]
├── params.yaml                # hiperparametros y rutas, leidos por DVC  [pendiente]
├── dvc.yaml                   # definicion del pipeline reproducible     [pendiente]
├── docker-compose.yml         # levanta la api en contenedor             [pendiente]
│
├── data/                      # versionado por DVC, NO por Git
│   ├── raw/                   # datos originales, inmutables
│   ├── interim/               # resultados intermedios del pipeline
│   └── processed/             # insumo final del entrenamiento
│
├── models/                    # artefactos empaquetados (.pkl / .joblib), DVC
│
├── notebooks/                 # exploracion y analisis
│
├── src/                       # libreria compartida (paquete instalable)
│   ├── data/                  # ingesta y limpieza
│   ├── features/              # transformaciones y construccion de variables
│   └── models/                # entrenamiento, evaluacion y empaquetado
│
├── api/                       # DESPLEGABLE — FastAPI + Dockerfile
├── tests/
│
└── docs/
    ├── guia-reporte.md        # que va en cada punto de los reportes de entrega
    ├── diccionario-variables.md  # referencia de las 50 columnas (generado)
    ├── maqueta/               # mockup del prototipo y sus iteraciones (E1)
    ├── entregas/              # reportes E1, E2, E3 (max 10 paginas c/u)
    └── soportes/              # evidencias: capturas de MLflow, DVC, Git
```

Las carpetas existen; los archivos marcados `[pendiente]` se crean al montar el pipeline.

### Reglas de la estructura

Cuatro decisiones que conviene no romper:

**`api/` vive en la raíz, no dentro de `src/`.** Es una unidad desplegable con su propio `Dockerfile` y su propio `requirements.txt`. Mantenerla aparte de `src/` evita que la imagen de servicio arrastre dependencias que solo usa el entrenamiento.

**`src/` es librería, no despliegue.** Contiene los pipelines de procesamiento y entrenamiento que la Entrega 3 exige tener versionados en el repositorio. Nadie la ejecuta como servicio.

**El tablero no importa de `src/`.** Vive en otro repositorio y se comunica con la API únicamente por HTTP. Es la frontera que evalúa el enunciado, y la separación en dos repos la vuelve estructural: el tablero no tiene forma de importar `src` ni de cargar el `.pkl`.

**`data/` y `models/` están fuera de Git.** Los versiona DVC; en Git solo viajan los punteros `.dvc`. Por eso ambas carpetas aparecen en `.gitignore` con excepciones para `.gitkeep` y `*.dvc`.

**`docs/soportes/` existe desde el día uno.** Los soportes son parte fundamental de cada entrega y su ausencia penaliza fuerte, así que se llenan sobre la marcha, no la víspera.

### Flujo de datos

```
data/raw  →  src/data  →  data/interim  →  src/features  →  data/processed
                                                                    ↓
                                                              src/models
                                                                    ↓
                                              models/  →  api/  →  HTTP  →  tablero
                                                    ↑
                                        MLflow registra cada experimento
```

## Entregas

| | Fecha | Contenido |
|---|---|---|
| 1 | dom 23 ago 2026 | Problema y contexto, pregunta de negocio, alcance, datasets, repos Git y DVC, exploración de datos, maqueta del prototipo, reporte de equipo |
| 2 | dom 6 sep 2026 | Modelos, experimentos en MLflow, tablero según la maqueta, repo con commits de todos, reporte de equipo |
| 3 | mar 22 sep 2026 | Todo integrado y desplegado en contenedores, datos en DVC, modelos empaquetados en la API, manual de usuario, manual de instalación, reporte de equipo, video ≤ 10 min |

Los reportes tienen un máximo de **10 páginas** (solo se califican las primeras 10). Los soportes son parte fundamental de cada entrega.

## Ejecución

Pendiente: documentar cómo entrenar, cómo levantar API y tablero (local y en contenedor), y cómo correr las pruebas.
