# Micro-proyecto — Desarrollo de Soluciones

Producto de datos desplegado: modelo supervisado empaquetado, API de inferencia y tablero, todo sobre contenedores Docker.

Maestría en Inteligencia Artificial (MAIA) — Universidad de los Andes.

> **Pendiente:** problema, pregunta de negocio y conjuntos de datos. Documentar aquí al definirlos.

## Arquitectura

```
tablero  →  API  →  modelo empaquetado
```

El tablero consume las predicciones **a través de la API**, nunca cargando el artefacto del modelo directamente. El tablero además visualiza datos descriptivos relevantes para el usuario.

Los tres componentes se despliegan en contenedores Docker.

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
| Git | Código: pipelines de procesamiento y entrenamiento, fuentes del tablero, artefactos de despliegue |
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
microproyecto-desarrollo-soluciones/
├── README.md
├── maia_pds_proy.pdf          # enunciado del curso
├── pyproject.toml             # dependencias y config del paquete src/   [pendiente]
├── params.yaml                # hiperparametros y rutas, leidos por DVC  [pendiente]
├── dvc.yaml                   # definicion del pipeline reproducible     [pendiente]
├── docker-compose.yml         # orquesta api + tablero                   [pendiente]
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
├── api/                       # DESPLEGABLE 1 — FastAPI + Dockerfile
├── dashboard/                 # DESPLEGABLE 2 — Streamlit + Dockerfile
├── tests/
│
└── docs/
    ├── maqueta/               # mockup del prototipo y sus iteraciones (E1)
    ├── entregas/              # reportes E1, E2, E3 (max 10 paginas c/u)
    └── soportes/              # evidencias: capturas de MLflow, DVC, Git
```

Las carpetas existen; los archivos marcados `[pendiente]` se crean al montar el pipeline.

### Reglas de la estructura

Cuatro decisiones que conviene no romper:

**`api/` y `dashboard/` viven en la raíz, no dentro de `src/`.** Cada uno es una unidad desplegable con su propio `Dockerfile` y su propio `requirements.txt`. Separarlos evita que la imagen del tablero arrastre `scikit-learn`, `mlflow` y demás dependencias de entrenamiento.

**`src/` es librería, no despliegue.** Contiene los pipelines de procesamiento y entrenamiento que la Entrega 3 exige tener versionados en el repositorio. Nadie la ejecuta como servicio.

**`dashboard/` no importa de `src/`.** Se comunica con la API únicamente por HTTP. Es la frontera que evalúa el enunciado, y la estructura la vuelve evidente: si el tablero necesita importar `src`, la arquitectura se rompió.

**`data/` y `models/` están fuera de Git.** Los versiona DVC; en Git solo viajan los punteros `.dvc`. Por eso ambas carpetas aparecen en `.gitignore` con excepciones para `.gitkeep` y `*.dvc`.

**`docs/soportes/` existe desde el día uno.** Los soportes son parte fundamental de cada entrega y su ausencia penaliza fuerte, así que se llenan sobre la marcha, no la víspera.

### Flujo de datos

```
data/raw  →  src/data  →  data/interim  →  src/features  →  data/processed
                                                                    ↓
                                                              src/models
                                                                    ↓
                                              models/  →  api/  →  dashboard/
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
