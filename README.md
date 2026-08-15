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
├── params.yaml           # hiperparametros y rutas (leidos por DVC)
├── dvc.yaml              # pipeline: procesar -> features -> entrenar -> evaluar
├── docker-compose.yml    # levanta api + tablero
├── data/                 # DVC, no Git: raw/ interim/ processed/
├── models/               # artefactos empaquetados (DVC)
├── notebooks/            # exploracion; el codigo que sobrevive migra a src/
├── src/                  # libreria compartida: data/ features/ models/
├── api/                  # despliegue 1: sirve inferencias + Dockerfile
├── dashboard/            # despliegue 2: consume la API + Dockerfile
├── tests/
└── docs/                 # maqueta, reportes de entrega, soportes, manuales
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
