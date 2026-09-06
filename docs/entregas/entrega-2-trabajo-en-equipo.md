# Entrega 2 — Reporte de trabajo en equipo

**Micro-proyecto · Desarrollo de Soluciones · MAIA — Universidad de los Andes**
Camilo Andres Rodriguez Duenas · Jasbyn Rainier Solano Carrillo ·
Leonardo Almanza Sanchez · Gineth Katerine Arias Carrillo

*(maximo 1 pagina — tope del enunciado)*

## Como nos organizamos

El trabajo se reparte por **item de trabajo**, no por persona: cada item vive en
su propia rama `feature/*`, sale de `develop` y vuelve a `develop` mediante un
pull request con revision de al menos un companero. Los merges conservan el
historial completo, sin *squash* ni *rebase* que colapsen la autoria, de modo
que el aporte de cada integrante queda verificable en el repositorio.

`main` conserva unicamente los estados integrados de la entrega. Entre el 14 de
agosto y el 5 de septiembre se abrieron ocho ramas de trabajo y siete pull
requests.

La revision interna se hace sobre el pull request antes de integrar. La revision
del modelo de regresion logistica, por ejemplo, identifico un archivo que no
compilaba, un umbral calibrado sobre un modelo distinto al final y una
dependencia de orden de ejecucion no documentada; los tres se corrigieron antes
del merge.

## Quien hizo que

**Camilo Rodriguez** — Modelo de regresion logistica. Preparacion de
caracteristicas, agrupacion de diagnosticos ICD-9, particion por paciente y las
seis versiones del modelo: base, balanceo de clases, regularizacion, peso de la
clase positiva, ajuste de umbral y Elastic Net. Evaluacion con ROC-AUC, PR-AUC,
recall, F1/F2 y matriz de confusion, mas el analisis de priorizacion por
capacidad. Redacto las secciones de modelos y de conclusiones del reporte.
*Evidencia: ramas `feature/camilo-validacion-entrega1` y
`feature/camilo-regresion-entrega2`, pull requests #1 y #4.*

**Leonardo Almanza** — Modelo de bosque aleatorio. Construccion del conjunto de
modelamiento a partir de las decisiones del EDA, particion y validacion cruzada
agrupadas por paciente, comparacion de siete tecnicas de manejo del desbalance,
cuatro escenarios de hiperparametros del arbol, seleccion de caracteristicas por
importancia de permutacion y evaluacion con calibracion. Cuaderno de analisis
ejecutado.
*Evidencia: rama `features/trees-random-boost-la`, pull request #6.*

**Rainer Solano** — `Entrega 1.` Diseno de la arquitectura de la solucion. Documento `INFRA-V-0.0.1.md` con la propuesta de infraestructura y los diagramas de arquitectura de desarrollo y produccion. Desarrollo del tablero. Evidencia: rama feature/infra-delivery-baseline-arq, pull request [#3](https://github.com/katterine2558/maia-pds-microproyecto-api/pull/3), repositorio maia-pds-microproyecto-ui. `Entrega 2.` Diseño de la arquitectura de software para el tablero, estructura modular de vistas, componentes y servicios, diseño de CSS, traducción de experiencia de usuario HTML de maquetas a `arquitectura streamlit`, despliegue a Railway para validación inicial.
*Evidencia: commits [working look and feel](https://github.com/katterine2558/maia-pds-microproyecto-ui/commit/ac5a2622889488566818cabc06b2e7f85d1ba494), [fix invoke fake api](https://github.com/katterine2558/maia-pds-microproyecto-ui/commit/734f8ef3afd5c96bd9b4dd6b077417a7297f8b22).*
*Evidencia: pull request [#1](https://github.com/katterine2558/maia-pds-microproyecto-ui/pull/1).*

**Katerine Arias** — Infraestructura de seguimiento de experimentos y
coordinacion de las entregas. Servidor de MLflow sobre AWS EC2 con
autenticacion, apagado automatico por inactividad y control de costo; usuarios
de acceso para el equipo; registro de los experimentos de regresion logistica en
el servidor compartido; figuras del reporte construidas desde MLflow; separacion
del repositorio en codigo y tablero. Consolidacion del reporte.
*Evidencia: ramas `feature/infra-mlflow-ec2`, `feature/mlflow-experimentos`,
`feature/reporte-entrega-2`, pull requests #5, #7 y #8.*

## Como discutimos los resultados

Las dos familias de modelos se desarrollaron en paralelo y por separado, y
llegaron de forma independiente a la misma conclusion: ninguna tecnica de
balanceo ni ajuste de hiperparametros mejora la separacion entre clases; todas
reubican el punto de operacion sobre la misma curva. Que dos personas lo
encontraran por caminos distintos le da mas peso al hallazgo que si fuera
resultado de un solo experimento.
