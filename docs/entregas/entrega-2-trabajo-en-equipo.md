# 6. Reporte de trabajo en equipo

## 6.1 Como nos organizamos

El trabajo se reparte por **item de trabajo**, no por persona: cada item vive en
su propia rama `feature/*`, sale de `develop` y vuelve a `develop` mediante un
pull request con revision de al menos un companero. Los merges conservan el
historial completo, sin *squash* ni *rebase* que colapsen la autoria, de modo
que el aporte de cada integrante queda verificable en el repositorio. `main`
conserva unicamente los estados integrados de la entrega. Entre el 14 de agosto
y el 6 de septiembre se abrieron 18 pull requests entre los dos repositorios, de
los cuales 17 se integraron.

La revision interna se hace sobre el pull request antes de integrar: la del
modelo de regresion logistica identifico un archivo que no compilaba, un umbral
calibrado sobre un modelo distinto al final y una dependencia de orden de
ejecucion no documentada; los tres se corrigieron antes del merge.

## 6.2 Quien hizo que

**Camilo Rodriguez** — Modelo de regresion logistica. Preparacion de
caracteristicas, agrupacion de diagnosticos ICD-9, particion por paciente y las
seis versiones del modelo: base, balanceo de clases, regularizacion, peso de la
clase positiva, ajuste de umbral y Elastic Net. Evaluacion con ROC-AUC, PR-AUC,
recall, F1/F2 y matriz de confusion, mas el analisis de priorizacion por
capacidad. Redacto las secciones de modelos y de conclusiones del reporte.
*Evidencia: ramas `feature/camilo-validacion-entrega1` y
`feature/camilo-regresion-entrega2`, pull requests #1 y #4.*

**Leonardo Almanza** — Modelo de bosque aleatorio. Conjunto de modelamiento a
partir de las decisiones del EDA, particion y validacion cruzada agrupadas por
paciente, comparacion de siete tecnicas de desbalance, cuatro escenarios de
hiperparametros del arbol, seleccion de caracteristicas por importancia de
permutacion y evaluacion con calibracion, en cuaderno ejecutado.
*Evidencia: rama `features/trees-random-boost-la`, pull request #6.*

**Rainer Solano** — Arquitectura de la solucion: documento `INFRA-V-0.0.1.md`
con la propuesta de infraestructura y los diagramas de desarrollo y produccion.
Arquitectura de software del tablero: estructura modular de vistas, componentes
y servicios, diseno del CSS, traduccion de la maqueta HTML a Streamlit y
despliegue en Railway para validacion inicial. Redacto la seccion del tablero
del reporte.
*Evidencia: rama `feature/infra-delivery-baseline-arq`, pull requests #3 y #10 en
`-api` y pull request #1 en `-ui`.*

**Katerine Arias** — Infraestructura de seguimiento de experimentos y coordinacion
de las entregas. Servidor de MLflow sobre AWS EC2 con autenticacion, apagado
automatico por inactividad y control de costo; usuarios de acceso para el equipo;
registro de los experimentos de regresion logistica; figuras del reporte desde
MLflow; separacion del repositorio en codigo y tablero. Consolidacion del reporte.
*Evidencia: ramas `feature/infra-mlflow-ec2`, `feature/mlflow-experimentos`,
`feature/reporte-entrega-2`, pull requests #5, #7 y #8.*

## 6.3 Como discutimos los resultados

Las dos familias de modelos se desarrollaron en paralelo y por separado, y
llegaron de forma independiente a la misma conclusion: ninguna tecnica de
balanceo ni ajuste de hiperparametros mejora la separacion entre clases; todas
reubican el punto de operacion sobre la misma curva. Que dos personas lo
encontraran por caminos distintos le da mas peso al hallazgo que si fuera
resultado de un solo experimento.
