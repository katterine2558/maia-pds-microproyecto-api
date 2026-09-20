# Predicción de reingreso hospitalario temprano en pacientes diabéticos

**Microproyecto de Desarrollo de Soluciones — Entrega 3**  
Camilo Andrés Rodríguez Dueñas · Jasbyn Rainier Solano Carrillo · Leonardo Almanza Sánchez · Gineth Katerine Arias Carrillo  
20 de septiembre de 2026

## 1. Resumen sobre el problema

Los pacientes diabéticos pueden requerir nuevas hospitalizaciones poco después del alta. Para el personal de enfermería, identificar a quién contactar primero es una tarea difícil cuando el número de egresos supera la capacidad disponible para seguimiento. La pregunta que orienta este proyecto es: **¿qué pacientes diabéticos presentan mayor riesgo de reingresar durante los 30 días posteriores al alta?**

El proyecto busca apoyar la organización de los contactos posteriores al egreso mediante un tablero que permita consultar el riesgo estimado de un paciente y ordenar un listado de egresos. La predicción sirve como apoyo para asignar la capacidad de seguimiento. No establece tratamientos ni reemplaza la valoración clínica.

Se utiliza el conjunto *Diabetes 130-US Hospitals for Years 1999-2008* del UCI Machine Learning Repository. De sus 101.766 encuentros originales se excluyeron 1.652 registros de pacientes fallecidos y 771 egresos a hospicio. La base analítica quedó conformada por **99.343 encuentros de 69.990 pacientes**. Aproximadamente el **11,4 %** de los encuentros registra un reingreso antes de 30 días, identificado mediante el valor `<30` de la variable `readmitted`.

En la Entrega 2 se compararon los modelos y se desarrollaron las primeras vistas del tablero. La consulta de Paciente aún mostraba un resultado ilustrativo. En esta entrega se preparó un modelo ajustado a los campos del formulario, se desarrolló la API y se comprobó localmente que la vista Paciente presenta una predicción calculada. La priorización de un archivo de egresos y el despliegue conjunto del tablero y la API siguen en desarrollo.

Los datos fueron recopilados en hospitales de Estados Unidos entre 1999 y 2008. Los resultados obtenidos con esta base no acreditan el desempeño del modelo en hospitales colombianos ni con pacientes actuales.

## 2. Modelos desarrollados y su evaluación

Para preparar los datos se trataron variables administrativas y clínicas, se agruparon diagnósticos ICD-9 y se transformaron las variables categóricas. El identificador `patient_nbr` se utilizó para separar los conjuntos sin compartir pacientes entre entrenamiento y prueba, pero no como predictor. La variable `race` se reservó para examinar el comportamiento entre grupos y tampoco se incorporó como entrada.

En la Entrega 2 se compararon modelos de regresión logística y bosque aleatorio. La regresión inicial alcanzó **88,24 % de exactitud** en validación, pero identificó apenas el **2,00 % de los reingresos**. Por ello se evaluaron configuraciones dirigidas a mejorar la detección de la clase positiva. La comparación final se realizó entre la regresión logística V5 y el bosque aleatorio V2 sobre **19.821 encuentros de prueba**, incluidos **2.207 reingresos**.

| Modelo | ROC-AUC | PR-AUC | Recall | Precisión | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| Regresión logística V5 | 0,6589 | 0,2133 | 81,56 % | 14,00 % | 407 |
| Bosque aleatorio V2 | 0,6673 | 0,2123 | 90,48 % | 12,83 % | 210 |

Se escogió el bosque V2 porque, al umbral evaluado, dejó sin identificar **210 reingresos**, frente a **407** de la regresión. La precisión de **12,83 %** advierte, sin embargo, que numerosos pacientes señalados para seguimiento no reingresaron en el periodo observado. El número de alertas debe examinarse en relación con los contactos que el equipo de enfermería pueda realizar.

En la regresión V5, el 10 % de encuentros con mayor riesgo estimado concentró el **22,97 % de los reingresos**, con un *lift* de **2,30**. Esta medición corresponde únicamente a dicha regresión.

Al preparar la integración se encontró que el bosque V2 había sido entrenado con más variables que las diez solicitadas en el formulario Paciente. Se entrenó entonces `bosque_formulario_e3_v1` con los campos disponibles en esa vista. La separación agrupada por paciente dejó **79.522 encuentros para entrenamiento** y **19.821 para prueba**, con **2.207 reingresos** en este último conjunto. El pipeline combina el procesamiento de categorías con un bosque de **400 árboles**, profundidad máxima de **12**, peso positivo de **5** y umbral de **0,30**.

| Modelo utilizado en la prueba local | ROC-AUC | PR-AUC | Recall | Precisión | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| `bosque_formulario_e3_v1` | 0,6310 | 0,1929 | 90,89 % | 12,26 % | 201 |

El modelo del formulario obtuvo valores menores de ROC-AUC y PR-AUC que el bosque V2. Las métricas de los dos modelos deben mantenerse diferenciadas: la predicción mostrada en Paciente proviene de `bosque_formulario_e3_v1`, mientras que los resultados de V2 corresponden al modelo estudiado en la Entrega 2 con más variables de entrada.

El modelo que sirve la API quedó registrado en el servidor de MLflow sobre EC2, en el experimento `readmision-diabetes`, bajo la corrida `bosque-formulario-e3-v1` (`1ca8cedf2b9b41378c222fc6129ac49d`), con sus once parámetros, sus métricas y el artefacto empaquetado. El registro se hizo con `src/models/experimento_formulario_e3.py`, que separa el experimento del entrenamiento: `entrenar_formulario_e3.py` debe poder correrse sin credenciales ni red al empaquetar la API, mientras que el registro exige el servidor arriba.

Al registrar la corrida se volvió a entrenar el modelo y las métricas no resultaron idénticas a las de la primera ejecución, pese a compartir `random_state=42`: ROC-AUC pasó de 0,6292 a 0,6310 y los falsos negativos bajaron de 223 a 201. La diferencia proviene del entorno de ejecución —versión de scikit-learn y el paralelismo de `n_jobs=-1` en el bosque—, no de un cambio en los datos ni en los hiperparámetros. Las cifras reportadas arriba son las de la corrida registrada en MLflow, que corresponden al artefacto que efectivamente sirve la API: la reproducibilidad exacta entre máquinas exigiría fijar también las versiones de las bibliotecas.

## 3. Descripción del tablero desarrollado y su funcionalidad

**Por Actualizar**
El tablero está desarrollado en Streamlit y contiene las vistas **Paciente**, **Priorización** y **Contexto**. La API, desarrollada en FastAPI, recibe los datos del encuentro, ejecuta el pipeline y devuelve la probabilidad estimada, el umbral y la versión del modelo. El tablero consulta este servicio por HTTP.

**Paciente** solicita diez datos del encuentro y presenta la respuesta de `POST /predict`. **Contexto** ofrece gráficas descriptivas del conjunto de datos; estas gráficas no representan predicciones individuales. **Priorización** muestra por ahora encuentros de demostración. Para completar esta última vista falta recibir un archivo de egresos, validar sus columnas, obtener las predicciones y ordenar los casos según la capacidad de seguimiento.

Realice en una prueba local, `/health` respondió con `{"estado":"ok","modelo":"bosque_formulario_e3_v1"}`. También se comprobó una respuesta de predicción y el rechazo de una estancia de cero días con estado **422**.

Se ejecutaron la API en `127.0.0.1:8000` y el tablero en `localhost:8501`. En Paciente se ingresó un encuentro con edad `[70-80)`, admisión `Emergency`, servicio `Nephrology`, nueve días de estancia, nueve diagnósticos, 21 medicamentos, cinco ingresos previos, dos visitas previas a urgencias, A1C no medido y cambio de medicación `Sí`. La pantalla presentó una **probabilidad de 0,64**, **riesgo alto**, umbral de **0,30** y versión **`bosque_formulario_e3_v1`**. Se comprobó así que el formulario consultó la API local y mostró su respuesta. También se retiró de la tarjeta una cifra fija de «cohorte comparable»; tras la corrección, el resultado se presentó sin código HTML visible.

## 4. Despliegue del tablero y la API

**Por ejecutar**.

En pruebas dentro en mi computador, El tablero y la API se encuentran en repositorios separados. Ambos cuentan con archivos para preparar su ejecución en contenedores: el tablero dispone de un `Dockerfile` y `railway.json`, y para la API se prepararon `api/Dockerfile` y `api/requirements.txt`. La dirección de la API se configura mediante `API_URL`; durante la prueba local se utilizó `http://localhost:8000`. Todavía no se ha comprobado el funcionamiento conjunto de los dos contenedores en nube. El tablero había sido presentado anteriormente en Railway, pero esa publicación no incluía la nueva API de esta entrega

## 5. Principales resultados y conclusiones

**Por ejecutar**.

## 6. Soportes de la entrega

El código del modelo, la API y la documentación se encuentran en el [repositorio de modelos y API](https://github.com/katterine2558/maia-pds-microproyecto-api); los cambios de esta etapa están propuestos en el [PR #19](https://github.com/katterine2558/maia-pds-microproyecto-api/pull/19). El tablero se encuentra en el [repositorio de la interfaz](https://github.com/katterine2558/maia-pds-microproyecto-ui), con la integración de Paciente propuesta en el [PR #7](https://github.com/katterine2558/maia-pds-microproyecto-ui/pull/7).

**Por actualizar** Se prepararon borradores del [manual de usuario](https://github.com/katterine2558/maia-pds-microproyecto-api/blob/feature/camilo-base-reporte-entrega3/docs/entregas/entrega-3-manual-usuario-borrador.md), el [manual de instalación](https://github.com/katterine2558/maia-pds-microproyecto-api/blob/feature/camilo-base-reporte-entrega3/docs/entregas/entrega-3-manual-instalacion-borrador.md) y el [guion del video](https://github.com/katterine2558/maia-pds-microproyecto-api/blob/feature/camilo-base-reporte-entrega3/docs/entregas/entrega-3-guion-video-borrador.md). Deben revisarse después de integrar y probar la versión final.


## 7. Reporte de trabajo en equipo
El trabajo se reparte por item de trabajo, no por persona: cada item vive en su propia rama feature/*, sale de develop y vuelve a develop mediante un pull request con revision de al menos un companero. Los merges conservan el historial completo, sin squash ni rebase que colapsen la autoria, de modo que el aporte de cada integrante queda verificable en el repositorio. main conserva unicamente los estados integrados de la entrega.
Entre el 17 y el 22 de septiembre se abrieron (Por completar X Numero) de pull requests entre los dos repositorios, de los cuales (Por completar X numero) se integraron.

## 7.1 Quien Hizo que
