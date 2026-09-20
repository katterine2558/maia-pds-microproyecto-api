# Predicción de reingreso hospitalario temprano en pacientes diabéticos

**Entrega 3 Micro-proyecto - Desarrollo de Soluciones  - MAIA**  
Camilo Andrés Rodríguez Dueñas · Jasbyn Rainier Solano Carrillo · Leonardo Almanza Sánchez · Gineth Katerine Arias Carrillo  
Repositorios: katterine2558 / maia-pds-microproyecto-api
## 1. Resumen del problema

### 1.1 Contexto y pregunta de negocio

El reingreso hospitalario dentro de los 30 días posteriores al alta señala una oportunidad de mejorar el seguimiento de los pacientes. Con capacidad limitada para llamadas y controles, enfermería necesita decidir a qué egresos atender primero. La pregunta de negocio es: **¿qué pacientes diabéticos presentan mayor riesgo de reingresar dentro de los 30 días siguientes al alta?** El prototipo pretende ordenar un listado de egresos mediante una estimación de riesgo y permitir la consulta de un encuentro individual. Su resultado apoya la asignación de cupos de seguimiento y no reemplaza el criterio clínico, no indica tratamientos ni estima la causa del reingreso.

### 1.2 Alcance y conjunto de datos

El usuario previsto es el personal de enfermería de gestión hospitalaria. El tablero contempla Priorización, Paciente y Contexto; la carga de egresos se realiza mediante archivo, sin conexión a una historia clínica electrónica. Se emplea *Diabetes 130-US Hospitals for Years 1999-2008* (UCI Machine Learning Repository, dataset 296). De 101.766 encuentros originales, se excluyeron 1.652 de pacientes fallecidos y 771 con egreso a hospicio: la base analítica contiene 99.343 encuentros de 69.990 pacientes, con aproximadamente 11,4% de reingresos antes de 30 días. La etiqueta positiva corresponde a `readmitted` igual a `<30`. Los datos se versionaron con DVC y el repositorio conserva el puntero.

Los registros proceden de hospitales de Estados Unidos entre 1999 y 2008. Las cifras observadas no garantizan desempeño con pacientes contemporáneos o colombianos. La variable `race` se reserva para evaluar el comportamiento entre grupos y no entra como predictora.

### 1.3 Cambios respecto de la segunda entrega

En la Entrega 2 se compararon dos familias de modelos, se eligió el bosque aleatorio V2 y se desarrollaron las vistas del tablero. La vista Paciente enviaba un formulario por HTTP a `/predict`, pero la tarjeta presentada en el informe mostraba valores ilustrativos porque la API real quedaba para la siguiente iteración. El tablero ya había sido mostrado en Railway. Esta Entrega 3 exige comprobar inferencias de un modelo empaquetado, su servicio por API, el consumo de la API desde el tablero y un despliegue conjunto mediante Docker en nube.

**Pendiente para versión final:** incluir enlaces a los commits de integración y distinguir el avance local descrito aquí de lo que efectivamente se incorpore a `develop`. La versión original de `develop` consultada no contenía la aplicación FastAPI ni el artefacto entrenado.

## 2. Modelos desarrollados y evaluación

### 2.1 Preparación, partición y características

La base de modelamiento excluye los egresos por fallecimiento y hospicio, transforma las variables administrativas y clínicas según el EDA y conserva `patient_nbr` para agrupar la partición sin introducirlo como predictor. Se trataron diagnósticos ICD-9 por grupos, edades como rangos ordinales, antecedentes por tramos y variables nominales mediante codificación adecuada al pipeline. Los hiperparámetros y el umbral se seleccionaron sin usar la prueba reservada para el ajuste. En la regresión logística se documentaron 63.670 encuentros para entrenamiento, 15.852 para validación y 19.821 para prueba, sin pacientes compartidos.

La regresión base logró 88,24% de exactitud en validación pero solo 2,00% de recall, lo que mostró que la exactitud no era suficiente con prevalencia cercana al 11,4%. Se compararon versiones con balanceo de clases, regularización, peso positivo, umbral y Elastic Net. La versión V5 conservó `C=0,5`, peso positivo 5 y umbral 0,30. Para bosque aleatorio se compararon cuatro escenarios de árbol y siete técnicas de desbalance con validación cruzada agrupada por paciente; el V2 acotó la profundidad a 12, con 400 árboles, peso positivo 5 y umbral 0,30.

### 2.2 Comparación en prueba reservada y elección

| Modelo | ROC-AUC | PR-AUC | Recall | Precisión | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| Regresión logística V5 | 0,6589 | 0,2133 | 81,56% | 14,00% | 407 |
| Bosque aleatorio V2 | 0,6673 | 0,2123 | 90,48% | 12,83% | 210 |

Ambos modelos se evaluaron sobre los mismos 19.821 encuentros, incluidos 2.207 reingresos. La capacidad de ordenamiento fue semejante: 0,0084 de ROC-AUC y 0,0010 de PR-AUC separaron a las familias. El bosque V2 se eligió porque, en el punto operativo evaluado, dejó escapar 210 reingresos frente a 407 de la regresión. La mejora en sensibilidad tiene un costo operativo: con precisión de 12,83%, muchas alertas consumirían cupos de seguimiento sin corresponder a reingresos. La lista de egresos debe permitir ajustar la capacidad diaria y mostrar quiénes quedan sin cubrir. No describir el recall como precisión ni extrapolar las métricas a otro modelo.

La comparación del 10% con mayor riesgo de la regresión V5 concentró el 22,97% de los reingresos, con lift 2,30. Este resultado pertenece a esa configuración y no debe confundirse con un indicador del V2 sin medirlo de nuevo. El límite de discriminación observado y la antigüedad del dataset impiden interpretar la herramienta como validada para uso clínico.

### 2.3 Experimentos MLflow y versión que será servida

Los experimentos de la segunda entrega se registraron en MLflow sobre una instancia EC2; el reporte previo muestra las corridas de ambas familias, el usuario de acceso y la IP pública. La versión final debe señalar **la corrida y el artefacto exactos que carga la API** junto con sus parámetros, columnas, preprocesamiento y umbral. Hay una incompatibilidad por resolver: el bosque V2 completo fue entrenado con más variables que los diez campos visibles en el formulario Paciente. Se puede ampliar la entrada al conjunto de variables del V2 o entrenar y evaluar una variante restringida al formulario, con identificación y métricas propias. Las métricas de la tabla anterior solo corresponden a las versiones evaluadas en Entrega 2.

Como avance aislado de Entrega 3 se entrenó `bosque_formulario_e3_v1` con las diez variables disponibles en Paciente (`src/models/entrenar_formulario_e3.py`). Se conservaron los 99.343 encuentros analíticos y se separó el 20 % para prueba agrupando por paciente: 79.522 encuentros para entrenamiento, 19.821 para prueba, 2.207 positivos en prueba y ningún paciente compartido. El modelo combina codificación de categorías y un bosque de 400 árboles, profundidad máxima 12, peso positivo 5 y umbral 0,30; el pipeline y su versión se guardaron juntos en `api/artifacts/modelo.joblib`.

En esta prueba, la variante restringida obtuvo ROC-AUC **0,6292**, PR-AUC **0,1913**, recall **89,90 %**, precisión **12,32 %** y **223 falsos negativos**. El bosque V2 de la sección 2.2 empleó más variables y tuvo ROC-AUC 0,6673 en la misma partición prevista para comparación; no debe tratarse a las dos versiones como el mismo modelo. Esta variante es un avance local reproducible que **aún no se registró en MLflow** ni se desplegó en nube; se requiere comprobación del equipo antes de sustituir formalmente el V2 para la entrega.

**Pendiente para versión final:** registrar el experimento en MLflow e identificar corrida, firma y ruta del artefacto; capturas requeridas de MLflow en EC2 con usuario e IP; evidencia de que la instancia queda detenida y no terminada.

## 3. Tablero y API desarrollados

### 3.1 Flujo previsto y necesidades del usuario

La vista Priorización debe recibir un archivo de egresos, verificar sus columnas, invocar la API y ordenar los encuentros por probabilidad estimada. Debe permitir ajustar la capacidad de seguimiento y separar los casos que caben en ella. La vista Paciente recoge las variables disponibles al alta para pedir una inferencia individual y mostrar probabilidad, umbral y versión de modelo. Contexto presenta la tasa histórica de reingreso según antecedentes, especialidad y edad; esas gráficas descriptivas no son predicciones individuales.

La API y el tablero se ubican en repositorios distintos. El tablero se comunica por HTTP y no carga directamente el modelo. La Entrega 2 mostró interfaz y separación de componentes, pero **no demostró todavía que la tarjeta de Paciente o el listado de Priorización presentaran predicciones reales**.

La inspección de las fuentes del tablero permite precisar qué está construido: `app.py` organiza las tres vistas; `views/paciente.py` recoge diez datos del encuentro y envía un diccionario a `services/api.py`, que implementa `POST /predict` y `GET /health` con un tiempo máximo de espera de diez segundos. La vista solo presenta un resultado si la respuesta incluye una probabilidad y un umbral numéricos válidos; ante un error informa al usuario. `views/contexto.py` ofrece visualizaciones descriptivas separadas de la predicción. En la copia consultada, `views/priorizacion.py` conserva ocho encuentros de ejemplo y cifras fijas de capacidad y cobertura; todavía no recibe un archivo ni consulta la API. Por ello, una captura de esa pantalla no serviría como evidencia de priorización calculada por el modelo.

Se preparó una API FastAPI en `api/main.py` para esta variante: `GET /health` informa la versión cargada y `POST /predict` recibe los diez campos del formulario, transforma las categorías visibles a los valores originales del conjunto de datos y devuelve probabilidad, umbral y versión. En una prueba HTTP local, `/health` respondió 200, `/predict` respondió 200 con una probabilidad calculada, y una estancia de cero días produjo 422. Esta prueba acredita el servicio **local** y el contrato del formulario; aún no acredita el consumo desde la versión publicada del tablero.

**Pendiente para versión final:** integrar y probar el tablero real contra esta API; capturas del mismo encuentro en ambos; prueba de un archivo con varios egresos; mensajes ante datos incompletos y API no disponible; eliminación de toda cifra ilustrativa presentada como inferencia.

## 4. Despliegue con Docker en nube

La arquitectura esperada es tablero Streamlit → API de inferencia → pipeline de modelo empaquetado. La segunda entrega incluyó un Dockerfile del tablero y describió su publicación en Railway. La tercera debe documentar el artefacto de despliegue de **ambos servicios**, la versión del modelo, sus URL, la configuración de comunicación entre servicios y una prueba funcional desde la interfaz pública. La existencia previa de la interfaz en Railway, por sí sola, no acredita el despliegue de la API.

El repositorio del tablero ya incluye un `Dockerfile` con Python 3.12, instalación mediante `uv.lock`, exposición del puerto 8501 y verificación de salud de Streamlit, además de `railway.json`. La variable `API_URL` determina a qué servicio HTTP se envían las solicitudes; su valor por defecto en `services/api.py` es `http://localhost:8000`, y el Dockerfile define `http://api:8000` para una red de contenedores. En un despliegue en Railway debe configurarse con la dirección realmente accesible de la API. Para la API se prepararon `api/Dockerfile` y `api/requirements.txt`; **la imagen todavía no se construyó ni se probó en Docker o en nube**.

**Pendiente para versión final:** plataforma efectiva, Dockerfiles y configuración, URL del tablero, URL de documentación de la API, prueba de salud y predicción, capturas y fecha de validación. Según la guía, detener las máquinas y servicios usados tras las pruebas sin terminarlos.

## 5. Principales resultados y conclusiones

Los resultados confirmados hasta la segunda entrega muestran que el bosque V2 identificó un mayor porcentaje de reingresos que la regresión V5 al umbral comparado, a costa de más seguimientos por cada caso correctamente detectado. La capacidad de discriminación se mantuvo moderada y los datos históricos limitan cualquier extrapolación. El valor del prototipo dependerá de servir el mismo pipeline evaluado, mostrar probabilidades realmente calculadas y permitir que enfermería interprete el ordenamiento dentro de su capacidad disponible.

La nueva variante limitada a los campos del formulario produjo inferencias reales por API local y presentó menor ROC-AUC y PR-AUC que el V2 original, aunque conservó una sensibilidad alta al umbral elegido. La comparación exige cuidado: se ha cambiado el conjunto de variables y queda pendiente el registro del experimento. **Pendiente para versión final:** resultados de pruebas con tablero integrado, comportamiento del despliegue, versión verdaderamente servida, incidencias y limitaciones.

## 6. Repositorios y soportes

- [Modelos y API](https://github.com/katterine2558/maia-pds-microproyecto-api).
- [Tablero](https://github.com/katterine2558/maia-pds-microproyecto-ui).
- Fuentes de referencia interna: `docs/entregas/Entrega-2-reporte.pdf`; guía `maia_pds_proy_e3.pdf` aportada al equipo; dataset UCI *Diabetes 130-US Hospitals for Years 1999-2008*.
- Borradores de soportes preparados a partir de las fuentes revisadas: `entrega-3-manual-usuario-borrador.md`, `entrega-3-manual-instalacion-borrador.md` y `entrega-3-guion-video-borrador.md`, en esta misma carpeta. Deben actualizarse tras las pruebas del producto integrado.

**Pendiente para versión final:** enlaces a commits y ramas integradas, manual de usuario, manual de instalación, reporte de trabajo en equipo de máximo una página, capturas MLflow, video de máximo diez minutos y registro de retroalimentación a cuatro grupos. Verificar que las primeras diez páginas del PDF contengan todo el texto evaluable.

## 7. Reporte de trabajo en equipo
El trabajo se reparte por item de trabajo, no por persona: cada item vive en su propia rama feature/*, sale de develop y vuelve a develop mediante un pull request con revision de al menos un companero. Los merges conservan el historial completo, sin squash ni rebase que colapsen la autoria, de modo que el aporte de cada integrante queda verificable en el repositorio. main conserva unicamente los estados integrados de la entrega.
Entre el 17 y el 22 de septiembre se abrieron (Por completar X Numero) de pull requests entre los dos repositorios, de los cuales (Por completar X numero) se integraron.

## 7.1 Quien Hizo que
