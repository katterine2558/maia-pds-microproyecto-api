# Predicción de reingreso hospitalario temprano en pacientes diabéticos

**Entrega 3 — informe preliminar para integración del equipo**  
Camilo Andrés Rodríguez Dueñas · Jasbyn Rainier Solano Carrillo · Leonardo Almanza Sánchez · Gineth Katerine Arias Carrillo  
Actualizado el 20 de septiembre de 2026 a partir del informe de Entrega 2, la guía de Entrega Final y las pruebas locales de integración. **No es el PDF final ni acredita un despliegue conjunto en nube.**

## 1. Resumen del problema

### 1.1 Contexto y pregunta de negocio

El reingreso hospitalario dentro de los 30 días posteriores al alta señala una oportunidad de mejorar el seguimiento de los pacientes. Con capacidad limitada para llamadas y controles, enfermería necesita decidir a qué egresos atender primero. La pregunta de negocio es: **¿qué pacientes diabéticos presentan mayor riesgo de reingresar dentro de los 30 días siguientes al alta?** El prototipo pretende ordenar un listado de egresos mediante una estimación de riesgo y permitir la consulta de un encuentro individual. Su resultado apoya la asignación de cupos de seguimiento; no reemplaza el criterio clínico, no indica tratamientos ni estima la causa del reingreso.

### 1.2 Alcance y conjunto de datos

El usuario previsto es el personal de enfermería de gestión hospitalaria. El tablero contempla Priorización, Paciente y Contexto; la carga de egresos se plantea mediante archivo, sin conexión a una historia clínica electrónica. Se emplea *Diabetes 130-US Hospitals for Years 1999-2008* (UCI Machine Learning Repository, dataset 296). De 101.766 encuentros originales, se excluyeron 1.652 de pacientes fallecidos y 771 con egreso a hospicio: la base analítica contiene 99.343 encuentros de 69.990 pacientes, con aproximadamente 11,4% de reingresos antes de 30 días. La etiqueta positiva corresponde a `readmitted` igual a `<30`. Los datos se versionaron con DVC y el repositorio conserva el puntero.

Los registros proceden de hospitales de Estados Unidos entre 1999 y 2008. Las cifras observadas no garantizan desempeño con pacientes contemporáneos o colombianos. La variable `race` se reserva para evaluar el comportamiento entre grupos y no entra como predictora.

### 1.3 Cambios respecto de la segunda entrega

En la Entrega 2 se compararon dos familias de modelos, se eligió el bosque aleatorio V2 y se desarrollaron las vistas del tablero. La vista Paciente enviaba un formulario por HTTP a `/predict`, pero la tarjeta presentada en el informe mostraba valores ilustrativos porque la API real quedaba para la siguiente iteración. El tablero ya había sido mostrado en Railway.

Para la Entrega 3 se preparó una variante del modelo compatible con los diez campos del formulario, se desarrolló una API de inferencia y se conectó la vista Paciente con su respuesta. El flujo completo se comprobó **en el equipo local**. El despliegue conjunto mediante Docker en nube y la integración de la lista de Priorización siguen pendientes.

## 2. Modelos desarrollados y evaluación

### 2.1 Preparación, partición y características

La base de modelamiento excluye los egresos por fallecimiento y hospicio, transforma las variables administrativas y clínicas según el EDA y conserva `patient_nbr` para agrupar la partición sin introducirlo como predictor. Se trataron diagnósticos ICD-9 por grupos, edades como rangos ordinales, antecedentes por tramos y variables nominales mediante codificación adecuada al pipeline. Los hiperparámetros y el umbral se seleccionaron sin usar la prueba reservada para el ajuste. En la regresión logística se documentaron 63.670 encuentros para entrenamiento, 15.852 para validación y 19.821 para prueba, sin pacientes compartidos.

La regresión base logró 88,24% de exactitud en validación pero solo 2,00% de recall, lo que mostró que la exactitud no era suficiente con prevalencia cercana al 11,4%. Se compararon versiones con balanceo de clases, regularización, peso positivo, umbral y Elastic Net. La versión V5 conservó `C=0,5`, peso positivo 5 y umbral 0,30. Para bosque aleatorio se compararon cuatro escenarios de árbol y siete técnicas de desbalance con validación cruzada agrupada por paciente; el V2 acotó la profundidad a 12, con 400 árboles, peso positivo 5 y umbral 0,30.

### 2.2 Comparación en prueba reservada y elección

| Modelo | ROC-AUC | PR-AUC | Recall | Precisión | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| Regresión logística V5 | 0,6589 | 0,2133 | 81,56% | 14,00% | 407 |
| Bosque aleatorio V2 | 0,6673 | 0,2123 | 90,48% | 12,83% | 210 |

Ambos modelos se evaluaron sobre los mismos 19.821 encuentros, incluidos 2.207 reingresos. La capacidad de ordenamiento fue semejante: 0,0084 de ROC-AUC y 0,0010 de PR-AUC separaron a las familias. El bosque V2 se eligió porque, en el punto operativo evaluado, dejó escapar 210 reingresos frente a 407 de la regresión. La mejora en sensibilidad tiene un costo operativo: con precisión de 12,83 %, muchas alertas consumirían cupos de seguimiento sin corresponder a reingresos. La lista de egresos debe permitir ajustar la capacidad diaria y mostrar quiénes quedan sin cubrir. El recall no debe describirse como precisión ni deben extrapolarse estas métricas a otro modelo.

La comparación del 10% con mayor riesgo de la regresión V5 concentró el 22,97% de los reingresos, con lift 2,30. Este resultado pertenece a esa configuración y no debe confundirse con un indicador del V2 sin medirlo de nuevo. El límite de discriminación observado y la antigüedad del conjunto de datos impiden interpretar la herramienta como validada para uso clínico.

### 2.3 Variante compatible con el formulario y MLflow

Los experimentos de la segunda entrega se registraron en MLflow sobre una instancia EC2. Hay una incompatibilidad que debe reconocerse: el bosque V2 completo fue entrenado con más variables que los diez campos visibles en el formulario Paciente. Las métricas de la sección 2.2 corresponden exclusivamente a los modelos evaluados en Entrega 2.

Como avance de Entrega 3 se entrenó `bosque_formulario_e3_v1` con las diez variables disponibles en Paciente (`src/models/entrenar_formulario_e3.py`). Se conservaron los 99.343 encuentros analíticos y se separó el 20 % para prueba agrupando por paciente: 79.522 encuentros para entrenamiento, 19.821 para prueba, 2.207 positivos en prueba y ningún paciente compartido. El modelo combina codificación de categorías y un bosque de 400 árboles, profundidad máxima 12, peso positivo 5 y umbral 0,30; el pipeline y su versión se guardaron juntos en `api/artifacts/modelo.joblib`.

En esta prueba, la variante restringida obtuvo ROC-AUC **0,6292**, PR-AUC **0,1913**, recall **89,90%**, precisión **12,32%** y **223 falsos negativos**. El bosque V2 empleó más variables y tuvo ROC-AUC 0,6673 en la comparación documentada: las dos versiones no deben presentarse como el mismo modelo. La variante del formulario **aún no se registró en MLflow** ni se desplegó en nube. El equipo debe decidir qué versión adoptará formalmente y documentar la corrida, firma, parámetros, artefacto y versión efectivamente servida.

**Pendiente para la versión final:** registrar la variante de Entrega 3 en MLflow; incorporar las evidencias requeridas de MLflow en EC2, con usuario e IP, y demostrar que la instancia queda detenida y no terminada.

## 3. Tablero y API desarrollados

### 3.1 Flujo y estado de las vistas

La vista Priorización debe recibir un archivo de egresos, verificar sus columnas, invocar la API y ordenar los encuentros por probabilidad estimada. Debe permitir ajustar la capacidad de seguimiento y separar los casos que caben en ella. **En el estado probado, esta vista sigue mostrando encuentros y cifras de ejemplo:** su pantalla indica expresamente que es demostrativa. No debe presentarse como una priorización calculada por el modelo.

La vista Paciente recoge diez variables disponibles al alta, solicita una inferencia individual y muestra la probabilidad, el nivel de riesgo, el umbral y la versión del modelo. Contexto presenta gráficas descriptivas separadas de las predicciones individuales.

La API y el tablero se encuentran en repositorios distintos. El tablero se comunica por HTTP y no carga directamente el artefacto del modelo. `services/api.py` realiza las llamadas a `POST /predict` y `GET /health`; el servicio FastAPI en `api/main.py` recibe los campos del formulario, transforma las categorías visibles a las del conjunto de datos y devuelve probabilidad, umbral y versión.

### 3.2 Pruebas locales de la API y del tablero

En la prueba HTTP local de la API, `/health` respondió con `{"estado":"ok","modelo":"bosque_formulario_e3_v1"}`. También se comprobó una respuesta de predicción y el rechazo de una estancia de cero días con estado 422.

El **20 de septiembre de 2026** se ejecutó la API localmente en `127.0.0.1:8000` y el tablero Streamlit en `localhost:8501`. En la vista Paciente se utilizaron los datos visibles del formulario: edad `[70-80)`, admisión `Emergency`, servicio `Nephrology`, nueve días de estancia, nueve diagnósticos, 21 medicamentos, cinco ingresos previos, dos visitas previas a urgencias, A1C no medido y cambio de medicación `Sí`. Al pulsar **Calcular riesgo**, la pantalla presentó una **probabilidad de 0,61**, nivel **riesgo alto**, **umbral de decisión 0,30** y versión **`bosque_formulario_e3_v1`**. Se verificó así el consumo de la API desde la pantalla Paciente **en entorno local**.

Se corrigió además la tarjeta del resultado para retirar una cifra fija de «cohorte comparable» y ocultar el apartado de factores cuando la API no aporta factores explicativos. La pantalla corregida volvió a ejecutarse localmente y mostró el resultado sin texto HTML visible. Esta prueba no acredita la integración de Priorización ni el funcionamiento de una interfaz pública en nube.

**Pendiente para la versión final:** probar mensajes ante API no disponible y entradas incompletas; implementar y probar el archivo con varios egresos en Priorización; documentar una prueba funcional de ambos servicios desplegados. Incorporar al reporte final las evidencias de interfaz y respuestas HTTP pertinentes.

## 4. Despliegue con Docker en nube

La arquitectura prevista es tablero Streamlit → API de inferencia → pipeline de modelo empaquetado. La segunda entrega incluyó un Dockerfile del tablero y describió su publicación en Railway. Para la tercera entrega deben documentarse **ambos servicios**, la versión del modelo, sus URL, la configuración de comunicación entre ellos y una prueba funcional desde la interfaz pública. La publicación anterior del tablero, por sí sola, no acredita el despliegue de la nueva API.

El repositorio del tablero incluye un `Dockerfile`, además de `railway.json`. La variable `API_URL` determina a qué servicio HTTP se envían las solicitudes; su valor por defecto en `services/api.py` es `http://localhost:8000`. Para la API se prepararon `api/Dockerfile` y `api/requirements.txt`. **Todavía no se ha comprobado la construcción y ejecución de ambos servicios con Docker ni su despliegue conjunto en nube.**

**Pendiente para la versión final:** plataforma efectiva, configuración y comprobación de los contenedores, URL del tablero, URL de documentación de la API, pruebas públicas de salud y predicción, capturas y fecha de validación. Seguir las indicaciones de la guía sobre detener los recursos utilizados al concluir las pruebas.

## 5. Principales resultados y conclusiones

En la segunda entrega, el bosque V2 identificó un mayor porcentaje de reingresos que la regresión V5 al umbral comparado, a costa de más seguimientos por cada caso correctamente detectado. La discriminación se mantuvo moderada y los datos históricos limitan cualquier extrapolación.

En esta tercera entrega se preparó una variante limitada a los campos del formulario. Sus métricas de prueba son inferiores en ROC-AUC y PR-AUC a las documentadas para el V2, aunque conserva una sensibilidad alta al umbral evaluado. Se comprobó **localmente** que el formulario Paciente consulta la API y muestra una probabilidad calculada con la versión servida. El equipo debe revisar si acepta esta variante o amplía la entrada para servir el modelo V2, registrar el experimento y completar la priorización por archivo y el despliegue conjunto.

## 6. Repositorios y soportes

- [Repositorio de modelos y API](https://github.com/katterine2558/maia-pds-microproyecto-api).
- [Repositorio del tablero](https://github.com/katterine2558/maia-pds-microproyecto-ui).
- [Borrador de integración del modelo, API y documentación: PR #19](https://github.com/katterine2558/maia-pds-microproyecto-api/pull/19).
- [Borrador de integración de la vista Paciente: PR #7](https://github.com/katterine2558/maia-pds-microproyecto-ui/pull/7).
- Fuentes de referencia interna: `docs/entregas/Entrega-2-reporte.pdf`; guía `maia_pds_proy_e3.pdf` aportada al equipo; dataset UCI *Diabetes 130-US Hospitals for Years 1999-2008*.
- Borradores de soportes preparados en esta carpeta: `entrega-3-manual-usuario-borrador.md`, `entrega-3-manual-instalacion-borrador.md` y `entrega-3-guion-video-borrador.md`. Requieren actualización al terminar el producto.

**Pendiente para la versión final:** integrar o revisar las ramas según decida el equipo; manuales definitivos, reporte de trabajo en equipo de máximo una página, evidencias de MLflow, video de máximo diez minutos y retroalimentación a cuatro grupos. Verificar que las primeras diez páginas del PDF contengan todo el texto evaluable.

## 7. Reporte de trabajo en equipo
El trabajo se reparte por item de trabajo, no por persona: cada item vive en su propia rama feature/*, sale de develop y vuelve a develop mediante un pull request con revision de al menos un companero. Los merges conservan el historial completo, sin squash ni rebase que colapsen la autoria, de modo que el aporte de cada integrante queda verificable en el repositorio. main conserva unicamente los estados integrados de la entrega.
Entre el 17 y el 22 de septiembre se abrieron (Por completar X Numero) de pull requests entre los dos repositorios, de los cuales (Por completar X numero) se integraron.

## 7.1 Quien Hizo que
