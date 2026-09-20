# Predicción de reingreso hospitalario temprano en pacientes diabéticos

**Informe de trabajo — Entrega 3**  
Camilo Andrés Rodríguez Dueñas · Jasbyn Rainier Solano Carrillo · Leonardo Almanza Sánchez · Gineth Katerine Arias Carrillo  
20 de septiembre de 2026

## 1. Problema y alcance

Los equipos de seguimiento hospitalario disponen de tiempo limitado para contactar a los pacientes después del alta. Este proyecto estudia si la información disponible al egreso permite ordenar los casos de pacientes diabéticos según su riesgo estimado de reingreso durante los siguientes 30 días.

La aplicación contempla una evaluación individual en la vista **Paciente** y una lista de egresos para organizar el seguimiento en **Priorización**. La estimación busca apoyar la distribución de la capacidad de atención; no determina tratamientos ni sustituye la valoración clínica.

Se utilizó el conjunto *Diabetes 130-US Hospitals for Years 1999-2008*, publicado por el UCI Machine Learning Repository. De los 101.766 encuentros originales se excluyeron 1.652 registros de pacientes fallecidos y 771 correspondientes a egresos a hospicio. La base analítica quedó conformada por **99.343 encuentros de 69.990 pacientes**. Cerca del **11,4 %** de los encuentros registraron un reingreso antes de 30 días, identificado en la variable `readmitted` con el valor `<30`.

Los datos proceden de hospitales de Estados Unidos entre 1999 y 2008. Por su origen y antigüedad, el desempeño medido en este conjunto no puede trasladarse directamente a hospitales colombianos ni a pacientes actuales.

## 2. Modelamiento y resultados

### 2.1 Resultados de la Entrega 2

La preparación de los datos incluyó el tratamiento de variables administrativas y clínicas, la agrupación de diagnósticos ICD-9 y la transformación de variables categóricas. Se conservó `patient_nbr` para separar los conjuntos sin compartir pacientes entre ellos, pero no se utilizó como predictor. La variable `race` se reservó para revisar el comportamiento del modelo entre grupos y tampoco se incorporó como entrada.

Se compararon una regresión logística y un bosque aleatorio. La regresión logística inicial alcanzó **88,24 % de exactitud** en validación, pero detectó solo el **2,00 % de los reingresos**. Dada la baja proporción de casos positivos, la selección posterior prestó especial atención a la capacidad de identificar a los pacientes que sí reingresaron.

Las configuraciones elegidas fueron la regresión logística V5 y el bosque aleatorio V2. Ambas se evaluaron sobre **19.821 encuentros de prueba**, entre ellos **2.207 reingresos**.

| Modelo | ROC-AUC | PR-AUC | Recall | Precisión | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| Regresión logística V5 | 0,6589 | 0,2133 | 81,56 % | 14,00 % | 407 |
| Bosque aleatorio V2 | 0,6673 | 0,2123 | 90,48 % | 12,83 % | 210 |

Se eligió el bosque V2 porque detectó una mayor proporción de reingresos en el umbral evaluado: dejó sin identificar **210 casos**, frente a **407** de la regresión. Su precisión de **12,83 %** muestra el costo operativo de esa elección: muchos pacientes señalados para seguimiento no registraron un reingreso en el conjunto de prueba. Por ello, una lista operativa debe considerar cuántos contactos puede asumir el equipo de enfermería y cuáles casos quedarían fuera de esa capacidad.

La regresión V5 concentró el **22,97 % de los reingresos** en el 10 % de encuentros con mayor riesgo estimado, con un *lift* de **2,30**. Esta medición corresponde a la regresión. Si el equipo quiere comparar la utilidad de ambos modelos para construir una lista de priorización, debe calcular el mismo indicador para el bosque.

### 2.2 Modelo preparado para el formulario de la Entrega 3

Durante la integración del tablero se identificó una diferencia entre el modelo seleccionado y la información solicitada en **Paciente**: el bosque V2 se entrenó con más variables que las diez disponibles en el formulario. Para probar el flujo completo se entrenó `bosque_formulario_e3_v1`, una nueva versión limitada a esas diez entradas.

El entrenamiento utilizó los **99.343 encuentros** de la base analítica. La separación agrupada por paciente dejó **79.522 encuentros para entrenamiento** y **19.821 para prueba**, con **2.207 casos positivos** en este último conjunto. El pipeline reúne la codificación de categorías y un bosque de **400 árboles**, profundidad máxima de **12**, peso positivo de **5** y umbral de decisión de **0,30**. El artefacto se guardó en `api/artifacts/modelo.joblib`.

| Modelo utilizado en la prueba local | ROC-AUC | PR-AUC | Recall | Precisión | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| `bosque_formulario_e3_v1` | 0,6292 | 0,1913 | 89,90 % | 12,32 % | 223 |

Estas métricas pertenecen al modelo del formulario y **no al bosque V2 de la Entrega 2**. La nueva versión conserva un recall alto al umbral probado, pero obtuvo valores menores de ROC-AUC y PR-AUC. Antes de elegir el modelo definitivo, el equipo debe decidir si mantiene el formulario de diez campos o si amplía las entradas del tablero para servir el V2. De esa decisión dependerán el artefacto que se despliegue y los resultados que se presenten como definitivos.

Los experimentos anteriores se documentaron en MLflow sobre EC2. **La variante `bosque_formulario_e3_v1` aún requiere registro en MLflow**, junto con sus parámetros, métricas, firma y artefacto.

## 3. Aplicación e integración

La solución se encuentra en dos repositorios. El tablero Streamlit presenta las vistas y envía solicitudes HTTP; la API FastAPI carga el pipeline y devuelve las predicciones.

La vista **Paciente** reúne diez datos del encuentro y consulta `POST /predict`. Presenta la probabilidad estimada, la categoría de riesgo, el umbral y la versión del modelo. La vista **Contexto** contiene información descriptiva del conjunto de datos.

La vista **Priorización** todavía utiliza encuentros y cifras de demostración. Para conectarla con el modelo falta recibir un archivo de egresos, validar sus columnas, calcular las probabilidades y ordenar los resultados según la capacidad de seguimiento disponible. Este desarrollo es necesario para completar el uso operativo planteado en el proyecto.

### Prueba de integración del 20 de septiembre

La API respondió localmente a `GET /health` con `{"estado":"ok","modelo":"bosque_formulario_e3_v1"}`. También se probó una solicitud de predicción y se verificó que una estancia de cero días produce una respuesta **422**.

Después se ejecutaron la API en `127.0.0.1:8000` y el tablero en `localhost:8501`. En **Paciente** se ingresó un encuentro con edad `[70-80)`, admisión `Emergency`, servicio `Nephrology`, nueve días de estancia, nueve diagnósticos, 21 medicamentos, cinco ingresos previos, dos visitas previas a urgencias, A1C no medido y cambio de medicación `Sí`.

La pantalla mostró una **probabilidad de 0,61**, categoría **riesgo alto**, umbral **0,30** y versión **`bosque_formulario_e3_v1`**. La prueba confirmó que la vista Paciente recibió y presentó una respuesta de la API ejecutada en el mismo computador.

Durante la revisión se retiró de la tarjeta una cifra fija de «cohorte comparable» y se ajustó la presentación de los factores explicativos, que la API actual no entrega. La pantalla se ejecutó nuevamente y mostró el resultado sin código HTML visible.

## 4. Estado del despliegue

El repositorio del tablero contiene un `Dockerfile` y `railway.json`. El de la API contiene `api/Dockerfile` y `api/requirements.txt`. El tablero consulta la dirección configurada en `API_URL`; su valor local por defecto es `http://localhost:8000`.

La integración descrita en este informe se probó con ambos procesos en un computador. Para completar el despliegue de la Entrega 3 falta construir y ejecutar los contenedores, publicar los dos servicios, configurar la dirección de la API en el tablero y comprobar una predicción desde la interfaz pública. El despliegue anterior del tablero en Railway corresponde a la Entrega 2 y no demuestra que la nueva API ya esté publicada.

## 5. Acuerdos necesarios para terminar la entrega

| Asunto | Decisión o comprobación | Resultado esperado |
|---|---|---|
| Modelo definitivo | Adoptar `bosque_formulario_e3_v1` o ampliar el formulario para utilizar V2. | Identificar una sola versión del modelo para el producto y el informe final. |
| Registro experimental | Registrar en MLflow la versión elegida. | Relacionar parámetros, métricas y artefacto con el servicio desplegado. |
| Priorización | Implementar carga, validación y evaluación de varios egresos; probar el ordenamiento y la capacidad de seguimiento. | Completar el flujo de trabajo previsto para enfermería. |
| Despliegue | Probar los contenedores, publicar tablero y API y verificar su comunicación. | Documentar una prueba desde la interfaz pública. |
| Documentación | Actualizar reporte, manuales y guion con base en las funciones terminadas. | Entregar instrucciones y evidencias consistentes con el producto. |

## 6. Conclusiones

La comparación de la Entrega 2 favoreció al bosque aleatorio V2 por su mayor detección de reingresos al umbral estudiado. La baja precisión observada exige, sin embargo, considerar la cantidad de contactos que producirían sus alertas.

Para la Entrega 3 se preparó una versión ajustada a los diez campos del formulario. El **20 de septiembre** se verificó localmente que la vista Paciente consulta la API y presenta la predicción calculada. La decisión inmediata del equipo es definir **qué modelo quedará como versión final**. A partir de ella se podrán cerrar el registro en MLflow, la integración de Priorización, el despliegue y las métricas del informe definitivo.

## 7. Repositorios y documentos de apoyo

- [Repositorio de modelos y API](https://github.com/katterine2558/maia-pds-microproyecto-api) y [PR #19 de integración](https://github.com/katterine2558/maia-pds-microproyecto-api/pull/19).
- [Repositorio del tablero](https://github.com/katterine2558/maia-pds-microproyecto-ui) y [PR #7 de la vista Paciente](https://github.com/katterine2558/maia-pds-microproyecto-ui/pull/7).
- Fuentes internas para revisar resultados y requisitos: `docs/entregas/Entrega-2-reporte.pdf` y `maia_pds_proy_e3.pdf`.
- Borradores que deben ajustarse al cerrar el producto: `entrega-3-manual-usuario-borrador.md`, `entrega-3-manual-instalacion-borrador.md` y `entrega-3-guion-video-borrador.md`.

## 8. Reporte de trabajo en equipo
El trabajo se reparte por item de trabajo, no por persona: cada item vive en su propia rama feature/*, sale de develop y vuelve a develop mediante un pull request con revision de al menos un companero. Los merges conservan el historial completo, sin squash ni rebase que colapsen la autoria, de modo que el aporte de cada integrante queda verificable en el repositorio. main conserva unicamente los estados integrados de la entrega.
Entre el 17 y el 22 de septiembre se abrieron (Por completar X Numero) de pull requests entre los dos repositorios, de los cuales (Por completar X numero) se integraron.

## 8.1 Quien Hizo que
