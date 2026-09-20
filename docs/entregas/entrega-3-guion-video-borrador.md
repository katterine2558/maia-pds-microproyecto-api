# Guion del video de Entrega 3 — borrador de hasta diez minutos

La grabación debe reflejar la versión efectivamente entregada. Los tiempos son una propuesta para una duración de aproximadamente ocho minutos; sustituir cada indicación entre corchetes con evidencia comprobada.

| Tiempo | Contenido a mostrar y explicar |
|---|---|
| 0:00–1:00 | Pregunta: a qué egresos diabéticos dar seguimiento con capacidad limitada. Dataset de 101.766 encuentros; cohorte analítica de 99.343. Aclarar alcance de prototipo orientativo. |
| 1:00–2:10 | Preparación de datos y partición agrupada por paciente. Explicar la clase minoritaria y por qué se evalúan recall, precisión, ROC-AUC y PR-AUC. |
| 2:10–3:30 | Comparar regresión logística V5 y bosque V2 con resultados de prueba: recall 81,56 % y 90,48 %; falsos negativos 407 y 210, respectivamente. Mostrar las corridas reales de MLflow y precisar cuál modelo se sirve [pendiente de confirmar]. |
| 3:30–5:40 | Mostrar `POST /predict` y la vista Paciente con el **mismo encuentro**, misma probabilidad y versión. Demostrar Priorización con egresos reales de prueba y el ajuste de capacidad **solo cuando esté implementado**. No filmar los encuentros ilustrativos como predicciones. |
| 5:40–6:40 | Recorrer Contexto y distinguir estadísticas descriptivas de predicciones individuales. |
| 6:40–7:40 | Mostrar URL del tablero y de la API, estado de ambos contenedores y respuesta de salud. Mostrar una interacción desde la interfaz publicada [pendiente de despliegue y prueba]. |
| 7:40–8:20 | Conclusiones: el V2 redujo falsos negativos en la prueba, con mayor costo de falsas alertas; datos históricos estadounidenses y desempeño moderado limitan la generalización. Mencionar cualquier cambio del modelo realmente desplegado. |

Antes de grabar: verificar URL, artefacto y esquema de variables, resultados reales del tablero, capturas de MLflow y duración final menor o igual a diez minutos. Si alguna parte no funciona, describirla honestamente como limitación y no como demostración lograda.
