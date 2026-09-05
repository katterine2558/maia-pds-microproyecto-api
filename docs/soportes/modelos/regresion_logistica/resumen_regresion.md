# Regresión logística - Entrega 2

## Población y partición

La población analítica contiene 99.343 encuentros correspondientes a 69.990 pacientes. La variable objetivo identifica el reingreso hospitalario dentro de los 30 días posteriores al alta.

La partición se realizó por `patient_nbr`, evitando que encuentros del mismo paciente quedaran simultáneamente en entrenamiento, validación y prueba.

- Entrenamiento: 63.670 encuentros, 44.793 pacientes, 11,39% positivos.
- Validación: 15.852 encuentros, 11.199 pacientes, 11,69% positivos.
- Prueba: 19.821 encuentros, 13.998 pacientes, 11,13% positivos.
- Pacientes compartidos entre particiones: 0.

## Selección y preparación de características

Se definieron 32 predictores antes de la codificación. Se excluyeron identificadores, la variable objetivo original, `weight`, `payer_code`, variables reservadas para análisis de sesgo y medicamentos de muy baja frecuencia. Los diagnósticos `diag_1`, `diag_2` y `diag_3` se agruparon en categorías clínicas para reducir su alta cardinalidad.

Las variables numéricas se estandarizaron. Los valores `?` se conservaron como categoría del dato original y las variables categóricas se codificaron mediante one-hot encoding.

## Estrategia de validación

Las primeras configuraciones se compararon utilizando el conjunto de validación. Los hiperparámetros, pesos de clase y umbral de decisión se seleccionaron sin utilizar el conjunto de prueba. Una vez fijada la configuración final de sensibilidad, el conjunto de prueba se utilizó para reportar su desempeño final.

## Versiones iniciales en validación

### V1 - Regresión logística base

- ROC-AUC: 0,6637
- PR-AUC: 0,2207
- Precision: 0,4353
- Recall: 0,0200
- F1: 0,0382
- Accuracy: 0,8824

La exactitud elevada se explica principalmente por el desbalance de clases. La versión base prácticamente no identifica reingresos.

### V2 - Regresión logística balanceada

- ROC-AUC: 0,6655
- PR-AUC: 0,2208
- Precision: 0,1875
- Recall: 0,5461
- F1: 0,2792
- Accuracy: 0,6703

El balance de clases incrementó de forma importante la identificación de reingresos.

### V3 - Ajuste de regularización

Se evaluaron valores de `C` iguales a 0,1; 0,5; 1,0; 2,0 y 10,0 sobre validación. El mejor resultado según PR-AUC correspondió a `C=0,5`.

Resultados de V3 en validación:

- ROC-AUC: 0,6658
- PR-AUC: 0,2208
- Precision: 0,1869
- Recall: 0,5440
- F1: 0,2783
- Accuracy: 0,6701

La mejora frente a V2 fue marginal.

## Ajuste orientado a sensibilidad

Posteriormente se evaluaron pesos adicionales para la clase positiva y distintos umbrales de decisión. El mejor peso según F2 de validación fue 5 y el mejor umbral fue 0,30.

### V5 - Configuración final orientada a sensibilidad

Resultados confirmados en prueba:

- `C`: 0,5
- Peso positivo: 5
- Umbral: 0,30
- ROC-AUC: 0,6589
- PR-AUC: 0,2133
- Precision: 0,1400
- Recall: 0,8156
- F2: 0,4151
- Accuracy: 0,4218
- Verdaderos positivos: 1.800
- Falsos negativos: 407
- Falsos positivos: 11.053
- Verdaderos negativos: 6.561

La configuración final aumenta de forma importante la sensibilidad, a costa de un mayor número de falsos positivos.

## V6 - Elastic Net y variables derivadas

Se evaluó una versión adicional con regularización Elastic Net y variables derivadas de utilización previa. Esta alternativa no superó de forma relevante la capacidad discriminativa observada en las versiones anteriores y no fue seleccionada como configuración final.

## Evaluación como herramienta de priorización

La V5 también se evaluó como mecanismo de priorización por probabilidad estimada.

- Priorizar el 10% superior captura el 22,97% de los reingresos, con precision de 25,58% y lift de 2,30x.
- Priorizar el 20% captura el 36,93%.
- Priorizar el 30% captura el 49,16%.
- Priorizar el 40% captura el 59,67%.
- Priorizar el 50% captura el 69,42%.

## Conclusión

El principal reto de la regresión fue el desbalance de la variable objetivo. La versión base alcanzó una exactitud elevada, pero prácticamente no identificó reingresos. El tratamiento del desbalance y el ajuste del punto de decisión permitieron elevar el recall hasta 81,56% en prueba.

Este resultado implica un costo en falsos positivos, por lo que la configuración seleccionada se interpreta como una alternativa orientada a reducir falsos negativos. Para el uso operativo se recomienda utilizar la probabilidad estimada como puntaje de riesgo y definir la cantidad de pacientes priorizados de acuerdo con la capacidad disponible.