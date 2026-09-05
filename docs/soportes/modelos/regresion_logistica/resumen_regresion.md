\# Regresión logística - Entrega 2



\## Población y partición



La población analítica contiene 99.343 encuentros correspondientes a 69.990 pacientes. La variable objetivo identifica el reingreso hospitalario dentro de los 30 días posteriores al alta.



La partición se realizó por `patient\_nbr`, evitando que encuentros del mismo paciente quedaran simultáneamente en entrenamiento, validación y prueba.



\- Entrenamiento: 63.670 encuentros, 44.793 pacientes, 11,39 % positivos.

\- Validación: 15.852 encuentros, 11.199 pacientes, 11,69 % positivos.

\- Prueba: 19.821 encuentros, 13.998 pacientes, 11,13 % positivos.

\- Pacientes compartidos entre particiones: 0.



\## Versiones evaluadas



\### V1 - Regresión logística base



\- `class\_weight`: None

\- `C`: 1,0

\- ROC-AUC: 0,6591

\- PR-AUC: 0,2140

\- Precision: 0,4884

\- Recall: 0,0190

\- F1: 0,0366

\- Accuracy: 0,8886



La elevada exactitud se explica principalmente por el desbalance de clases. La versión base identificó solo 42 de los 2.207 reingresos presentes en prueba.



\### V2 - Regresión logística balanceada



\- `class\_weight`: balanced

\- `C`: 1,0

\- ROC-AUC: 0,6594

\- PR-AUC: 0,2142

\- Precision: 0,1787

\- Recall: 0,5360

\- F1: 0,2680

\- Accuracy: 0,6740



El balance de clases aumentó de forma importante la identificación de reingresos, al pasar de 1,9 % a 53,6 % de recall.



\### V3 - Regresión logística balanceada con ajuste de regularización



Se evaluaron los valores de `C`: 0,1; 0,5; 1,0; 2,0 y 10,0 utilizando el conjunto de validación.



El mejor resultado según PR-AUC de validación correspondió a `C = 0,5`.



Resultados finales en prueba:



\- `class\_weight`: balanced

\- `C`: 0,5

\- ROC-AUC: 0,6595

\- PR-AUC: 0,2141

\- Precision: 0,1789

\- Recall: 0,5365

\- F1: 0,2683

\- Accuracy: 0,6741



El ajuste de regularización produjo cambios pequeños frente a V2. La principal mejora observada frente a la versión base provino del balance de clases.



\## Evaluación como herramienta de priorización



El modelo V3 se evaluó también ordenando los encuentros por probabilidad estimada:



\- Priorizar el 10 % superior captura el 22,84 % de los reingresos, con un lift de 2,28 veces la tasa base.

\- Priorizar el 20 % captura el 37,29 % de los reingresos.

\- Priorizar el 30 % captura el 49,21 % de los reingresos.

\- Priorizar el 40 % captura el 59,67 % de los reingresos.

\- Priorizar el 50 % captura el 69,78 % de los reingresos.



Este análisis complementa las métricas tradicionales porque el propósito del prototipo es ordenar los egresos por riesgo para apoyar la priorización del seguimiento.

## Ajustes orientados a sensibilidad

Después de las primeras tres versiones se evaluaron configuraciones adicionales para mejorar la detección de reingresos.

### V4 - Ajuste del peso de la clase positiva

Se evaluaron pesos de 2, 3, 4 y 5 para la clase positiva. El mejor resultado de validación según F2 se obtuvo con peso positivo igual a 5.

### V5 - Peso de clase y ajuste del umbral

Sobre la configuración anterior se evaluaron umbrales entre 0,20 y 0,50. El mejor F2 de validación se obtuvo con umbral 0,30.

Resultados en prueba:

- Peso positivo: 5
- C: 0,5
- Umbral: 0,30
- ROC-AUC: 0,6594
- PR-AUC: 0,2143
- Precision: 0,1387
- Recall: 0,8201
- F2: 0,4137
- Accuracy: 0,4130
- Verdaderos positivos: 1.810
- Falsos negativos: 397
- Falsos positivos: 11.237
- Verdaderos negativos: 6.377

El aumento del recall implica un costo importante en falsos positivos. Por esta razón esta configuración se interpreta como una alternativa orientada a sensibilidad y no como una mejora general de todas las métricas.

### V6 - Elastic Net y variables derivadas

Se evaluó una versión con regularización Elastic Net y variables derivadas de utilización previa.

Resultados en prueba con umbral 0,50:

- ROC-AUC: 0,6603
- PR-AUC: 0,2139
- Precision: 0,3211
- Recall: 0,1382
- F2: 0,1560
- Accuracy: 0,8715

Aunque V6 obtuvo el ROC-AUC más alto, la diferencia fue marginal y el PR-AUC no mejoró. Además, su recall con el umbral estándar fue considerablemente menor que el de V5. Por lo tanto, no se seleccionó como configuración final.

## Selección de la regresión

La configuración seleccionada para el uso operativo es V5, con peso positivo 5, C igual a 0,5 y umbral 0,30. La selección responde al objetivo de aumentar la detección de pacientes con reingreso temprano.

Sin embargo, para la priorización diaria se recomienda utilizar principalmente la probabilidad estimada como puntaje de riesgo y ordenar los egresos según la capacidad disponible, en lugar de depender exclusivamente de una clasificación binaria fija.

La evaluación por capacidad mostró que el 10 % de mayor riesgo concentra el 22,84% de los reingresos y el 30% concentra aproximadamente el 49,4%.
