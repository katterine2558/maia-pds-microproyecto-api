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

