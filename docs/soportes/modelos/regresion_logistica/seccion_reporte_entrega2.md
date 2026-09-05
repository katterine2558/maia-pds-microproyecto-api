\# \*\*Modelo de regresión logística\*\*



\## \*\*Entrenamiento y selección de características\*\*



Se desarrolló un modelo de regresión logística para estimar la probabilidad de reingreso hospitalario dentro de los 30 días posteriores al alta. Se mantuvo la población analítica definida en la primera entrega: 99.343 encuentros correspondientes a 69.990 pacientes, con 11.314 reingresos tempranos (11,39%).

Frente a la Entrega 1, centrada en la caracterización y exploración de los datos, en esta entrega se avanzó hacia la construcción y evaluación del modelo predictivo. A partir de los hallazgos previos se definieron las variables de entrada, se trataron variables de alta cardinalidad y baja frecuencia y se realizó la partición por paciente. Uno de los principales retos fue el desbalance de la variable objetivo: solo el 11,39% de los encuentros corresponde a reingresos antes de 30 días, condición que afectó especialmente el desempeño de la regresión base.

La partición de los datos se realizó por paciente utilizando `patient\_nbr`, de forma que un mismo paciente no pudiera aparecer simultáneamente en entrenamiento, validación y prueba. El conjunto final quedó compuesto por 63.670 encuentros de entrenamiento, 15.852 de validación y 19.821 de prueba, sin pacientes compartidos entre particiones.



La preparación inicial produjo 32 predictores antes de la codificación. Se excluyeron los identificadores, la variable objetivo original, `weight`, `payer\_code`, las variables reservadas para análisis de sesgo y los medicamentos con menos de 100 registros diferentes de `No`. Los diagnósticos `diag\_1`, `diag\_2` y `diag\_3` se agruparon en categorías clínicas para reducir su alta cardinalidad. Las variables numéricas se estandarizaron y las categóricas se codificaron mediante one-hot encoding.



\## \*\*Versiones evaluadas\*\*



Se evaluaron varias configuraciones con el propósito de analizar el efecto del desbalance de clases y de los hiperparámetros del modelo.



La versión base presentó ROC-AUC de 0,6591 y PR-AUC de 0,2140. Aunque alcanzó una exactitud de 88,86%, su recall fue únicamente de 1,90%, lo cual evidencia que la alta exactitud se explicaba principalmente por la clase mayoritaria.



La segunda versión incorporó ponderación balanceada de clases. El recall aumentó a 53,60%, con ROC-AUC de 0,6594 y PR-AUC de 0,2142. Posteriormente se ajustó la regularización y se seleccionó C=0,5 mediante el conjunto de validación. Esta modificación produjo cambios pequeños frente a la versión balanceada.



Con el fin de priorizar la identificación de reingresos se evaluaron pesos adicionales para la clase positiva y diferentes umbrales de clasificación. La combinación de peso positivo igual a 5 y umbral de 0,30 obtuvo el mejor F2 en validación. En prueba alcanzó recall de 82,01%, identificando 1.810 de los 2.207 reingresos observados, con 397 falsos negativos.



El aumento de sensibilidad se produjo a costa de una reducción de la precisión a 13,87% y un incremento de falsos positivos. Por lo tanto, esta versión debe interpretarse como una configuración orientada a detectar la mayor cantidad posible de pacientes con riesgo y no como una mejora simultánea de todas las métricas.



También se evaluó una sexta versión con regularización Elastic Net y variables derivadas de utilización previa. Esta versión obtuvo ROC-AUC de 0,6603, pero PR-AUC de 0,2139 y recall de 13,82% con el umbral estándar. La mejora en ROC-AUC fue marginal y no justificó su selección.



\## \*\*Evaluación de priorización\*\*



Dado que el objetivo del prototipo es apoyar la asignación de una capacidad limitada de seguimiento, se evaluó también el modelo como mecanismo de ordenamiento por riesgo.



Al priorizar el 10% de encuentros con mayor probabilidad estimada se captura el 22,84% de los reingresos observados, con una tasa de reingreso 2,28 veces superior a la tasa base. Al ampliar la cobertura al 20 % se captura aproximadamente el 37% de los reingresos y al 30% se alcanza aproximadamente el 49,4%.



Estos resultados indican que la principal utilidad de la regresión no está en producir una decisión clínica binaria aislada, sino en generar un puntaje de riesgo que permita ordenar los egresos y adaptar la cantidad de pacientes priorizados a la capacidad disponible.



\## \*\*Conclusión del modelo\*\*



La regresión logística mostró una capacidad discriminativa moderada, con ROC-AUC alrededor de 0,66 y PR-AUC alrededor de 0,21 en las diferentes versiones. El tratamiento del desbalance permitió aumentar sustancialmente la detección de reingresos, mientras que los ajustes adicionales de regularización produjeron mejoras marginales.



La configuración seleccionada para sensibilidad utiliza C=0,5, peso positivo igual a 5 y umbral de 0,30. Sin embargo, para el uso operativo del prototipo se recomienda utilizar la probabilidad estimada como puntaje de priorización y definir el corte según la capacidad disponible de seguimiento.

