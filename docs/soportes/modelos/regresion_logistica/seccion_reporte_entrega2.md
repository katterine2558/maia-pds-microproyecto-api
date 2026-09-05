# Regresión logística

A partir de la base analítica definida en la Entrega 1 se desarrolló una regresión logística para estimar el riesgo de reingreso hospitalario antes de 30 días. Frente a la Entrega 1, centrada en la caracterización y exploración de los datos, en esta entrega se avanzó hacia la preparación, entrenamiento y evaluación del modelo predictivo.

Se trabajó con 99.343 encuentros de 69.990 pacientes y se definieron 32 predictores antes de la codificación. Los diagnósticos `diag_1`, `diag_2` y `diag_3` se agruparon para reducir su alta cardinalidad y se excluyeron identificadores, variables con alta ausencia o baja información. La partición entrenamiento/validación/prueba se realizó por paciente, con 63.670/15.852/19.821 encuentros respectivamente y cero pacientes compartidos entre particiones.

El principal reto fue el desbalance de la variable objetivo: solo el 11,39% de los encuentros corresponde a reingresos antes de 30 días. Los hiperparámetros, pesos de clase y umbral de decisión se seleccionaron utilizando el conjunto de validación; el conjunto de prueba se reservó para reportar el desempeño final de la configuración seleccionada.

| Versión | Conjunto | Cambio principal | ROC-AUC | PR-AUC | Recall |
|---|---|---|---:|---:|---:|
| V1 | Validación | Regresión base | 0,6637 | 0,2207 | 2,00% |
| V2 | Validación | Balanceo de clases | 0,6655 | 0,2208 | 54,61% |
| V3 | Validación | Regularización C=0,5 | 0,6658 | 0,2208 | 54,40% |
| **V5** | **Prueba** | **Peso positivo=5; umbral=0,30** | **0,6589** | **0,2133** | **81,56%** |

**Conclusión.** El desbalance tuvo un efecto directo sobre la regresión base, que alcanzó una exactitud alta pero apenas 2,00% de recall en validación. El balanceo y el ajuste del punto de decisión permitieron aumentar sustancialmente la sensibilidad. La configuración final V5 alcanzó 81,56% de recall en prueba, identificando 1.800 de los 2.207 reingresos, con 407 falsos negativos. Este resultado tiene como contrapartida una precisión de 14,00% y un mayor número de falsos positivos, por lo que la configuración se seleccionó para un escenario en el que se busca reducir reingresos no identificados. Adicionalmente, el 10% de los casos con mayor riesgo estimado concentra el 22,97% de los reingresos, con un lift de 2,30x, lo que respalda utilizar la probabilidad estimada para priorizar el seguimiento según la capacidad disponible.