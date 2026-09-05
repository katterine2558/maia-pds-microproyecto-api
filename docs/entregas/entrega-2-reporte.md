# Entrega 2 — Reporte

**Micro-proyecto · Desarrollo de Soluciones · MAIA — Universidad de los Andes**
Katerine Arias · Camilo Ardila · Rainer Solano · Leonardo

> **Restriccion del enunciado: maximo 10 paginas.** Si se entrega mas, solo se
> califican las primeras 10. El reporte de trabajo en equipo va aparte y tiene
> su propio limite de 1 pagina.
>
> Presupuesto sugerido: seccion 1 = 1 pagina (tope del enunciado), seccion 2 =
> 4 paginas, seccion 3 = 2 paginas, seccion 4 = 2 paginas, deja 1 de holgura.
>
> Este archivo es el borrador de trabajo. La version que se entrega se exporta
> a documento con el formato del curso.

---

## 1. Resumen del problema

*(maximo 1 pagina — es tope del enunciado, no sugerencia)*

### Contexto del problema

TODO — Rainer / Leonardo. Reingreso hospitalario temprano en pacientes
diabeticos: por que importa clinica y economicamente.

### Pregunta de negocio y alcance

TODO — retomar de la Entrega 1 y ajustar si cambio.

### Breve descripcion de los conjuntos de datos

Base analitica derivada del dataset de readmision en diabetes: 99.343
encuentros de 69.990 pacientes, 32 predictores antes de codificacion. Los
diagnosticos `diag_1`, `diag_2` y `diag_3` se agruparon para reducir su alta
cardinalidad; se excluyeron identificadores, variables con alta ausencia o
baja informacion.

### Cambios respecto a la Entrega 1

**El enunciado pide explicitamente resaltar esto.** No puede faltar.

- La Entrega 1 se centro en caracterizacion y exploracion. Esta entrega avanza
  a preparacion, entrenamiento y evaluacion del modelo predictivo.
- TODO — cambios en el alcance o en los datos, si los hubo.
- TODO — cambios en la maqueta del tablero frente a la version de la Entrega 1.

---

## 2. Modelos desarrollados y su evaluacion

*(texto de Camilo, ya redactado — revisar y recortar si excede el presupuesto)*

A partir de la base analitica definida en la Entrega 1 se desarrollo una
regresion logistica para estimar el riesgo de reingreso hospitalario antes de
30 dias.

La particion entrenamiento / validacion / prueba se realizo **por paciente**,
con 63.670 / 15.852 / 19.821 encuentros respectivamente y cero pacientes
compartidos entre particiones.

El principal reto fue el desbalance de la variable objetivo: solo el 11,39% de
los encuentros corresponde a reingresos antes de 30 dias. Los hiperparametros,
pesos de clase y umbral de decision se seleccionaron utilizando el conjunto de
**validacion**; el conjunto de **prueba** se reservo para reportar el
desempeno final de la configuracion seleccionada.

La regresion base alcanzo 88,24% de exactitud en validacion, pero solamente
2,00% de recall, evidenciando que una exactitud elevada no representaba una
buena identificacion de la clase de interes. A partir de este resultado se
evaluaron balanceo de clases, regularizacion, diferentes pesos para la clase
positiva, ajuste del umbral de decision y una alternativa con Elastic Net.

| Version | Conjunto | Cambio principal | ROC-AUC | PR-AUC | Recall |
|---|---|---|---|---|---|
| V1 | Validacion | Regresion base | 0,6637 | 0,2207 | 2,00% |
| V2 | Validacion | Balanceo de clases | 0,6655 | 0,2208 | 54,61% |
| V3 | Validacion | Regularizacion C=0,5 | 0,6658 | 0,2208 | 54,40% |
| V5 | Prueba | Peso positivo=5; umbral=0,30 | 0,6589 | 0,2133 | 81,56% |

TODO — agregar V4 (barrido de peso) y V6 (Elastic Net) a la tabla, o explicar
por que se omiten. Los resultados de las 76 corridas estan en MLflow.

TODO — una figura. Candidata mas util: la curva recall/precision contra el
umbral, que hace visible el intercambio que justifica elegir V5.

---

## 3. Observaciones y conclusiones sobre los modelos

*(texto de Camilo)*

El desbalance tuvo un efecto directo sobre la regresion base, que alcanzo una
exactitud alta pero apenas 2,00% de recall en validacion. El balanceo y el
ajuste del punto de decision permitieron aumentar sustancialmente la
sensibilidad.

La configuracion final V5 alcanzo 81,56% de recall en prueba, identificando
1.800 de los 2.207 reingresos, con 407 falsos negativos. Este resultado tiene
como contrapartida una precision de 14,00% y un mayor numero de falsos
positivos, por lo que la configuracion se selecciono para un escenario en el
que se busca reducir reingresos no identificados.

Adicionalmente, el 10% de los casos con mayor riesgo estimado concentra el
22,97% de los reingresos, con un lift de 2,30x, lo que respalda utilizar la
probabilidad estimada para priorizar el seguimiento segun la capacidad
disponible.

TODO — vale la pena agregar una observacion que salio del registro en MLflow:
la regularizacion no es la palanca relevante. El barrido de C entre 0,1 y 10
mueve el PR-AUC menos de 0,0005. Lo que mueve el resultado es el peso de clase
y el umbral. Es una conclusion justificada con evidencia, que es justo lo que
pide la rubrica.

---

## 4. Descripcion del tablero y la funcionalidad que ofrece

**FALTA POR COMPLETO. Es entregable obligatorio de la Entrega 2.**

TODO — describir el tablero desarrollado de acuerdo con la maqueta
(`docs/maqueta/`) y la funcionalidad que ofrece.

Recordar la frontera que exige el enunciado: el tablero consume el modelo
**a traves de la API**, no importandolo.

---

## Entregables que van aparte del reporte

- [ ] Repositorio Git accesible, con evidencia de uso por parte de **cada**
      integrante via commits
- [ ] Fuentes de los modelos desarrollados
- [ ] Fuentes del tablero desarrollado
- [ ] **Pantallazos de MLflow**, donde se vean el usuario y la IP de la maquina
      EC2, y la IP dentro de la interfaz de MLflow → `docs/entregas/figuras/`
- [ ] Reporte de trabajo en equipo, **maximo 1 pagina**

### Los tres pantallazos exigidos

1. Terminal SSH conectada, donde se lea el prompt `ubuntu@ip-172-31-x-x`
2. Consola de EC2 con la instancia y su IP publica
3. Interfaz de MLflow con los experimentos y la URL visible en la barra de
   direcciones
