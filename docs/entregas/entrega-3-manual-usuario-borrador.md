# Manual de usuario del tablero — borrador verificable

**Producto:** priorización de seguimiento tras el alta para pacientes diabéticos. **Estado:** basado en el código de la rama `develop` revisado el 20 de septiembre de 2026. Revisar tras integrar la API y la carga de egresos. El resultado es orientativo y requiere criterio clínico.

## Acceso y navegación

Abra la dirección del tablero facilitada por el equipo. En el menú lateral, elija **Priorización**, **Paciente** o **Contexto**. La vista Contexto muestra información histórica descriptiva; no calcula el riesgo de un paciente particular.

## Consultar un encuentro individual

1. Abra **Paciente**. Complete rango de edad, tipo de admisión, servicio de alta, días de estancia, número de diagnósticos y medicamentos, ingresos y urgencias previos, resultado de A1C y cambio de medicación.
2. Pulse **Calcular riesgo**. El tablero envía los datos a `POST /predict` de la API configurada. Si la API responde con probabilidad y umbral válidos, verá la estimación, la comparación con el umbral y, si está disponible, el nombre del modelo.
3. Si aparece un error de conexión o de respuesta, registre el mensaje y comuníquelo al equipo técnico; no interprete una tarjeta anterior como resultado del nuevo encuentro. La vista evita mostrar un resultado tras ese error.

**Limitación actual:** el repositorio de la API revisado no incluye todavía el servicio para responder a la solicitud. Hasta integrarlo, el botón no permite obtener una predicción real. No use datos reales de pacientes en un servicio de demostración sin autorización institucional.

## Priorización y contexto

En **Priorización** se muestran campos para fecha de alta, servicio y capacidad (10, 20, 30 o 50 seguimientos), con una línea que separa los registros dentro y fuera de la capacidad. **En la versión revisada los encuentros y cifras son ejemplos fijos**, no resultados de inferencia ni de un archivo cargado. No los use para decidir a quién contactar. Esta sección se actualizará cuando exista el listado real calculado mediante la API.

En **Contexto**, consulte las gráficas históricas como apoyo para entender el problema poblacional; sus proporciones no representan la probabilidad estimada para un encuentro. Verificar títulos y filtros contra la versión finalmente publicada.

## Antes de publicar este manual

Sustituir la dirección de acceso por la URL comprobada y la fecha de la prueba; actualizar los pasos de carga de archivo, filtros, descarga y mensajes de error solo si esas funciones se implementan; adjuntar una captura de una predicción real del mismo encuentro en API y tablero.
