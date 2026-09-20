# Despliegue de la API en Railway

La API de inferencia se despliega como un servicio Railway construido desde
`api/Dockerfile`. El tablero, en el repositorio `-ui`, ya corre asi.

## Antes de empezar

El servicio necesita el artefacto del modelo, que viaja dentro de la imagen:
`api/artifacts/modelo.joblib` esta versionado en Git, asi que no hay que montar
volumenes ni descargar nada en tiempo de despliegue.

## Pasos

Desde el panel de Railway, en la cuenta donde vive el tablero:

1. **New Project → Deploy from GitHub repo** → `maia-pds-microproyecto-api`.
2. En **Settings → Source**, fijar la rama `develop`. Railway reconstruye sola
   en cada push, igual que el tablero.
3. En **Settings → Build**, confirmar que el builder es Dockerfile y la ruta es
   `api/Dockerfile`. El `api/railway.json` de este repositorio ya lo declara.
4. **Settings → Networking → Generate Domain.** El puerto destino es el que
   aparece en los logs del despliegue (`Uvicorn running on 0.0.0.0:XXXX`), no el
   del `EXPOSE`: Railway inyecta su propia variable `PORT` en tiempo de
   ejecucion y pisa el `ENV PORT` de la imagen. Por eso el `CMD` usa
   `${PORT:-8000}`.
5. Comprobar el despliegue:

   ```bash
   curl https://<dominio-de-la-api>/health
   # {"estado":"ok","modelo":"bosque_formulario_e3_v1"}
   ```

## Conectar el tablero

En el proyecto del **tablero** (repositorio `-ui`), agregar la variable:

```
API_URL=https://<dominio-de-la-api>
```

Sin barra final: `services/api.py` la agrega. Railway reinicia el servicio al
guardar la variable. Despues, en la vista Paciente, el boton "Calcular riesgo"
debe devolver una probabilidad en lugar de un error de conexion.

## Quien puede hacerlo

Quien tenga acceso al proyecto de Railway donde vive el tablero. La variable
`API_URL` se edita en ese panel, no en el repositorio, asi que el paso de
conectar los dos servicios lo hace esa misma persona.

## Verificacion de extremo a extremo

Con ambos servicios arriba:

```bash
# la API responde por si sola
curl -s https://<dominio-de-la-api>/health

# una prediccion completa
curl -s -X POST https://<dominio-de-la-api>/predict \
  -H 'Content-Type: application/json' \
  -d '{"rango_edad":"[70-80)","tipo_admision":"Emergency","servicio_alta":"Nephrology","dias_estancia":9,"num_diagnosticos":9,"num_medicamentos":21,"ingresos_previos":5,"urgencias_previas":2,"resultado_a1c":"No medido","cambio_medicacion":"Sí"}'
```

Y en el tablero desplegado, las dos vistas que consumen el modelo:

- **Paciente**: llenar el formulario y pulsar "Calcular riesgo".
- **Priorizacion**: subir un CSV de egresos y ver la lista ordenada.

Esas dos pantallas son la evidencia de que la cadena `tablero -> API -> modelo
empaquetado` funciona desplegada, que es lo que evalua el enunciado.
