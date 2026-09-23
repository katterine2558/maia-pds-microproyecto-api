# Despliegue de la API en Railway

La API de inferencia se despliega como un servicio Railway construido desde
`api/Dockerfile`. El tablero, en el repositorio `-ui`, ya corre asi.

> **Estado final (Entrega 3).** La API quedo como servicio interno, **sin
> dominio publico**: el tablero la alcanza por la red privada (IPv6) del
> proyecto de Railway. El paso 4 y los `curl` con `<dominio-de-la-api>` solo
> aplican si se le genera un dominio para probarla desde fuera; sin el, la
> verificacion se hace desde el tablero.

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
4. *(Opcional)* **Settings → Networking → Generate Domain.** El puerto destino es el que
   aparece en los logs del despliegue (`Uvicorn running on http://[::]:XXXX`), no el
   del `EXPOSE`: Railway inyecta su propia variable `PORT` en tiempo de
   ejecucion y pisa el `ENV PORT` de la imagen. Por eso el `CMD` usa
   `${PORT:-8000}`.
5. Comprobar el despliegue:

   ```bash
   curl https://<dominio-de-la-api>/health
   # {"estado":"ok","modelo":"bosque_formulario_e3_v1"}
   ```

## Variables de ambiente

La API no lee ninguna variable propia: el artefacto viaja dentro de la imagen y
su ruta se resuelve desde `__file__`. La unica variable que interviene es
`PORT`, y la inyecta Railway.

### Servicio de la API (este repositorio)

| Variable | Quien la define | Valor |
|---|---|---|
| `PORT` | Railway, en tiempo de ejecucion | No agregarla a mano. El `CMD` la lee con `${PORT:-8000}`; fijarla en el panel desincroniza el puerto real del puerto destino del dominio. |

No hacen falta credenciales. En particular **no** van aqui las llaves de AWS
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) ni la URI de MLflow: DVC y
MLflow pertenecen al ciclo de entrenamiento, y el servicio desplegado solo
sirve un `modelo.joblib` ya empaquetado.

Ajustes del panel que no son variables, pero sin los cuales no despliega:
rama `develop` en **Settings → Source**, builder Dockerfile con ruta
`api/Dockerfile` en **Settings → Build**, y el dominio generado en
**Settings → Networking** apuntando al puerto que imprimen los logs.

### Servicio del tablero (repositorio `-ui`)

| Variable | Quien la define | Valor |
|---|---|---|
| `API_URL` | A mano, en el panel del tablero | `http://<servicio-api>.railway.internal:<puerto>`, sin barra final: `services/api.py` la agrega. La API no tiene dominio publico: el tablero la alcanza por la red privada (IPv6) del proyecto, y `<puerto>` es el que Uvicorn anuncia en los logs de la API |
| `PORT` | Railway, en tiempo de ejecucion | No agregarla a mano |

`API_URL` es obligatoria. El `Dockerfile` del tablero trae
`ENV API_URL=http://api:8000`, que es el nombre del servicio dentro de
`docker-compose`. En Railway ese nombre no resuelve, asi que si la variable no
se sobreescribe el tablero apunta a un host inexistente y las dos vistas que
consumen el modelo fallan con error de conexion. Railway reinicia el servicio
al guardar la variable.

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
