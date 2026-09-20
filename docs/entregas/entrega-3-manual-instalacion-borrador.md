# Manual de instalación del tablero — borrador técnico

**Fuente:** repositorio [`maia-pds-microproyecto-ui`](https://github.com/katterine2558/maia-pds-microproyecto-ui), rama `develop` revisada el 20 de septiembre de 2026. Requisitos: Git, Python 3.12 y `uv`, o Docker para la opción de contenedor. La interfaz necesita además una API de inferencia operativa para calcular predicciones.

## Ejecución local

1. Clone el repositorio y entre en su carpeta: `git clone https://github.com/katterine2558/maia-pds-microproyecto-ui.git` y `cd maia-pds-microproyecto-ui`. La rama predeterminada observada es `develop`; verifique con `git branch --show-current`.
2. Instale dependencias con `uv sync --frozen` (Python 3.12). Si trabaja con un entorno Python preparado, el README también contempla `pip install -e .`, pero esa ruta no fija las mismas versiones del archivo `uv.lock`.
3. Configure `API_URL` para que apunte a la API; por ejemplo, `http://localhost:8000` si ya está funcionando en la misma máquina. La plantilla `.env.example` describe la variable, pero el módulo `services/api.py` lee variables de entorno directamente: expórtela en la sesión o configúrela en la plataforma. No suponga que copiar `.env` cargará automáticamente la variable.
4. Arranque desde la raíz con `uv run streamlit run app.py`. Abra `http://localhost:8501`. Revise las vistas. Para verificar una predicción, la API debe responder correctamente a `GET /health` y `POST /predict`.

## Ejecución del tablero en Docker

Desde la raíz del repositorio UI: `docker build -t reingreso-ui .`. Con la API accesible desde el contenedor, ejecute `docker run --rm -p 8501:8501 -e API_URL=http://host.docker.internal:8000 reingreso-ui`. La dirección `host.docker.internal` depende del entorno Docker; en una red Compose compartida, `API_URL` puede apuntar a `http://api:8000` si el servicio se llama `api`. Visite `http://localhost:8501`.

El `Dockerfile` usa Python 3.12, `uv.lock`, el puerto 8501 y una comprobación de salud en `/_stcore/health`. `railway.json` indica a Railway que construya con ese Dockerfile. Para desplegar, configure `API_URL` con una dirección que el contenedor del tablero pueda alcanzar y pruebe una predicción desde la URL publicada. Si Railway asigna `PORT`, el comando de arranque del contenedor lo utiliza.

## Límite de esta versión

El repositorio API revisado no trae aún el servicio FastAPI, su imagen ni el artefacto del modelo. Por ello, estos pasos permiten iniciar el tablero, **pero no completan la instalación del producto integrado**. Tras integrar ambos repositorios, añadir versión de imagen, comando real de arranque de API, configuración de artefactos del modelo y una prueba repetible de extremo a extremo. No registrar claves ni datos de pacientes en este manual.
