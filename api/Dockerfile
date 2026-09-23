FROM python:3.12-slim
WORKDIR /app
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ ./api/
ENV PORT=8000
EXPOSE 8000
CMD uvicorn api.main:app --host :: --port ${PORT:-8000}
