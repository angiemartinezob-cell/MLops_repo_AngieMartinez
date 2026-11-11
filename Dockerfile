# Imagen de la API de riesgo de crédito (FastAPI + Uvicorn)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código fuente, configuración y datos de ejemplo
COPY config.json Base_de_datos.csv ./
COPY mlops_pipeline/ ./mlops_pipeline/

# El modelo se entrena dentro de la imagen para que el .joblib quede
# generado con las mismas versiones de librerías con las que se va a cargar.
RUN python mlops_pipeline/model_evaluacion.py

# Usuario sin privilegios para ejecutar el servicio
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "deploy:app", "--app-dir", "mlops_pipeline", "--host", "0.0.0.0", "--port", "8000"]
