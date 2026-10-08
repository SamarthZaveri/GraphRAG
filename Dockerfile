FROM python:3.11-slim
WORKDIR /workspace
COPY backend/requirements-api.txt /workspace/backend/requirements-api.txt
RUN pip install --no-cache-dir -r backend/requirements-api.txt
COPY backend/app /workspace/backend/app
COPY frontend /workspace/frontend
COPY RL /workspace/RL
ENV PYTHONPATH=/workspace/backend LEDGER_DATA_DIR=/data PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
