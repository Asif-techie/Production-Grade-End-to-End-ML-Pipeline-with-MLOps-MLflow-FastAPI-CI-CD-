# ------------------------
# Base image
# ------------------------
FROM python:3.11-slim

# ------------------------
# Set working directory
# ------------------------
WORKDIR /app

# ------------------------
# Copy source code, models, and frontend
# ------------------------
COPY src/ ./src
COPY models/ ./models
COPY frontend/ ./frontend

# ------------------------
# Set environment variables for paths
# ------------------------
ENV MODELS_PATH=/app/models
ENV FRONTEND_PATH=/app/frontend

# ------------------------
# Install dependencies
# ------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------
# Expose FastAPI port
# ------------------------
EXPOSE 8000

# ------------------------
# Run FastAPI with Prometheus monitoring
# ------------------------
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
