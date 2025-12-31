FROM python:3.11-slim

# -----------------------------
# Environment safety
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.pkl

# -----------------------------
# Working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application source
# -----------------------------
COPY src/ ./src

# -----------------------------
# Copy trained model (from CI)
# Expected path: /app/models/model.pkl
# -----------------------------
COPY models/ ./models

# -----------------------------
# Validate model exists at build time
# (Fail fast if CI is broken)
# -----------------------------
RUN test -f /app/models/model.pkl

# -----------------------------
# Expose port
# -----------------------------
EXPOSE 80

# -----------------------------
# Start FastAPI
# -----------------------------
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
