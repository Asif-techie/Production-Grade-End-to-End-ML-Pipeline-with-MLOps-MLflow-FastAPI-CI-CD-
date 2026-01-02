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
COPY frontend/ ./frontend
COPY models/ ./models

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
# Run FastAPI with uvicorn
# ------------------------
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
