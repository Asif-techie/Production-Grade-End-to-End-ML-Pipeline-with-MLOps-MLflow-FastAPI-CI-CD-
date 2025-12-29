# Base image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model
COPY src/ src/
COPY model.pkl model.pkl

# Set environment variable
ENV MODEL_PATH=/app/model.pkl

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]