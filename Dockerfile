FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src

# Copy trained model artifacts (ROOT-level models/)
COPY models/ ./models

# Expose API port
EXPOSE 80

# Start FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
