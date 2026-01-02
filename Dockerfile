# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy source code and frontend
COPY src/ ./src
COPY frontend/ ./frontend

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the app (adjust the module path to src)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
