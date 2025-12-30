# Use official Python base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src
COPY models/ ./models

# Expose FastAPI port
EXPOSE 80

# Start FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]