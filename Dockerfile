# Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src 
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 80

# Start FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
