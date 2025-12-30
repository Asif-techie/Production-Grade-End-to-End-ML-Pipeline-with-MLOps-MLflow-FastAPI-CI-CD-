# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Copy source code
COPY src/ ./src
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 80

# Start FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
