# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app and frontend
COPY app/ ./app
COPY frontend/ ./frontend

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
