# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Copy trained model (ensure model exists before Docker build)
ARG MODEL_DIR=models
ARG MODEL_NAME=LogisticRegression.pkl
COPY ${MODEL_DIR}/ ./models

# Set environment variable for model path
ENV MODEL_PATH=models/${MODEL_NAME}

# Expose port for FastAPI
EXPOSE 80

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
