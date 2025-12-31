# Use slim Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained model
COPY src/ ./src
COPY models/ ./models

# Environment variable for model path
ENV MODEL_PATH=/app/models/LogisticRegression.pkl

# Expose port
EXPOSE 80

# Start the API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
