FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app/hybrid_rag/

# Copy config to app root for easy access
COPY config.yaml /app/config.yaml

# Create indexes directory
RUN mkdir -p /app/indexes

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-m", "hybrid_rag", "serve", "--config", "/app/config.yaml", "--host", "0.0.0.0", "--port", "8000", "--index-dir", "/app/indexes"]
