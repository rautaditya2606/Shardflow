FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application source code
COPY . .

# Expose HTTP port
EXPOSE 8000

# Default command launches API Gateway
CMD ["uvicorn", "shardflow.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
