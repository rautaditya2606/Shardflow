FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy application source code & install (using CPU PyTorch for lightweight gateway/registry container)
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Expose HTTP port
EXPOSE 8000

# Default command launches API Gateway
CMD ["uvicorn", "shardflow.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
