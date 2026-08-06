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

# ── Pre-cache the tokenizer at BUILD TIME ──────────────────────────────────────
# This bakes the tokenizer vocab into the image so Render never downloads it at
# runtime. The Docker build host has no RAM limit, so the ~500 MB /tmp spike
# during download is harmless here. At runtime the tokenizer loads from the
# HF cache in ~50 ms with zero network calls and <10 MB RAM.
#
# Set SHARDFLOW_MODEL_PATH build-arg to override the model (default: TinyLlama).
ARG SHARDFLOW_MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
ENV HF_HOME=/app/.hf_cache
RUN python - <<'EOF'
import os
from transformers import AutoTokenizer
model_path = os.environ.get("SHARDFLOW_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"Pre-caching tokenizer: {model_path}")
AutoTokenizer.from_pretrained(model_path, use_fast=True)
print("Tokenizer cached successfully.")
EOF
# ────────────────────────────────────────────────────────────────────────────────

# Expose HTTP port
EXPOSE 8000

# Default command launches API Gateway
CMD ["uvicorn", "shardflow.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]

