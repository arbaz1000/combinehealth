# Multi-stage build: keeps the final image small
# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only runtime deps (no docling — that's for offline preprocessing only)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY data/qdrant_store/ ./data/qdrant_store/
COPY data/chunks/ ./data/chunks/
COPY start.sh .

RUN chmod +x start.sh

ENV PYTHONPATH=/app

# OPENAI_API_KEY must be passed at runtime (not baked into image).
# On HuggingFace Spaces: Settings → Secrets → add OPENAI_API_KEY
# Locally: docker run -e OPENAI_API_KEY=sk-... ...
EXPOSE 7860

CMD ["./start.sh"]
