# ---------- Build stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools only for this stage
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a temporary layer
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ---------- Final stage ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app source and model
COPY src ./src
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

# Add a Docker healthcheck (works without docker-compose)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -s http://localhost:8000/health | grep '"status": "ok"' || exit 1

# Default command (production)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

