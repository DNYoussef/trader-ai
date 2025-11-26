# Multi-stage Dockerfile for Trader-AI Dashboard
# Stage 1: Build React frontend
# Stage 2: Python runtime with built frontend

# =============================================================================
# STAGE 1: Build React Frontend
# =============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files (cache bust: 2025-11-26)
COPY src/dashboard/frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production=false

# Copy frontend source
COPY src/dashboard/frontend/ ./

# Build production bundle
RUN npm run build

# =============================================================================
# STAGE 2: Python Runtime
# =============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./src/dashboard/frontend/dist

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will override with PORT env)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

# Start the dashboard server
CMD ["python", "-m", "src.dashboard.run_server_simple"]
