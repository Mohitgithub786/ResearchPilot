# syntax=docker/dockerfile:1
# Use Python 3.11 slim image for smaller size and security
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (e.g. for building specific packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install python dependencies utilizing BuildKit cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Create a non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Run the application using uvicorn, utilizing Uvicorn's worker optimization
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
