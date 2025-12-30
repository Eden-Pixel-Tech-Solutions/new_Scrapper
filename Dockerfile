
# Use the official Playwright Python image
# It includes Python 3.10+, browsers, and system dependencies
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app

# Install system dependencies if you need any extra (e.g. for mysql client sometimes)
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache pip install
COPY requirements-worker.txt .

# Install python dependencies
# Using --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements-worker.txt

# Install Playwright browsers (already in base image, but 'install' ensures deps are linked)
RUN playwright install chromium
RUN playwright install-deps

# Copy the rest of the application
COPY . .

# Default command (can be overridden in compose)
CMD ["celery", "-A", "celery_worker", "worker", "--loglevel=info"]
