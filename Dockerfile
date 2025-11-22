# Video Activity Recognition - Dockerfile
# Base image: Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROJECT_ROOT=/app

# Install system dependencies required for OpenCV, MediaPipe, and GUI applications
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY Entrega3/requirements.txt /app/Entrega3/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/Entrega3/requirements.txt

# Copy the entire project
COPY . /app/

# Expose any port if needed (optional, for future web interfaces)
# EXPOSE 8000

# Set the entrypoint to run the application
CMD ["python", "-m", "Entrega3.src.online.ui_app"]
