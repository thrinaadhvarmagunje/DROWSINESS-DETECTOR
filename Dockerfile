FROM python:3.10-bullseye

WORKDIR /app

# System dependencies for OpenCV & Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Cloud Run sets PORT env; default 8080
ENV PORT=8080

# Use gunicorn in container
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
