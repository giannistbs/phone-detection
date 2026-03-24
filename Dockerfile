FROM python:3.12-slim

# System deps for OpenCV headless + FFMPEG for RTSP
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install deps — swap opencv-python for the headless build (no GUI libs needed)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        $(grep -v '^opencv-python$' requirements.txt) \
        opencv-python-headless

COPY src/ ./src/

USER appuser

ENTRYPOINT ["python", "src/detect.py", "--headless"]
