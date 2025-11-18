# Use Python 3.10 (TensorFlow compatible)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for TensorFlow, OpenCV, and PyAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    libavutil-dev \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip

# Install requirements AFTER system libs
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
