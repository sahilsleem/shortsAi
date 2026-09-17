FROM python:3.11-slim

# Install system dependencies (FFmpeg is required by the renderer)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Render dynamically passes the PORT environment variable
ENV PORT=8000

# Start the FastAPI server using the specified PORT
CMD uvicorn src.server:app --host 0.0.0.0 --port $PORT
