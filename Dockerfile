FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY api/ api/
COPY workers/ workers/
COPY preprocessing/ preprocessing/
COPY pipelines/ pipelines/

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    redis \
    pydantic \
    Pillow

# Create storage directories
RUN mkdir -p /storage/uploads /storage/previews /storage/outputs /storage/artifacts /storage/jobs

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]

