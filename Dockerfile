FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgl1 + libglib2 needed by opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY api/ api/
COPY workers/ workers/
COPY preprocessing/ preprocessing/
COPY pipelines/ pipelines/

# Install Python dependencies
# Core API deps + pipeline deps (numpy, opencv, scipy, scikit-image)
# rembg is imported lazily for background removal in preprocessing
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    redis \
    pydantic \
    Pillow \
    numpy \
    opencv-python-headless \
    scipy \
    scikit-image \
    rembg

# Create storage directories
RUN mkdir -p /storage/uploads /storage/previews /storage/outputs /storage/artifacts /storage/jobs

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
