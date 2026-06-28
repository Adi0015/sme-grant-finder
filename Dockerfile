# RAISE — SME grant finder UI.
# Builds an image that serves the Streamlit app. The app needs two things at RUNTIME that
# are deliberately NOT baked into the image (see .dockerignore):
#   1. a built Chroma index at /app/data/chroma  -> mount your local data/ as a volume
#   2. a reachable Ollama with the model pulled   -> set OLLAMA_HOST to the host
# The embedding model (~1GB) downloads from Hugging Face on first run (or mount a cache).
#
# Build:
#   docker build -t raise-app .
# Run (demo mode needs neither index nor Ollama):
#   docker run --rm -p 8501:8501 raise-app
#   # then open http://localhost:8501/?demo=1
# Run (full pipeline — mount data + point at host Ollama):
#   docker run --rm -p 8501:8501 \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     -v "$PWD/data:/app/data" \
#     raise-app

FROM python:3.13-slim

WORKDIR /app

# System deps kept minimal; wheels cover torch/chromadb on linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY data/samples/ ./data/samples/

ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    PYTHONUNBUFFERED=1

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
