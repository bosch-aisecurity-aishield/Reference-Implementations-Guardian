FROM ubuntu:22.04

WORKDIR /app

# Install dependencies (zstd is required by Ollama installer)
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# (Optional) Pre-pull only the required models at build time
# NOTE: This makes the image very large and build slower.
RUN ollama serve & \
    OLLAMA_PID=$! && \
    sleep 10 && \
    OLLAMA_HOST=http://127.0.0.1:11434 ollama pull llama3.2 && \
    OLLAMA_HOST=http://127.0.0.1:11434 ollama pull llama3.2-vision && \
    OLLAMA_HOST=http://127.0.0.1:11434 ollama pull qwen2.5-coder && \
    kill $OLLAMA_PID

# Expose Ollama API port
EXPOSE 11434

# Start Ollama server
CMD ["ollama", "serve"]