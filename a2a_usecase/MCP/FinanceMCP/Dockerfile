FROM python:3.11-slim

# Better logging / no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies (use requirements.txt if present)
COPY requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose MCP HTTP port
EXPOSE 8000

# Optional runtime env defaults (can override with docker run -e)
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Run your MCP server
CMD ["python", "-m", "mcp_server"]