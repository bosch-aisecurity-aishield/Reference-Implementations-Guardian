FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your agent code
COPY . .

# Expose the default LangGraph port
EXPOSE 2026

# Run the server on 0.0.0.0 so Docker can see it
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2026"]