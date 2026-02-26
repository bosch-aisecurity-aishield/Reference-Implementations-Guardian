#!/usr/bin/env sh
set -e

echo "🚀 Starting inventory MCP container..."
echo "📄 Using environment variables from Docker Compose"

echo "⏳ Waiting for PostgreSQL to be reachable..."
python - <<'PY'
import os, time, psycopg2, re

dsn = os.getenv("POSTGRES_DSN")
if not dsn:
    raise SystemExit("❌ POSTGRES_DSN is not set")

# Optional safety warning
if "localhost" in dsn or "127.0.0.1" in dsn:
    print("⚠️ Warning: POSTGRES_DSN points to localhost/127.0.0.1. In Docker, use host=postgres (service name).")

# Mask password in logs (supports URL and key=value styles loosely)
masked = re.sub(r'://([^:/@\s]+):([^@/\s]+)@', r'://\1:***@', dsn)
masked = re.sub(r'(password=)(\S+)', r'\1***', masked)
print(f"🔎 POSTGRES_DSN detected: {masked}")

for i in range(30):
    try:
        conn = psycopg2.connect(dsn)
        conn.close()
        print("✅ PostgreSQL is reachable")
        break
    except Exception as e:
        print(f"Attempt {i+1}/30: DB not ready yet ({e})")
        time.sleep(2)
else:
    raise SystemExit("❌ PostgreSQL not reachable after retries")
PY

echo "🗄️ Running DB seed..."
python /app/setup_db.py

echo "🧠 Running RAG seed..."
python /app/setup_rag.py

echo "🌐 Starting MCP server..."
python /app/mcp_server.py