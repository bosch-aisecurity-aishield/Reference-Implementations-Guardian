#!/usr/bin/env sh
set -e

echo "🚀 Starting inventory MCP container..."

# Load .env into environment (if present)
if [ -f /app/.env ]; then
  echo "📄 Loading environment from /app/.env"
  set -a
  . /app/.env
  set +a
fi

echo "⏳ Waiting for PostgreSQL to be reachable..."
python - <<'PY'
import os, time, psycopg2
dsn = os.getenv("POSTGRES_DSN")
if not dsn:
    raise SystemExit("❌ POSTGRES_DSN is not set")

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
python /app/setup_rag.py   # or setup_rag.py if that's your actual filename

echo "🌐 Starting MCP server..."
python /app/mcp_server.py