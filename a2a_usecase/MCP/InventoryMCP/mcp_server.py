"""Inventory MCP Tool Server.

Provides SQL query tools for vehicle search exposed over the Model Context Protocol:
    - SQL queries for factual data (price, color, model, VIN)

Tools:
    run_sql_query: Execute read-only SQL queries on vehicle database

Usage:
    This server is launched automatically by the Inventory Agent.
    Direct usage: python -m inventory.mcp_server
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Inventory Service")


mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
mcp.settings.port = int(os.getenv("MCP_PORT", "8081"))

# -----------------------------------------------------------------------------
# CORS / Transport security
# -----------------------------------------------------------------------------
# Set your public host/IP (or DNS) here. Prefer DNS if you have it.

EC2_PUBLIC_HOST = os.getenv("MCP_PUBLIC_HOST", "16.58.206.13")
PORT = mcp.settings.port

# If you REALLY want to allow any origin/host (not recommended), set:
#   MCP_ALLOW_ANY_ORIGIN=true
allow_any = os.getenv("MCP_ALLOW_ANY_ORIGIN", "false").lower() == "true"
allowed_hosts = [
    f"{EC2_PUBLIC_HOST}:{PORT}",
    "localhost:*",
    "127.0.0.1:*",
]

allowed_origins = [
    f"http://{EC2_PUBLIC_HOST}:{PORT}",
    f"https://{EC2_PUBLIC_HOST}:{PORT}",
    "http://localhost:*",
    "http://127.0.0.1:*",
]

if allow_any:
    # ⚠️ Dangerous on public servers
    allowed_hosts.append("*:*")
    allowed_origins.append("*")
    
mcp.settings.transport_security.allowed_hosts.extend(allowed_hosts)
mcp.settings.transport_security.allowed_origins.extend(allowed_origins)


# ── Configuration ────────────────────────────────────────────────────────────

# Use environment variable (passed by Docker) or fallback to host.docker.internal for Docker environments
POSTGRES_DSN: str = os.getenv(
    "POSTGRES_DSN",
    "dbname=dealership user=admin password=password123 host=host.docker.internal port=5432",
)


# ── SQL Tool ─────────────────────────────────────────────────────────────────

@mcp.tool()
def run_sql_query(query: str) -> str:
    """Execute read-only SQL query against the vehicles database.
    Args:
        query: SELECT statement to execute (only read queries allowed)

    Returns:
        JSON string of query results or error message

    Security:
        Only SELECT queries are permitted for read-only access

    Example:
        >>> run_sql_query("SELECT * FROM vehicles WHERE color='Red')
        "[{'vin': '...', 'make': 'Toyota', ...}, ...]"
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        
        if not rows:
            return "No matching records found."
        
        return str([dict(r) for r in rows])
    except Exception as exc:
        return f"SQL Error: {exc}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")