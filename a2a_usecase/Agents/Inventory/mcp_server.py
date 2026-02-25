"""Inventory MCP Tool Server.

Provides dual-mode vehicle search tools exposed over the Model Context Protocol:
    1. SQL queries for factual data (price, color, model, VIN)
    2. RAG vector search for visual features and descriptions

Tools:
    run_sql_query: Execute read-only SQL queries on vehicle database
    search_vehicle_images: Semantic search for visual features using ChromaDB

Usage:
    This server is launched automatically by the Inventory Agent.
    Direct usage: python -m inventory.mcp_server
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file if it exists
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

mcp = FastMCP("Inventory Service")

# ── Configuration ────────────────────────────────────────────────────────────

# Use environment variable (passed by Docker) or fallback to host.docker.internal for Docker environments
POSTGRES_DSN: str = os.getenv(
    "POSTGRES_DSN",
    "dbname=dealership user=admin password=password123 host=host.docker.internal port=5432",
)
CHROMA_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")


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


# ── RAG Vector Search Tool ───────────────────────────────────────────────────

@mcp.tool()
def search_vehicle_images(query: str, max_results: int = 5) -> str:
    """Search vehicle visual features using semantic vector search.

    Searches ChromaDB embeddings to find vehicles matching visual descriptions
    such as leather seats, dashboard styles, exterior looks, etc.

    Args:
        query: Natural language description of visual features
        max_results: Maximum number of results to return (default: 5)

    Returns:
        JSON string with top matching vehicles, descriptions, and base64-encoded images

    Example:
        >>> search_vehicle_images("leather seats with wood trim", max_results=3)
        '{"results": [{"vin": "ABC123", "type": "interior", "image_base64": "...", "description": "..."}]}'
    """
    import chromadb
    import base64
    import json
    import urllib.request
    from urllib.parse import urlparse

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        collection = client.get_collection("vehicle_visuals")
    except Exception:
        return json.dumps({"error": "Vector collection 'vehicle_visuals' not found. Run setup_rag.py first."})

    # Validate max_results
    n_results = max(1, min(max_results, 20))  # Clamp between 1 and 20
    
    results = collection.query(query_texts=[query], n_results=n_results)

    if not results["ids"] or not results["ids"][0]:
        return json.dumps({"error": "No matching images found."})

    result_list = []
    for idx, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), 1
    ):
        # Extract image URL from metadata
        image_url = meta.get('image_url') or meta.get('path', '')
        image_base64 = ""
        
        # Try to fetch and encode the image
        if image_url:
            try:
                # Handle both local files and URLs
                parsed = urlparse(image_url)
                if parsed.scheme in ('http', 'https'):
                    # Remote URL
                    with urllib.request.urlopen(image_url, timeout=10) as response:
                        image_data = response.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                elif parsed.scheme == 'file' or not parsed.scheme:
                    # Local file path
                    file_path = parsed.path if parsed.scheme == 'file' else image_url
                    with open(file_path, 'rb') as f:
                        image_data = f.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                # If fetching fails, just note the error but continue
                image_base64 = f"ERROR_FETCHING_IMAGE: {str(e)}"
        
        result_list.append({
            "vin": meta.get('vin', '?'),
            "type": meta.get('type', '?'),
            "image_base64": image_base64,
            "description": doc
        })
    
    return json.dumps({"results": result_list})


# ── Bulk & Advanced SQL Tools ────────────────────────────────────────────────
@mcp.tool()
def get_vehicle_details_with_images(vin: str) -> str:
    """Get complete vehicle details combining SQL data and visual images.

    Fetches both structured data (price, specs) and visual content (images)
    for a comprehensive vehicle profile.

    Args:
        vin: Vehicle Identification Number

    Returns:
        JSON with vehicle data and associated images

    Example:
        >>> get_vehicle_details_with_images("ABC123")
        '{"vehicle": {...}, "images": [{"type": "exterior", ...}]}'
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import chromadb
    import base64
    import json
    import urllib.request
    from urllib.parse import urlparse

    # Get SQL data
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM vehicles WHERE vin = %s", (vin,))
        vehicle = cur.fetchone()
        conn.close()
        
        if not vehicle:
            return json.dumps({"error": f"Vehicle with VIN {vin} not found"})
        
        vehicle_dict = dict(vehicle)
    except Exception as exc:
        return json.dumps({"error": f"SQL Error: {exc}"})

    # Get RAG images
    images = []
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection("vehicle_visuals")
        
        # Search for this specific VIN
        results = collection.get(
            where={"vin": vin}
        )
        
        if results["ids"]:
            for idx, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
                image_url = meta.get('image_url') or meta.get('path', '')
                image_base64 = ""
                
                if image_url:
                    try:
                        parsed = urlparse(image_url)
                        if parsed.scheme in ('http', 'https'):
                            with urllib.request.urlopen(image_url, timeout=10) as response:
                                image_data = response.read()
                                image_base64 = base64.b64encode(image_data).decode('utf-8')
                        elif parsed.scheme == 'file' or not parsed.scheme:
                            file_path = parsed.path if parsed.scheme == 'file' else image_url
                            with open(file_path, 'rb') as f:
                                image_data = f.read()
                                image_base64 = base64.b64encode(image_data).decode('utf-8')
                    except Exception:
                        pass
                
                images.append({
                    "type": meta.get('type', '?'),
                    "image_base64": image_base64,
                    "description": doc
                })
    except Exception:
        pass  # RAG data is optional
    
    return json.dumps({"vehicle": vehicle_dict, "images": images})


if __name__ == "__main__":
    mcp.run()