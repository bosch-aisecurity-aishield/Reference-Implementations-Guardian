"""Inventory Agent – Manual routing with MCP tool integration.

This agent searches vehicle inventory using MCP-exposed tools via manual routing.
The LLM determines the query type (SQL/RAG/combined) and the graph routes accordingly.

MCP Tools (from mcp_server.py):
    - run_sql_query: Execute read-only SQL queries
    - search_vehicle_images: Semantic visual search with max_results parameter
    - bulk_vehicle_search: Multi-filter search (make, model, color, price range, year, status)
    - get_vehicle_statistics: Aggregated counts and prices by dimension
    - get_available_makes_models: List all available makes and their models
    - search_by_price_range: Optimized price-based search
    - get_vehicle_details_with_images: Complete vehicle profile (SQL+RAG combined)

Architecture:
    - Uses manual StateGraph routing (compatible with qwen2.5-coder)
    - Router node analyzes intent and determines route
    - Specialized nodes call appropriate MCP tools
    - Synthesizer node formats final response
    - Works with models that don't support structured tool calling

Environment Variables:
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    INVENTORY_MODEL: LLM model name (default: qwen2.5-coder)

Launch:
    langgraph dev --config langgraph.json --port 2025
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# ── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("InventoryAgent")

# ── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
INVENTORY_MODEL = os.getenv("INVENTORY_MODEL", "qwen2.5-coder")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "your_litellm_api_key_here")
INVENTORY_MCP_SERVER_URL = os.getenv("INVENTORY_MCP_SERVER_URL", "http://localhost:8000")

# ── State Definition ─────────────────────────────────────────────────────────

class Context:
    """Runtime configuration context."""
    pass

@dataclass
class State:
    """Graph state for inventory agent workflow."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    route: str = "direct"
    query: str = ""
    user_intent: str = ""
    sql_query: str = ""
    rag_query: str = ""
    sql_data: str = ""
    rag_data: str = ""
    images: List[str] = field(default_factory=list)


# ── LLM Setup ────────────────────────────────────────────────────────────────

_llm = ChatOllama(
    model=INVENTORY_MODEL,
    base_url=OLLAMA_BASE_URL,
    format="json",
    temperature=0,
    disable_streaming=True
)

ROUTER_PROMPT = """\
You are an Inventory Agent router. Analyze the user's request and determine the best execution path.

DATABASE SCHEMA:
Table: vehicles
Columns:
  - vin (VARCHAR, PRIMARY KEY): Vehicle Identification Number
  - make (VARCHAR): Manufacturer (e.g., 'Toyota', 'Honda', 'Ford', 'Chevrolet')
  - model (VARCHAR): Model name (e.g., 'Camry', 'Civic', 'F-150', 'Silverado')
  - year (INTEGER): Model year (e.g., 2020, 2021, 2022, 2023)
  - color (VARCHAR): Exterior color (e.g., 'Red', 'Blue', 'White', 'Black', 'Silver')
  - price (NUMERIC): Price in dollars (e.g., 25000.00, 35999.99)
  - status (VARCHAR): Availability ('AVAILABLE', 'SOLD', 'RESERVED')

SQL TIPS:
- Use ILIKE for case-insensitive string matching (e.g., WHERE color ILIKE 'red')
- Use BETWEEN for price ranges (e.g., price BETWEEN 20000 AND 30000)

ROUTES:
1. **sql_only**: Factual queries about price, color, model, year, VIN, availability, stock
   Query format: Complete SQL SELECT statement using the vehicles table
   Example: "SELECT * FROM vehicles WHERE price < 30000 AND color ILIKE 'red' LIMIT 10"

2. **rag_only**: Visual/descriptive features like leather seats, dashboard design, sporty look
   Query format: Natural language (e.g., "modern dashboard with touchscreen")

3. **images_only**: Direct image requests without needing descriptions ("show me pictures", "images of")
   Query format: Vehicle/feature description (e.g., "Honda Civic exterior")

4. **combined**: Complex queries needing both data AND visuals (e.g., "red SUV under 40k with leather interior")
   Query format: JSON with "sql" and "rag" keys
   Example: {{"sql": "SELECT * FROM vehicles WHERE price <= 40000 AND color ILIKE '%red%' LIMIT 10", "rag": "SUV with leather interior"}}

5. **direct**: Simple questions answerable without lookup (greetings, general info)
   Query format: Original text

Respond ONLY with JSON:
{{
  "route": "sql_only" | "rag_only" | "images_only" | "combined" | "direct",
  "query": "<formatted query based on route>",
  "sql_query": "<Complete SQL SELECT statement if route is combined>",
  "rag_query": "<RAG search phrase if route is combined>"
}}
"""

SQL_SUMMARIZE_PROMPT = """\
You are an inventory assistant. Convert this database result into a customer-friendly response.
Note: Data comes from vehicles table with columns: vin, make, model, year, color, price, status
Data: {data}

Respond ONLY with JSON: {{ "message": "<your text>" }}
"""

RAG_SUMMARIZE_PROMPT = """\
You are an inventory assistant. Describe these vehicle features to the customer.

Features: {data}
Image Count: {image_count}

Respond ONLY with JSON: {{ "message": "<your text>" }}
"""

COMBINED_SUMMARIZE_PROMPT = """\
You are an inventory assistant. Combine factual data and visual features into one response.
Note: SQL data is from vehicles table (vin, make, model, year, color, price, status)
      Visual data is from image search describing vehicle features
Database Info: {sql_data}
Visual Features: {rag_data}
Images Available: {image_count}

Respond ONLY with JSON: {{ "message": "<your text>" }}
"""

DIRECT_RESPONSE_PROMPT = """\
You are a helpful car dealership inventory assistant.
Answer this general question directly without database lookup.

Question: {query}

Respond ONLY with JSON: {{ "message": "<your text>" }}
"""

server_config = {
    "finance": {
        "transport": "streamable_http",
        "url": INVENTORY_MCP_SERVER_URL,
        "headers": {"x-litellm-api-key": f"Bearer {LITELLM_API_KEY}"}
    }
}

# ── Core Agent Logic ─────────────────────────────────────────────────────────
async def _call_mcp_sql(query: str) -> str:
    """Execute SQL query via MCP server over streamable HTTP.

    Args:
        query: Complete SQL SELECT statement

    Returns:
        JSON string with query results
    """
    cfg = server_config["finance"]
    print(f"Calling MCP SQL with query: {query}")

    try:
        async with streamable_http_client(
            url=cfg["url"],
            headers=cfg.get("headers", {})
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                res = await session.call_tool(
                    "run_sql_query",
                    arguments={"query": query}
                )
                print(f"MCP SQL response: {res}")

                if getattr(res, "content", None) and hasattr(res.content[0], "text"):
                    return res.content[0].text
                return str(res)

    except Exception as e:
        logger.error(f"MCP SQL call failed: {e}")
        return json.dumps({"error": str(e)})


async def _call_mcp_rag(query: str, max_results: int = 5) -> Tuple[str, List[str]]:
    """Execute RAG vector search via MCP server over streamable HTTP.

    Args:
        query: Natural language search phrase
        max_results: Maximum number of image results (1-20)

    Returns:
        Tuple of (raw_data_json, list_of_base64_images)
    """
    cfg = server_config["finance"]

    try:
        async with streamable_http_client(
            url=cfg["url"],
            headers=cfg.get("headers", {})
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                res = await session.call_tool(
                    "search_vehicle_images",
                    arguments={"query": query, "max_results": max_results}
                )

                raw_data = (
                    res.content[0].text
                    if getattr(res, "content", None) and hasattr(res.content[0], "text")
                    else str(res)
                )

                images: List[str] = []
                try:
                    data_json = json.loads(raw_data)
                    if "results" in data_json:
                        for result in data_json["results"]:
                            img_b64 = result.get("image_base64", "")
                            if img_b64 and not img_b64.startswith("ERROR"):
                                images.append(img_b64)
                except json.JSONDecodeError:
                    pass

                return raw_data, images

    except Exception as e:
        logger.error(f"MCP RAG call failed: {e}")
        return json.dumps({"error": str(e)}), []

async def router_node(state: State) -> Dict[str, Any]:
    """Analyze user intent and determine execution route.
    
    Returns:
        Updated state with route, query, and user_intent
    """
    if not state.messages:
        return {"route": "direct", "query": "", "user_intent": ""}
    
    latest_message = state.messages[-1]
    user_text = latest_message.content if hasattr(latest_message, "content") else latest_message.get("content", "")
    
    logger.info(f"Router analyzing: {user_text[:100]}")
    
    decision_response = await _llm.ainvoke([
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": user_text},
    ])
    
    try:
        decision = json.loads(decision_response.content)
    except json.JSONDecodeError:
        decision = {"route": "direct", "query": user_text}
    
    route = decision.get("route", "direct")
    query = decision.get("query", user_text)
    
    logger.info(f"Route decision: {route}, Query: {query[:100] if isinstance(query, str) else query}")
    
    result = {
        "route": route,
        "query": query,
        "user_intent": user_text
    }
    
    if route == "combined":
        result["sql_query"] = decision.get("sql_query", "")
        result["rag_query"] = decision.get("rag_query", "")
    
    return result


async def sql_node(state: State) -> Dict[str, Any]:
    """Execute SQL database query.
    
    Returns:
        Updated state with sql_data
    """
    logger.info(f"SQL Node executing: {state.query[:100]}")
    
    try:
        sql_data = await asyncio.wait_for(
            _call_mcp_sql(state.query),
            timeout=30.0
        )
        logger.debug(f"SQL returned {len(sql_data)} chars")
    except asyncio.TimeoutError:
        sql_data = '{"error": "Query timed out"}'
        logger.warning("SQL query timed out")
    except Exception as exc:
        logger.error(f"SQL query failed: {exc}")
        sql_data = f'{{"error": "{str(exc)}"}}'
    
    return {"sql_data": sql_data}


async def rag_node(state: State) -> Dict[str, Any]:
    """Execute RAG vector search.
    
    Returns:
        Updated state with rag_data and images
    """
    logger.info(f"RAG Node executing: {state.query[:100]}")
    
    try:
        rag_data, images = await asyncio.wait_for(
            _call_mcp_rag(state.query),
            timeout=45.0
        )
        logger.info(f"RAG returned {len(rag_data)} chars, {len(images)} images")
    except asyncio.TimeoutError:
        rag_data = '{"error": "Search timed out"}'
        images = []
        logger.warning("RAG search timed out")
    except Exception as exc:
        logger.error(f"RAG search failed: {exc}")
        rag_data = f'{{"error": "{str(exc)}"}}'
        images = []
    
    return {"rag_data": rag_data, "images": images}


async def images_only_node(state: State) -> Dict[str, Any]:
    """Retrieve only images without text summarization.
    
    Returns:
        Updated state with images and needs_summary=False
    """
    logger.info(f"Images-Only Node executing: {state.query[:100]}")
    
    try:
        _, images = await asyncio.wait_for(
            _call_mcp_rag(state.query),
            timeout=45.0
        )
        logger.info(f"Images-Only returned {len(images)} images")
    except Exception as exc:
        logger.error(f"Image retrieval failed: {exc}")
        images = []
    
    return {"images": images, "needs_summary": False}


async def combined_node(state: State) -> Dict[str, Any]:
    """Execute both SQL and RAG queries in parallel.
    
    Returns:
        Updated state with sql_data, rag_data, and images
    """
    logger.info("Combined Node executing SQL + RAG")
    
    sql_query = state.sql_query or state.query
    rag_query = state.rag_query or state.query
    
    # Handle dict query from router (combined route returns a dict)
    if isinstance(sql_query, dict):
        sql_query = sql_query.get('sql', '')
    if isinstance(rag_query, dict):
        rag_query = rag_query.get('rag', '')
    
    logger.info(f"SQL Query: {str(sql_query)[:100]}, RAG Query: {str(rag_query)[:100]}")
    
    sql_task = _call_mcp_sql(sql_query)
    rag_task = _call_mcp_rag(rag_query)
    
    try:
        results = await asyncio.gather(
            asyncio.wait_for(sql_task, timeout=30.0),
            asyncio.wait_for(rag_task, timeout=45.0),
            return_exceptions=True
        )
        
        sql_data = results[0] if not isinstance(results[0], Exception) else '{"error": "SQL failed"}'
        rag_result = results[1] if not isinstance(results[1], Exception) else ('{"error": "RAG failed"}', [])
        
        rag_data, images = rag_result if isinstance(rag_result, tuple) else (str(rag_result), [])
        
        logger.info(f"Combined returned: SQL={len(sql_data)} chars, RAG={len(rag_data)} chars, Images={len(images)}")
    except Exception as exc:
        logger.error(f"Combined query failed: {exc}")
        sql_data = '{"error": "Combined query failed"}'
        rag_data = '{"error": "Combined query failed"}'
        images = []
    
    return {"sql_data": sql_data, "rag_data": rag_data, "images": images}


async def direct_node(state: State) -> Dict[str, Any]:
    """Handle simple queries without database lookup.
    
    Returns:
        Final response message
    """
    logger.info("Direct Node responding")
    
    prompt = DIRECT_RESPONSE_PROMPT.format(query=state.user_intent or state.query)
    
    try:
        response = await _llm.ainvoke([
            {"role": "system", "content": prompt}
        ])
        content = json.loads(response.content).get("message", "How can I help you find a vehicle?")
    except:
        content = "I'm here to help you search our inventory. What are you looking for?"
    
    return {"messages": [AIMessage(content=content)]}


async def summarizer_node(state: State) -> Dict[str, Any]:
    """Summarize results based on route type.
    
    Returns:
        Final response message with optional images
    """
    logger.info(f"Summarizer Node: route={state.route}, images={len(state.images)}")
    
    if state.route == "sql_only":
        prompt = SQL_SUMMARIZE_PROMPT.format(data=state.sql_data)
    elif state.route == "rag_only":
        prompt = RAG_SUMMARIZE_PROMPT.format(data=state.rag_data, image_count=len(state.images))
    elif state.route == "combined":
        prompt = COMBINED_SUMMARIZE_PROMPT.format(
            sql_data=state.sql_data,
            rag_data=state.rag_data,
            image_count=len(state.images)
        )
    elif state.route == "images_only":
        if state.images:
            image_uris = [f"data:image/jpeg;base64,{img}" for img in state.images[:5]]
            response_payload = {
                "text": f"Here are {len(image_uris)} images matching your request.",
                "images": image_uris
            }
            return {"messages": [AIMessage(content=json.dumps(response_payload))]}
        else:
            return {"messages": [AIMessage(content="No images found matching your request.")]}
    else:
        return {"messages": [AIMessage(content="Unable to process request.")]}
    
    try:
        response = await _llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": state.user_intent},
        ])
        try:
            text_reply = json.loads(response.content).get("message", str(response.content))
        except:
            text_reply = str(response.content)
    except Exception as e:
        logger.error(f"Summarizer LLM failed: {e}")
        text_reply = "Here is the inventory data."
    
    image_uris = []
    if state.images:
        for img in state.images[:2]:
            if img.startswith("data:image"):
                image_uris.append(img)
            else:
                image_uris.append(f"data:image/jpeg;base64,{img}")
        response_payload = {
            "text": text_reply,
            "images": image_uris
        }
        final_message = AIMessage(content=json.dumps(response_payload))
        logger.info(f"Returning response with {len(image_uris)} images")
    else:
        final_message = AIMessage(content=text_reply)
        logger.debug("Returning text-only response")
    
    return {"messages": [final_message]}



# ── Conditional Edge Logic ──────────────────────────────────────────────────

def route_from_router(state: State) -> str:
    """Determine next node based on router decision.
    
    Returns:
        Node name to execute next
    """
    route_map = {
        "sql_only": "sql_executor",
        "rag_only": "rag_executor",
        "images_only": "images_retriever",
        "combined": "combined_executor",
        "direct": "direct_responder"
    }
    
    next_node = route_map.get(state.route, "direct_responder")
    logger.info(f"Routing to: {next_node}")
    return next_node


def should_summarize(state: State) -> str:
    """Determine if summarization is needed.
    
    Returns:
        "summarize" or "end"
    """
    if state.needs_summary:
        return "summarize"
    else:
        return "end"


# ── Graph Definition ─────────────────────────────────────────────────────────

inventory_retry_policy = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=30.0,
    max_attempts=3,
    jitter=True,
)

builder = StateGraph(State)

builder.add_node("router", router_node)
builder.add_node("sql_executor", sql_node)
builder.add_node("rag_executor", rag_node)
builder.add_node("images_retriever", images_only_node)
builder.add_node("combined_executor", combined_node)
builder.add_node("direct_responder", direct_node)
builder.add_node("summarizer", summarizer_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route_from_router,
    {
        "sql_executor": "sql_executor",
        "rag_executor": "rag_executor",
        "images_retriever": "images_retriever",
        "combined_executor": "combined_executor",
        "direct_responder": "direct_responder"
    }
)

builder.add_edge("sql_executor", "summarizer")
builder.add_edge("rag_executor", "summarizer")
builder.add_edge("combined_executor", "summarizer")

builder.add_conditional_edges(
    "images_retriever",
    should_summarize,
    {
        "summarize": "summarizer",
        "end": END
    }
)

builder.add_edge("direct_responder", END)
builder.add_edge("summarizer", END)

graph = builder.compile()