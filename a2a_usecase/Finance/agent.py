"""Finance Agent – ReAct-based autonomous agent with MCP tool auto-discovery.

This agent autonomously handles loan calculations using MCP-exposed tools.
The LLM dynamically decides which tools to call based on user queries.

MCP Tools (Auto-discovered from mcp_server.py):
    - calculate_monthly_payment: Single loan scenario calculation
    - compare_loan_options: Multi-scenario comparison (rates × terms)
    - calculate_affordability: Max price from monthly budget (reverse calculation)
    - get_rate_recommendations: Credit score-based rate guidance
    - calculate_total_cost_comparison: Full breakdown with down payment

Architecture:
    - Uses create_react_agent() for autonomous tool selection
    - LLM sees all MCP tools and chooses appropriate ones
    - No manual parameter extraction needed
    - Fully autonomous decision-making

Environment Variables:
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    FINANCE_MODEL: LLM model name (default: qwen2.5-coder)

Launch:
    langgraph dev --config langgraph.json --port 2024
"""

from __future__ import annotations
from pprint import pprint
import logging
import os
import asyncio

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("FinanceAgent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
FINANCE_MODEL = os.getenv("FINANCE_MODEL", "llama3.2")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")  #\

llm = ChatOllama(
    model=FINANCE_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    disable_streaming=True
)

mcp_client = None

async def load_mcp_tools():
    """Connect to the MCP server and fetch tools dynamically."""
    global mcp_client
    server_config = {
        "finance": {
            "transport": "streamable_http",  
            "url": "http://host.docker.internal:4000/MCPFinance/mcp",
            "headers": {
                "x-litellm-api-key": f"Bearer {LITELLM_API_KEY}"
            }
        }
    }
    try:
        mcp_client = MultiServerMCPClient(server_config)
        tools = await mcp_client.get_tools()
        logger.info(f"Discovered {len(tools)} tools via MCP:")
        for tool in tools:
            logger.info(f" - {tool.name}: {pprint(tool.args_schema)}")
        return tools
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        return []

    
try:
    tools = asyncio.run(load_mcp_tools())
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    tools = []


graph = create_react_agent(
    llm,
    tools=tools,
)

logger.info("Finance ReAct Agent initialized with MCP auto-discovery")
