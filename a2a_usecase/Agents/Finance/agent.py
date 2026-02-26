"""
Finance Agent – ReAct-based autonomous agent with MCP tool integration.

This agent provides financial analysis and loan calculations using tools 
dynamically discovered via the Model Context Protocol (MCP). It features 
a specialized security router to intercept and block forbidden content 
(virus detections) from tool responses.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Annotated, TypedDict, List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("FinanceAgent")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
FINANCE_MODEL = os.getenv("FINANCE_MODEL", "llama3.2")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")
FINANCE_MCP_SERVER_URL = os.getenv("FINANCE_MCP_SERVER_URL", "http://host.docker.internal:4000/MCPFinance/mcp")



class AgentState(TypedDict):
    """
    State schema for the finance agent.
    
    Attributes:
        messages: Ongoing list of conversation messages.
    """
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]



def get_mcp_tools():
    """
    Synchronously fetches tools from the MCP server.
    
    This function handles the event loop lifecycle to ensure tool discovery 
    succeeds even when called from within non-async thread pools used 
    by the LangGraph runtime.
    
    Returns:
        List of discovered LangChain tools.
    """
    server_config = {
        "finance": {
            "transport": "streamable_http",
            "url": FINANCE_MCP_SERVER_URL,
            "headers": {"x-litellm-api-key": f"Bearer {LITELLM_API_KEY}"}
        }
    }
    
    async def discover():
        client = MultiServerMCPClient(server_config)
        return await client.get_tools()

    try:
        return asyncio.run(discover())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(discover())
    except Exception as e:
        logger.error(f"Critical failure during MCP tool discovery: {e}")
        return []



mcp_tools = get_mcp_tools()
llm = ChatOllama(
    model=FINANCE_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
).bind_tools(mcp_tools)



def call_model(state: AgentState):
    """
    Passes the current state to the LLM to determine the next message or tool call.
    
    Returns:
        Update to the state messages.
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def router(state: AgentState):
    """
    Routes the execution flow based on the most recent message.
    
    Implements security logic to terminate the graph immediately if 
    'Forbidden content: virus found' is detected in a tool response.
    
    Returns:
        Destination node name or END.
    """
    last_message = state["messages"][-1]
    
    if isinstance(last_message, ToolMessage):
        if "AIShield Content blocked" in str(last_message.content):
            logger.warning("Security Interceptor: Virus detected. Stopping graph.")
            return END
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return END



builder = StateGraph(AgentState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(mcp_tools))

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools", 
        END: END
    }
)

builder.add_conditional_edges(
    "tools",
    router,
    {
        "agent": "agent", 
        END: END
    }
)

graph = builder.compile()