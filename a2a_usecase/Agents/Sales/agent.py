"""Sales Agent – Orchestrator-based LangGraph agent for customer interactions.

This agent acts as the central router (Orchestrator). It analyzes customer intent
and conditionally routes the workflow to specialized worker nodes (Inventory, Finance)
or handles the query directly.

Architecture:
    1. Orchestrator Node: Analyzes input and determines the 'next_step'.
    2. Conditional Edge: Routes based on 'next_step'.
    3. Worker Nodes:
       - Inventory Node: Calls Inventory Agent via A2A + runs Vision Analysis.
       - Finance Node: Calls Finance Agent via A2A.
       - Direct Node: Handles chit-chat/general queries.
    4. Synthesizer Node: Formats specialist data into a warm customer response.

Launch:
    langgraph dev --config sales_langgraph.json --port 5001
"""

from __future__ import annotations

import asyncio
import base64
import json
import io
import os
import uuid
import time
import random
import logging
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union
from typing_extensions import TypedDict

import aiohttp
import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langgraph.config import get_config
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SalesAgent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
SALES_MODEL = os.getenv("SALES_MODEL", "llama3.2")
VISION_MODEL = os.getenv("VISION_MODEL", "llama3.2-vision")

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://host.docker.internal:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")
ROOT_AGENT_IDENTIFIER = os.getenv("ROOT_AGENT_IDENTIFIER", "SalesAgent-Orchestrator")

SPECIALIST_AID = {
    "inventory":  os.getenv("INVENTORY_A2A_AID", "AIS:InventoryAgent"),
    "finance": os.getenv("FINANCE_A2A_AID", "AIS:FinanceAgent"),
}

TRANSIENT_STATUS = {429, 502, 503, 504}

class Context(TypedDict):
    """Configuration context passed at runtime."""
    customer_name: str

@dataclass
class State:
    """Graph state tracking conversation and internal processing data."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    next_step: Literal["inventory", "finance", "direct", "combined"] = "direct"
    current_intent_query: str = ""
    specialist_data: Dict[str, Any] = field(default_factory=dict)
    customer_name: str = ""
    needs_vision: bool = False

_llm = ChatOllama(
    model=SALES_MODEL,
    base_url=OLLAMA_BASE_URL,
    format="json",
    temperature=0,
    disable_streaming=True
)

_vision_llm = ChatOllama(
    model=VISION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    disable_streaming=True
)

ROUTER_PROMPT = """\
You are the Orchestrator for a car dealership sales system.
Analyze the user's latest message and route it to the correct specialist.

ROUTING RULES:
- "inventory": Questions about cars, stock, features, models, specs, or requesting images.
- "finance": Questions about loans, interest rates, monthly payments, approval, or credit.
- "combined": Queries requiring both inventory AND finance information (e.g., "red SUV under 40k with financing").
- "direct": Greetings, chit-chat, or questions unrelated to cars/finance.

OUTPUT FORMAT (JSON ONLY):
{{
    "route": "inventory" | "finance" | "combined" | "direct",
    "query": "<refined search query for the specialist (or original text)>",
    "needs_vision": true | false
}}
"""

DIRECT_RESPONSE_PROMPT = """\
You are a helpful car dealership receptionist. 
The user sent a message that doesn't require a specialist (Inventory/Finance).
Respond warmly and professionally. If they are greeting you, greet them back and ask how you can help with our cars or financing.

Output JSON: {{ "response": "<your text>" }}
"""

SYNTHESIZER_PROMPT = """\
You are a Sales Agent. You have received raw data from a specialist department.
Synthesize this into a final, helpful response for the customer.
If AIShield Content Blocking is detected in the specialist data, respond with a polite message about content restrictions and don't answer anything further.

Customer Query: {query}
Specialist Data: {specialist_text}
Visual Analysis (if any): {vision_text}

Guidelines:
1. Answer the customer's query directly using the data.
2. If Visual Analysis exists, enthusiastically describe the car's looks.
3. Be professional but conversational.
4. If AIShield Content Blocking is detected in the specialist data, respond with a polite message about content restrictions and don't answer anything further.

Output JSON: {{ "response": "<your text>" }}
"""

def _get_aishield_headers() -> dict:
    """Read trace_id from LangGraph config and return headers to forward."""
    try:
        cfg = get_config()
        trace_id = cfg.get("configurable", {}).get("x-aishield-trace-id")
        if trace_id:
            return trace_id
    except Exception:
        pass
    return None

def compress_base64_image(base64_string: str, max_size: int = 800) -> str:
    """
    Decodes a base64 image, resizes it to max_size (width/height), 
    and re-encodes it as a compressed JPEG to reduce string length.
    """
    try:
        if "base64," in base64_string:
            header, data = base64_string.split("base64,", 1)
        else:
            header = None
            data = base64_string

        image_data = base64.b64decode(data)
        img = Image.open(io.BytesIO(image_data))

        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))

        buffer = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        img.save(buffer, format="JPEG", quality=60) 
        compressed_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{compressed_data}"
    except Exception as e:
        print(f"Image compression failed: {e}")
        if header and not base64_string.startswith("data:"):
             return f"{header}base64,{data}"
        return base64_string

def _merge_aishield_headers(headers: Dict[str, str]) -> Dict[str, str]:
    aishield = _get_aishield_headers()
    if not aishield:
        return headers
    if isinstance(aishield, dict):
        headers.update({str(k): str(v) for k, v in aishield.items()})
    else:
        headers["x-aishield-trace-id"] = str(aishield)
    return headers


# async def _a2a_call(base_url: str, text: str, aid: str, api_key: str = None) -> Dict[str, Any]:
#     """Execute A2A JSON-RPC call to a specialist agent."""
#     base = base_url.rstrip("/")
#     headers = {"x-api-key": api_key, "x-caller-agent": ROOT_AGENT_IDENTIFIER} if api_key else {}
#     aishield_headers = _get_aishield_headers()
#     if aishield_headers:
#         headers["x-aishield-trace-id"] = aishield_headers
#         logger.info(f"Forwarding AISHIELD trace ID: {aishield_headers}")
#     try:
#         async with aiohttp.ClientSession() as session:

#             payload = {
#                 "jsonrpc": "2.0",
#                 "id": str(uuid.uuid4()),
#                 "method": "message/send",
#                 "params": {
#                     "message": {
#                         "role": "user",
#                         "parts": [{"kind": "text", "text": text}],
#                         "messageId": str(uuid.uuid4()),
#                     }
#                 },
#             }
            
#             async with session.post(f"{base}/a2a/{aid}", json=payload, headers=headers) as resp:
#                 resp.raise_for_status()
#                 result = await resp.json()

#         response_text = "(No text response)"
#         image_data = []
#         with open("debug_a2a_response.json", "w") as f:
#             json.dump(result, f, indent=2)
        
#         artifacts = result.get("result", {}).get("artifacts", [])

#         status_msg = (result.get("result", {}).get("status") or {}).get("message")
        
#         logger.info(f"[A2A Response] artifacts count: {len(artifacts)}, has status: {bool(status_msg)}")
#         sources = artifacts if artifacts else ([status_msg] if status_msg else [])
#         if sources and isinstance(sources, list) and len(sources) > 0:
#             for idx, source in enumerate(sources):
#                 parts = source.get("parts", []) if source else []
#                 for part_idx, part in enumerate(parts):
#                     part_type = part.get("kind") or part.get("type")
#                     logger.info(f"[A2A Part {idx}.{part_idx}] type={part_type}, keys={list(part.keys())}")
#                     if part_type == "text":
#                         raw_text = part.get("text", "")
#                         logger.info(f"[A2A Part {idx}.{part_idx}] Got text: {len(raw_text)} chars")
                        
#                         try:
#                             parsed = json.loads(raw_text)
#                             if isinstance(parsed, dict) and "images" in parsed:
#                                 response_text = parsed.get("text", "")
#                                 image_data.extend(parsed.get("images", []))
#                                 logger.info(f"[A2A Part {idx}.{part_idx}] ✓ Parsed JSON: text={len(response_text)} chars, images={len(parsed['images'])}")
#                             else:
#                                 response_text = raw_text
#                         except (json.JSONDecodeError, TypeError):
#                             response_text = raw_text
#                     elif part_type == "image_url":
#                         url = part.get("image_url", {}).get("url")
#                         logger.info(f"[A2A Part {idx}.{part_idx}] image_url field: {url[:80] if url else 'None'}...")
#                         if url:
#                             if url.startswith("data:image/"):
#                                 image_data.append(url)
#                                 logger.info(f"[A2A Part {idx}.{part_idx}] ✓ Added base64 data URI")
#                             else:
#                                 image_data.append(url)
#                                 logger.info(f"[A2A Part {idx}.{part_idx}] ✓ Added URL")
#         else:
#             logger.warning(f"No artifacts or status message found in A2A response.")
#             parts = result.get("result", {}).get("message", {}).get("parts", [])
#             parts = json.loads(parts) if isinstance(parts, str) else parts
#             text_parts = []
#             for idx, part in enumerate(parts):
#                 logger.info(f"[A2A Fallback Part {idx}] keys={list(part.keys())}")
#                 part = part.get("text", {}) if isinstance(part, dict) else {}
#                 if "images" in part or "image_url" in part:
#                     part = json.loads(part) if isinstance(part, str) else part
#                     img_part = part.get("images") or part.get("image_url")
#                     for img in (img_part if isinstance(img_part, list) else []):
#                         logger.info(f"[A2A Fallback Part {idx}] Found image: {img[:80]}...")
#                         if img.startswith("data:image/"):
#                             image_data.append(img)
#                             logger.info(f"[A2A Fallback Part {idx}]  Added base64 data URI")
#                         else:
#                             image_data.append(img)
#                             logger.info(f"[A2A Fallback Part {idx}]  Added URL")
#                     if "text" in part:
#                         text = part.get("text") if isinstance(part, dict) else None
#                         if text:
#                             text = json.loads(text) if isinstance(text, str) else text
#                             text_parts.append(text.get("description", "") if isinstance(text, dict) else text)
#                 else:
#                     text_parts.append(part)
#             response_text = "\n".join(text_parts)


#         logger.info(f"[A2A Extraction Complete] text={len(response_text)} chars, images={len(image_data)}")
#         return {"text": response_text, "images": image_data}

#     except Exception as e:
#         logger.error(f"A2A Call Failed: {e}")
#         return {"text": "I'm having trouble connecting to that department right now.", "images": []}

async def _a2a_call(
    base_url: str,
    text: str,
    aid: str,
    api_key: str = None,
    session: Optional[aiohttp.ClientSession] = None,
    max_retries: int = 2,
    timeout_total_s: float = 300.0,
) -> Dict[str, Any]:
    """Execute A2A JSON-RPC call to a specialist agent with timeouts + retries."""
    base = base_url.rstrip("/")

    headers: Dict[str, str] = {}
    if api_key:
        headers.update({
            "x-api-key": api_key,
            "x-caller-agent": ROOT_AGENT_IDENTIFIER,
        })
    headers = _merge_aishield_headers(headers)

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }

    timeout = aiohttp.ClientTimeout(
    total=300.0,
    connect=10.0,
    sock_connect=10.0,
    sock_read=300.0,
)

    owns_session = session is None
    if owns_session:
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    try:
        url = f"{base}/a2a/{aid}"

        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            start = time.perf_counter()
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    # Capture body for debugging on errors
                    raw_body = await resp.text()
                    elapsed = time.perf_counter() - start

                    if resp.status in TRANSIENT_STATUS and attempt < max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            sleep_s = float(retry_after)
                        else:
                            # exp backoff + jitter
                            sleep_s = (2 ** attempt) + random.random()
                        logger.warning(
                            f"A2A transient HTTP {resp.status} from {url} in {elapsed:.2f}s; "
                            f"retrying in {sleep_s:.2f}s (attempt {attempt+1}/{max_retries})"
                        )
                        await asyncio.sleep(sleep_s)
                        continue

                    if resp.status >= 400:
                        logger.error(
                            f"A2A HTTP {resp.status} from {url} in {elapsed:.2f}s. "
                            f"Body (truncated): {raw_body[:2000]}"
                        )
                        resp.raise_for_status()

                    # Normal success path
                    result = json.loads(raw_body)

                response_text = "(No text response)"
                image_data = []

                artifacts = result.get("result", {}).get("artifacts", [])
                status_msg = (result.get("result", {}).get("status") or {}).get("message")

                logger.info(f"[A2A Response] artifacts count: {len(artifacts)}, has status: {bool(status_msg)}")
                sources = artifacts if artifacts else ([status_msg] if status_msg else [])

                if sources and isinstance(sources, list) and len(sources) > 0:
                    for idx, source in enumerate(sources):
                        parts = source.get("parts", []) if source else []
                        for part_idx, part in enumerate(parts):
                            part_type = part.get("kind") or part.get("type")
                            logger.info(f"[A2A Part {idx}.{part_idx}] type={part_type}, keys={list(part.keys())}")

                            if part_type == "text":
                                raw_text = part.get("text", "")
                                logger.info(f"[A2A Part {idx}.{part_idx}] Got text: {len(raw_text)} chars")

                                try:
                                    parsed = json.loads(raw_text)
                                    if isinstance(parsed, dict) and "images" in parsed:
                                        response_text = parsed.get("text", "")
                                        image_data.extend(parsed.get("images", []))
                                        logger.info(
                                            f"[A2A Part {idx}.{part_idx}] ✓ Parsed JSON: "
                                            f"text={len(response_text)} chars, images={len(parsed['images'])}"
                                        )
                                    else:
                                        response_text = raw_text
                                except (json.JSONDecodeError, TypeError):
                                    response_text = raw_text

                            elif part_type == "image_url":
                                url = part.get("image_url", {}).get("url")
                                logger.info(f"[A2A Part {idx}.{part_idx}] image_url field: {url[:80] if url else 'None'}...")
                                if url:
                                    image_data.append(url)

                else:
                    logger.warning("No artifacts or status message found in A2A response.")
                    parts = result.get("result", {}).get("message", {}).get("parts", [])
                    parts = json.loads(parts) if isinstance(parts, str) else parts
                    text_parts = []

                    for idx, part in enumerate(parts):
                        logger.info(f"[A2A Fallback Part {idx}] keys={list(part.keys())}")
                        part = part.get("text", {}) if isinstance(part, dict) else {}
                        if "images" in part or "image_url" in part:
                            pprint("*" * 20)
                            print()
                            pprint(part)
                            print()
                            pprint("*" * 20)
                            with open("part.txt", "w+") as f:
                                f.write(part)
                            part = json.loads(part) if isinstance(part, str) else part
                            img_part = part.get("images") or part.get("image_url")
                            for img in (img_part if isinstance(img_part, list) else []):
                                image_data.append(img)
                            if "text" in part:
                                t = part.get("text")
                                if t:
                                    t = json.loads(t) if isinstance(t, str) else t
                                    text_parts.append(t.get("description", "") if isinstance(t, dict) else t)
                        else:
                            text_parts.append(part)

                    response_text = "\n".join([p for p in text_parts if p])

                logger.info(f"[A2A Extraction Complete] text={len(response_text)} chars, images={len(image_data)}")
                return {"text": response_text, "images": image_data}

            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError, aiohttp.ClientResponseError) as e:
                elapsed = time.perf_counter() - start
                last_err = e
                if attempt < max_retries:
                    sleep_s = (2 ** attempt) + random.random()
                    logger.warning(
                        f"A2A exception {type(e).__name__} after {elapsed:.2f}s; "
                        f"retrying in {sleep_s:.2f}s (attempt {attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                break

        logger.error(f"A2A Call Failed after retries: {last_err}")
        return {"text": "I'm having trouble connecting to that department right now.", "images": []}

    finally:
        if owns_session:
            await session.close()

async def _analyze_images(image_data: List[str], query: str) -> str:
    """Run Vision LLM on images if present.
    
    Args:
        image_data: List of base64 data URIs or URLs
        query: User's query for context
    """
    if not image_data:
        return ""
    
    descriptions = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i, img in enumerate(image_data[:2]):
            try:
                if img.startswith("data:image/"):
                    data_uri = img
                else:
                    resp = await client.get(img)
                    resp.raise_for_status()
                    b64_img = base64.b64encode(resp.content).decode('utf-8')
                    data_uri = f"data:image/jpeg;base64,{b64_img}"
                
                msg = HumanMessage(content=[
                    {"type": "text", "text": f"Briefly describe this car feature relevant to: '{query}'"},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ])
                
                res = await _vision_llm.ainvoke([msg])
                descriptions.append(res.content)
            except Exception as e:
                logger.warning(f"Vision failed for image {i}: {e}")

    return "\n".join(descriptions)

async def orchestrator_node(state: State) -> Dict[str, Any]:
    """Analyzes intent and routes the conversation."""
    last_msg = state.messages[-1]["content"] if state.messages else ""
    
    logger.info(f"Orchestrator analyzing: {last_msg}")
    
    try:
        response = await _llm.ainvoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_msg)
        ])
        decision = json.loads(response.content)
        logger.info(f"Route decision: {decision.get('route')}, needs_vision: {decision.get('needs_vision', False)}")
    except Exception as e:
        logger.warning(f"Router parsing failed: {e}")
        decision = {"route": "direct", "query": last_msg, "needs_vision": False}
        
    return {
        "next_step": decision.get("route", "direct"),
        "current_intent_query": decision.get("query", last_msg),
        "needs_vision": decision.get("needs_vision", False)
    }

async def inventory_node(state: State) -> Dict[str, Any]:
    """Worker: Handles Inventory A2A calls."""
    logger.info("Executing Inventory Node")
    data = await _a2a_call(LITELLM_BASE_URL, state.current_intent_query, aid=SPECIALIST_AID["inventory"], api_key=LITELLM_API_KEY)
    logger.info(f"Received from Inventory: text_length={len(data.get('text', ''))}, images_count={len(data.get('images', []))}")
    if data.get('images'):
        logger.info(f"First image preview: {data['images'][0][:100]}...")
    return {"specialist_data": data}

async def finance_node(state: State) -> Dict[str, Any]:
    """Worker: Handles Finance A2A calls."""
    logger.info("Executing Finance Node")
    data = await _a2a_call(LITELLM_BASE_URL, state.current_intent_query, aid=SPECIALIST_AID["finance"], api_key=LITELLM_API_KEY)
    return {"specialist_data": data}

async def combined_node(state: State) -> Dict[str, Any]:
    """Worker: Handles queries requiring both Inventory AND Finance."""
    logger.info("Executing Combined Node - parallel specialist calls")
    
    inventory_task = _a2a_call(LITELLM_BASE_URL, state.current_intent_query, aid=SPECIALIST_AID["inventory"], api_key=LITELLM_API_KEY)
    finance_task = _a2a_call(LITELLM_BASE_URL, state.current_intent_query, aid=SPECIALIST_AID["finance"], api_key=LITELLM_API_KEY)
    
    inventory_data, finance_data = await asyncio.gather(inventory_task, finance_task)
    
    logger.info(f"Combined results: inventory={len(inventory_data.get('text', ''))}, finance={len(finance_data.get('text', ''))}")
    
    combined_text = f"Vehicle Information:\n{inventory_data.get('text', '')}\n\nFinancing Options:\n{finance_data.get('text', '')}"
    
    return {
        "specialist_data": {
            "text": combined_text,
            "images": inventory_data.get("images", [])
        }
    }

async def direct_node(state: State) -> Dict[str, Any]:
    """Worker: Handles General/Direct queries internally."""
    logger.info("Executing Direct Node")
    last_msg = state.messages[-1]["content"]
    
    try:
        response = await _llm.ainvoke([
            SystemMessage(content=DIRECT_RESPONSE_PROMPT),
            HumanMessage(content=last_msg)
        ])
        content = json.loads(response.content).get("response", "How can I help you?")
    except:
        content = "I'm here to help with cars and financing. What do you need?"
        
    return {
        "messages": [AIMessage(content=content)]
    }

async def synthesizer_node(state: State) -> Dict[str, Any]:
    """Synthesizes specialist data + vision analysis into a final response."""
    logger.info("Executing Synthesizer Node")
    
    data = state.specialist_data
    images = data.get("images", [])
    text_data = data.get("text", "")
    
    logger.info(f"Synthesizer received: text_length={len(text_data)}, images_count={len(images)}")
    
    vision_text = ""
    if images:
        logger.info(f"Running vision analysis on {len(images)} images")
        vision_text = await _analyze_images(images, state.current_intent_query)
        logger.info(f"Vision analysis result length: {len(vision_text)}")
    else:
        logger.info("No images to analyze")
        
    prompt = SYNTHESIZER_PROMPT.format(
        query=state.current_intent_query,
        specialist_text=text_data,
        vision_text=vision_text
    )
    
    try:
        response = await _llm.ainvoke([HumanMessage(content=prompt)])
        final_text = json.loads(response.content).get("response", str(response.content))
    except:
        final_text = text_data
        
    if images:
        markdown_content = final_text + "\n\n"
        
        for i, img in enumerate(images):
            # Ensure proper prefix for base64 if missing
            if not img.startswith("http") and not img.startswith("data:"):
                img = f"data:image/jpeg;base64,{img}"
            if img.startswith("http"):
                clean_img = img
            else:
                # 2. Compress the Base64 string
                clean_img = await asyncio.to_thread(compress_base64_image, img)
            
            # Format as Markdown Image
            markdown_content += f"![Vehicle Image {i+1}]({clean_img})\n\n"

        # Return as a single string message (bypasses LiteLLM list error)
        return {
            "messages": [AIMessage(content=markdown_content)]
        }
    else:
        return {
            "messages": [AIMessage(content=final_text)]
        }


async def route_request(state: State) -> str:
    """Conditional Edge Logic: Determines which node to visit next."""
    route_map = {
        "inventory": "inventory_agent",
        "finance": "finance_agent",
        "combined": "combined_agent",
        "direct": "direct_responder"
    }
    next_node = route_map.get(state.next_step, "direct_responder")
    logger.info(f"Routing to: {next_node}")
    return next_node



builder = StateGraph(State, context_schema=Context)

builder.add_node("orchestrator", orchestrator_node)
builder.add_node("inventory_agent", inventory_node)
builder.add_node("finance_agent", finance_node)
builder.add_node("combined_agent", combined_node)
builder.add_node("direct_responder", direct_node)
builder.add_node("synthesizer", synthesizer_node)

builder.add_edge(START, "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    route_request,
    {
        "inventory_agent": "inventory_agent",
        "finance_agent": "finance_agent",
        "combined_agent": "combined_agent",
        "direct_responder": "direct_responder"
    }
)

builder.add_edge("inventory_agent", "synthesizer")
builder.add_edge("finance_agent", "synthesizer")
builder.add_edge("combined_agent", "synthesizer")
builder.add_edge("synthesizer", END)
builder.add_edge("direct_responder", END)

graph = builder.compile()