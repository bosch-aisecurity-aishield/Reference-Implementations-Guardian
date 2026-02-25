import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def debug():
    url = "http://localhost:4000/MCPFinance/mcp"
    print(f"1. Attempting to connect to: {url}")
    
    client = MultiServerMCPClient({
        "finance": {
            "transport": "streamable_http",
            "url": url,
            "headers": {"x-litellm-api-key": "Bearer sk-1234"}
        }
    })

    try:
        print("2. Fetching tools...")
        tools = await client.get_tools()
        print(f"3. ✅ Connection Successful!")
        print(f"4. Tools Found ({len(tools)}): {[t.name for t in tools]}")
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")
        print("\nTroubleshooting:")
        print(" - Is 'litellm' running?")
        print(" - Does 'config.yaml' have 'mcp_servers: MCPFinance'?")
        print(" - Is the URL port (4000) correct?")

if __name__ == "__main__":
    asyncio.run(debug())