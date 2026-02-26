import os
import hashlib
from dotenv import load_dotenv
from langgraph_sdk import Auth

# Load environment variables
load_dotenv()

auth = Auth()

EXPECTED_API_KEY = os.getenv("API_KEY", "my-secret-key-123")

@auth.authenticate
async def authenticate(headers: dict):
    """
    Check if the x-api-key header matches our secret.
    """
    user_key_bytes = headers.get(b'x-api-key') or headers.get(b'authorization') or \
                     headers.get('x-api-key') or headers.get('authorization')
    
    if isinstance(user_key_bytes, bytes):
        user_key = user_key_bytes.decode('utf-8')
    else:
        user_key = user_key_bytes

    # 2. Validate
    if user_key and (user_key == EXPECTED_API_KEY or user_key == f"Bearer {EXPECTED_API_KEY}"):
        key_hash = hashlib.sha256(user_key.encode()).hexdigest()[:16]
        
        return {
            "identity": f"api-key-{key_hash}",
            "is_authenticated": True,
            "display_name": f"API User ({key_hash})",
            "permissions": ["admin"] # Optional, but useful
        }
    
    # 3. Fail
    raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid API Key")