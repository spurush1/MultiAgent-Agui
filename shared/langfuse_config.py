"""
Shared Langfuse configuration and helper utilities for all agents.
"""
import os
from langfuse import Langfuse

# Initialize Langfuse client
def get_langfuse_client():
    """Get configured Langfuse client"""
    return Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-secret"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-public"),
        host=os.getenv("LANGFUSE_HOST", "http://langfuse:3000"),
    )

# Common configuration
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
