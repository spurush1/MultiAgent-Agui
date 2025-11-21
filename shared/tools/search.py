import os
from typing import Dict, Any, Optional
import google.generativeai as genai
from .base import SearchInterface

class GoogleSearchTool(SearchInterface):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    @property
    def name(self) -> str:
        return "google_search"

    @property
    def description(self) -> str:
        return "Search the web using Google Search via Gemini Grounding."

    def search(self, query: str) -> Dict[str, Any]:
        if not self.model:
            return {"error": "GOOGLE_API_KEY not set."}
        
        try:
            # Use the tools argument to enable Google Search
            response = self.model.generate_content(
                query,
                tools='google_search_retrieval'
            )
            return {"query": query, "result": response.text}
        except Exception as e:
            return {"error": f"Error searching with Gemini: {e}"}

class BraveSearchTool(SearchInterface):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")

    @property
    def name(self) -> str:
        return "brave_search"

    @property
    def description(self) -> str:
        return "Search the web using Brave Search."

    def search(self, query: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "BRAVE_API_KEY not set."}
        
        # Mock implementation for now as requested by pluggability requirement
        return {
            "query": query, 
            "result": f"[MOCK] Brave Search result for: {query}. (Implement actual API call here)"
        }

def get_search_tool(provider: str = "google") -> SearchInterface:
    """Factory to get the configured search tool."""
    if provider.lower() == "brave":
        return BraveSearchTool()
    return GoogleSearchTool()
