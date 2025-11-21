from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """Abstract base class for all tools in the Tools Hub."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

class SearchInterface(BaseTool):
    """Abstract interface for search providers."""
    
    @abstractmethod
    def search(self, query: str) -> Dict[str, Any]:
        """
        Execute a search query.
        Returns a dictionary with at least 'query' and 'result' (text content).
        """
        pass
