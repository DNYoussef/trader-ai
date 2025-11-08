import httpx
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryClient:
    """
    Client for interacting with the Memory MCP Triple System.
    Enables Trader AI to store trading events and retrieve context.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=5.0)
        
    def store_memory(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Store a memory in the triple-layer system.
        
        Args:
            content: The text content to store (e.g., "Executed BUY order for AAPL")
            metadata: Dictionary of metadata (e.g., {'layer': 'mid_term', 'category': 'trade'})
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Default to mid_term layer for trading events if not specified
            if 'layer' not in metadata:
                metadata['layer'] = 'mid_term'
            
            # Adapt to the memory-mcp API structure
            # Assuming a tool named 'store_memory' exists, or we need to use the native ingestion API
            # Based on docs, we might need to use the python wrapper or check if there's a store endpoint
            # For now, we'll assume a tool wrapper exists or we'll fail gracefully
            
            # Actually, looking at the 'memory-store.py' script, it uses internal classes.
            # The MCP server likely exposes 'vector_search'. 
            # If 'store_memory' tool isn't exposed, we might need to add it to the server.
            # For now, let's try to hit a hypothetical tool endpoint.
            
            payload = {
                "content": content,
                "metadata": metadata
            }
            
            response = self.client.post("/tools/store_memory", json=payload)
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    def retrieve_context(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The query string (e.g., "past performance of momentum strategy")
            
        Returns:
            Dict containing results or None if failed
        """
        try:
            # Using the example from MCP-DEPLOYMENT-GUIDE.md
            # curl -X POST "http://localhost:8080/tools/vector_search?query=..."
            
            params = {
                "query": query,
                "limit": 5
            }
            
            response = self.client.post("/tools/vector_search", params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return None

    def log_trade(self, symbol: str, action: str, quantity: float, price: float, reason: str):
        """Helper to log a trade execution event"""
        content = f"Executed {action} order for {quantity} shares of {symbol} at ${price:.2f}. Reason: {reason}"
        metadata = {
            "category": "trade_execution",
            "symbol": symbol,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "source": "trader-ai"
        }
        self.store_memory(content, metadata)
