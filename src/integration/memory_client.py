"""
ISS-014: Async Memory Client for Memory MCP Triple System.

Fixed to use AsyncClient with proper timeouts and graceful degradation.
"""
import httpx
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryClient:
    """
    Async client for interacting with the Memory MCP Triple System.
    Enables Trader AI to store trading events and retrieve context.

    ISS-014 FIX: Uses AsyncClient with timeouts to prevent blocking.
    """

    def __init__(self, base_url: str = "http://localhost:8080", enabled: bool = True):
        """
        Initialize memory client.

        Args:
            base_url: Base URL for memory MCP service
            enabled: Whether memory integration is enabled (default: True)
        """
        self.base_url = base_url
        self.enabled = enabled
        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False

        # Timeouts to prevent blocking
        self._timeout = httpx.Timeout(
            connect=2.0,    # 2s to establish connection
            read=5.0,       # 5s to read response
            write=5.0,      # 5s to send request
            pool=2.0        # 2s to acquire connection from pool
        )

    async def connect(self) -> bool:
        """
        Initialize the async HTTP client.

        Returns:
            True if client created successfully
        """
        if not self.enabled:
            logger.info("Memory client disabled")
            return False

        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout
            )
            self._connected = True
            logger.info(f"Memory client connected to {self.base_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to create memory client: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Close the async HTTP client."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing memory client: {e}")
            finally:
                self._client = None
                self._connected = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None

    async def store_memory(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Store a memory in the triple-layer system.

        Args:
            content: The text content to store
            metadata: Dictionary of metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            # Default to mid_term layer for trading events
            if 'layer' not in metadata:
                metadata['layer'] = 'mid_term'

            payload = {
                "content": content,
                "metadata": metadata
            }

            response = await self._client.post("/tools/store_memory", json=payload)
            response.raise_for_status()
            return True

        except httpx.TimeoutException:
            logger.warning("Memory store timed out - service may be unavailable")
            return False
        except httpx.ConnectError:
            logger.warning("Memory service unavailable - disabling for this session")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def retrieve_context(self, query: str, limit: int = 5) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: The query string
            limit: Maximum number of results

        Returns:
            Dict containing results or None if failed
        """
        if not self.is_connected:
            return None

        try:
            params = {
                "query": query,
                "limit": limit
            }

            response = await self._client.post("/tools/vector_search", params=params)
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            logger.warning("Memory retrieve timed out")
            return None
        except httpx.ConnectError:
            logger.warning("Memory service unavailable")
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return None

    async def log_trade(self, symbol: str, action: str, quantity: float,
                       price: float, reason: str) -> bool:
        """
        Helper to log a trade execution event.

        Args:
            symbol: Trading symbol
            action: BUY/SELL
            quantity: Number of shares
            price: Execution price
            reason: Trade reason

        Returns:
            True if logged successfully
        """
        content = f"Executed {action} order for {quantity} shares of {symbol} at ${price:.2f}. Reason: {reason}"
        metadata = {
            "category": "trade_execution",
            "symbol": symbol,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "source": "trader-ai"
        }
        return await self.store_memory(content, metadata)

    async def log_event(self, event_type: str, details: str,
                       extra_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a general trading event.

        Args:
            event_type: Type of event (e.g., 'rebalance', 'kill_switch', 'error')
            details: Event description
            extra_metadata: Additional metadata to include

        Returns:
            True if logged successfully
        """
        metadata = {
            "category": event_type,
            "timestamp": datetime.now().isoformat(),
            "source": "trader-ai"
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return await self.store_memory(details, metadata)


# Synchronous wrapper for backwards compatibility
class SyncMemoryClient:
    """
    Synchronous wrapper around MemoryClient for backwards compatibility.

    Note: This should only be used in non-async contexts.
    Prefer MemoryClient in async code.
    """

    def __init__(self, base_url: str = "http://localhost:8080", enabled: bool = True):
        self.base_url = base_url
        self.enabled = enabled
        self._client: Optional[httpx.Client] = None

        if enabled:
            try:
                self._client = httpx.Client(
                    base_url=base_url,
                    timeout=httpx.Timeout(5.0, connect=2.0)
                )
            except Exception as e:
                logger.warning(f"Failed to create sync memory client: {e}")
                self.enabled = False

    def store_memory(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Store memory synchronously (legacy)."""
        if not self.enabled or not self._client:
            return False

        try:
            if 'layer' not in metadata:
                metadata['layer'] = 'mid_term'

            payload = {"content": content, "metadata": metadata}
            response = self._client.post("/tools/store_memory", json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to store memory (sync): {e}")
            return False

    def __del__(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
