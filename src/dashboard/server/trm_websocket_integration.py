"""
TRM WebSocket Integration Module
Add this to your websocket_server.py to broadcast TRM predictions
"""

import asyncio
import logging
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intelligence.trm_streaming_integration import TRMStreamingPredictor

logger = logging.getLogger(__name__)


class TRMWebSocketBroadcaster:
    """
    Broadcasts TRM predictions via WebSocket server
    Integrates seamlessly with existing ConnectionManager
    """

    def __init__(self, connection_manager, update_interval: int = 60):
        """
        Initialize TRM broadcaster

        Args:
            connection_manager: WebSocket ConnectionManager instance
            update_interval: Prediction update interval in seconds (default: 60)
        """
        self.connection_manager = connection_manager
        self.update_interval = update_interval
        self.predictor = None
        self.is_running = False
        self.broadcast_task = None

    async def start(self):
        """Start TRM predictions and broadcasting"""
        try:
            logger.info("Initializing TRM predictor...")

            # Initialize predictor
            self.predictor = TRMStreamingPredictor(update_interval=self.update_interval)

            logger.info(f"TRM predictor initialized (interval={self.update_interval}s)")
            logger.info("Starting TRM broadcast loop...")

            # Start broadcast loop
            self.is_running = True
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())

            logger.info("TRM broadcasting started successfully")

        except Exception as e:
            logger.error(f"Failed to start TRM broadcasting: {e}")
            raise

    async def _broadcast_loop(self):
        """Continuous broadcasting loop"""
        import numpy as np

        while self.is_running:
            try:
                # Get TRM prediction
                prediction = await self.predictor.predict_streaming()

                # Fallback to mock data if database is empty
                if not prediction:
                    logger.warning("Database empty - using mock features for demo")

                    # Generate mock features (10 values representing current market state)
                    mock_features = np.array([
                        14.5,   # VIX level (moderate volatility)
                        -0.02,  # SPY 5-day returns (slight decline)
                        -0.05,  # SPY 20-day returns (downtrend)
                        1.1,    # Volume ratio (above average)
                        0.45,   # Market breadth
                        0.28,   # Correlation
                        1.3,    # Put/call ratio (defensive sentiment)
                        0.99,   # Gini coefficient
                        0.03,   # Sector dispersion
                        0.95    # Signal quality
                    ])

                    # Make prediction with mock data
                    prediction = self.predictor.predict(mock_features)
                    prediction['mock_data'] = True  # Flag as mock

                if prediction:
                    # Format for WebSocket
                    message = {
                        'type': 'trm_prediction',
                        'data': prediction,
                        'timestamp': prediction['timestamp']
                    }

                    # Broadcast to all connected clients
                    await self.connection_manager.broadcast(message)

                    mode = " (MOCK DATA)" if prediction.get('mock_data') else ""
                    logger.info(
                        f"Broadcasted TRM prediction{mode}: {prediction['strategy_name']} "
                        f"(confidence={prediction['confidence']:.2%})"
                    )

            except Exception as e:
                logger.error(f"Error in TRM broadcast loop: {e}")

            # Wait for next interval
            await asyncio.sleep(self.update_interval)

    async def stop(self):
        """Stop TRM broadcasting"""
        self.is_running = False

        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass

        if self.predictor:
            self.predictor.stop_streaming()

        logger.info("TRM broadcasting stopped")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of TRM predictions"""
        if self.predictor:
            return self.predictor.get_prediction_summary()
        return {'status': 'not_initialized'}


# ============================================================================
# INTEGRATION EXAMPLE FOR websocket_server.py
# ============================================================================

"""
Add this to src/dashboard/server/websocket_server.py:

1. Import at top of file:
   ----------------------------------------------------------------------------
   from .trm_websocket_integration import TRMWebSocketBroadcaster

2. Add to app startup (in main() or startup event):
   ----------------------------------------------------------------------------
   # Initialize TRM broadcaster
   trm_broadcaster = TRMWebSocketBroadcaster(
       connection_manager=manager,
       update_interval=60  # 1-minute intervals
   )

   # Start TRM broadcasting
   await trm_broadcaster.start()

3. Add to app shutdown (if you have one):
   ----------------------------------------------------------------------------
   @app.on_event("shutdown")
   async def shutdown():
       await trm_broadcaster.stop()

4. Optional: Add REST endpoint for TRM summary:
   ----------------------------------------------------------------------------
   @app.get("/api/trm/summary")
   async def get_trm_summary():
       return trm_broadcaster.get_summary()

5. Optional: Add to WebSocket message handler for manual refresh:
   ----------------------------------------------------------------------------
   @app.websocket("/ws/{client_id}")
   async def websocket_endpoint(websocket: WebSocket, client_id: str):
       # ... existing code ...

       # Handle TRM refresh request
       if data.get('type') == 'request_trm_prediction':
           summary = trm_broadcaster.get_summary()
           await manager.send_personal_message(
               {'type': 'trm_summary', 'data': summary},
               client_id
           )

THAT'S IT! TRM predictions will now broadcast automatically every 60 seconds.

WebSocket message format received by clients:
{
    "type": "trm_prediction",
    "data": {
        "timestamp": "2025-11-07T16:45:23",
        "strategy_id": 0,
        "strategy_name": "ultra_defensive",
        "confidence": 0.5432,
        "probabilities": {...},
        "raw_features": [...],
        "halt_probability": 0.0123,
        "model_metadata": {...}
    },
    "timestamp": "2025-11-07T16:45:23"
}
"""


# ============================================================================
# STANDALONE SERVER EXAMPLE (For Testing)
# ============================================================================

async def run_standalone_server():
    """
    Standalone WebSocket server with TRM broadcasting
    For testing TRM integration without full dashboard
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    app = FastAPI(title="TRM Streaming Server")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Simple connection manager
    class SimpleConnectionManager:
        def __init__(self):
            self.active_connections = {}

        async def connect(self, websocket: WebSocket, client_id: str):
            await websocket.accept()
            self.active_connections[client_id] = websocket
            logger.info(f"Client {client_id} connected")

        def disconnect(self, client_id: str):
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"Client {client_id} disconnected")

        async def broadcast(self, message: Dict):
            for client_id, ws in list(self.active_connections.items()):
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to {client_id}: {e}")
                    self.disconnect(client_id)

        async def send_personal_message(self, message: Dict, client_id: str):
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to {client_id}: {e}")
                    self.disconnect(client_id)

    manager = SimpleConnectionManager()
    trm_broadcaster = None

    @app.on_event("startup")
    async def startup():
        nonlocal trm_broadcaster
        trm_broadcaster = TRMWebSocketBroadcaster(manager, update_interval=30)
        await trm_broadcaster.start()
        logger.info("TRM Standalone Server started")

    @app.on_event("shutdown")
    async def shutdown():
        if trm_broadcaster:
            await trm_broadcaster.stop()
        logger.info("TRM Standalone Server stopped")

    @app.get("/")
    async def root():
        return {
            "service": "TRM Streaming Server",
            "status": "running",
            "websocket": "ws://localhost:8001/ws/{client_id}",
            "summary": "GET /api/trm/summary"
        }

    @app.get("/api/trm/summary")
    async def get_summary():
        if trm_broadcaster:
            return trm_broadcaster.get_summary()
        return {"error": "TRM not initialized"}

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await manager.connect(websocket, client_id)

        try:
            while True:
                # Keep connection alive
                data = await websocket.receive_text()
                logger.debug(f"Received from {client_id}: {data}")

                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")

        except WebSocketDisconnect:
            manager.disconnect(client_id)

    # Run server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("="*80)
    print("TRM STREAMING WEBSOCKET SERVER")
    print("="*80)
    print()
    print("Starting standalone server for TRM predictions...")
    print("WebSocket: ws://localhost:8001/ws/{client_id}")
    print("REST API: http://localhost:8001/api/trm/summary")
    print()
    print("Press Ctrl+C to stop")
    print("="*80)
    print()

    asyncio.run(run_standalone_server())
