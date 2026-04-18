"""
Tests for HTTP wrapper with circuit breaker integration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import importlib
import sys
import types
import pytest


MODULE_PATH = 'components.utilities.circuit_breaker.http_wrapper'
EXPORTS = ['HttpResponse', 'CircuitBreakerHttpClient', 'HttpClientManager', 'with_circuit_breaker']


def _library_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent:
        if (root / 'catalog.json').exists() and (root / 'components').exists():
            return root
        root = root.parent
    raise RuntimeError('Library root not found')


def _ensure_library_package() -> None:
    root = _library_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if 'library' not in sys.modules:
        library = types.ModuleType('library')
        library.__path__ = [str(root)]
        sys.modules['library'] = library


def _import_module():
    _ensure_library_package()
    try:
        return importlib.import_module(MODULE_PATH)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name in MODULE_PATH:
            raise
        pytest.skip(f'Missing dependency: {exc.name}')
    except ImportError as exc:
        pytest.skip(f'Import error: {exc}')


def test_module_imports():
    """Test that module can be imported."""
    _import_module()


def test_exports_present():
    """Test that all expected exports are present."""
    module = _import_module()
    if not EXPORTS:
        return
    missing = [name for name in EXPORTS if not hasattr(module, name)]
    assert not missing, f"Missing exports: {missing}"


def test_http_response_creation():
    """Test HttpResponse dataclass."""
    module = _import_module()
    response = module.HttpResponse(
        status_code=200,
        text='{"success": true}',
        json_data={"success": True},
        headers={"content-type": "application/json"}
    )
    assert response.status_code == 200
    assert response.ok is True
    assert response.json() == {"success": True}


def test_http_response_not_ok():
    """Test HttpResponse.ok property for error codes."""
    module = _import_module()

    # 4xx errors
    response_400 = module.HttpResponse(status_code=400, text="Bad Request")
    assert response_400.ok is False

    # 5xx errors
    response_500 = module.HttpResponse(status_code=500, text="Server Error")
    assert response_500.ok is False

    # 2xx success
    response_201 = module.HttpResponse(status_code=201, text="Created")
    assert response_201.ok is True


def test_circuit_breaker_http_client_init():
    """Test CircuitBreakerHttpClient initialization."""
    module = _import_module()

    # Import config from main module
    cb_module = importlib.import_module('components.utilities.circuit_breaker.circuit_breaker')

    client = module.CircuitBreakerHttpClient(
        name="test_service",
        base_url="http://localhost:8080",
        config=cb_module.CircuitBreakerConfig(failure_threshold=3),
        timeout=10.0,
        headers={"X-Test": "true"}
    )

    assert client.name == "test_service"
    assert client.base_url == "http://localhost:8080"
    assert client.timeout == 10.0
    assert client.is_available is True  # Circuit starts closed


def test_http_client_manager_init():
    """Test HttpClientManager initialization."""
    module = _import_module()
    manager = module.HttpClientManager()

    status = manager.get_system_status()
    assert status["total_services"] == 0
    assert status["available_services"] == 0
    assert status["unavailable_services"] == []


@pytest.mark.asyncio
async def test_http_client_manager_register():
    """Test registering clients with manager."""
    module = _import_module()
    manager = module.HttpClientManager()

    # Register a client
    client = await manager.register(
        "test_service",
        "http://localhost:9999",
        timeout=5.0
    )

    assert client.name == "test_service"

    # Verify registration
    status = manager.get_system_status()
    assert status["total_services"] == 1
    assert "test_service" in status["services"]

    # Get registered client
    retrieved = manager.get("test_service")
    assert retrieved.name == "test_service"


@pytest.mark.asyncio
async def test_http_client_manager_duplicate_register():
    """Test that duplicate registration raises error."""
    module = _import_module()
    manager = module.HttpClientManager()

    await manager.register("test_service", "http://localhost:9999")

    with pytest.raises(ValueError, match="already registered"):
        await manager.register("test_service", "http://localhost:8888")


def test_http_client_manager_get_unknown():
    """Test getting unknown client raises KeyError."""
    module = _import_module()
    manager = module.HttpClientManager()

    with pytest.raises(KeyError, match="not registered"):
        manager.get("unknown_service")


def test_with_circuit_breaker_decorator():
    """Test the circuit breaker decorator exists."""
    module = _import_module()
    cb_module = importlib.import_module('components.utilities.circuit_breaker.circuit_breaker')

    breaker = cb_module.CircuitBreaker("test_decorator")
    decorator = module.with_circuit_breaker(breaker, fallback={"error": "fallback"})

    # Verify it returns a callable decorator
    assert callable(decorator)


@pytest.mark.asyncio
async def test_circuit_breaker_status_in_client():
    """Test getting circuit breaker status from HTTP client."""
    module = _import_module()

    client = module.CircuitBreakerHttpClient(
        name="status_test",
        base_url="http://localhost:8080"
    )

    status = client.get_status()
    assert status["name"] == "status_test"
    assert status["available"] is True
    assert status["circuit_state"] == "closed"
    assert "failure_count" in status
    assert "metrics" in status
