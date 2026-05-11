import pytest

from src.dashboard.run_server_simple import resolve_bind_host


@pytest.mark.unit
def test_resolve_bind_host_keeps_local_default_loopback():
    assert resolve_bind_host({}) == "127.0.0.1"


@pytest.mark.unit
def test_resolve_bind_host_prefers_explicit_host():
    env = {
        "HOST": "127.0.0.2",
        "RAILWAY_ENVIRONMENT": "production",
    }

    assert resolve_bind_host(env) == "127.0.0.2"


@pytest.mark.unit
@pytest.mark.parametrize(
    "marker",
    [
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_DEPLOYMENT_ID",
        "RAILWAY_PUBLIC_DOMAIN",
        "RAILWAY_PRIVATE_DOMAIN",
    ],
)
def test_resolve_bind_host_uses_all_interfaces_on_railway(marker):
    assert resolve_bind_host({marker: "present"}) == "0.0.0.0"
