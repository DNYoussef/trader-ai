from src.dashboard.constants import get_cors_origins


def test_dashboard_cors_ignores_wildcard_for_credentialed_requests(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGIN", "*,https://app.example.com")
    monkeypatch.setenv("CORS_ORIGINS", "https://admin.example.com,*")

    origins = get_cors_origins()

    assert "*" not in origins
    assert "https://app.example.com" in origins
    assert "https://admin.example.com" in origins


def test_dashboard_cors_includes_railway_public_domain(monkeypatch):
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "trader-ai-production.up.railway.app")
    monkeypatch.delenv("CORS_ALLOW_ORIGIN", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    assert "https://trader-ai-production.up.railway.app" in get_cors_origins()
