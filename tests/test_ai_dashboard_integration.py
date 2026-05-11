import logging

import pytest

from src.dashboard import ai_dashboard_integration as module


@pytest.mark.asyncio
async def test_ai_dashboard_start_skips_when_stream_integrator_unavailable(monkeypatch, caplog):
    integrator = module.AIDashboardIntegrator()
    monkeypatch.setattr(module, "AI_SYSTEMS_AVAILABLE", True)
    monkeypatch.setattr(module, "ai_data_stream_integrator", None)

    with caplog.at_level(logging.INFO):
        await integrator.start_ai_dashboard_integration()

    assert integrator.is_streaming is False
    assert "background AI services disabled" in caplog.text
