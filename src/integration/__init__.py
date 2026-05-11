"""Integration package exports."""

from .mieza_quant_bridge import (
    MiezaBridgeError,
    MiezaNonceStore,
    MiezaReplayError,
    MiezaSignalBridge,
    MiezaSignalEnvelope,
    MiezaSignalValidationError,
    MiezaSignatureError,
    alpha_events_from_envelope,
    sign_mieza_envelope,
)
from .mieza_signal_ingestion import MiezaIngestionResult, MiezaSignalIngestionService
from .mieza_signal_store import MiezaSQLiteStore


def __getattr__(name):
    if name == 'Phase2SystemFactory':
        from .phase2_factory import Phase2SystemFactory

        return Phase2SystemFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Phase2SystemFactory',
    'MiezaBridgeError',
    'MiezaNonceStore',
    'MiezaReplayError',
    'MiezaSignalBridge',
    'MiezaSignalEnvelope',
    'MiezaSignalValidationError',
    'MiezaSignatureError',
    'MiezaIngestionResult',
    'MiezaSignalIngestionService',
    'MiezaSQLiteStore',
    'alpha_events_from_envelope',
    'sign_mieza_envelope',
]
