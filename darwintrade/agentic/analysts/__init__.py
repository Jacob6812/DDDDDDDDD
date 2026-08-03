from __future__ import annotations

from typing import Any

from .graph import DarwinTradeAgenticAnalystBackend, build_analyst_graph
from .role_execution import BackendRoleSnapshot, DarwinTradeRoleExecutionLayer
from .validation import ReportValidationResult, validate_report_schema

DARWINTRADE_AGENTIC_BACKENDS = {"darwintrade_agentic", "agentic"}


def normalize_analyzer_backend_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in DARWINTRADE_AGENTIC_BACKENDS:
        return "darwintrade_agentic"
    return normalized


def create_analyzer_backend(name: str, **kwargs: Any) -> Any:
    normalized = normalize_analyzer_backend_name(name)
    if normalized == "darwintrade_agentic":
        return DarwinTradeAgenticAnalystBackend(**kwargs)
    raise ValueError(f"Unknown analyzer backend: {name}")


__all__ = [
    "BackendRoleSnapshot",
    "DARWINTRADE_AGENTIC_BACKENDS",
    "DarwinTradeAgenticAnalystBackend",
    "DarwinTradeRoleExecutionLayer",
    "ReportValidationResult",
    "build_analyst_graph",
    "create_analyzer_backend",
    "normalize_analyzer_backend_name",
    "validate_report_schema",
]
