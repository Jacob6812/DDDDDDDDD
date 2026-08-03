from __future__ import annotations

import csv
import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

from ...contracts import EvidenceLedger, ToolCallTrace
from .real_data_provider import DarwinTradeMCPProvider


@dataclass(slots=True)
class MCPToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MCPTool(Protocol):
    spec: MCPToolSpec

    def call(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(slots=True)
class MCPRequest:
    tool_name: str
    payload: dict[str, Any]


@dataclass(slots=True)
class MCPResponse:
    tool_name: str
    output: dict[str, Any]
    server_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MCPServer:
    name: str
    tools: dict[str, MCPTool] = field(default_factory=dict)

    def register_tool(self, tool: MCPTool) -> None:
        self.tools[tool.spec.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.spec.to_dict() for tool in self.tools.values()]

    def handle_request(self, request: MCPRequest) -> MCPResponse:
        if request.tool_name not in self.tools:
            raise KeyError(
                f"Unknown MCP tool on server {self.name}: {request.tool_name}"
            )
        output = self.tools[request.tool_name].call(request.payload)
        return MCPResponse(
            tool_name=request.tool_name, output=output, server_name=self.name
        )


class MCPTransport(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...

    def send(self, request: MCPRequest) -> MCPResponse: ...


class InMemoryMCPTransport:
    def __init__(self, servers: list[MCPServer] | None = None) -> None:
        self._servers = servers or []
    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for server in self._servers:
            for tool in server.list_tools():
                tools.append({"server": server.name, **tool})
        return tools

    def send(self, request: MCPRequest) -> MCPResponse:
        for server in self._servers:
            if request.tool_name in server.tools:
                return server.handle_request(request)
        raise KeyError(f"Unknown MCP tool across attached servers: {request.tool_name}")


class MCPClient:
    def __init__(self, transport: MCPTransport) -> None:
        self.transport = transport

    def list_tools(self) -> list[dict[str, Any]]:
        return self.transport.list_tools()

    def call_tool(self, name: str, payload: dict[str, Any]) -> MCPResponse:
        return self.transport.send(MCPRequest(tool_name=name, payload=payload))


@dataclass(slots=True)
class MCPProviderSelection:
    provider_key: str
    provider_name: str
    server_name: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MCPDataProvider(Protocol):
    provider_name: str

    def execute(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class MCPProviderError(RuntimeError):
    """Raised when a backing provider returns an explicit failure."""


_RESULT_KEYS = {
    "get_stock_data": "stock_data",
    "get_indicators": "indicators",
    "get_balance_sheet": "balance_sheet",
    "get_cashflow": "cashflow",
    "get_income_statement": "income_statement",
    "get_news": "company_news",
    "get_global_news": "global_news",
    "get_insider_transactions": "insider_transactions",
}
class ProviderBackedTool:
    tool_name: str = ""
    result_key: str = "result"
    provider_preference: tuple[str, ...] = ()
    server_name: str = "market-data-server"

    def __init__(
        self,
        providers: dict[str, MCPDataProvider],
        *,
        default_provider: str,
    ) -> None:
        self.providers = providers
        self.default_provider = default_provider

    def provider_selection(self, payload: dict[str, Any]) -> MCPProviderSelection:
        preference = payload.get("provider")
        candidates: list[str] = []
        explicit_provider = ""
        if isinstance(preference, str) and preference.strip():
            explicit_provider = preference.strip()
            provider = self.providers.get(explicit_provider)
            if provider is None:
                raise MCPProviderError(
                    f"Requested MCP provider '{explicit_provider}' is unavailable for {self.tool_name}"
                )
            details = {
                "requested_provider": explicit_provider,
                "default_provider": self.default_provider,
                "tool_name": self.tool_name,
            }
            return MCPProviderSelection(
                provider_key=explicit_provider,
                provider_name=provider.provider_name,
                server_name=self.server_name_for_provider(explicit_provider),
                details=details,
            )
        candidates.extend(self.provider_preference)
        if self.default_provider not in candidates:
            candidates.append(self.default_provider)
        for candidate in candidates:
            provider = self.providers.get(candidate)
            if provider is None:
                continue
            details = {
                "requested_provider": explicit_provider,
                "default_provider": self.default_provider,
                "tool_name": self.tool_name,
            }
            return MCPProviderSelection(
                provider_key=candidate,
                provider_name=provider.provider_name,
                server_name=self.server_name_for_provider(candidate),
                details=details,
            )
        raise MCPProviderError(f"No MCP provider available for {self.tool_name}")

    def server_name_for_provider(self, provider_key: str) -> str:
        return self.server_name

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        selection = self.provider_selection(payload)
        provider = self.providers[selection.provider_key]
        output = provider.execute(self.tool_name, payload)
        if not isinstance(output, dict):
            raise MCPProviderError(
                f"{self.tool_name} returned non-dict output from provider {selection.provider_name}"
            )
        if output.get("available") is False:
            raise MCPProviderError(
                str(output.get("reason") or output.get("error") or f"{self.tool_name} unavailable")
            )
        if self.result_key not in output:
            raise MCPProviderError(
                f"{self.tool_name} response missing required key '{self.result_key}'"
            )
        audit = dict(selection.details)
        audit.update(
            {
                "provider": selection.provider_name,
                "provider_key": selection.provider_key,
                "server_name": selection.server_name,
            }
        )
        return {
            **output,
            "_provider_selection": audit,
        }


class EvidenceCoverageTool:
    spec = MCPToolSpec(
        name="evidence_coverage_check",
        description="Summarize evidence coverage, unresolved flags, and verification readiness.",
        input_schema={"type": "object", "required": ["ledger"]},
    )

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        ledger = payload.get("ledger")
        if isinstance(ledger, EvidenceLedger):
            verification_status = ledger.verification_status
            verify_required = ledger.verify_required
            verify_flags = list(ledger.verify_flags)
            conflict_density = ledger.conflict_density
            claim_count = len(ledger.entries)
            entry_count = len(ledger.entries)
        else:
            verification_status = str(payload.get("verification_status", "unknown"))
            verify_required = bool(payload.get("verify_required"))
            verify_flags = list(payload.get("verify_flags", []) or [])
            conflict_density = float(payload.get("conflict_density", 0.0) or 0.0)
            claim_count = int(payload.get("claim_count", 0) or 0)
            entry_count = int(payload.get("entry_count", 0) or 0)
        coverage_state = (
            "verified"
            if verification_status == "verified"
            else "needs_review"
            if verify_required or verify_flags
            else "informational"
        )
        return {
            "coverage_state": coverage_state,
            "verification_status": verification_status,
            "verify_required": verify_required,
            "verify_flags": verify_flags,
            "claim_count": claim_count,
            "entry_count": entry_count,
            "conflict_density": conflict_density,
        }


class ConflictCheckTool:
    spec = MCPToolSpec(
        name="conflict_density_check",
        description="Checks whether claim conflict density breaches the risk gate threshold.",
        input_schema={"type": "object", "required": ["ledger"]},
    )

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        ledger = payload.get("ledger")
        conflict_density = (
            ledger.conflict_density
            if isinstance(ledger, EvidenceLedger)
            else float(payload.get("disagreement", 0.0))
        )
        return {
            "conflict_density": conflict_density,
            "risk_gate": "block" if conflict_density >= 0.55 else "pass",
        }


class TraceConsistencyTool:
    spec = MCPToolSpec(
        name="trace_consistency_check",
        description="Checks whether real tool traces exist when verification is required.",
        input_schema={"type": "object", "required": ["tool_traces", "verify_required"]},
    )

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        traces = payload.get("tool_traces") or []
        verify_required = bool(payload.get("verify_required"))
        return {
            "trace_count": len(traces),
            "verification_status": "observed"
            if traces
            else ("missing" if verify_required else "not_required"),
        }
class GetStockDataTool(ProviderBackedTool):
    spec = MCPToolSpec(
        name="get_stock_data",
        description="Fetch OHLCV stock data via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["symbol", "trade_date"]},
    )
    tool_name = "get_stock_data"
    result_key = "stock_data"
    provider_preference = ("darwintrade", "tradingagents")
    server_name = "market-data-server"


class GetIndicatorsTool(ProviderBackedTool):
    spec = MCPToolSpec(
        name="get_indicators",
        description="Fetch technical indicators via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["symbol", "trade_date"]},
    )
    tool_name = "get_indicators"
    result_key = "indicators"
    provider_preference = ("darwintrade", "tradingagents")
    server_name = "market-data-server"


class GetStatementTool(ProviderBackedTool):
    result_key = "statement"
    provider_preference = ("darwintrade", "tradingagents")
    server_name = "fundamentals-server"


class GetBalanceSheetTool(GetStatementTool):
    spec = MCPToolSpec(
        name="get_balance_sheet",
        description="Fetch balance sheet data via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["symbol", "trade_date"]},
    )
    tool_name = "get_balance_sheet"
    result_key = "balance_sheet"


class GetCashflowTool(GetStatementTool):
    spec = MCPToolSpec(
        name="get_cashflow",
        description="Fetch cashflow data via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["symbol", "trade_date"]},
    )
    tool_name = "get_cashflow"
    result_key = "cashflow"


class GetIncomeStatementTool(GetStatementTool):
    spec = MCPToolSpec(
        name="get_income_statement",
        description="Fetch income statement data via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["symbol", "trade_date"]},
    )
    tool_name = "get_income_statement"
    result_key = "income_statement"


class GetNewsTool(ProviderBackedTool):
    spec = MCPToolSpec(
        name="get_news",
        description="Fetch company news via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["symbol", "trade_date"]},
    )
    tool_name = "get_news"
    result_key = "company_news"
    provider_preference = ("darwintrade", "tradingagents")
    server_name = "news-server"


class GetGlobalNewsTool(ProviderBackedTool):
    spec = MCPToolSpec(
        name="get_global_news",
        description="Fetch macro news via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["trade_date"]},
    )
    tool_name = "get_global_news"
    result_key = "global_news"
    provider_preference = ("darwintrade", "tradingagents")
    server_name = "news-server"


class GetInsiderTransactionsTool(ProviderBackedTool):
    spec = MCPToolSpec(
        name="get_insider_transactions",
        description="Fetch insider transactions via DarwinTrade-owned real providers.",
        input_schema={"type": "object", "required": ["symbol"]},
    )
    tool_name = "get_insider_transactions"
    result_key = "insider_transactions"
    provider_preference = ("darwintrade", "tradingagents")
    server_name = "news-server"


def build_default_mcp_client(
    *,
    default_data_provider: str = "darwintrade",
    providers: dict[str, MCPDataProvider] | None = None,
) -> MCPClient:
    if providers is not None:
        data_providers = providers
    else:
        data_providers = {"darwintrade": DarwinTradeMCPProvider()}
    market_server = MCPServer(name="market-data-server")
    market_server.register_tool(
        GetStockDataTool(data_providers, default_provider=default_data_provider)
    )
    market_server.register_tool(
        GetIndicatorsTool(data_providers, default_provider=default_data_provider)
    )
    news_server = MCPServer(name="news-server")
    news_server.register_tool(
        GetNewsTool(data_providers, default_provider=default_data_provider)
    )
    news_server.register_tool(
        GetGlobalNewsTool(data_providers, default_provider=default_data_provider)
    )
    news_server.register_tool(
        GetInsiderTransactionsTool(
            data_providers, default_provider=default_data_provider
        )
    )
    fundamentals_server = MCPServer(name="fundamentals-server")
    fundamentals_server.register_tool(
        GetBalanceSheetTool(data_providers, default_provider=default_data_provider)
    )
    fundamentals_server.register_tool(
        GetCashflowTool(data_providers, default_provider=default_data_provider)
    )
    fundamentals_server.register_tool(
        GetIncomeStatementTool(data_providers, default_provider=default_data_provider)
    )
    verification_server = MCPServer(name="verification-server")
    verification_server.register_tool(EvidenceCoverageTool())
    verification_server.register_tool(ConflictCheckTool())
    verification_server.register_tool(TraceConsistencyTool())
    transport = InMemoryMCPTransport(
        servers=[
            market_server,
            news_server,
            fundamentals_server,
            verification_server,
        ]
    )
    return MCPClient(transport=transport)


def build_mcp_trace(
    step: int,
    tool_name: str,
    arguments: dict[str, Any],
    output: dict[str, Any],
    server_name: str,
) -> ToolCallTrace:
    return ToolCallTrace(
        step=step,
        tool_name=f"{server_name}:{tool_name}",
        arguments=arguments,
        output=output,
    )


__all__ = [
    "ConflictCheckTool",
    "EvidenceCoverageTool",
    "GetBalanceSheetTool",
    "GetCashflowTool",
    "GetGlobalNewsTool",
    "GetIncomeStatementTool",
    "GetIndicatorsTool",
    "GetInsiderTransactionsTool",
    "GetNewsTool",
    "GetStockDataTool",
    "InMemoryMCPTransport",
    "MCPClient",
    "MCPDataProvider",
    "MCPProviderError",
    "MCPProviderSelection",
    "MCPRequest",
    "MCPResponse",
    "MCPServer",
    "MCPTool",
    "MCPToolSpec",
    "MCPTransport",
    "ProviderBackedTool",
    "TraceConsistencyTool",
    "TradingAgentsMCPProvider",
    "build_default_mcp_client",
    "build_mcp_trace",
]
