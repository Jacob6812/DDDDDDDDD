from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from time import time_ns
from typing import Any, Callable

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from ._helpers import build_asset_research_packet
from .role_execution import (
    BackendRoleSnapshot,
    DarwinTradeRoleExecutionLayer,
)
from darwintrade.contracts import AssetResearchPacket, PortfolioState
from darwintrade.integrations.llm import DarwinTradeLLMClient, _invoke_with_retries
from darwintrade.integrations.llm.llm import (
    build_json_schema_response_format,
    parse_llm_json_dict,
)
from darwintrade.integrations.llm.mcp import MCPClient, build_default_mcp_client, build_mcp_trace
from darwintrade.integrations.llm.prompting import build_role_instruction
from darwintrade.integrations.llm.real_data_provider import sanitize_runtime_context
from darwintrade.interfaces import AnalyzerResult, RoleReport
from darwintrade.serialization import to_jsonable

from ..tooling import (
    EvidenceLedger,
    EvidenceRequest,
    ToolExecutor,
    ToolPlanner,
    ToolRegistry,
    load_default_registry,
)
from .fundamentals import build_fundamentals_request
from .market import build_market_request
from .news import build_news_request
from .state import AnalystState


RESEARCH_ROLE_ORDER = (
    "market_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "social_analyst",
)

logger = logging.getLogger(__name__)


_parse_json_dict = parse_llm_json_dict


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _build_structured_asset_metadata(
    *,
    signal: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Project the Bayesian aggregator's summary into the metadata shape
    expected by downstream consumers (allocator, reflection, reporting).
    `summary` is always the Bayesian aggregate produced inside _research_node;
    the legacy heuristic that built this dict locally has been removed."""
    research_action = str(summary.get("direction") or "HOLD").upper()
    metadata = {
        "confidence": float(summary["aggregate_confidence"]),
        "decision_confidence": float(summary["decision_confidence"]),
        "disagreement": float(summary["disagreement"]),
        "downside_risk": float(summary["downside_risk"]),
        "expected_return_view": float(summary["expected_return_view"]),
        "expected_return_model": "bayesian_capsule_aggregate",
        "aggregate_research_score": float(summary["aggregate_score"]),
        "role_score_map": dict(summary.get("role_score_map", {})),
        "role_confidence_map": dict(summary.get("role_confidence_map", {})),
        "verification_verdict": str(summary["verification_verdict"]),
        "risk_uncertainty": float(summary["risk_uncertainty"]),
        "portfolio_signal": str(signal or "HOLD").upper(),
        "hypothesis_action": research_action,
        "signal_authority": "non_executable_hypothesis",
        "execution_requires_pipeline_governance": True,
    }
    # Bayesian-only fields: per-role precision/beta and posterior diagnostics.
    for key in (
        "information_ratio", "tau_total", "role_precision_map",
        "role_warm_map", "role_beta_map", "role_implied_return_map",
        "n_contributing_roles", "aggregator",
    ):
        if key in summary:
            metadata[key] = summary[key]
    return metadata


class DarwinTradeAgenticAnalystBackend:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        planner: ToolPlanner | None = None,
        llm_client: DarwinTradeLLMClient | None = None,
        mcp_client: MCPClient | None = None,
        analyst_memory: Any = None,
        **_: Any,
    ) -> None:
        self.registry = registry or load_default_registry()
        self.planner = planner or ToolPlanner(self.registry)
        self.executor = executor or ToolExecutor(self.registry)
        self.llm_client = llm_client or DarwinTradeLLMClient()
        self.mcp_client = mcp_client or build_default_mcp_client()
        self.role_execution = DarwinTradeRoleExecutionLayer()
        # Default to NullAnalystMemory so the Bayesian aggregator always has a
        # capsule lookup. Real AnalystMemory is injected via set_analyst_memory
        # by the pipeline when memory_enabled=True; otherwise capsules stay at
        # prior, which collapses the aggregator to confidence-weighted.
        if analyst_memory is None:
            from darwintrade.memory.analyst_memory import NullAnalystMemory
            analyst_memory = NullAnalystMemory()
        self.analyst_memory = analyst_memory
        self._graph = build_analyst_graph(
            self.planner,
            self.executor,
            self.registry,
            self.llm_client,
            self.mcp_client,
            aggregator_hook=self._build_aggregator_hook(),
        )
        self.session_context: dict[str, Any] = {}
        # Aggregator-only context bypass. Carries data the aggregator needs
        # (regime, verification verdict) WITHOUT routing through the
        # sanitize_runtime_context filter that scrubs the LLM-facing
        # session_context. This keeps memory out of LLM prompts (the
        # research-team LLM stays uninfluenced by memory) while letting
        # the deterministic aggregator below the LLM use the portfolio
        # regime to look up the correct (regime, role) capsule.
        self.aggregator_runtime: dict[str, Any] = {}

    def set_aggregator_runtime(self, context: dict[str, Any]) -> None:
        """Inject regime / verification overrides for the aggregator.
        These never reach the analyst LLM prompts."""
        self.aggregator_runtime = dict(context or {})

    def set_analyst_memory(self, analyst_memory: Any) -> None:
        """Late-bind the analyst memory and rebuild the graph so the
        aggregator hook is active. Used when the pipeline constructs the
        backend before the memory exists. Passing None resets to
        NullAnalystMemory (no-learning ablation)."""
        if analyst_memory is None:
            from darwintrade.memory.analyst_memory import NullAnalystMemory
            analyst_memory = NullAnalystMemory()
        self.analyst_memory = analyst_memory
        self._graph = build_analyst_graph(
            self.planner,
            self.executor,
            self.registry,
            self.llm_client,
            self.mcp_client,
            aggregator_hook=self._build_aggregator_hook(),
        )

    def _build_aggregator_hook(self):
        """Closure binding the aggregator to the backend's analyst memory
        and the live session_context (for regime + verification verdict)."""
        from darwintrade.agents.bayesian_aggregator import aggregate as _bayes_aggregate

        backend = self

        def _hook(role_payloads: dict[str, Any], shared_state: dict[str, Any]) -> dict[str, Any]:
            # Regime is taken from the aggregator runtime bypass, not from
            # shared_state or session_context (both of which are sanitized
            # for LLM consumption and strip "regime"). The bypass is set
            # by the pipeline once per bar via set_aggregator_runtime.
            regime = str(
                backend.aggregator_runtime.get("regime")
                or shared_state.get("regime")
                or "unknown"
            )
            verification_verdict = str(
                backend.aggregator_runtime.get("verification_verdict")
                or shared_state.get("verification_verdict", "PARTIAL")
                or "PARTIAL"
            )
            risk_uncertainty = float(shared_state.get("risk_uncertainty", 0.0) or 0.0)
            memory = backend.analyst_memory

            def _lookup(reg: str, role: str):
                return memory.get_or_create(reg, role)

            summary = _bayes_aggregate(
                role_payloads={
                    role: payload for role, payload in role_payloads.items()
                    if role in RESEARCH_ROLE_ORDER and isinstance(payload, dict)
                },
                regime=regime,
                capsule_lookup=_lookup,
                verification_verdict=verification_verdict,
                risk_uncertainty=risk_uncertainty,
            )
            return summary

        return _hook

    def is_available(self) -> bool:
        return self.llm_client.is_available()

    def set_runtime_context(self, context: dict[str, Any]) -> None:
        self.session_context = sanitize_runtime_context(context)

    def _resolve_query(self, symbol: str, trade_date: str, context: dict[str, Any]) -> str:
        return str(
            context.get("query")
            or context.get("question")
            or f"Analyze {symbol} for {trade_date}"
        )

    def _analyze_with_context(
        self, symbol: str, trade_date: str, context: dict[str, Any]
    ) -> AnalyzerResult:
        raw_context = dict(context)
        sanitized_context = {
            **sanitize_runtime_context(context),
            **{
                key: value
                for key, value in raw_context.items()
                if key in {
                    "analysis_scope",
                    "portfolio_state",
                    "universe",
                    "focus_symbol",
                    "analyzer_memory",
                    "episode_memory",
                    "strategy_memory",
                    "control_memory",
                    "memory_hint",
                }
            },
        }
        query = self._resolve_query(symbol, trade_date, sanitized_context)
        state = self._graph.invoke(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "query": query,
                "snapshots": [],
                "tool_traces": [],
                "ledgers": [],
                "role_outputs": {},
                "role_payloads": {},
                "shared_state": dict(sanitized_context),
                "signal": "HOLD",
                "raw_report": "",
            },
            {
                "configurable": {
                    "thread_id": f"agentic::{symbol}::{trade_date}::{time_ns()}"
                }
            },
        )
        snapshots = [
            BackendRoleSnapshot(**item) for item in state.get("snapshots", [])
        ]
        role_payloads = dict(state.get("role_payloads", {}))
        shared_state = dict(state.get("shared_state", {}))
        metadata = {
            "source": "darwintrade_agentic",
            "tool_traces": list(state.get("tool_traces", [])),
            "evidence_ledgers": list(state.get("ledgers", [])),
            "role_payloads": role_payloads,
            "shared_state": shared_state,
            **_build_structured_asset_metadata(
                signal=str(state.get("signal", "HOLD")),
                summary=state.get("aggregator_summary") or {},
            ),
        }
        return AnalyzerResult(
            backend_name="darwintrade_agentic",
            symbol=symbol,
            trade_date=trade_date,
            signal=str(state.get("signal", "HOLD")),
            raw_report=str(state.get("raw_report", "")),
            role_reports=self.role_execution.build_role_reports(
                symbol, trade_date, snapshots
            ),
            claims=self.role_execution.build_claims(symbol, trade_date, snapshots),
            metadata=metadata,
        )

    def analyze(self, symbol: str, trade_date: str) -> AnalyzerResult:
        return self._analyze_with_context(symbol, trade_date, self.session_context)

    def _symbol_parallelism(self, symbol_count: int) -> int:
        configured = os.getenv("DARWINTRADE_SYMBOL_PARALLELISM")
        if configured:
            try:
                return max(1, min(symbol_count, int(configured)))
            except ValueError:
                pass
        return max(1, symbol_count)

    def _analyze_symbol_packet(
        self,
        symbol: str,
        trade_date: str,
        symbols: list[str],
        portfolio_state: PortfolioState,
    ) -> AssetResearchPacket:
        context = {
            **self.session_context,
            "analysis_scope": "portfolio",
            "portfolio_state": portfolio_state.to_dict(),
            "universe": list(symbols),
            "focus_symbol": symbol,
        }
        result = self._analyze_with_context(symbol, trade_date, context)
        result.metadata["portfolio_runtime"] = {
            "analysis_scope": "portfolio",
            "universe": list(symbols),
            "focus_symbol": symbol,
        }
        return build_asset_research_packet(result)

    def analyze_universe_packets(
        self,
        symbols: list[str],
        trade_date: str,
        portfolio_state: PortfolioState,
    ) -> list[AssetResearchPacket]:
        if len(symbols) <= 1:
            return [
                self._analyze_symbol_packet(symbols[0], trade_date, symbols, portfolio_state)
            ] if symbols else []
        max_workers = self._symbol_parallelism(len(symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(
                pool.map(
                    lambda symbol: self._analyze_symbol_packet(
                        symbol, trade_date, symbols, portfolio_state
                    ),
                    symbols,
                )
            )

    def analyze_universe(
        self,
        symbols: list[str],
        trade_date: str,
        portfolio_state: PortfolioState,
    ) -> list[Any]:
        return [
            packet.asset_view
            for packet in self.analyze_universe_packets(
                symbols,
                trade_date,
                portfolio_state,
            )
        ]

    def reenter_role(
        self,
        role: str,
        symbol: str,
        trade_date: str,
        *,
        session_context: dict[str, Any] | None = None,
        role_outputs: dict[str, str] | None = None,
        callback_context: dict[str, Any] | None = None,
        debate_context: dict[str, Any] | None = None,
    ) -> tuple[RoleReport, dict[str, Any]]:
        shared_state = sanitize_runtime_context(session_context)
        shared_state["callback_context"] = dict(callback_context or {})
        shared_state["debate_context"] = dict(debate_context or {})
        query = str(
            shared_state.get("query")
            or shared_state.get("question")
            or f"Reenter {role} for {symbol}"
        )
        if role not in {"market_analyst", "news_analyst", "social_analyst", "fundamentals_analyst"}:
            raise ValueError(
                f"reenter_role only supports research roles after 2-node simplification, got {role}"
            )
        snapshots: list[BackendRoleSnapshot] = []
        tool_traces: list[dict[str, Any]] = []
        ledgers: list[dict[str, Any]] = []
        role_payloads = dict(shared_state.get("role_payloads", {}) or {})
        snapshot, traces, ledger, _analysis = _run_research_role(
            symbol,
            trade_date,
            query,
            role,
            self.planner,
            self.executor,
            self.registry,
            self.llm_client,
        )
        snapshots.append(snapshot)
        tool_traces.extend(traces)
        ledgers.append(ledger)
        report = self.role_execution.build_role_reports(symbol, trade_date, snapshots)[0]
        metadata = {
            "source": "darwintrade_agentic",
            "tool_traces": tool_traces,
            "evidence_ledgers": ledgers,
            "role_payloads": role_payloads,
            "shared_state": shared_state,
        }
        return report, metadata


def build_analyst_graph(
    planner: ToolPlanner,
    executor: ToolExecutor,
    registry: ToolRegistry,
    llm_client: DarwinTradeLLMClient,
    mcp_client: MCPClient,
    *,
    aggregator_hook,
):
    """
    Single-node graph. The research node aggregates 4-role concurrent
    analysis into a signal via the supplied `aggregator_hook` — currently
    always the Bayesian online-capsule aggregator. The previous risk /
    portfolio / verification LLM nodes were removed (they duplicated the
    score aggregation now done in the hook, and self-validating LLM checks
    rarely caught real problems). The Bayesian hook is the single source of
    truth for converting role-level scores into a directional signal.
    """
    builder = StateGraph(AnalystState)
    builder.add_node(
        "research",
        lambda state: _research_node(
            state, planner, executor, registry, llm_client,
            aggregator_hook=aggregator_hook,
        ),
    )
    builder.add_edge(START, "research")
    builder.add_edge("research", END)
    return builder.compile(checkpointer=MemorySaver())


def _research_node(
    state: AnalystState,
    planner: ToolPlanner,
    executor: ToolExecutor,
    registry: ToolRegistry,
    llm_client: DarwinTradeLLMClient,
    *,
    aggregator_hook,
) -> dict[str, Any]:
    symbol = state["symbol"]
    trade_date = state["trade_date"]
    query = state["query"]
    max_workers = _research_parallelism()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_by_role = {
            role: pool.submit(
                _run_research_role,
                symbol,
                trade_date,
                query,
                role,
                planner,
                executor,
                registry,
                llm_client,
            )
            for role in RESEARCH_ROLE_ORDER
        }
    snapshots: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    role_outputs: dict[str, str] = {}
    shared_state: dict[str, Any] = dict(state.get("shared_state", {}))
    role_payloads: dict[str, Any] = {}
    for role in RESEARCH_ROLE_ORDER:
        snapshot, role_traces, ledger, analysis = future_by_role[role].result()
        snapshots.append(asdict(snapshot))
        traces.extend(role_traces)
        ledgers.append(ledger)
        role_outputs[role] = snapshot.summary
        role_payloads[role] = dict(analysis)
        shared_state.update(_shared_state_for_research_role(role, analysis))

    # Aggregate 4-role outputs into a directional signal via the Bayesian
    # online-capsule aggregator. The hook is always supplied by the backend;
    # NullAnalystMemory is used as the no-learning ablation control.
    summary = aggregator_hook(role_payloads, shared_state)
    signal = str(summary.get("direction") or "HOLD").upper()
    shared_state["proposed_action"] = signal
    shared_state["proposed_action_authority"] = "research_aggregate"
    shared_state["aggregate_research_score"] = summary["aggregate_score"]
    shared_state["aggregate_research_confidence"] = summary["aggregate_confidence"]
    shared_state["risk_uncertainty"] = summary.get("risk_uncertainty", 0.0)
    shared_state["expected_return_view"] = summary.get("expected_return_view", 0.0)
    # Bayesian-only diagnostic fields surfaced for downstream logging / paper.
    if "information_ratio" in summary:
        shared_state["information_ratio"] = summary["information_ratio"]
    if "tau_total" in summary:
        shared_state["tau_total"] = summary["tau_total"]
    if "role_precision_map" in summary:
        shared_state["role_precision_map"] = summary["role_precision_map"]

    raw_report_lines = [
        f"{role}: {role_outputs[role][:200]}"
        for role in RESEARCH_ROLE_ORDER
        if role in role_outputs
    ]
    raw_report = (
        f"Signal={signal} score={summary['aggregate_score']:.3f} "
        f"conf={summary['aggregate_confidence']:.3f} "
        f"risk_uncertainty={summary.get('risk_uncertainty', 0.0):.3f}\n"
        + "\n".join(raw_report_lines)
    )

    return {
        "snapshots": snapshots,
        "tool_traces": traces,
        "ledgers": ledgers,
        "role_outputs": role_outputs,
        "role_payloads": role_payloads,
        "shared_state": shared_state,
        "signal": signal,
        "raw_report": raw_report,
        "aggregator_summary": summary,
    }


def _run_research_role(
    symbol: str,
    trade_date: str,
    query: str,
    role: str,
    planner: ToolPlanner,
    executor: ToolExecutor,
    registry: ToolRegistry,
    llm_client: DarwinTradeLLMClient,
) -> tuple[BackendRoleSnapshot, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    request = _build_request(role, symbol, trade_date, query)
    plan = planner.score_and_rank(request)
    payload = {
        "symbol": symbol,
        "trade_date": trade_date,
        "ledger": {"symbol": symbol, "trade_date": trade_date},
        "disagreement": 0.25,
        "tool_traces": [],
        "verify_required": True,
    }
    ledger = executor.execute_plan(request, plan, payload)
    plan_dict = plan.to_dict()
    traces = _execution_traces_from_plan_dict(ledger, plan_dict, registry)
    outputs = _outputs_by_tool_name(ledger, registry)
    analysis = _llm_research_analysis(
        role=role,
        symbol=symbol,
        trade_date=trade_date,
        query=query,
        outputs=outputs,
        traces=traces,
        llm_client=llm_client,
    )
    confidence = max(0.0, min(float(analysis["confidence"]), 1.0))
    score = float(analysis["score"])
    snapshot = BackendRoleSnapshot(
        role=role,
        summary=str(analysis["summary"]),
        provenance=f"darwintrade_agentic.{role}",
        timestamp=trade_date,
        support=max(0.5, min(0.95, 0.5 + confidence * 0.45)),
        semantic_uncertainty=round(max(0.02, 1.0 - confidence), 4),
        evidence_uncertainty=0.18,
        tool_trace_refs=[str(item.get("tool_name", item.get("tool_id", ""))) for item in traces],
    )
    return snapshot, traces, ledger.to_dict(), {
        "summary": snapshot.summary,
        "score": score,
        "confidence": confidence,
        "market_regime": str(analysis["market_regime"]),
    }


def _llm_research_analysis(
    *,
    role: str,
    symbol: str,
    trade_date: str,
    query: str,
    outputs: dict[str, Any],
    traces: list[dict[str, Any]],
    llm_client: DarwinTradeLLMClient,
) -> dict[str, Any]:
    role_instruction = build_role_instruction(role)
    system_prompt = (
        "You are a real DarwinTrade research analyst. "
        "Use ONLY the provided tool outputs and produce a concise, evidence-grounded assessment. "
        "Your score is graded against the NEXT TRADING DAY's return: a positive score means "
        "you expect this asset to OUTPERFORM over the next 1 bar, negative means underperform. "
        "Anchor your conviction to that short horizon — a strong long-term thesis with no "
        "near-term catalyst warrants a modest score and lower confidence, not a maximal one. "
        "Return strict JSON only.\n\n"
        f"{role_instruction}"
    )
    user_prompt = json.dumps(
        to_jsonable(
            {
                "task": "research_role_analysis",
                "role": role,
                "symbol": symbol,
                "trade_date": trade_date,
                "query": query,
                "required_output_schema": {
                    "summary": "string",
                    "score": "float in [-1,1] where positive is bullish and negative is bearish",
                    "confidence": "float in [0,1]",
                    "market_regime": "bull|bear|sideways; required for every research role; use sideways when the role has no direct regime evidence",
                },
                "tool_traces": traces,
            }
        ),
        ensure_ascii=True,
        sort_keys=True,
    )
    logger.debug(
        "[LLM_PROMPT][research] role=%s symbol=%s trade_date=%s system_chars=%s user_chars=%s total_chars=%s est_tokens=%s trace_count=%s",
        role,
        symbol,
        trade_date,
        len(system_prompt),
        len(user_prompt),
        len(system_prompt) + len(user_prompt),
        _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt),
        len(traces),
    )
    payload = _invoke_llm_json(
        llm_client,
        system_prompt,
        user_prompt,
        caller_tag=f"research:{role}:{symbol}:{trade_date}",
        validator=lambda item: _validate_research_llm_payload(
            item,
            caller_tag=f"research:{role}:{symbol}:{trade_date}",
        ),
        response_format=_research_response_format(),
    )
    return payload


_TOOL_ENABLED_CONTROL_ROLES = {
    "risk_manager",
    "portfolio_manager",
    "verification_agent",
}
def _research_parallelism() -> int:
    configured = os.getenv("DARWINTRADE_ROLE_PARALLELISM")
    if configured:
        try:
            return max(1, min(len(RESEARCH_ROLE_ORDER), int(configured)))
        except ValueError:
            pass
    return len(RESEARCH_ROLE_ORDER)


def _shared_state_for_research_role(
    role: str, analysis: dict[str, Any]
) -> dict[str, Any]:
    score = float(analysis.get("score", 0.0) or 0.0)
    confidence = _clip(float(analysis.get("confidence", 0.0) or 0.0), 0.0, 1.0)
    if role == "market_analyst":
        return {
            "market_score": score,
            "market_confidence": confidence,
            "market_regime": str(analysis["market_regime"]),
        }
    if role == "news_analyst":
        return {"news_score": score, "news_confidence": confidence}
    if role == "fundamentals_analyst":
        return {"fundamentals_score": score, "fundamentals_confidence": confidence}
    if role == "social_analyst":
        return {"social_score": score, "social_confidence": confidence}
    return {}


def _research_response_format() -> dict[str, Any]:
    return build_json_schema_response_format(
        "research_role_analysis",
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "score": {"type": "number"},
                "confidence": {"type": "number"},
                "market_regime": {"type": "string", "enum": ["bull", "bear", "sideways"]},
            },
            "required": ["summary", "score", "confidence", "market_regime"],
            "additionalProperties": False,
        },
    )
def _invoke_llm_json(
    llm_client: DarwinTradeLLMClient,
    system_prompt: str,
    user_prompt: str,
    *,
    caller_tag: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def _call_and_parse() -> dict[str, Any]:
        def _full_validator(content: str) -> None:
            payload = parse_llm_json_dict(content, repair_invalid_escapes=True)
            if validator is not None:
                validator(payload)
        invoke_once = getattr(llm_client, "invoke_text_once", None)
        if callable(invoke_once):
            try:
                raw = invoke_once(
                    system_prompt,
                    user_prompt,
                    caller_tag=caller_tag,
                    response_format=response_format,
                    validator=_full_validator,
                )
            except TypeError:
                raw = invoke_once(
                    system_prompt,
                    user_prompt,
                    caller_tag=caller_tag,
                )
        else:
            try:
                raw = llm_client.invoke(
                    system_prompt,
                    user_prompt,
                    caller_tag=caller_tag,
                    response_format=response_format,
                    validator=_full_validator,
                )
            except TypeError:
                raw = llm_client.invoke(
                    system_prompt,
                    user_prompt,
                    caller_tag=caller_tag,
                )
        try:
            payload = parse_llm_json_dict(raw, repair_invalid_escapes=True)
        except Exception:
            raise ValueError(f"LLM did not return valid JSON: {raw[:1200]}")
        return validator(payload) if validator is not None else payload

    retries = getattr(llm_client, "max_retries", 100)
    retry_interval_seconds = getattr(llm_client, "retry_interval_seconds", 3.0)
    if retries is None:
        retries = 100
    if retry_interval_seconds is None:
        retry_interval_seconds = 3.0
    return _invoke_with_retries(
        _call_and_parse,
        retries=int(retries),
        retry_interval_seconds=float(retry_interval_seconds),
        caller_tag=caller_tag,
    )
def _require_payload_string(payload: dict[str, Any], key: str, *, caller_tag: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{caller_tag} missing required string field: {key}")
    return value.strip()


def _require_payload_float(payload: dict[str, Any], key: str, *, caller_tag: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{caller_tag} missing required numeric field: {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{caller_tag} invalid numeric field: {key}") from exc
def _validate_research_llm_payload(
    payload: dict[str, Any],
    *,
    caller_tag: str,
) -> dict[str, Any]:
    market_regime = _require_payload_string(payload, "market_regime", caller_tag=caller_tag).lower()
    if market_regime not in {"bull", "bear", "sideways"}:
        raise ValueError(f"{caller_tag} invalid market_regime: {market_regime}")
    return {
        "summary": _require_payload_string(payload, "summary", caller_tag=caller_tag),
        "score": max(-1.0, min(_require_payload_float(payload, "score", caller_tag=caller_tag), 1.0)),
        "confidence": max(
            0.0,
            min(_require_payload_float(payload, "confidence", caller_tag=caller_tag), 1.0),
        ),
        "market_regime": market_regime,
    }
def _build_request(role: str, symbol: str, trade_date: str, query: str) -> EvidenceRequest:
    if role == "market_analyst":
        return build_market_request(symbol, trade_date, query)
    if role in {"news_analyst", "social_analyst"}:
        return build_news_request(symbol, trade_date, query)
    return build_fundamentals_request(symbol, trade_date, query)


def _outputs_by_tool_name(
    ledger: EvidenceLedger, registry: ToolRegistry
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for execution in ledger.executions:
        outputs[registry.get(execution.tool_id).tool_name] = execution.output
    return outputs


def _execution_traces_from_plan_dict(
    ledger: EvidenceLedger, plan_dict: dict[str, Any], registry: ToolRegistry
) -> list[dict[str, Any]]:
    ranked_tools = {
        str(item.get("tool_id", "")): item
        for item in list(plan_dict.get("ranked_tools", []))
        if item.get("tool_id")
    }
    traces: list[dict[str, Any]] = []
    for execution in ledger.executions:
        record = registry.get(execution.tool_id)
        item = ranked_tools.get(execution.tool_id, {})
        traces.append(
            {
                "step": execution.step,
                "tool_name": f"{record.server}:{record.tool_name}",
                "tool_id": execution.tool_id,
                "arguments": execution.arguments,
                "output": execution.output,
                "plan_trace": dict(item.get("trace", {})),
                "compliant": bool(item.get("compliant", False)),
                "status": execution.status,
                "error": execution.error,
            }
        )
    traces.sort(key=lambda item: int(item.get("step", 0) or 0))
    return traces
__all__ = ["DarwinTradeAgenticAnalystBackend", "build_analyst_graph"]
