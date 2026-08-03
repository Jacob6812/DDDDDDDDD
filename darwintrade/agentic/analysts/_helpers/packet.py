from __future__ import annotations

import math
from typing import Any

from ....contracts import (
    AssetResearchPacket,
    AssetView,
    DisagreementProfile,
    EvidenceLedger,
    EvidenceLedgerEntry,
    PortfolioState,
    ToolCallTrace,
)
from ....interfaces import AnalyzerBackend, AnalyzerResult
from .state import empty_portfolio_state
from .views import analyzer_result_to_asset_view


def build_asset_research_packet(
    result: AnalyzerResult,
    *,
    analysis_mode: str = "full",
) -> AssetResearchPacket:
    asset_view = analyzer_result_to_asset_view(result)
    tool_traces = _normalize_tool_traces(result.metadata.get("tool_traces"))
    verification_summary = _verification_summary(result, tool_traces)
    evidence_ledger = _build_evidence_ledger(result, tool_traces, verification_summary, horizon_days=asset_view.horizon_days)
    disagreement_profile = _build_disagreement_profile(result)
    role_reports = [report.to_dict() for report in result.role_reports]
    claims = [claim.to_dict() for claim in result.claims]
    packet_metadata = {
        "analysis_mode": analysis_mode,
        "source_backend": result.backend_name,
        "signal": result.signal,
        "role_report_count": len(role_reports),
        "claim_count": len(claims),
        "tool_trace_count": len(tool_traces),
        **dict(result.metadata),
    }
    asset_view.metadata = {
        **dict(asset_view.metadata),
        "research_packet": {
            "analysis_mode": analysis_mode,
            "role_report_count": len(role_reports),
            "claim_count": len(claims),
            "tool_trace_count": len(tool_traces),
            "verification_status": verification_summary["verification_status"],
        },
    }
    return AssetResearchPacket(
        symbol=result.symbol,
        trade_date=result.trade_date,
        asset_view=asset_view,
        role_reports=role_reports,
        claims=claims,
        tool_traces=tool_traces,
        evidence_ledger=evidence_ledger,
        disagreement_profile=disagreement_profile,
        verification_summary=verification_summary,
        metadata=packet_metadata,
    )
def _normalize_tool_traces(raw_value: Any) -> list[ToolCallTrace]:
    traces: list[ToolCallTrace] = []
    for index, item in enumerate(raw_value or [], start=1):
        if isinstance(item, ToolCallTrace):
            traces.append(item)
            continue
        if not isinstance(item, dict):
            continue
        traces.append(
            ToolCallTrace(
                step=int(item.get("step", index) or index),
                tool_name=str(item.get("tool_name", "tool") or "tool"),
                arguments=dict(item.get("arguments", {}) or {}),
                output=dict(item.get("output", {}) or {}),
            )
        )
    return traces


def _verification_summary(
    result: AnalyzerResult,
    tool_traces: list[ToolCallTrace],
) -> dict[str, Any]:
    verify_required = any(getattr(claim, "verify_flags", None) for claim in result.claims)
    verification_status = str(result.metadata.get("verification_status", "") or "")
    if not verification_status:
        if verify_required:
            verification_status = "observed" if tool_traces else "missing"
        else:
            verification_status = "not_required"
    return {
        "verification_status": verification_status,
        "verify_required": verify_required,
        "tool_trace_count": len(tool_traces),
        "claim_count": len(result.claims),
        "role_report_count": len(result.role_reports),
    }


def _build_evidence_ledger(
    result: AnalyzerResult,
    tool_traces: list[ToolCallTrace],
    verification_summary: dict[str, Any],
    *,
    horizon_days: int,
) -> EvidenceLedger:
    entries: list[EvidenceLedgerEntry] = []
    support_values: list[float] = []
    for index, claim in enumerate(result.claims, start=1):
        refs = list(getattr(claim, "evidence_refs", []) or [])
        ref_supports = [float(ref.support) for ref in refs if getattr(ref, "support", None) is not None]
        support_values.extend(ref_supports)
        entries.append(
            EvidenceLedgerEntry(
                key=f"claim:{index}:{claim.stance}",
                source_count=max(len(refs), 1),
                avg_support=round(sum(ref_supports) / len(ref_supports), 6) if ref_supports else 0.5,
                timestamps=[str(ref.timestamp) for ref in refs if getattr(ref, "timestamp", None)],
                refs=refs,
            )
        )
    if not entries:
        entries.append(
            EvidenceLedgerEntry(
                key="report:summary",
                source_count=max(len(result.role_reports), 1),
                avg_support=round(max(0.0, 1.0 - float(result.metadata.get("disagreement", 0.0) or 0.0)), 6),
                timestamps=[result.trade_date],
                refs=[],
            )
        )
    support_mean = round(sum(support_values) / len(support_values), 6) if support_values else round(max(0.0, 1.0 - float(result.metadata.get("disagreement", 0.0) or 0.0)), 6)
    return EvidenceLedger(
        asset=result.symbol,
        horizon=f"{horizon_days}d",
        entries=entries,
        source_coverage=round(min(1.0, len(entries) / max(len(result.role_reports), 1)), 6),
        support_mean=support_mean,
        conflict_density=round(float(result.metadata.get("disagreement", 0.0) or 0.0), 6),
        verify_required=bool(verification_summary["verify_required"]),
        verify_flags=sorted({flag for claim in result.claims for flag in getattr(claim, "verify_flags", [])}),
        verification_status=str(verification_summary["verification_status"]),
    )


def _build_disagreement_profile(result: AnalyzerResult) -> DisagreementProfile:
    disagreement = float(result.metadata.get("disagreement", result.metadata.get("uncertainty", 0.0)) or 0.0)
    action_probs = _action_probabilities(str(result.signal or "hold"), disagreement)
    top_action = max(action_probs, key=lambda action: action_probs[action])
    return DisagreementProfile(
        action_probs=action_probs,
        entropy=_entropy(action_probs),
        top_action=top_action,
        top_action_prob=round(action_probs[top_action], 6),
        vote_variance=round(disagreement * 0.25, 6),
        support_weight_mean=round(max(0.0, 1.0 - float(result.metadata.get("downside_risk", disagreement) or disagreement)), 6),
        directional_dispersion=round(disagreement, 6),
    )


def _action_probabilities(signal: str, disagreement: float) -> dict[str, float]:
    normalized = str(signal or "hold").strip().lower()
    disagreement = max(0.0, min(float(disagreement), 1.0))
    primary = max(0.34, 1.0 - disagreement)
    residual = max(0.0, 1.0 - primary)
    if normalized in {"buy", "overweight", "strong_buy", "accumulate"}:
        return {
            "buy": round(primary, 6),
            "hold": round(residual * 0.65, 6),
            "sell": round(residual * 0.35, 6),
        }
    if normalized in {"sell", "underweight", "strong_sell", "reduce"}:
        return {
            "buy": round(residual * 0.35, 6),
            "hold": round(residual * 0.65, 6),
            "sell": round(primary, 6),
        }
    hold_share = max(0.34, 1.0 - disagreement * 0.5)
    residual = max(0.0, 1.0 - hold_share)
    return {
        "buy": round(residual * 0.5, 6),
        "hold": round(hold_share, 6),
        "sell": round(residual * 0.5, 6),
    }


def _entropy(action_probs: dict[str, float]) -> float:
    normalized_total = sum(action_probs.values()) or 1.0
    entropy = 0.0
    for probability in action_probs.values():
        normalized = probability / normalized_total
        if normalized <= 0:
            continue
        entropy -= normalized * math.log(normalized, 2)
    return round(entropy, 6)
