from __future__ import annotations
from typing import Any
from ....contracts import AssetView, DownsideProfile, DownsideTailPoint, PortfolioState
from ....interfaces import AnalyzerBackend, AnalyzerResult
from .state import empty_portfolio_state, optional_float, optional_int
POSITIVE_SIGNALS = {"buy", "overweight", "strong_buy", "accumulate"}
NEGATIVE_SIGNALS = {"sell", "underweight", "strong_sell", "reduce"}


def analyzer_result_to_asset_view(result: AnalyzerResult) -> AssetView:
    signal = str(result.signal or "hold").strip().lower()
    confidence = _resolve_confidence(metadata=result.metadata)
    uncertainty = _metadata_float(result.metadata, "uncertainty", 1.0 - confidence)
    disagreement = _resolve_metric(
        metadata=result.metadata,
        key="disagreement",
        default=uncertainty,
    )
    downside_risk = _resolve_metric(
        metadata=result.metadata,
        key="downside_risk",
        default=uncertainty,
    )
    expected_return_view = _resolve_expected_return_view(metadata=result.metadata)
    proposed_direction = _resolve_proposed_direction(
        signal=signal,
        metadata=result.metadata,
    )
    return AssetView(
        symbol=result.symbol,
        trade_date=result.trade_date,
        expected_return_view=expected_return_view,
        confidence=_clip(confidence, 0.0, 1.0),
        downside_risk=_clip(downside_risk, 0.0, 1.0),
        disagreement=_clip(disagreement, 0.0, 1.0),
        horizon_days=optional_int(result.metadata.get("horizon_days")) or 5,
        proposed_direction=proposed_direction,
        event_type=str(result.metadata.get("event_type", "") or "") or None,
        event_half_life_days=optional_int(result.metadata.get("event_half_life_days")),
        liquidity_risk=_clip_optional(optional_float(result.metadata.get("liquidity_risk")), 0.0, 1.0),
        regime_sensitivity=_normalized_regime_sensitivity(result.metadata.get("regime_sensitivity")),
        downside_distribution=_normalized_downside_distribution(result.metadata.get("downside_distribution")),
        downside_profile=_normalized_downside_profile(
            result.metadata.get("downside_profile"),
            fallback_distribution=result.metadata.get("downside_distribution"),
        ),
        rationale=result.raw_report,
        source=result.backend_name,
        metadata={
            "raw_signal": result.signal,
            "role_report_count": len(result.role_reports),
            "claim_count": len(result.claims),
            **dict(result.metadata),
        },
    )


def _metadata_float(metadata: dict[str, Any], key: str, default: float) -> float:
    value = metadata.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _required_metadata_float(metadata: dict[str, Any], key: str) -> float:
    if key not in metadata:
        raise ValueError(f"AnalyzerResult metadata missing structured {key}")
    parsed = optional_float(metadata.get(key))
    if parsed is None:
        raise ValueError(f"AnalyzerResult metadata has invalid structured {key}")
    return parsed


def _resolve_confidence(*, metadata: dict[str, Any]) -> float:
    return _required_metadata_float(metadata, "confidence")


def _resolve_metric(
    *, metadata: dict[str, Any], key: str, default: float
) -> float:
    del default
    return _required_metadata_float(metadata, key)


def _normalize_direction(signal: str) -> str:
    if signal in POSITIVE_SIGNALS:
        return "buy"
    if signal in NEGATIVE_SIGNALS:
        return "sell"
    return "hold"


def _resolve_expected_return_view(*, metadata: dict[str, Any]) -> float:
    for key in (
        "expected_return_view",
        "expected_return",
        "alpha",
        "alpha_view",
        "forecast_return",
    ):
        if key in metadata:
            value = _normalized_expected_return_value(metadata.get(key))
            if value is not None:
                return value
    raise ValueError("AnalyzerResult metadata missing structured expected_return_view")


def _resolve_proposed_direction(
    *, signal: str, metadata: dict[str, Any]
) -> str:
    for key in ("proposed_direction", "direction"):
        if key in metadata:
            value = _normalize_direction(str(metadata.get(key, "hold") or "hold").strip().lower())
            if value != "hold":
                return value
    return _normalize_direction(signal)


def _normalized_expected_return_value(value: Any) -> float | None:
    parsed = optional_float(value)
    if parsed is None:
        return None
    if abs(parsed) > 1.0:
        parsed /= 100.0
    return round(_clip(parsed, -1.0, 1.0), 6)


def _mapping_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _clip_optional(value: float | None, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    return _clip(value, lower, upper)


def _normalized_regime_sensitivity(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, float] = {}
    for key, item in value.items():
        parsed = optional_float(item)
        if parsed is not None:
            normalized[str(key)] = _clip(parsed, -1.0, 1.0)
    return normalized or None


def _normalized_downside_distribution(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, float] = {}
    for key, item in value.items():
        parsed = optional_float(item)
        if parsed is not None:
            normalized[str(key)] = float(parsed)
    return normalized or None


def _normalized_downside_profile(value: Any, *, fallback_distribution: Any = None) -> DownsideProfile | None:
    if isinstance(value, DownsideProfile):
        return value
    if isinstance(value, dict):
        raw_tail_points = value.get("tail_points")
        if raw_tail_points is None:
            reserved = {"tail_points", "source", "horizon_days", "confidence", "metadata"}
            raw_tail_points = {
                label: tail_value
                for label, tail_value in value.items()
                if label not in reserved and _parse_tail_level(str(label)) is not None
            }
        tail_points = _tail_points_from_any(raw_tail_points, source=str(value.get("source", "typed_metadata") or "typed_metadata"))
        if tail_points:
            return DownsideProfile(
                tail_points=tail_points,
                horizon_days=optional_int(value.get("horizon_days")),
                source=str(value.get("source", "typed_metadata") or "typed_metadata"),
                confidence=_clip_optional(optional_float(value.get("confidence")), 0.0, 1.0),
                metadata=_mapping_dict(value.get("metadata")),
            )
    fallback = _normalized_downside_distribution(fallback_distribution)
    if fallback:
        tail_points = _tail_points_from_any(fallback, source="distribution_fallback")
        if tail_points:
            return DownsideProfile(tail_points=tail_points, source="distribution_fallback")
    return None


def _tail_points_from_any(value: Any, *, source: str) -> list[DownsideTailPoint]:
    tail_points: list[DownsideTailPoint] = []
    if isinstance(value, dict):
        iterable = [{"label": label, "return_value": tail_value} for label, tail_value in value.items()]
    elif isinstance(value, list):
        iterable = value
    else:
        iterable = []
    for item in iterable:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "")
        if not label:
            continue
        level = optional_float(item.get("level"))
        if level is None:
            level = _parse_tail_level(label)
        return_value = optional_float(item.get("return_value"))
        if return_value is None:
            continue
        tail_points.append(
            DownsideTailPoint(
                label=label,
                level=level if level is not None else 0.05,
                return_value=min(float(return_value), 0.0),
                probability_mass=optional_float(item.get("probability_mass")),
                source=str(item.get("source", source) or source),
                metadata=_mapping_dict(item.get("metadata")),
            )
        )
    return sorted(tail_points, key=lambda point: point.level)


def _parse_tail_level(label: str) -> float | None:
    text = str(label).lower().strip()
    if text.startswith("q"):
        text = text[1:]
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        value /= 100.0
    if 0.0 < value < 0.5:
        return value
    return None
