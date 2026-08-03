from __future__ import annotations

from datetime import datetime
from typing import Any


class _FallbackFrame:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.columns = list(records[0].keys()) if records else []

    def head(self, limit: int):
        return _FallbackFrame(self._records[:limit])

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        if orient != "records":
            raise ValueError("Only records orientation is supported")
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


def _unavailable(reason: str, payload: dict[str, Any], *, function_name: str = "") -> dict[str, Any]:
    return {
        "available": False,
        "provider": "akshare",
        "function": function_name,
        "reason": reason,
        "payload": dict(payload),
    }


def _normalize_frame(frame: Any, *, limit: int = 5) -> dict[str, Any]:
    if hasattr(frame, "head") and hasattr(frame, "to_dict"):
        sample = frame.head(limit)
        records = sample.to_dict(orient="records")
        return {
            "records": records,
            "columns": [str(item) for item in getattr(frame, "columns", [])],
            "row_count": len(frame) if hasattr(frame, "__len__") else len(records),
        }
    if isinstance(frame, list):
        return {
            "records": frame[:limit],
            "columns": list(frame[0].keys()) if frame and isinstance(frame[0], dict) else [],
            "row_count": len(frame),
        }
    if isinstance(frame, dict):
        return {
            "records": [frame],
            "columns": sorted(frame.keys()),
            "row_count": 1,
        }
    fallback = _FallbackFrame([{"value": str(frame)}])
    return {
        "records": fallback.to_dict(),
        "columns": fallback.columns,
        "row_count": len(fallback),
    }


def _load_akshare() -> Any | None:
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return None
    return ak


def _call_akshare(function_name: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ak = _load_akshare()
    if ak is None:
        return _unavailable("akshare_unavailable", payload, function_name=function_name)
    func = getattr(ak, function_name, None)
    if not callable(func):
        return _unavailable("function_not_available", payload, function_name=function_name)
    try:
        result = func(**kwargs)
    except Exception as exc:
        return {
            **_unavailable("execution_failed", payload, function_name=function_name),
            "error": str(exc),
            "arguments": kwargs,
        }
    return {
        "available": True,
        "provider": "akshare",
        "function": function_name,
        "arguments": kwargs,
        **_normalize_frame(result),
    }
def stock_us_daily(payload: dict[str, Any]) -> dict[str, Any]:
    # Anti-lookahead: akshare's stock_us_daily returns the full history including
    # today's close. In backtest mode (trade_date supplied) that is a hard leak,
    # so we filter the returned frame down to bars strictly before trade_date.
    symbol = str(payload.get("symbol") or "AAPL").upper()
    adjust = str(payload.get("adjust") or "")
    trade_date = str(payload.get("trade_date") or "").strip()
    result = _call_akshare("stock_us_daily", payload, symbol=symbol, adjust=adjust)
    if not trade_date or not result.get("available"):
        return result
    from datetime import datetime as _dt, timedelta as _td
    try:
        cutoff = (_dt.strptime(trade_date, "%Y-%m-%d") - _td(days=1)).date()
    except ValueError:
        return result
    records = result.get("records") or []
    date_key = None
    for key in ("date", "日期", "trade_date"):
        if records and key in records[0]:
            date_key = key
            break
    if date_key is None:
        # Cannot verify — treat as unusable in backtest mode to avoid leaking.
        return _unavailable(
            "no_date_column_for_lookahead_filter",
            payload,
            function_name="stock_us_daily",
        )
    def _parse(row: dict[str, Any]) -> Any:
        val = row.get(date_key)
        try:
            return _dt.strptime(str(val)[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None
    kept = [r for r in records if (d := _parse(r)) is not None and d <= cutoff]
    result["records"] = kept
    result["row_count"] = len(kept)
    return result


def stock_us_hist_min_em(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "AAPL").upper()
    secid = symbol if "." in symbol else f"105.{symbol}"
    trade_date = str(payload.get("trade_date") or datetime.utcnow().strftime("%Y-%m-%d"))
    # Anti-lookahead: intraday data must not cross the decision boundary. Our
    # orders execute at trade_date's open, so nothing after trade_date's opening
    # bell (or, safer, any bar dated trade_date) is admissible. Default the
    # window to the calendar day BEFORE trade_date. Callers who explicitly want
    # intraday-of-trade_date data can still pass start_date/end_date, but the
    # backtest never should.
    from datetime import datetime as _dt, timedelta as _td
    try:
        prev_day = (_dt.strptime(trade_date, "%Y-%m-%d") - _td(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        prev_day = trade_date
    start_date = str(payload.get("start_date") or f"{prev_day} 09:30:00")
    end_date = str(payload.get("end_date") or f"{prev_day} 16:00:00")
    return _call_akshare(
        "stock_us_hist_min_em",
        payload,
        symbol=secid,
        start_date=start_date,
        end_date=end_date,
    )


def stock_us_famous_spot_em(payload: dict[str, Any]) -> dict[str, Any]:
    # Anti-lookahead: this is a REALTIME spot snapshot with no historical
    # filter — using it during a backtest leaks the current market state.
    # Refuse when a historical trade_date is supplied.
    if str(payload.get("trade_date") or "").strip():
        return _unavailable(
            "realtime_snapshot_blocked_in_backtest",
            payload,
            function_name="stock_us_famous_spot_em",
        )
    category = str(payload.get("category") or "科技类")
    return _call_akshare("stock_us_famous_spot_em", payload, symbol=category)


def stock_value_em(payload: dict[str, Any]) -> dict[str, Any]:
    # Anti-lookahead: this valuation snapshot reflects the latest daily update
    # with no historical accessor, so it leaks into backtest queries. Refuse
    # when a historical trade_date is supplied.
    if str(payload.get("trade_date") or "").strip():
        return _unavailable(
            "point_in_time_snapshot_blocked_in_backtest",
            payload,
            function_name="stock_value_em",
        )
    symbol = str(payload.get("symbol") or "").strip()
    if not symbol:
        return _unavailable("missing_symbol", payload, function_name="stock_value_em")
    digits = "".join(ch for ch in symbol if ch.isdigit())
    if not digits:
        return _unavailable("unsupported_symbol_format", payload, function_name="stock_value_em")
    return _call_akshare("stock_value_em", payload, symbol=digits)
