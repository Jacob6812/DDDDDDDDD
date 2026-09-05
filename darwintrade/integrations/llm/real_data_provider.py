from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import importlib
import json
import math
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from darwintrade.paths import PACKAGE_ROOT, REPO_ROOT


def _as_date(value: str | None) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.utcnow()
    return datetime.strptime(text, "%Y-%m-%d")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _filter_statement_frame_by_trade_date(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    normalized_trade_date = _as_date(trade_date).date()
    keep_columns: list[Any] = []
    for column in frame.columns:
        parsed = _parse_article_timestamp(column)
        if parsed is None:
            continue
        if parsed.date() <= normalized_trade_date:
            keep_columns.append(column)
    if not keep_columns:
        raise RuntimeError(f"no statement periods available on or before {trade_date}")
    return frame.loc[:, keep_columns]


def _latest_numeric_value(records: list[dict[str, Any]], field: str) -> float | None:
    for record in records:
        value = record.get(field)
        try:
            if value is None or pd.isna(value):
                continue
            return float(value)
        except Exception:
            continue
    return None


def _previous_numeric_value(records: list[dict[str, Any]], field: str) -> float | None:
    seen = 0
    for record in records:
        value = record.get(field)
        try:
            if value is None or pd.isna(value):
                continue
            seen += 1
            if seen == 2:
                return float(value)
        except Exception:
            continue
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return float(numerator) / float(denominator)


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}"


def _summarize_statement_fundamentals(output: dict[str, Any]) -> tuple[str, float]:
    balance_sheet = dict(output.get("get_balance_sheet", {}) or {}).get("balance_sheet", {})
    cashflow = dict(output.get("get_cashflow", {}) or {}).get("cashflow", {})
    income_statement = dict(output.get("get_income_statement", {}) or {}).get("income_statement", {})
    bs_records = list(balance_sheet.get("records", []) or [])
    cf_records = list(cashflow.get("records", []) or [])
    is_records = list(income_statement.get("records", []) or [])

    latest_revenue = _latest_numeric_value(is_records, "Total Revenue")
    previous_revenue = _previous_numeric_value(is_records, "Total Revenue")
    latest_net_income = _latest_numeric_value(is_records, "Net Income")
    latest_operating_cash_flow = _latest_numeric_value(cf_records, "Operating Cash Flow")
    latest_free_cash_flow = _latest_numeric_value(cf_records, "Free Cash Flow")
    latest_current_assets = _latest_numeric_value(bs_records, "Current Assets")
    latest_current_liabilities = _latest_numeric_value(bs_records, "Current Liabilities")
    latest_total_debt = _latest_numeric_value(bs_records, "Total Debt")
    latest_stockholders_equity = _latest_numeric_value(bs_records, "Stockholders Equity")

    revenue_growth = None
    if latest_revenue is not None and previous_revenue not in (None, 0.0):
        revenue_growth = (latest_revenue / previous_revenue) - 1.0
    current_ratio = _safe_ratio(latest_current_assets, latest_current_liabilities)
    debt_to_equity = _safe_ratio(latest_total_debt, latest_stockholders_equity)

    score = 0.0
    if revenue_growth is not None:
        score += 0.15 if revenue_growth > 0 else -0.15
    if latest_net_income is not None:
        score += 0.15 if latest_net_income > 0 else -0.15
    if latest_operating_cash_flow is not None:
        score += 0.15 if latest_operating_cash_flow > 0 else -0.15
    if latest_free_cash_flow is not None:
        score += 0.1 if latest_free_cash_flow > 0 else -0.1
    if current_ratio is not None:
        score += 0.1 if current_ratio >= 1.0 else -0.1
    if debt_to_equity is not None:
        score += 0.05 if debt_to_equity <= 2.0 else -0.05

    summary = (
        "statement_revenue_growth="
        f"{_format_ratio(revenue_growth)}, net_income={_format_value(latest_net_income)}, "
        f"operating_cash_flow={_format_value(latest_operating_cash_flow)}, "
        f"free_cash_flow={_format_value(latest_free_cash_flow)}, "
        f"current_ratio={_format_ratio(current_ratio)}, debt_to_equity={_format_ratio(debt_to_equity)}"
    )
    return summary, max(-1.0, min(score, 1.0))


def summarize_statement_fundamentals(output: dict[str, Any]) -> tuple[str, float]:
    return _summarize_statement_fundamentals(output)
def _parse_article_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().replace(tzinfo=timezone.utc)
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            pass
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 1e12:
            raw /= 1000.0
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in ("%Y-%m-%d", "%Y%m%dT%H%M", "%Y%m%dT%H%M%S"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                parsed = None
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _filter_articles_by_trade_date(
    items: list[dict[str, Any]],
    trade_date: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in items:
        published_at = item.get("published_utc") or item.get("published") or item.get("pubDate")
        if _is_future_article(published_at, trade_date):
            continue
        filtered.append(item)
        if len(filtered) >= max(limit, 1):
            break
    return filtered
def _sanitize_runtime_context(context: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(context or {})
    allowed_keys = {
        "query",
        "question",
        "analyzer_memory",
        "episode_memory",
        "strategy_memory",
        "control_memory",
        "memory_hint",
    }
    return {key: raw[key] for key in allowed_keys if key in raw}
def sanitize_runtime_context(context: dict[str, Any] | None) -> dict[str, Any]:
    return _sanitize_runtime_context(context)
def _is_future_article(published_at: Any, trade_date: str) -> bool:
    # Anti-lookahead: since orders execute at trade_date's OPEN, we cannot
    # generally know intraday-of-trade_date news at decision time (pre-market
    # articles are the exception, but distinguishing them by timestamp alone
    # is fragile — some feeds only give day-precision, and some tag with
    # market-close time). Treat any article dated ON OR AFTER trade_date as
    # future information.
    published = _parse_article_timestamp(published_at)
    if published is None:
        return False
    return published.date() >= _as_date(trade_date).date()
def _sentiment_score(texts: list[str]) -> float:
    positive = {
        "beat",
        "beats",
        "growth",
        "strong",
        "record",
        "surge",
        "up",
        "gain",
        "bull",
        "approval",
        "partnership",
        "expands",
        "expansion",
        "profit",
        "profits",
    }
    negative = {
        "miss",
        "misses",
        "weak",
        "drop",
        "down",
        "fall",
        "falls",
        "bear",
        "lawsuit",
        "probe",
        "fraud",
        "delay",
        "cut",
        "cuts",
        "layoff",
        "risk",
        "risks",
    }
    score = 0.0
    tokens = 0
    for text in texts:
        words = [item.strip(".,:;!?()[]{}\"'").lower() for item in str(text or "").split()]
        for word in words:
            if not word:
                continue
            tokens += 1
            if word in positive:
                score += 1.0
            elif word in negative:
                score -= 1.0
    if tokens <= 0:
        return 0.0
    return score / max(tokens, 1)


def _load_data_hub() -> Any:
    return importlib.import_module("backtest.stockbench.core.data_hub")


def _format_alpha_vantage_datetime(date_input: str | datetime) -> str:
    if isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    text = str(date_input).strip()
    if len(text) == 13 and "T" in text:
        return text
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y%m%dT0000")
    except ValueError:
        return datetime.strptime(text, "%Y-%m-%d %H:%M").strftime("%Y%m%dT%H%M")


_STATEMENT_TOOL_SPECS = {
    "balance_sheet": {
        "tool_name": "get_balance_sheet",
        "section": "bs",
    },
    "cashflow": {
        "tool_name": "get_cashflow",
        "section": "cf",
    },
    "income_statement": {
        "tool_name": "get_income_statement",
        "section": "ic",
    },
}
_STATEMENT_RESULT_KEY_BY_TOOL_NAME = {
    str(spec["tool_name"]): result_key
    for result_key, spec in _STATEMENT_TOOL_SPECS.items()
}


def _provider_error(source: str, exc: Exception) -> dict[str, Any]:
    return {
        "source": source,
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }


def _normalized_news_articles(
    provider_items: list[dict[str, Any]],
    trade_date: str,
    *,
    limit: int,
    include_symbol: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    articles: list[dict[str, Any]] = []
    texts: list[str] = []
    for item in _filter_articles_by_trade_date(provider_items, trade_date, limit=limit):
        title = str(item.get("title") or "").strip()
        summary = str(item.get("description") or "").strip()
        if not title and not summary:
            continue
        texts.append(f"{title} {summary}".strip())
        article = {
            "title": title,
            "summary": summary,
            "link": str(item.get("url") or "").strip(),
            "source": str(item.get("source") or "").strip(),
            "published": str(item.get("published_utc") or "").strip(),
        }
        if include_symbol:
            article["symbol"] = str(item.get("symbol") or "").strip()
        articles.append(article)
    return articles, _sentiment_score(texts)


# Cache root for the analyst tools. Defaults to the repo's `storage/cache`, which
# is what backtests use. The live tool lets the user relocate it via
# `set_cache_root`, so these stay functions rather than module constants.
_CACHE_ROOT_OVERRIDE: Path | None = None

# Data mode handed to `data_hub` when the analyst tools fetch bars. Backtests pin
# this to `offline_only` so a 4-hour run can never stall on a rate limit, and so
# an experiment reads exactly the bars that were pre-cached. The live tool needs
# the opposite: today's bars are not in the cache yet, so it must be allowed to
# call out.
_BARS_DATA_MODE: str = "offline_only"


def set_cache_root(root: str | Path | None) -> None:
    global _CACHE_ROOT_OVERRIDE
    _CACHE_ROOT_OVERRIDE = Path(root) if root else None


def set_bars_data_mode(mode: str) -> None:
    """Set the data mode the analyst price-history tool passes to `data_hub`."""
    global _BARS_DATA_MODE
    normalized = str(mode or "").strip().lower()
    _BARS_DATA_MODE = (
        "auto" if normalized == "auto" else "offline_only"
    )


def _cache_root() -> Path:
    return _CACHE_ROOT_OVERRIDE or (REPO_ROOT / "storage" / "cache")


def _fundamentals_cache_root() -> Path:
    return _cache_root() / "fundamentals"


def _news_by_day_cache_root() -> Path:
    return _cache_root() / "news_by_day"


def _filing_text_cache_root() -> Path:
    return _cache_root() / "filing_text"


def _load_filing_text(symbol: str, trade_date: str) -> dict[str, Any]:
    """
    Read pre-fetched 10-K/10-Q full text from
    `storage/cache/filing_text/<SYMBOL>/<YYYY-MM-DD>.json`.

    File schema: `{"filing_k": "...", "filing_q": "...", "as_of": "..."}` (any
    field optional). Acts as a fundamentals fall-back: when present, callers
    feed the long-form filing text directly to the LLM in lieu of the
    structured statement records.

    Resolution: trade-date file first, otherwise the most recent file on or
    before trade_date in the same symbol directory. Returns
    `{"available": False}` when nothing is cached.
    """
    base = _filing_text_cache_root() / symbol.upper()
    if not base.exists():
        return {"available": False, "symbol": symbol, "trade_date": trade_date}
    target = _as_date(trade_date).date()
    chosen: Path | None = None
    chosen_date = None
    for path in sorted(base.glob("*.json")):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date > target:
            continue
        if chosen_date is None or file_date > chosen_date:
            chosen_date = file_date
            chosen = path
    if chosen is None:
        return {"available": False, "symbol": symbol, "trade_date": trade_date}
    try:
        with chosen.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        return {
            "available": False,
            "symbol": symbol,
            "trade_date": trade_date,
            "error": str(exc),
        }
    return {
        "available": True,
        "symbol": symbol,
        "trade_date": trade_date,
        "as_of": str(payload.get("as_of") or chosen_date.strftime("%Y-%m-%d")),
        "filing_k": str(payload.get("filing_k") or ""),
        "filing_q": str(payload.get("filing_q") or ""),
        "source_path": str(chosen),
    }


def _load_news_by_day_window(
    symbol: str,
    trade_date: str,
    *,
    look_back_days: int,
) -> list[dict[str, Any]]:
    """
    Read pre-fetched news from `storage/cache/news_by_day/<SYMBOL>/<YYYY-MM-DD>.json`.
    Returns the union of items in [trade_date - look_back_days + 1, trade_date].
    Each cached file is `{"items": [{title, description, published_utc, source, url, ...}]}`.
    """
    base = _news_by_day_cache_root() / symbol.upper()
    if not base.exists():
        return []
    end_date = _as_date(trade_date).date()
    start_date = end_date - timedelta(days=max(look_back_days - 1, 0))
    items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    cursor = start_date
    while cursor <= end_date:
        path = base / f"{cursor.strftime('%Y-%m-%d')}.json"
        cursor = cursor + timedelta(days=1)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        for it in (payload.get("items") or []):
            if not isinstance(it, dict):
                continue
            key = str(it.get("url") or it.get("id") or it.get("title") or "")
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            items.append(it)
    return items


def _load_global_news_window(
    trade_date: str,
    *,
    look_back_days: int,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Aggregate global news by unioning all per-symbol per-day cache entries.
    Stamps each item with the symbol whose folder it came from (when missing).
    """
    base = _news_by_day_cache_root()
    if not base.exists():
        return []
    end_date = _as_date(trade_date).date()
    start_date = end_date - timedelta(days=max(look_back_days - 1, 0))
    items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name
        cursor = start_date
        while cursor <= end_date:
            path = symbol_dir / f"{cursor.strftime('%Y-%m-%d')}.json"
            cursor = cursor + timedelta(days=1)
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            for it in (payload.get("items") or []):
                if not isinstance(it, dict):
                    continue
                key = str(it.get("url") or it.get("id") or it.get("title") or "")
                if key and key in seen_keys:
                    continue
                if key:
                    seen_keys.add(key)
                if not it.get("symbol"):
                    related = it.get("related_symbols") or []
                    it = dict(it)
                    it["symbol"] = (related[0] if related else symbol)
                items.append(it)
                if len(items) >= max(limit * 4, limit):
                    return items
    return items


def _statement_cache_file(result_key: str, symbol: str, trade_date: str) -> Path:
    tool_name = str(_STATEMENT_TOOL_SPECS[result_key]["tool_name"])
    return _fundamentals_cache_root() / tool_name / symbol.upper() / f"{trade_date}.json"


def _finnhub_reported_cache_file(symbol: str, freq: str = "quarterly") -> Path:
    return _fundamentals_cache_root() / "_sources" / "finnhub_financials_reported" / freq / f"{symbol.upper()}.json"


def _write_cache_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _read_cache_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _finnhub_request(path: str, params: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY environment variable is not set.")
    request = Request(
        "https://finnhub.io/api/v1" + path + "?" + urlencode({**params, "token": api_key}),
        headers={"User-Agent": "darwintrade/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"Finnhub HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Finnhub network error: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Finnhub returned non-JSON payload: {body[:200]}") from exc
    error_message = str(parsed.get("error") or "").strip()
    if error_message:
        raise RuntimeError(error_message)
    return dict(parsed)


def _fetch_finnhub_reported_financials(symbol: str, freq: str = "quarterly") -> dict[str, Any]:
    payload = _finnhub_request(
        "/stock/financials-reported",
        {"symbol": symbol.upper(), "freq": freq},
    )
    cache_payload = {
        "symbol": symbol.upper(),
        "freq": freq,
        "vendor": "finnhub_financials_reported",
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload": payload,
    }
    _write_cache_json(_finnhub_reported_cache_file(symbol, freq=freq), cache_payload)
    return payload


def _load_finnhub_reported_financials(symbol: str, freq: str = "quarterly") -> dict[str, Any]:
    cached = _read_cache_json(_finnhub_reported_cache_file(symbol, freq=freq))
    if cached is not None:
        return dict(cached.get("payload") or {})
    return _fetch_finnhub_reported_financials(symbol, freq=freq)


def _eligible_finnhub_reports(reported_payload: dict[str, Any], trade_date: str) -> list[dict[str, Any]]:
    cutoff = _as_date(trade_date).date()
    eligible: list[tuple[datetime, dict[str, Any]]] = []
    for item in list(reported_payload.get("data") or []):
        end_date = _parse_article_timestamp(item.get("endDate"))
        if end_date is None or end_date.date() > cutoff:
            continue
        eligible.append((end_date, dict(item)))
    eligible.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in eligible]


def _statement_aliases(result_key: str, concept: str, label: str) -> list[str]:
    aliases: list[str] = []
    concept_lower = concept.lower()
    label_lower = label.lower()
    if result_key == "income_statement":
        if concept_lower in {
            "us-gaap_revenuefromcontractwithcustomerexcludingassessedtax",
            "us-gaap_salesrevenuenet",
            "us-gaap_revenues",
        } or "total revenue" in label_lower or label_lower == "net sales":
            aliases.append("Total Revenue")
        if concept_lower == "us-gaap_netincomeloss" or label_lower == "net income":
            aliases.append("Net Income")
    elif result_key == "cashflow":
        if concept_lower == "us-gaap_netcashprovidedbyusedinoperatingactivities" or "operating activities" in label_lower:
            aliases.append("Operating Cash Flow")
        if concept_lower == "us-gaap_paymentstoacquirepropertyplantandequipment" or "capital expenditure" in label_lower or "property, plant and equipment" in label_lower:
            aliases.append("Capital Expenditures")
    elif result_key == "balance_sheet":
        if concept_lower == "us-gaap_assetscurrent" or "current assets" in label_lower:
            aliases.append("Current Assets")
        if concept_lower == "us-gaap_liabilitiescurrent" or "current liabilities" in label_lower:
            aliases.append("Current Liabilities")
        if concept_lower == "us-gaap_stockholdersequity" or concept_lower == "us-gaap_stockholdersequityincludingportionattributabletononcontrollinginterest" or "stockholders' equity" in label_lower or "shareholders’ equity" in label_lower or "shareholders' equity" in label_lower:
            aliases.append("Stockholders Equity")
        if concept_lower in {
            "us-gaap_debtandfinanceleaseobligations",
            "us-gaap_longtermdebtandfinanceleaseobligations",
            "us-gaap_longtermdebtcurrent",
            "us-gaap_longtermdebtnoncurrent",
        } or "total debt" in label_lower:
            aliases.append("Total Debt")
    return aliases


def _build_statement_row(report: dict[str, Any], result_key: str) -> dict[str, Any]:
    row = {
        "period": str(report.get("endDate") or "").split(" ")[0],
        "filedDate": str(report.get("filedDate") or "").split(" ")[0],
        "acceptedDate": str(report.get("acceptedDate") or "").strip(),
        "form": str(report.get("form") or "").strip(),
        "fiscalYear": report.get("year"),
        "fiscalQuarter": report.get("quarter"),
    }
    section = str(_STATEMENT_TOOL_SPECS[result_key]["section"])
    report_section = dict(report.get("report") or {}).get(section) or []
    for item in report_section:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        label = str(item.get("label") or item.get("concept") or "").strip()
        concept = str(item.get("concept") or "").strip()
        if label:
            row[label] = value
        for alias in _statement_aliases(result_key, concept, label):
            row[alias] = value
    if result_key == "cashflow" and row.get("Free Cash Flow") in {None, ""}:
        operating_cash_flow = row.get("Operating Cash Flow")
        capital_expenditures = row.get("Capital Expenditures")
        if operating_cash_flow not in {None, ""} and capital_expenditures not in {None, ""}:
            try:
                row["Free Cash Flow"] = float(operating_cash_flow) + float(capital_expenditures)
            except Exception:
                pass
    return row


def _build_statement_payload_from_finnhub(symbol: str, trade_date: str, result_key: str, reported_payload: dict[str, Any]) -> dict[str, Any]:
    eligible = _eligible_finnhub_reports(reported_payload, trade_date)
    if not eligible:
        raise RuntimeError(f"no {result_key} available on or before {trade_date} for {symbol}")
    rows = [_build_statement_row(report, result_key) for report in eligible[:12]]
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    latest = eligible[0]
    return {
        result_key: {
            "symbol": symbol,
            "trade_date": trade_date,
            "source_report_end_date": str(latest.get("endDate") or "").split(" ")[0],
            "source_report_filed_date": str(latest.get("filedDate") or "").split(" ")[0],
            "records": rows,
            "columns": columns,
            "row_count": len(rows),
        },
        "available": True,
        "provider": "darwintrade",
        "vendor": "finnhub_financials_reported",
    }


def _build_statement_cache_entry(symbol: str, trade_date: str, result_key: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_version": 1,
        "symbol": symbol,
        "trade_date": trade_date,
        "tool_name": _STATEMENT_TOOL_SPECS[result_key]["tool_name"],
        "status": status,
        "vendor": payload.get("vendor") or "finnhub_financials_reported",
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload": payload,
    }


def _load_cached_statement_payload(symbol: str, trade_date: str, result_key: str) -> dict[str, Any] | None:
    cached = _read_cache_json(_statement_cache_file(result_key, symbol, trade_date))
    if cached is None:
        return None
    payload = dict(cached.get("payload") or {})
    if str(cached.get("status") or "") == "error":
        raise RuntimeError(str(payload.get("error") or f"cached {result_key} error for {symbol} on {trade_date}"))
    return payload


def _materialize_statement_payload(symbol: str, trade_date: str, result_key: str) -> dict[str, Any]:
    cached = _load_cached_statement_payload(symbol, trade_date, result_key)
    if cached is not None:
        return cached
    try:
        reported_payload = _load_finnhub_reported_financials(symbol)
        payload = _build_statement_payload_from_finnhub(symbol, trade_date, result_key, reported_payload)
        _write_cache_json(
            _statement_cache_file(result_key, symbol, trade_date),
            _build_statement_cache_entry(symbol, trade_date, result_key, "ok", payload),
        )
        return payload
    except Exception as exc:
        error_payload = {
            "available": False,
            "provider": "darwintrade",
            "vendor": "finnhub_financials_reported",
            "error": str(exc),
        }
        _write_cache_json(
            _statement_cache_file(result_key, symbol, trade_date),
            _build_statement_cache_entry(symbol, trade_date, result_key, "error", error_payload),
        )
        raise


class AlphaVantageRateLimitError(RuntimeError):
    pass


def _alpha_vantage_request(function_name: str, params: dict[str, Any]) -> Any:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")
    query = {
        **params,
        "function": function_name,
        "apikey": api_key,
        "source": "darwintrade",
    }
    request = Request(
        "https://www.alphavantage.co/query?" + urlencode(query),
        headers={"User-Agent": "darwintrade/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"Alpha Vantage HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Alpha Vantage network error: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body
    info_message = str(parsed.get("Information") or "")
    if info_message and ("rate limit" in info_message.lower() or "api key" in info_message.lower()):
        raise AlphaVantageRateLimitError(info_message)
    error_message = str(parsed.get("Error Message") or "")
    if error_message:
        raise RuntimeError(error_message)
    return parsed


class DarwinTradeMCPProvider:
    provider_name = "darwintrade"

    @staticmethod
    def _require_symbol(symbol: str, tool_name: str) -> str:
        if not symbol:
            raise RuntimeError(f"{tool_name} requires symbol")
        return symbol

    def execute(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        symbol = str(payload.get("symbol") or payload.get("focus_symbol") or "").strip().upper()
        trade_date = str(payload.get("trade_date") or payload.get("curr_date") or datetime.utcnow().strftime("%Y-%m-%d"))

        result_key = _STATEMENT_RESULT_KEY_BY_TOOL_NAME.get(tool_name)
        if result_key is not None:
            return self._get_statement(
                self._require_symbol(symbol, tool_name),
                result_key,
                trade_date,
            )
        if tool_name == "get_stock_data":
            return self._get_stock_data(
                self._require_symbol(symbol, tool_name),
                payload,
                trade_date,
            )
        if tool_name == "get_indicators":
            return self._get_indicators(self._require_symbol(symbol, tool_name), trade_date)
        if tool_name == "get_news":
            return self._get_news(
                self._require_symbol(symbol, tool_name),
                payload,
                trade_date,
            )
        if tool_name == "get_global_news":
            return self._get_global_news(
                trade_date,
                limit=int(payload.get("limit") or 10),
                look_back_days=int(payload.get("look_back_days") or 7),
            )
        if tool_name == "get_filing_text":
            return self._get_filing_text(
                self._require_symbol(symbol, tool_name),
                trade_date,
            )
        if tool_name == "get_insider_transactions":
            import yfinance as yf

            return self._get_insider_transactions(
                yf.Ticker(self._require_symbol(symbol, tool_name)),
                symbol,
                trade_date,
            )
        raise RuntimeError(f"Unsupported DarwinTrade MCP tool: {tool_name}")

    def _get_history(
        self,
        symbol: str,
        *,
        end_date: str,
        look_back_days: int,
        include_trade_date_open: bool = False,
    ) -> pd.DataFrame:
        # Anti-lookahead contract for bar T (= end_date):
        #   - orders execute at T's OPEN, so T's open is legitimate decision-time
        #     information (the market already fixed it before we act).
        #   - T's close/high/low/volume are still in the future at decision time
        #     and must never enter the LLM prompt.
        # Two callers:
        #   - _get_stock_data (analyst LLM prompt) → include_trade_date_open=True.
        #     We fetch bars through T, but MASK T's close/high/low/volume so only
        #     the open survives. The analyst sees a synthetic partial bar for T
        #     with just `open` populated.
        #   - _get_indicators (SMA/MACD/RSI/ATR) → include_trade_date_open=False.
        #     Technical indicators need full OHLC, so we only feed them fully-
        #     realised T-1 bars and stop there.
        data_hub = _load_data_hub()
        trade_day = _as_date(end_date).date()
        if include_trade_date_open:
            fetch_end = _as_date(end_date)
        else:
            fetch_end = _as_date(end_date) - timedelta(days=1)
        start = fetch_end - timedelta(days=max(look_back_days, 30))
        history = data_hub.get_bars(
            ticker=symbol,
            start=start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            multiplier=1,
            timespan="day",
            adjusted=True,
            cfg={"data": {"mode": _BARS_DATA_MODE}},
        )

        # data_hub returns OHLCV schema: date/open/high/low/close/volume/vwap
        if history is None or getattr(history, "empty", True):
            raise RuntimeError("no price history returned")
        history = history.copy()
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Date": "date",
        }
        history = history.rename(columns=rename_map)
        if "date" not in history.columns:
            history = history.reset_index().rename(columns={"index": "date"})
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history = history.dropna(subset=["date"])
        for column in ["open", "high", "low", "close", "volume"]:
            if column in history.columns:
                history[column] = pd.to_numeric(history[column], errors="coerce")

        if include_trade_date_open:
            # Mask T's close/high/low/volume so the analyst only sees T's open.
            # The row must survive downstream dropna(subset=["close"]) filters:
            # we do NOT drop rows on close here (that would delete the partial
            # T-day bar). Callers that rely on a fully-realised close series
            # must filter to date < trade_day themselves.
            trade_day_mask = history["date"].dt.date == trade_day
            for column in ("close", "high", "low", "volume"):
                if column in history.columns:
                    history.loc[trade_day_mask, column] = pd.NA
            # Rows where even OPEN is missing (data feed hiccup) are useless.
            history = history.dropna(subset=["open"])
        else:
            # Historical-only mode: require a valid close.
            history = history.dropna(subset=["close"])

        if history.empty:
            raise RuntimeError("no price history returned")
        return history.sort_values("date").reset_index(drop=True)

    def _get_stock_data(self, symbol: str, payload: dict[str, Any], trade_date: str) -> dict[str, Any]:
        # The analyst is allowed to see T's OPEN (that's the execution price and
        # was fixed by the market before we act) but must NOT see T's close /
        # high / low / volume. `_get_history(include_trade_date_open=True)`
        # returns a partial T-day bar with only `open` populated; close/high/
        # low/volume are NA and serialized as None in `records`.
        look_back_days = int(payload.get("look_back_days") or 60)
        history = self._get_history(
            symbol,
            end_date=trade_date,
            look_back_days=look_back_days,
            include_trade_date_open=True,
        )
        trade_day = _as_date(trade_date).date()

        def _optional(value: Any) -> Any:
            v = _safe_float(value, float("nan"))
            return None if math.isnan(v) else round(v, 4)

        records = []
        for _, row in history.tail(20).iterrows():
            row_date = row.get("date")
            is_trade_day_partial = (
                hasattr(row_date, "date") and row_date.date() == trade_day
            )
            records.append(
                {
                    "date": _normalize_timestamp(row_date),
                    "open": round(_safe_float(row.get("open")), 4),
                    "high": _optional(row.get("high")),
                    "low": _optional(row.get("low")),
                    "close": _optional(row.get("close")),
                    "volume": None if is_trade_day_partial else int(_safe_float(row.get("volume"), 0.0)),
                    **({"note": "trade_date open only; close/high/low/volume not yet realized"}
                       if is_trade_day_partial else {}),
                }
            )

        # `close` and `change_pct` summarize the last FULLY-REALISED bar (T-1),
        # never T's partial bar. Filter to rows with a real close.
        close_series = history["close"].dropna()
        if len(close_series) < 2:
            raise RuntimeError("insufficient close history returned")
        close_price = _safe_float(close_series.iloc[-1])
        prev_close = _safe_float(close_series.iloc[-2], close_price)
        change_pct = 0.0 if prev_close == 0 else round(((close_price / prev_close) - 1.0) * 100.0, 4)
        anchor = _safe_float(close_series.iloc[max(0, len(close_series) - 5)], close_price)
        return {
            "stock_data": {
                "symbol": symbol,
                "trade_date": trade_date,
                "records": records,
                "columns": ["date", "open", "high", "low", "close", "volume"],
                "row_count": len(history),
                "close": round(close_price, 4),          # T-1 close
                "change_pct": change_pct,                # (T-1 close vs T-2 close)
                "trend": "up" if close_price >= anchor else "down",
            },
            "available": True,
            "provider": "darwintrade",
            "vendor": "backtest_parquet",
        }

    def _get_indicators(self, symbol: str, trade_date: str) -> dict[str, Any]:
        # Technical indicators need full OHLC — the partial T-day bar would
        # break SMA/MACD/RSI/ATR. Feed them only fully-realised bars up to T-1.
        history = self._get_history(
            symbol, end_date=trade_date, look_back_days=260,
            include_trade_date_open=False,
        )
        close = history["close"].astype(float)
        high = history["high"].astype(float)
        low = history["low"].astype(float)
        if len(close) < 30:
            raise RuntimeError("insufficient history for indicator calculation")
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta.clip(upper=0.0)).mask(lambda series: series == 0.0)
        avg_gain = gains.rolling(window=14, min_periods=14).mean()
        avg_loss = losses.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean()
        sma50 = close.rolling(window=50, min_periods=50).mean()
        sma200 = close.rolling(window=200, min_periods=200).mean()
        ema10 = close.ewm(span=10, adjust=False).mean()
        latest_close = _safe_float(close.iloc[-1])
        latest_atr = _safe_float(atr.iloc[-1])
        values = {
            "close_50_sma": round(_safe_float(sma50.iloc[-1], latest_close), 4),
            "close_200_sma": round(_safe_float(sma200.iloc[-1], latest_close), 4),
            "close_10_ema": round(_safe_float(ema10.iloc[-1], latest_close), 4),
            "macd": round(_safe_float(macd.iloc[-1]), 6),
            "macds": round(_safe_float(macd_signal.iloc[-1]), 6),
            "macdh": round(_safe_float(macd_hist.iloc[-1]), 6),
            "rsi": round(_safe_float(rsi.iloc[-1], 50.0), 4),
            "atr": round(latest_atr, 4),
        }
        return {
            "indicators": {
                "symbol": symbol,
                "trade_date": trade_date,
                "indicator_count": len(values),
                "values": values,
                "rsi": values["rsi"],
                "macd_bias": "bullish" if values["macd"] >= values["macds"] else "bearish",
                "atr": values["atr"],
                "atr_regime": "high" if latest_close > 0 and latest_atr / latest_close >= 0.03 else "normal",
                "support_resistance": "support intact" if latest_close >= values["close_10_ema"] else "resistance overhead",
            },
            "available": True,
            "provider": "darwintrade",
            "vendor": "backtest_parquet",
        }

    def _alpha_vantage_news_items(
        self,
        symbol: str,
        trade_date: str,
        *,
        look_back_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not os.getenv("ALPHA_VANTAGE_API_KEY"):
            return []
        start_date = (_as_date(trade_date) - timedelta(days=max(look_back_days - 1, 0))).strftime("%Y-%m-%d")
        parsed = _alpha_vantage_request(
            "NEWS_SENTIMENT",
            {
                "tickers": symbol,
                "time_from": _format_alpha_vantage_datetime(start_date),
                "time_to": _format_alpha_vantage_datetime(trade_date),
                "limit": str(max(limit, 1)),
            },
        )
        feed = list((parsed or {}).get("feed") or [])
        articles: list[dict[str, Any]] = []
        for item in feed[: max(limit, 1)]:
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not title and not summary:
                continue
            articles.append(
                {
                    "title": title,
                    "description": summary,
                    "url": str(item.get("url") or "").strip(),
                    "source": str(item.get("source") or item.get("source_domain") or "").strip(),
                    "published_utc": str(item.get("time_published") or "").strip(),
                }
            )
        return articles

    def _alpha_vantage_global_news_items(
        self,
        trade_date: str,
        *,
        look_back_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not os.getenv("ALPHA_VANTAGE_API_KEY"):
            return []
        start_date = (_as_date(trade_date) - timedelta(days=max(look_back_days, 0))).strftime("%Y-%m-%d")
        parsed = _alpha_vantage_request(
            "NEWS_SENTIMENT",
            {
                "topics": "financial_markets,economy_macro,economy_monetary",
                "time_from": _format_alpha_vantage_datetime(start_date),
                "time_to": _format_alpha_vantage_datetime(trade_date),
                "limit": str(max(limit, 1)),
            },
        )
        feed = list((parsed or {}).get("feed") or [])
        articles: list[dict[str, Any]] = []
        for item in feed[: max(limit, 1)]:
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not title and not summary:
                continue
            articles.append(
                {
                    "title": title,
                    "description": summary,
                    "url": str(item.get("url") or "").strip(),
                    "source": str(item.get("source") or item.get("source_domain") or "").strip(),
                    "published_utc": str(item.get("time_published") or "").strip(),
                    "symbol": str(item.get("ticker_sentiment", [{}])[0].get("ticker") or "").strip() if item.get("ticker_sentiment") else "",
                }
            )
        return articles
    def _get_news(self, symbol: str, payload: dict[str, Any], trade_date: str) -> dict[str, Any]:
        look_back_days = int(payload.get("look_back_days") or 7)
        limit = int(payload.get("limit") or 10)
        provider_error: dict[str, Any] | None = None
        vendor = "news_by_day_cache"

        # Cache-first: read pre-fetched per-day news from storage/cache/news_by_day/<SYMBOL>/<date>.json
        provider_items = _load_news_by_day_window(
            symbol, trade_date, look_back_days=look_back_days
        )

        if not provider_items:
            # Cache miss → fall back to Alpha Vantage live fetch.
            vendor = "alpha_vantage_news"
            try:
                provider_items = self._alpha_vantage_news_items(
                    symbol,
                    trade_date,
                    look_back_days=look_back_days,
                    limit=limit,
                )
            except Exception as exc:
                provider_items = []
                provider_error = _provider_error("alpha_vantage_news", exc)

        articles, sentiment = _normalized_news_articles(
            provider_items,
            trade_date,
            limit=limit,
        )
        result = {
            "company_news": {
                "symbol": symbol,
                "trade_date": trade_date,
                "articles": articles,
                "article_count": len(articles),
                "headline_bias": "positive" if sentiment > 0.01 else "negative" if sentiment < -0.01 else "neutral",
                "confidence": round(min(0.9, 0.5 + abs(sentiment) * 5.0), 4) if articles else 0.0,
            },
            "available": bool(articles),
            "provider": "darwintrade",
            "vendor": vendor,
        }
        if provider_error is not None:
            result["provider_error"] = provider_error
        return result

    def _get_global_news(self, trade_date: str, *, limit: int, look_back_days: int = 7) -> dict[str, Any]:
        provider_error: dict[str, Any] | None = None
        vendor = "news_by_day_cache"

        provider_items = _load_global_news_window(
            trade_date, look_back_days=look_back_days, limit=limit
        )

        if not provider_items:
            vendor = "alpha_vantage_news"
            try:
                provider_items = self._alpha_vantage_global_news_items(
                    trade_date,
                    look_back_days=look_back_days,
                    limit=limit,
                )
            except Exception as exc:
                provider_items = []
                provider_error = _provider_error("alpha_vantage_news", exc)

        articles, sentiment = _normalized_news_articles(
            provider_items,
            trade_date,
            limit=limit,
            include_symbol=True,
        )
        result = {
            "global_news": {
                "trade_date": trade_date,
                "articles": articles,
                "article_count": len(articles),
                "macro_bias": "risk_on" if sentiment > 0.01 else "risk_off" if sentiment < -0.01 else "mixed",
            },
            "available": bool(articles),
            "provider": "darwintrade",
            "vendor": vendor,
        }
        if provider_error is not None:
            result["provider_error"] = provider_error
        return result

    def _get_insider_transactions(self, ticker: Any, symbol: str, trade_date: str) -> dict[str, Any]:
        frame = getattr(ticker, "insider_transactions", None)
        if frame is None or getattr(frame, "empty", True):
            raise RuntimeError(f"no insider transactions returned for {symbol}")
        normalized = frame.copy()

        # find the date column (yfinance uses "Start Date" or similar)
        date_col = next(
            (c for c in normalized.columns if "date" in str(c).lower() or "start" in str(c).lower()),
            None,
        )

        # filter out transactions after trade_date to prevent lookahead bias
        if date_col is not None:
            cutoff = _as_date(trade_date).date()
            normalized[date_col] = pd.to_datetime(normalized[date_col], errors="coerce")
            normalized = normalized[
                normalized[date_col].dt.date <= cutoff
            ]

        for column in normalized.columns:
            normalized[column] = normalized[column].map(
                _normalize_timestamp
                if "date" in str(column).lower() or "start" in str(column).lower()
                else lambda value: value
            )
        records = normalized.head(20).to_dict(orient="records")
        return {
            "insider_transactions": {
                "symbol": symbol,
                "trade_date": trade_date,
                "records": records,
                "columns": [str(item) for item in normalized.columns],
                "row_count": int(len(normalized)),
            },
            "available": True,
            "provider": "darwintrade",
            "vendor": "yfinance",
        }

    def _get_statement(self, symbol: str, result_key: str, trade_date: str) -> dict[str, Any]:
        return _materialize_statement_payload(symbol.upper(), trade_date, result_key)

    def _get_filing_text(self, symbol: str, trade_date: str) -> dict[str, Any]:
        result = _load_filing_text(symbol.upper(), trade_date)
        result.setdefault("provider", "darwintrade")
        result.setdefault("vendor", "filing_text_cache")
        return result
__all__ = ["DarwinTradeMCPProvider"]
