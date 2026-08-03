"""One-shot backfill for storage/cache/news_by_day/<SYM>/<YYYY-MM-DD>.json.

Uses Polygon `/v2/reference/news` (Finnhub free tier caps retention around 12
months, insufficient for a 2025-03..2025-06 window queried in 2026-07). Reads
POLYGON_API_KEY from the environment (source .envmm before running). For every
(symbol, trade_day) that has no cache file within the requested window, calls
Polygon news over a padded range and writes normalized items bucketed by
publication date.

Not a replacement for the online path in data_hub — this exclusively populates
the pre-fetched cache so the backtest can run with data.mode=offline_only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from backtest.stockbench.adapters.polygon_client import PolygonClient

CACHE_ROOT = REPO_ROOT / "storage" / "cache" / "news_by_day"
DEFAULT_UNIVERSE = [
    "GS", "MSFT", "HD", "V", "SHW", "CAT", "MCD", "UNH", "AXP", "AMGN",
    "TRV", "CRM", "JPM", "IBM", "HON", "BA", "AMZN", "AAPL", "PG", "JNJ",
]


def _trade_days(start: str, end: str, reference_symbol: str = "SPY") -> list[str]:
    """Enumerate trading days from the reference symbol's parquet cache.
    Defaults to SPY. Falls back to the first symbol that has parquet coverage
    for the requested window if the reference doesn't cover it.
    """
    def _dates_for(sym: str) -> list[str]:
        d = REPO_ROOT / "storage" / "cache" / "parquet" / sym / "day"
        return sorted(p.stem for p in d.glob("*.parquet"))

    dates = _dates_for(reference_symbol)
    filt = [d for d in dates if start <= d <= end]
    if filt:
        return filt
    for candidate in ("TSLA", "AMZN", "MSFT", "NFLX", "AAPL"):
        dates = _dates_for(candidate)
        filt = [d for d in dates if start <= d <= end]
        if filt:
            return filt
    return []


def _missing_days(symbol: str, trade_days: list[str]) -> list[str]:
    """A day is 'missing' if either the file is absent OR it's a previously
    written empty stub (source==polygon_backfill, items==[]). Stubs from a
    failed earlier attempt should be re-fetched, not accepted as ground truth.
    """
    d = CACHE_ROOT / symbol.upper()
    if not d.exists():
        return list(trade_days)
    missing: list[str] = []
    for t in trade_days:
        path = d / f"{t}.json"
        if not path.exists():
            missing.append(t)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            missing.append(t)
            continue
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        source = payload.get("source") if isinstance(payload, dict) else None
        if not items and source == "polygon_backfill":
            missing.append(t)
    return missing


def _news_unique_key(item: dict) -> str:
    _id = str(item.get("id") or "").strip()
    if _id:
        return f"id:{_id}"
    url = str(item.get("url") or item.get("article_url") or "").strip()
    if url:
        return f"url:{url}"
    return f"tt:{item.get('title','').strip()}|{item.get('published_utc','').strip()}"


def _normalize_polygon_item(item: dict, symbol: str) -> dict:
    return {
        "id": str(item.get("id") or "").strip(),
        "title": str(item.get("title") or "").strip(),
        "description": str(item.get("description") or "").strip(),
        "published_utc": str(item.get("published_utc") or "").strip(),
        "source": str(item.get("publisher", {}).get("name") if isinstance(item.get("publisher"), dict) else item.get("publisher") or "").strip(),
        "url": str(item.get("article_url") or item.get("url") or "").strip(),
        "category": "",
        "image": str(item.get("image_url") or "").strip(),
        "symbol": symbol,
    }


def _write_day_cache(symbol: str, day: str, items: list[dict]) -> None:
    d = CACHE_ROOT / symbol.upper()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{day}.json"
    existing: list[dict] = []
    seen: set[str] = set()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            existing = list(payload.get("items", []) if isinstance(payload, dict) else payload)
            for it in existing:
                seen.add(_news_unique_key(it))
        except Exception:
            existing = []
            seen = set()
    merged = list(existing)
    for it in items:
        key = _news_unique_key(it)
        if key not in seen:
            merged.append(it)
            seen.add(key)
    merged.sort(key=lambda x: str(x.get("published_utc", "")), reverse=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"items": merged, "source": "polygon_backfill"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def _fetch_symbol_window(client: PolygonClient, symbol: str, gte: str, lte: str, max_pages: int = 40) -> list[dict]:
    """Paginate Polygon news over a full range in one call chain."""
    items: list[dict] = []
    cursor: str | None = None
    for page in range(max_pages):
        batch, next_cursor = client.list_ticker_news(
            symbol,
            gte=gte,
            lte=lte,
            limit=1000,
            page_token=cursor,
        )
        if batch:
            items.extend(batch)
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.15)  # polite spacing
    return items


def backfill(symbol: str, missing_days: list[str], client: PolygonClient, window_start: str, window_end: str) -> tuple[int, int]:
    if not missing_days:
        return 0, 0
    # One padded fetch per symbol across the whole window — cheaper than
    # chunking, and Polygon caps at 1000 items per page.
    pad_start = (datetime.strptime(window_start, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
    pad_end = (datetime.strptime(window_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = _fetch_symbol_window(client, symbol, pad_start, pad_end)
    print(f"  [{symbol}] fetched {len(raw)} items over {pad_start}..{pad_end}", flush=True)
    if not raw:
        return 0, 0

    trade_day_set = set(missing_days)
    buckets: dict[str, list[dict]] = {}
    for it in raw:
        ts = str(it.get("published_utc") or "").strip()
        if not ts:
            continue
        try:
            d = pd.to_datetime(ts).tz_localize(None).date().isoformat()
        except Exception:
            continue
        if d in trade_day_set:
            buckets.setdefault(d, []).append(_normalize_polygon_item(it, symbol))
    # Write stub files (empty items) for missing days that Polygon reported
    # zero news on. The analyst-side reader tolerates empty item lists — the
    # stub distinguishes "no news that day per Polygon" from "cache miss".
    for d in trade_day_set:
        if d not in buckets:
            buckets[d] = []
    for d, day_items in buckets.items():
        _write_day_cache(symbol, d, day_items)
    return len(buckets), sum(len(v) for v in buckets.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill news_by_day cache from Polygon")
    parser.add_argument("--start", default="2025-03-03")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_UNIVERSE)
    args = parser.parse_args()

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not in env; source .envmm first", file=sys.stderr)
        return 2
    client = PolygonClient(api_key)
    if not client.api_key:
        print("ERROR: PolygonClient reports no key", file=sys.stderr)
        return 2

    trade_days = _trade_days(args.start, args.end)
    print(f"trading days in window: {len(trade_days)} ({trade_days[0]}..{trade_days[-1]})", flush=True)

    total_days = 0
    total_items = 0
    for sym in args.symbols:
        missing = _missing_days(sym, trade_days)
        print(f"[{sym}] missing days pre-backfill: {len(missing)}", flush=True)
        if not missing:
            continue
        d, n = backfill(sym, missing, client, args.start, args.end)
        total_days += d
        total_items += n
        # Polygon rate-limits fairly aggressively; pause between symbols so a
        # 429 avalanche doesn't wipe out later symbols.
        time.sleep(2.0)

    print(f"\nbackfill done: {total_days} days written, {total_items} items", flush=True)

    still_missing: dict[str, list[str]] = {}
    for sym in args.symbols:
        miss = _missing_days(sym, trade_days)
        if miss:
            still_missing[sym] = miss
    if still_missing:
        print("\n=== REMAINING GAPS (no news published on these trading days per Polygon) ===")
        for sym, days in still_missing.items():
            print(f"  {sym}: {len(days)} days ({days[0]}..{days[-1]})")
        return 1
    print("\n=== zero gaps ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
