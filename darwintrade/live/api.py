"""
HTTP API for the live decision tool, plus the static single-page front end.

No authentication: this is a local research tool that binds to 127.0.0.1 by
default, and every decision spends LLM tokens and third-party API quota. Do not
expose it to a network without putting auth and rate limiting in front of it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DEFAULT_UNIVERSE, MAX_SYMBOLS_PER_REQUEST, LiveSettings
from .context import MarketDataUnavailable, resolve_trade_date
from .session import SessionError, SessionStore, _normalize_symbols

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class DecideRequest(BaseModel):
    symbols: list[str] = Field(
        default_factory=list,
        description="Tickers to analyse. Comma- or space-separated entries are split.",
    )
    trade_date: str | None = Field(
        default=None,
        description="YYYY-MM-DD. Defaults to the latest NYSE trading day.",
    )
    capital: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Starting equity for a new session. Omit for continued sessions."
    )
    session_id: str | None = Field(
        default=None,
        description="Reuse a session so memory carries over. Omit to create one.",
    )


class SessionRequest(BaseModel):
    capital: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    label: str = ""


def create_app(settings: LiveSettings | None = None) -> FastAPI:
    resolved = settings or LiveSettings.from_env()
    _apply_data_settings(resolved)
    store = SessionStore(resolved)

    app = FastAPI(
        title="DarwinTrade",
        description=(
            "Self-evolving multi-agent long/short equity decisions. Research "
            "output only — not investment advice."
        ),
        version="1.0.0",
    )
    app.state.settings = resolved
    app.state.store = store

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "llm_configured": resolved.llm_configured(),
            "data_mode": resolved.data_mode,
            "cache_root": str(resolved.cache_root),
            "benchmark": resolved.benchmark,
            "default_universe": list(DEFAULT_UNIVERSE),
            "default_capital": resolved.default_capital,
            "max_symbols": MAX_SYMBOLS_PER_REQUEST,
        }

    @app.post("/api/sessions")
    def create_session(payload: SessionRequest) -> dict[str, Any]:
        session = store.create(capital=payload.capital, label=payload.label)
        return {
            "session_id": session.session_id,
            "capital": session.state.capital,
            "label": session.state.label,
            "created_at": session.state.created_at,
        }

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": store.list()}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            session = store.get(session_id)
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return session.state.to_dict()

    @app.post("/api/decide")
    def decide(payload: DecideRequest) -> dict[str, Any]:
        try:
            _normalize_symbols(payload.symbols)
            resolve_trade_date(payload.trade_date, resolved)
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MarketDataUnavailable as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not resolved.llm_configured():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No LLM_API_KEY configured. Copy .env.example to .env and "
                    "fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL."
                ),
            )
        try:
            session = (
                store.get(payload.session_id)
                if payload.session_id
                else store.create(capital=payload.capital)
            )
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            return session.decide(
                symbols=payload.symbols,
                trade_date=payload.trade_date,
                capital=payload.capital,
            )
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MarketDataUnavailable as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("decision failed")
            raise HTTPException(
                status_code=500, detail="Analysis failed. Check the server log and provider configuration before retrying."
            ) from exc

    if STATIC_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
        )

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _apply_data_settings(settings: LiveSettings) -> None:
    """Point the data layer at the configured cache and data mode.

    Must run before the first data access: `data_hub` reads its cache constants
    at call time, so switching mid-run would split the cache across two roots.
    """
    from backtest.stockbench.core import data_hub
    from darwintrade.integrations.llm import real_data_provider

    settings.cache_root.mkdir(parents=True, exist_ok=True)
    data_hub.set_cache_root(str(settings.cache_root))
    data_hub.set_data_mode(settings.data_mode)
    data_hub.refresh_api_clients()
    real_data_provider.set_cache_root(settings.cache_root)
    # Backtests pin the analyst price tool to the cache so a long run cannot
    # stall on a rate limit. Live decisions need today's bars, which are not
    # cached yet, so the analyst tools must be allowed to call out too.
    real_data_provider.set_bars_data_mode(settings.data_mode)


__all__ = ["DecideRequest", "SessionRequest", "create_app"]
