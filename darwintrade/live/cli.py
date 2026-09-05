"""
Console entry points: `darwintrade serve` and `darwintrade predict`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from darwintrade.paths import REPO_ROOT


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env", override=False)


def _settings(args: argparse.Namespace):
    from .config import LiveSettings

    settings = LiveSettings.from_env()
    overrides = {}
    if getattr(args, "cache_dir", None):
        overrides["cache_root"] = Path(args.cache_dir).expanduser()
    if getattr(args, "session_dir", None):
        overrides["session_root"] = Path(args.session_dir).expanduser()
    if getattr(args, "offline", False):
        overrides["data_mode"] = "offline_only"
    if not overrides:
        return settings
    import dataclasses

    return dataclasses.replace(settings, **overrides)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Market-data cache root (default: storage/cache).",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Where session memory is persisted (default: storage/live/sessions).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never call third-party APIs; read only what is already cached.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darwintrade",
        description="Self-evolving multi-agent long/short equity decisions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the web UI and HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Auto-reload on edits.")
    _add_data_args(serve)

    predict = sub.add_parser("predict", help="Print one decision as JSON.")
    predict.add_argument("symbols", nargs="+", help="Tickers, e.g. AAPL MSFT NVDA")
    predict.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest).")
    predict.add_argument("--capital", type=float, default=None)
    predict.add_argument(
        "--session",
        default=None,
        help="Reuse a session id so memory carries over between runs.",
    )
    _add_data_args(predict)

    return parser


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    settings = _settings(args)
    if not settings.llm_configured():
        print(
            "[warn] No LLM_API_KEY found. The UI will load, but decisions will "
            "fail until you fill in .env",
            file=sys.stderr,
        )
    if args.host not in {"127.0.0.1", "localhost"}:
        print(
            f"[warn] Binding to {args.host}. This API has no authentication and "
            "each request spends LLM and market-data quota — put auth in front "
            "of it before exposing it.",
            file=sys.stderr,
        )
    print(f"DarwinTrade UI: http://{args.host}:{args.port}")
    if args.reload:
        import os
        os.environ["DARWINTRADE_CACHE_DIR"] = str(settings.cache_root.resolve())
        os.environ["DARWINTRADE_SESSION_DIR"] = str(settings.session_root.resolve())
        os.environ["DARWINTRADE_DATA_MODE"] = settings.data_mode
        # The reload worker rebuilds the app from these resolved overrides.
        uvicorn.run(
            "darwintrade.live.api:create_app",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True,
        )
    else:
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def _predict(args: argparse.Namespace) -> int:
    from .api import _apply_data_settings
    from .context import MarketDataUnavailable
    from .session import SessionError, SessionStore

    settings = _settings(args)
    if not settings.llm_configured():
        print(
            "No LLM_API_KEY configured. Copy .env.example to .env and fill in "
            "LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.",
            file=sys.stderr,
        )
        return 2

    _apply_data_settings(settings)
    store = SessionStore(settings)
    try:
        session = (
            store.get(args.session) if args.session else store.create(capital=args.capital)
        )
        payload = session.decide(
            symbols=list(args.symbols),
            trade_date=args.date,
            capital=args.capital,
        )
    except (SessionError, MarketDataUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"\nsession: {payload['session_id']}  "
        f"(pass --session {payload['session_id']} to keep evolving it)",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "predict":
        return _predict(args)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
