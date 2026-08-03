from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import local
import json
import os
import sys
from typing import Any, Dict, List

from backtest.stockbench.core.executor import plan_orders


ROOT = Path(__file__).resolve().parents[3]
TRADINGAGENTS_ROOT = ROOT / "external" / "TradingAgents"
_TRADINGAGENTS_HOME = Path.home() / ".tradingagents"
_DEFAULT_RESULTS_DIR = _TRADINGAGENTS_HOME / "logs"
if str(TRADINGAGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRADINGAGENTS_ROOT))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_repo_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)
    shared_key = str(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("API_KEY")
        or ""
    )
    if shared_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = shared_key
    if shared_key and not os.getenv("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = shared_key
    base_url = str(os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL") or "").strip().lower()
    model = str(os.getenv("LLM_MODEL") or os.getenv("MODEL") or "").strip()
    if model == "deepseek/deepseek-chat" and "openrouter" in base_url:
        os.environ["MODEL"] = "deepseek-chat"
    elif model == "deepseek/deepseek-reasoner" and "openrouter" in base_url:
        os.environ["MODEL"] = "deepseek-reasoner"


_TA_DATA_RETRY_MAX_ATTEMPTS = 100
_TA_DATA_RETRY_INTERVAL_SEC = 3.0


def _patch_tradingagents_data_retry() -> None:
    """Wrap AlphaVantage + yfinance calls with 100 attempts / 3s retry.

    Free-tier AV rate-limits at 5 rpm and yfinance's unofficial endpoint
    aggressively throttles when 20 tickers hammer it in parallel. The
    LLM-level retry the graph already implements does not cover the tool
    executor; without this patch a single 429 kills the entire backtest.
    """
    import time as _time
    try:
        from tradingagents.dataflows import alpha_vantage_common as _avc  # noqa: WPS433
    except Exception:
        _avc = None
    try:
        import yfinance.data as _yfd  # noqa: WPS433
        from yfinance.exceptions import YFRateLimitError as _YFRateLimitError  # noqa: WPS433
    except Exception:
        _yfd = None
        _YFRateLimitError = None

    if _avc is not None and not getattr(_avc, "_stockbench_retry_guard", False):
        _orig_make_request = _avc._make_api_request

        def _retry_make_api_request(function_name, params):
            last_exc: Exception | None = None
            for attempt in range(1, _TA_DATA_RETRY_MAX_ATTEMPTS + 1):
                try:
                    return _orig_make_request(function_name, params)
                except _avc.AlphaVantageRateLimitError as exc:
                    last_exc = exc
                except Exception as exc:
                    last_exc = exc
                if attempt >= _TA_DATA_RETRY_MAX_ATTEMPTS:
                    break
                try:
                    sys.stderr.write(
                        f"[ta-av-retry] {function_name} attempt {attempt}/"
                        f"{_TA_DATA_RETRY_MAX_ATTEMPTS} failed: "
                        f"{type(last_exc).__name__}: {last_exc}. Sleeping "
                        f"{_TA_DATA_RETRY_INTERVAL_SEC}s.\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                _time.sleep(_TA_DATA_RETRY_INTERVAL_SEC)
            assert last_exc is not None
            raise last_exc

        _avc._make_api_request = _retry_make_api_request
        _avc._stockbench_retry_guard = True

        # Each alpha_vantage_* submodule does
        #   from .alpha_vantage_common import _make_api_request
        # which binds the pre-patch reference into its own namespace, so
        # rebind them explicitly so retries actually fire from every call
        # site (stock / indicator / fundamentals / news).
        try:
            from tradingagents.dataflows import (
                alpha_vantage_stock,
                alpha_vantage_indicator,
                alpha_vantage_fundamentals,
                alpha_vantage_news,
            )
            for _mod in (alpha_vantage_stock, alpha_vantage_indicator, alpha_vantage_fundamentals, alpha_vantage_news):
                if hasattr(_mod, "_make_api_request"):
                    _mod._make_api_request = _retry_make_api_request
        except Exception:
            pass

    if _yfd is not None and _YFRateLimitError is not None and not getattr(_yfd, "_stockbench_retry_guard", False):
        _orig_cache_get = _yfd.YfData.cache_get

        def _retry_cache_get(self, *args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, _TA_DATA_RETRY_MAX_ATTEMPTS + 1):
                try:
                    return _orig_cache_get(self, *args, **kwargs)
                except _YFRateLimitError as exc:
                    last_exc = exc
                except Exception as exc:
                    last_exc = exc
                if attempt >= _TA_DATA_RETRY_MAX_ATTEMPTS:
                    break
                try:
                    sys.stderr.write(
                        f"[ta-yf-retry] cache_get attempt {attempt}/"
                        f"{_TA_DATA_RETRY_MAX_ATTEMPTS} failed: "
                        f"{type(last_exc).__name__}: {last_exc}. Sleeping "
                        f"{_TA_DATA_RETRY_INTERVAL_SEC}s.\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                _time.sleep(_TA_DATA_RETRY_INTERVAL_SEC)
            assert last_exc is not None
            raise last_exc

        _yfd.YfData.cache_get = _retry_cache_get
        _yfd._stockbench_retry_guard = True


def _tradingagents_modules():
    _load_repo_env()
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    _patch_tradingagents_data_retry()

    # Register the repo-local polygon vendor so config data_vendors.<cat>
    # can point at "polygon" and route through DarwinTrade's parquet cache.
    try:
        from backtest.stockbench.strategies.tradingagents_polygon_vendor import (
            register_polygon_vendor,
        )

        register_polygon_vendor()
    except Exception as exc:
        sys.stderr.write(f"[ta-polygon-vendor] register failed: {exc}\n")

    return {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "TradingAgentsGraph": TradingAgentsGraph,
    }


class Strategy:
    def __init__(self, cfg: Dict) -> None:
        modules = _tradingagents_modules()
        self._default_graph_config = self._build_graph_config(modules["DEFAULT_CONFIG"])
        self._graph_cls = modules["TradingAgentsGraph"]

        self.cfg = cfg
        self.total_equity = float(
            (cfg or {}).get("portfolio", {}).get("total_cash", 100000.0)
        )
        self.records: list[dict] = []
        self.daily_output_dir: Path | None = None
        self.pending_decisions: dict[str, dict] = {}
        self._graph_state = local()

    def _selected_analysts(self) -> list[str]:
        configured = str(os.getenv("TRADINGAGENTS_SELECTED_ANALYSTS") or "").strip()
        if not configured:
            return ["market", "social", "news", "fundamentals"]
        selected = [item.strip().lower() for item in configured.split(",") if item.strip()]
        return selected or ["market", "social", "news", "fundamentals"]

    def _build_graph_config(self, default_config: dict, run_id: str | None = None) -> dict:
        config = dict(default_config)
        env_model = str(os.getenv("LLM_MODEL") or os.getenv("MODEL") or "").strip()
        env_base_url = str(os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL") or "").strip()
        normalized_model = env_model
        if env_model == "deepseek/deepseek-chat" and "openrouter" in env_base_url.lower():
            normalized_model = "deepseek-chat"
        elif env_model == "deepseek/deepseek-reasoner" and "openrouter" in env_base_url.lower():
            normalized_model = "deepseek-reasoner"
        if normalized_model:
            config["deep_think_llm"] = normalized_model
            config["quick_think_llm"] = normalized_model
        if env_base_url:
            config["backend_url"] = env_base_url
        config["llm_provider"] = str(
            os.getenv("TRADINGAGENTS_LLM_PROVIDER") or "openai"
        ).strip() or "openai"
        config["llm_timeout_seconds"] = float(
            os.getenv("TRADINGAGENTS_LLM_TIMEOUT") or config.get("llm_timeout_seconds", 300)
        )
        config["llm_max_retries"] = 0
        config["llm_retry_attempts"] = int(
            os.getenv("TRADINGAGENTS_LLM_MAX_RETRIES") or 100
        )
        config["llm_retry_interval_seconds"] = float(
            os.getenv("TRADINGAGENTS_LLM_RETRY_INTERVAL") or 3.0
        )
        config["max_debate_rounds"] = int(
            os.getenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS") or 1
        )
        config["max_risk_discuss_rounds"] = int(
            os.getenv("TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS") or 1
        )
        config["max_recur_limit"] = int(
            os.getenv("TRADINGAGENTS_MAX_RECUR_LIMIT") or 300
        )
        config["output_language"] = str(
            os.getenv("TRADINGAGENTS_OUTPUT_LANGUAGE") or "English"
        ).strip() or "English"
        # yfinance's public endpoint rate-limits under parallel load and
        # alpha_vantage's free tier caps at 25 requests/day — neither can
        # sustain a 20-symbol, 80-day backtest. Default to the repo-local
        # `polygon` vendor (see tradingagents_polygon_vendor.py) which
        # reads from our parquet cache and hits the Polygon client already
        # provisioned via POLYGON_API_KEY in .envmm.
        _default_vendor = "polygon"
        config["data_vendors"] = {
            "core_stock_apis": str(
                os.getenv("TRADINGAGENTS_CORE_STOCK_VENDOR") or _default_vendor
            ).strip() or _default_vendor,
            "technical_indicators": str(
                os.getenv("TRADINGAGENTS_TECHNICAL_VENDOR") or _default_vendor
            ).strip() or _default_vendor,
            "fundamental_data": str(
                os.getenv("TRADINGAGENTS_FUNDAMENTAL_VENDOR") or _default_vendor
            ).strip() or _default_vendor,
            "news_data": str(
                os.getenv("TRADINGAGENTS_NEWS_VENDOR") or _default_vendor
            ).strip() or _default_vendor,
        }
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            os.environ.setdefault(
                "ALPHAVANTAGE_API_KEY", os.environ["ALPHA_VANTAGE_API_KEY"]
            )
        if run_id:
            safe_run_id = str(run_id).replace("/", "_").replace("\\", "_")
            base_results_dir = Path(str(config.get("results_dir") or _DEFAULT_RESULTS_DIR))
            config["results_dir"] = str(base_results_dir / safe_run_id)
        return config

    def _parallel_workers(self) -> int:
        configured = (self.cfg or {}).get("tradingagents", {}).get("parallel_workers")
        if configured is None:
            configured = os.getenv("TRADINGAGENTS_PARALLEL_WORKERS")
        if configured is None:
            configured = os.cpu_count() or 4
        return max(int(configured), 1)

    def _graph(self, run_id: str | None = None):
        cache_key = str(run_id or "default")
        graph_cache = getattr(self._graph_state, "graphs", None)
        if graph_cache is None:
            graph_cache = {}
            self._graph_state.graphs = graph_cache
        if cache_key not in graph_cache:
            graph_cache[cache_key] = self._graph_cls(
                selected_analysts=self._selected_analysts(),
                debug=False,
                config=self._build_graph_config(self._default_graph_config, run_id=run_id),
            )
        return graph_cache[cache_key]

    def _analyze_symbol(
        self,
        *,
        symbol: str,
        date_str: str,
        total_equity: float,
        current_value: float,
        current_symbols: int,
        run_id: str | None,
    ) -> dict[str, Any]:
        # Broadcast the current trade date so the polygon vendor can
        # clamp any downstream tool query against it. Env-var based
        # because the vendor callables live in a separate module and
        # threads share the process env.
        os.environ["TA_RUN_TRADE_DATE"] = date_str
        _, raw_signal = self._graph(run_id=run_id).propagate(symbol, date_str)
        rating = self._extract_rating(raw_signal)
        target_cash_amount = self._target_cash_amount(
            rating=rating,
            total_equity=total_equity,
            current_value=current_value,
            current_symbols=current_symbols,
        )
        if rating in {"BUY", "OVERWEIGHT"}:
            action = "increase"
        elif rating in {"SELL", "UNDERWEIGHT"}:
            action = "close" if target_cash_amount <= 0 else "decrease"
        else:
            action = "hold"
        return {
            "symbol": symbol,
            "action": action,
            "rating": rating,
            "target_cash_amount": round(target_cash_amount, 2),
            "cash_change": round(target_cash_amount - current_value, 2),
            "confidence": 0.6,
            "reason": f"{raw_signal} | {self._graph_profile_label()}",
            "trade_date": date_str,
            "input_order": symbol,
        }

    def preflight_readiness(self) -> dict:
        issues: list[str] = []
        if not TRADINGAGENTS_ROOT.exists():
            issues.append("tradingagents_repo_missing")
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_API_KEY"):
            issues.append("llm_api_key_missing")
        return {
            "ready": not issues,
            "issues": issues,
            "strategy": self.__class__.__name__,
            "tradingagents_root": str(TRADINGAGENTS_ROOT),
            "selected_analysts": self._selected_analysts(),
        }

    def set_daily_output_dir(self, output_dir: str | Path) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.daily_output_dir = out_dir

    def _append_jsonl_row(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _extract_rating(self, decision_text: str) -> str:
        normalized = str(decision_text or "").strip().upper()
        for token in ["OVERWEIGHT", "UNDERWEIGHT", "BUY", "SELL", "HOLD"]:
            if token in normalized:
                return token
        return normalized or "HOLD"

    def _target_cash_amount(
        self,
        *,
        rating: str,
        total_equity: float,
        current_value: float,
        current_symbols: int,
    ) -> float:
        max_positions = max(
            int((self.cfg or {}).get("risk", {}).get("max_positions", 5) or 5), 1
        )
        universe_size = max(current_symbols, 1)
        base_weight = min(1.0 / max_positions, 1.0 / universe_size)
        overweight_weight = min(base_weight * 1.5, 0.35)
        underweight_weight = base_weight * 0.5
        if rating == "BUY":
            return round(total_equity * base_weight, 2)
        if rating == "OVERWEIGHT":
            return round(total_equity * overweight_weight, 2)
        if rating == "UNDERWEIGHT":
            return round(min(current_value, total_equity * underweight_weight), 2)
        if rating == "SELL":
            return 0.0
        return round(current_value, 2)

    def _graph_profile_label(self) -> str:
        return (
            f"analysts={self._selected_analysts()} "
            f"timeout={os.getenv('TRADINGAGENTS_LLM_TIMEOUT', '60')} "
            f"retries={os.getenv('TRADINGAGENTS_LLM_MAX_RETRIES', '1')}"
        )

    def on_bar(self, ctx: Dict) -> List[Dict]:
        date_str = ctx["date"].strftime("%Y-%m-%d")
        open_map = dict(ctx.get("open_map") or ctx.get("open_price_map") or {})
        portfolio = ctx.get("portfolio")
        total_equity = float(
            ctx.get("equity_for_sizing")
            or getattr(portfolio, "total_equity", 0.0)
            or self.total_equity
        )
        orders: list[dict] = []
        self.pending_decisions = {}
        run_id = str(ctx.get("run_id") or "").strip() or None
        symbols = [
            symbol
            for symbol in (ctx.get("symbols") or [])
            if float(open_map.get(symbol) or 0.0) > 0
        ]
        current_values = {
            symbol: (
                float(position.shares * float(open_map.get(symbol) or 0.0))
                if (position := portfolio.positions.get(symbol) if portfolio else None)
                else 0.0
            )
            for symbol in symbols
        }
        worker_count = min(self._parallel_workers(), max(len(symbols), 1))
        if worker_count == 1:
            analyzed = [
                self._analyze_symbol(
                    symbol=symbol,
                    date_str=date_str,
                    total_equity=total_equity,
                    current_value=current_values[symbol],
                    current_symbols=len(symbols),
                    run_id=run_id,
                )
                for symbol in symbols
            ]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        self._analyze_symbol,
                        symbol=symbol,
                        date_str=date_str,
                        total_equity=total_equity,
                        current_value=current_values[symbol],
                        current_symbols=len(symbols),
                        run_id=run_id,
                    )
                    for symbol in symbols
                ]
                analyzed = [future.result() for future in futures]
        analyzed.sort(key=lambda item: symbols.index(str(item.get("input_order") or item.get("symbol") or "")))
        for decision in analyzed:
            symbol = str(decision["symbol"])
            self.pending_decisions[symbol] = {key: value for key, value in decision.items() if key != "input_order"}
            planned = plan_orders(
                decision=self.pending_decisions[symbol],
                snapshot_price=float(open_map.get(symbol) or 0.0),
                cfg=self.cfg,
                portfolio={
                    "equity": total_equity,
                    "positions": {
                        ticker: {
                            "shares": float(getattr(pos, "shares", 0.0) or 0.0),
                            "position_value": float(getattr(pos, "shares", 0.0) or 0.0)
                            * float(
                                open_map.get(ticker)
                                or getattr(pos, "avg_price", 0.0)
                                or 0.0
                            ),
                        }
                        for ticker, pos in (portfolio.positions.items() if portfolio else [])
                    },
                },
            )
            orders.extend(planned)
            self.records.append({"date": date_str, **self.pending_decisions[symbol]})
            if self.daily_output_dir is not None:
                self._append_jsonl_row(
                    self.daily_output_dir / "tradingagents_portfolio_decisions.daily.jsonl",
                    {"date": date_str, **self.pending_decisions[symbol]},
                )
        return orders

    def record_executed_decisions(
        self, executed_symbols: list[str], portfolio=None
    ) -> None:
        if self.daily_output_dir is not None:
            self._append_jsonl_row(
                self.daily_output_dir / "tradingagents_execution_summary.daily.jsonl",
                {
                    "executed_symbols": list(executed_symbols),
                    "pending_symbols": sorted(self.pending_decisions.keys()),
                },
            )
        self.pending_decisions = {}
