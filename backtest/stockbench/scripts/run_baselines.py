"""Run all six L/S quant baselines through the StockBench engine.

Each strategy is run with the same config / period / universe as the
DarwinTrade default config, so reports under
storage/reports/STOCKBENCH/<STRATEGY>_LS_<TIMESTAMP>/ are directly
comparable to the DarwinTrade run.

Usage:
    python -m backtest.stockbench.scripts.run_baselines [--start 2025-03-01 --end 2025-07-31]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass

from backtest.stockbench.pipeline import run_backtest
from backtest.stockbench.utils.logging_setup import setup_json_logging, Metrics
from backtest.stockbench.core.data_hub import refresh_api_clients, set_data_mode
from backtest.stockbench.strategies.rule_based import (
    BuyAndHoldStrategy,
    MonthEndEffectStrategy,
    SMACrossoverStrategy,
    WMACrossoverStrategy,
    BollingerBandsStrategy,
    ATRBandsStrategy,
)
from backtest.stockbench.strategies.predictive import (
    ARIMAStrategy,
    XGBoostStrategy,
)
from backtest.stockbench.strategies.classical_quant import (
    ClassicalHRPStrategy,
    ClassicalMaxSharpeStrategy,
    ClassicalEqualWeightStrategy,
)


# Report-dir label -> (strategy class, config-strategy-name).
# The label controls storage/reports/STOCKBENCH/<label>{suffix}/ naming and
# matches the pre-existing folder convention (no _LS suffix; keypool suffix
# only for the rule-based long-only pair). The config-strategy-name is what
# gets written into cfg["strategy"]["name"] as metadata.
LS_BASELINES: dict[str, tuple[type, str]] = {
    "SMA_CROSSOVER": (SMACrossoverStrategy, "sma_crossover_ls"),
    "WMA_CROSSOVER": (WMACrossoverStrategy, "wma_crossover_ls"),
    "BOLLINGER_BANDS": (BollingerBandsStrategy, "bollinger_bands_ls"),
    "ATR_BANDS": (ATRBandsStrategy, "atr_bands_ls"),
    "ARIMA": (ARIMAStrategy, "arima_ls"),
    "XGBOOST": (XGBoostStrategy, "xgboost_ls"),
    "BUY_AND_HOLD_KEYPOOL": (BuyAndHoldStrategy, "buy_and_hold"),
    "MONTH_END_EFFECT_KEYPOOL": (MonthEndEffectStrategy, "month_end_effect"),
    "CLASSICAL_HRP": (ClassicalHRPStrategy, "classical_quant_hrp"),
    "CLASSICAL_MAXSHARPE": (ClassicalMaxSharpeStrategy, "classical_quant_max_sharpe"),
    "CLASSICAL_EQUALWEIGHT": (ClassicalEqualWeightStrategy, "classical_quant_equal_weight"),
}

DEFAULT_CFG = REPO_ROOT / "backtest" / "stockbench" / "config_darwintrade.yaml"
REPORTS_ROOT = REPO_ROOT / "storage" / "reports" / "STOCKBENCH"


def _resolve_env_value(value: object) -> str:
    text = str(value or "")
    if text.startswith("${") and text.endswith("}"):
        return os.getenv(text[2:-1], "")
    return text


def _setup_data(config: dict, offline: bool) -> None:
    api_cfg = dict(config.get("api", {}) or {})
    polygon_cfg = dict(api_cfg.get("polygon", {}) or {})
    finnhub_cfg = dict(api_cfg.get("finnhub", {}) or {})
    polygon_key = _resolve_env_value(
        polygon_cfg.get("api_key") or os.getenv("POLYGON_API_KEY", "")
    )
    finnhub_key = _resolve_env_value(
        finnhub_cfg.get("api_key")
        or os.getenv("FINNUB_API_KEY", "")
        or os.getenv("FINNHUB_API_KEY", "")
    )
    if polygon_key:
        os.environ["POLYGON_API_KEY"] = polygon_key
    if finnhub_key:
        os.environ["FINNUB_API_KEY"] = finnhub_key
    refresh_api_clients(
        polygon_api_key=polygon_key or None,
        finnhub_api_key=finnhub_key or None,
    )

    data_cfg = config.setdefault("data", {})
    if offline:
        data_cfg["mode"] = "offline_only"
        set_data_mode("offline_only")
    elif data_cfg.get("mode"):
        set_data_mode(str(data_cfg["mode"]))


def _run_one(
    name: str,
    strategy_cls,
    strategy_key: str,
    config: dict,
    start: str,
    end: str,
    symbols: List[str],
    dir_suffix: str,
) -> Path:
    # Reports land at storage/reports/STOCKBENCH/<NAME>{suffix}/ so different
    # backtest windows don't clobber each other. Convention: suffix carries
    # the date range (e.g. _2025-03-03_2025-06-30).
    dir_name = f"{name}{dir_suffix}"
    out_dir = REPORTS_ROOT / dir_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The engine's reports.write_outputs hard-codes
    # `cwd/storage/reports/backtest/<run_id>/`, so we use a unique run_id
    # to avoid collisions, then move the artifacts into out_dir.
    run_id = f"{dir_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # redirect stockbench engine output to our STOCKBENCH/<name> folder
    config = dict(config)
    config["strategy"] = {"name": strategy_key}
    backtest_cfg = config.setdefault("backtest", {})
    backtest_cfg["output_dir"] = str(out_dir)
    backtest_cfg["timespan"] = "day"
    # baseline runs always start fresh; no resume from a prior summary
    resume_cfg = backtest_cfg.setdefault("resume", {})
    resume_cfg["enabled"] = False

    os.environ["TA_RUN_ID"] = run_id

    print(f"[{name}] starting period={start}→{end}", flush=True)

    strategy = strategy_cls(config)
    if hasattr(strategy, "set_daily_output_dir"):
        strategy.set_daily_output_dir(out_dir)

    res = run_backtest(
        config,
        strategy,
        start,
        end,
        symbols,
        run_id=run_id,
        timespan="day",
    )
    actual_out = res.get("output_dir") if isinstance(res, dict) else None
    if actual_out:
        engine_out = Path(actual_out)
        if engine_out.exists() and engine_out.resolve() != out_dir.resolve():
            for item in engine_out.iterdir():
                target = out_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            try:
                engine_out.rmdir()
            except OSError:
                pass
    print(f"[{name}] done -> {out_dir}", flush=True)
    return out_dir


def _worker_entry(payload: dict) -> tuple[str, str | None, str | None]:
    """ProcessPoolExecutor worker: re-imports strategy class by name in the
    child process so we don't have to pickle the class object."""
    name = payload["name"]
    try:
        strategy_cls, strategy_key = LS_BASELINES[name]
        out = _run_one(
            name=name,
            strategy_cls=strategy_cls,
            strategy_key=strategy_key,
            config=payload["config"],
            start=payload["start"],
            end=payload["end"],
            symbols=payload["symbols"],
            dir_suffix=payload["dir_suffix"],
        )
        return name, str(out), None
    except Exception as exc:
        import traceback

        return name, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run six L/S quant baselines.")
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument("--start", default="2025-03-01")
    p.add_argument("--end", default="2025-07-31")
    p.add_argument("--symbols", default="")
    p.add_argument(
        "--strategies",
        default="",
        help=f"Comma-separated subset; default = all of {','.join(LS_BASELINES)}",
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers; 0 = one per strategy (full parallel), 1 = serial",
    )
    p.add_argument(
        "--period-suffix",
        default=None,
        help="Suffix appended to each output folder (default: _<start>_<end>). "
             "Pass empty string to write to bare <NAME>/ (legacy behaviour).",
    )
    args = p.parse_args(argv)

    if args.period_suffix is None:
        dir_suffix = f"_{args.start}_{args.end}"
    else:
        dir_suffix = args.period_suffix

    if not args.cfg.exists():
        raise SystemExit(f"Config file not found: {args.cfg}")
    config = yaml.safe_load(args.cfg.read_text(encoding="utf-8")) or {}

    _setup_data(config, args.offline)
    setup_json_logging(config)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(config.get("symbols_universe", []))

    if args.strategies:
        wanted = {s.strip().upper() for s in args.strategies.split(",") if s.strip()}
        runs = {k: v for k, v in LS_BASELINES.items() if k in wanted}
    else:
        runs = LS_BASELINES

    if not runs:
        raise SystemExit("No matching strategies. Available: " + ",".join(LS_BASELINES))

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    Metrics().incr("baselines_runner.start", 1)

    workers = args.workers if args.workers > 0 else len(runs)
    print(
        f"\n{'=' * 70}\n"
        f"  Running {len(runs)} baselines with {workers} parallel workers\n"
        f"  Period: {args.start} → {args.end} | symbols: {len(symbols)}\n"
        f"{'=' * 70}\n",
        flush=True,
    )

    payloads = [
        {
            "name": name,
            "config": dict(config),
            "start": args.start,
            "end": args.end,
            "symbols": symbols,
            "dir_suffix": dir_suffix,
        }
        for name in runs
    ]

    started = time.monotonic()
    results: list[tuple[str, Path | None, str | None]] = []

    if workers == 1:
        # Serial fallback (useful for debugging)
        for payload in payloads:
            name, out, err = _worker_entry(payload)
            if err:
                print(f"[FAIL] {name}: {err}", flush=True)
            results.append((name, Path(out) if out else None, err))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker_entry, payload): payload["name"]
                for payload in payloads
            }
            for fut in as_completed(futures):
                name, out, err = fut.result()
                if err:
                    print(f"[FAIL] {name}: {err}", flush=True)
                results.append((name, Path(out) if out else None, err))

    elapsed = time.monotonic() - started
    print(
        f"\n\n{'=' * 70}\n"
        f"  Baseline runs complete ({len(results)} strategies in {elapsed:.1f}s)\n"
        f"{'=' * 70}",
        flush=True,
    )
    for name, out, err in results:
        status = str(out) if out else f"FAILED ({err.splitlines()[0] if err else 'unknown'})"
        print(f"  {name}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
