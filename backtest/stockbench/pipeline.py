from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import os
import shutil

from backtest.stockbench.datasets import Datasets
from backtest.stockbench.engine import BacktestEngine
from backtest.stockbench.slippage import Slippage
from backtest.stockbench.reports import write_outputs
from backtest.stockbench.agents.backtest_report_llm import generate_backtest_report


def _sync_daily_outputs(result: Dict, out_dir: str) -> None:
    daily_output_dir = result.get("daily_output_dir")
    if not daily_output_dir:
        return
    source_dir = Path(str(daily_output_dir))
    target_dir = Path(out_dir)
    if not source_dir.exists() or source_dir.resolve() == target_dir.resolve():
        return
    for name in (
        "daily_run_summary.jsonl",
        "darwintrade_portfolio_payloads.daily.jsonl",
        "darwintrade_decision_records.daily.jsonl",
        "darwintrade_portfolio_decisions.daily.jsonl",
    ):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)


def run_backtest(
    cfg: Dict,
    strategy,
    start: str,
    end: str,
    symbols: List[str],
    run_id: str | None = None,
    timespan: Optional[str] = None,
) -> Dict:
    datasets = Datasets(cfg)
    slippage = Slippage.from_cfg(cfg)
    engine = BacktestEngine(cfg, datasets, slippage)
    # Select timespan: prioritize CLI input; otherwise read from config; finally fallback to "day"
    effective_timespan = (
        timespan or (cfg.get("backtest", {}) or {}).get("timespan") or "day"
    )
    # Run backtest
    result = engine.run(
        strategy=strategy,
        start=start,
        end=end,
        symbols=symbols,
        timespan=effective_timespan,
        run_id=run_id,
    )
    # Write timespan back to cfg for report display
    try:
        cfg.setdefault("backtest", {})["timespan"] = effective_timespan
    except Exception:
        pass
    out_dir = write_outputs(result, run_id=run_id, cfg=cfg)
    _sync_daily_outputs(result, out_dir)
    result["output_dir"] = out_dir
    # Backtest natural language summary (optional)
    try:
        enable_llm = bool((cfg or {}).get("backtest", {}).get("summary_llm", False))
        if enable_llm:
            summary_txt_path = os.path.join(out_dir, "summary.txt")
            summary_text = ""
            try:
                if os.path.exists(summary_txt_path):
                    with open(summary_txt_path, "r", encoding="utf-8") as f:
                        summary_text = f.read()
            except Exception:
                summary_text = ""
            metrics_dict = result.get("metrics") or {}
            payload = {
                "metrics": metrics_dict,
                "summary_text": summary_text,
                "period": {"start": start, "end": end},
                "timespan": effective_timespan,
                "run_id": run_id or os.path.basename(out_dir),
                "symbols": symbols,
            }
            profile_name = None
            try:
                profiles = cfg.get("llm_profiles", {})
                if profiles:
                    if "openai" in profiles:
                        profile_name = "openai"
                    else:
                        profile_name = next(iter(profiles.keys()))
            except Exception:
                pass

            text = generate_backtest_report(
                payload, cfg=cfg, run_id=run_id, profile_name=profile_name
            )
            nl_path = os.path.join(out_dir, "nl_summary.txt")
            with open(nl_path, "w", encoding="utf-8") as f:
                f.write(text)
            result["nl_summary"] = nl_path
    except Exception:
        pass
    return result
