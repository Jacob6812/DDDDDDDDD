"""
Live-layer runtime settings.

Everything here is read once at process start. The cache root is configurable so
a user can point the tool at an existing `storage/cache/` (the one shipped for
paper reproduction) or at a fresh directory that the third-party APIs will fill
in as they go.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from darwintrade.paths import REPO_ROOT

# The 20-name Dow-subset universe used in the paper. Offered as the default so a
# first-time user gets a working run without knowing which symbols are cached.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "GS", "MSFT", "HD", "V", "SHW", "CAT", "MCD", "UNH", "AXP", "AMGN",
    "TRV", "CRM", "JPM", "IBM", "HON", "BA", "AMZN", "AAPL", "PG", "JNJ",
)

DEFAULT_BENCHMARK = "SPY"
MAX_SYMBOLS_PER_REQUEST = 30


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key, "").strip()
    return Path(raw).expanduser() if raw else default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class LiveSettings:
    """Resolved live-layer configuration.

    cache_root      where bars / news / fundamentals are read and written
    session_root    where per-session memory state is persisted
    data_mode       "auto" hits third-party APIs on cache miss; "offline_only"
                    never leaves the cache (useful for demos without keys)
    """
    cache_root: Path
    session_root: Path
    data_mode: str = "auto"
    benchmark: str = DEFAULT_BENCHMARK
    default_capital: float = 100_000.0
    max_gross_exposure: float = 0.95
    max_single_position: float = 1.0
    min_confidence_threshold: float = 0.35
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LiveSettings":
        storage = _env_path("DARWINTRADE_STORAGE_DIR", Path.cwd() / "storage")
        cache_root = _env_path("DARWINTRADE_CACHE_DIR", storage / "cache")
        session_root = _env_path(
            "DARWINTRADE_SESSION_DIR", storage / "live" / "sessions"
        )
        mode = (os.getenv("DARWINTRADE_DATA_MODE", "auto").strip().lower() or "auto")
        if mode in {"offline", "cache", "cache_only", "offline_only"}:
            mode = "offline_only"
        else:
            mode = "auto"
        return cls(
            cache_root=cache_root,
            session_root=session_root,
            data_mode=mode,
            benchmark=os.getenv("DARWINTRADE_BENCHMARK", DEFAULT_BENCHMARK).strip().upper()
            or DEFAULT_BENCHMARK,
            default_capital=_env_float("DARWINTRADE_DEFAULT_CAPITAL", 100_000.0),
            max_gross_exposure=_env_float("DARWINTRADE_MAX_GROSS_EXPOSURE", 0.95),
            max_single_position=_env_float("DARWINTRADE_MAX_SINGLE_POSITION", 1.0),
            min_confidence_threshold=_env_float("DARWINTRADE_MIN_CONFIDENCE", 0.35),
        )

    def data_hub_cfg(self) -> dict:
        """Config dict in the shape `data_hub` expects.

        `data_hub` resolves API keys from the environment and the data mode from
        `cfg["data"]["mode"]`, so this is the whole contract.
        """
        return {
            "data": {"mode": self.data_mode},
            "bars": {"day": {"adjusted": True}},
            "backtest": {"benchmark": {"symbol": self.benchmark, "field": "close"}},
        }

    def llm_configured(self) -> bool:
        return bool(os.getenv("LLM_API_KEY", "").strip())


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_UNIVERSE",
    "MAX_SYMBOLS_PER_REQUEST",
    "LiveSettings",
]
