"""DarwinTrade — two-layer self-evolving trading system."""

from __future__ import annotations

from .pipeline import DarwinTradePipeline, PipelineConfig, PipelineResult, build_pipeline
from .core.contracts import (
    AssetSignal,
    EpisodeRecord,
    ExecutionPlan,
    MemoryInfluence,
    PortfolioState,
    RegimeState,
    StrategicPatch,
    StrategicReflection,
    TacticalInfluence,
    TacticalReflection,
)
from .memory import (
    NullStrategicMemory,
    NullTacticalMemory,
    StrategicMemory,
    StrategicReflectionAgent,
    TacticalMemory,
    TacticalReflectionAgent,
    combine_influence,
)
from .agents.regime import RegimeAgent
from .agents.market import MarketAgent
from .agents.evolution import StrategicEvolutionAgent, TacticalEvolutionAgent

__all__ = [
    "DarwinTradePipeline",
    "PipelineConfig",
    "PipelineResult",
    "build_pipeline",
    "AssetSignal",
    "EpisodeRecord",
    "ExecutionPlan",
    "MemoryInfluence",
    "PortfolioState",
    "RegimeState",
    "StrategicPatch",
    "StrategicReflection",
    "TacticalInfluence",
    "TacticalReflection",
    "TacticalMemory",
    "NullTacticalMemory",
    "StrategicMemory",
    "NullStrategicMemory",
    "TacticalReflectionAgent",
    "StrategicReflectionAgent",
    "TacticalEvolutionAgent",
    "StrategicEvolutionAgent",
    "combine_influence",
    "RegimeAgent",
    "MarketAgent",
]
