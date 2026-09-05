"""
Test configuration. Stubs LLM calls so tests run without API keys.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from darwintrade.core.llm import LLMClient


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Stub all LLM calls unless test is marked real_llm."""
    if request.node.get_closest_marker("real_llm") is not None:
        return

    def _fake_invoke(
        self: LLMClient,
        system_prompt: str,
        user_prompt: str,
        *,
        caller_tag: str = "unknown",
        response_format=None,
        validator=None,
    ) -> str:
        if caller_tag == "regime_agent":
            resp = json.dumps({
                "regime": "bull",
                "confidence": 0.7,
                "evidence": ["stub evidence"],
                "raw_scores": {"bull": 0.7, "bear": 0.1, "sideways": 0.1, "volatile": 0.1},
            })
        elif caller_tag == "market_agent":
            try:
                payload = json.loads(user_prompt)
                assets = payload.get("assets", [])
            except Exception:
                assets = []
            signals = [
                {"symbol": a["symbol"], "direction": "long", "confidence": 0.6, "thesis": "stub"}
                for a in assets
            ]
            resp = json.dumps({"signals": signals})
        elif caller_tag == "tactical_reflection":
            resp = json.dumps({
                "mistake_type": "none",
                "immediate_lessons": [],
                "summary": "stub tactical reflection",
            })
        elif caller_tag == "tactical_evolution":
            resp = json.dumps({
                "tactical_influence": {
                    "avoid_symbols": [],
                    "reduce_only_symbols": [],
                    "position_haircut": 1.0,
                    "expires_in_days": 1,
                },
                "rationale": "stub",
                "urgency": "normal",
            })
        elif caller_tag == "strategic_reflection":
            resp = json.dumps({
                "dominant_pattern": "none",
                "persistent_winners": [],
                "persistent_losers": [],
                "regime_observation": "",
                "lessons": [],
                "strategic_signal_strength": 0.0,
                "summary": "stub strategic reflection",
            })
        elif caller_tag == "strategic_diagnosis":
            resp = json.dumps({
                "dominant_failure_mode": "none",
                "failure_modes": [],
                "regime_hypotheses": [],
                "patch_goals": ["improve returns"],
                "confidence": 0.7,
                "summary": "stub diagnosis",
            })
        elif caller_tag == "strategic_policy_author":
            resp = json.dumps({
                "patch": {"max_gross_exposure": 0.9},
                "rationale": "stub patch",
                "expected_effects": [],
                "risk_notes": [],
                "rollback_trigger": "5d nav drop > 2%",
            })
        else:
            resp = json.dumps({"summary": "stub", "score": 0.5, "confidence": 0.5})

        if validator is not None:
            validator(resp)
        return resp

    monkeypatch.setattr(LLMClient, "invoke", _fake_invoke)
