from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...paths import SKILLS_ROOT
from .skills import SkillEntry, SkillLoader


ROLE_SKILL_MAP = {
    "market_regime_analyst": "market-regime-analyst",
    "market_analyst": "market-analyst",
    "news_analyst": "news-analyst",
    "social_analyst": "social-analyst",
    "fundamentals_analyst": "fundamentals-analyst",
    "risk_manager": "risk-manager",
    "portfolio_manager": "portfolio-manager",
    "verification_agent": "verification-policy",
    "callback_judge": "verification-policy",
    "bull_research_agent": "market-analyst",
    "bear_research_agent": "market-analyst",
    "risk_guardian": "risk-manager",
    "risk_opportunist": "risk-manager",
    "risk_moderator": "risk-moderator",
    "risk_chair": "risk-manager",
    "execution_chair": "portfolio-manager",
    "regime_chair": "market-regime-analyst",
    "verification_chair": "verification-policy",
    "board_chair": "portfolio-manager",
    "board_moderator": "board-moderator",
    "board_secretary": "portfolio-manager",
    "allocator_agent": "allocator-agent",
}


@dataclass(slots=True)
class RolePromptSpec:
    role: str
    title: str
    objective: str
    required_outputs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    skill_slug: str = ""
    skill_text: str = ""
    tool_order: list[str] = field(default_factory=list)
    report_schema: list[str] = field(default_factory=list)
    input_schema: list[str] = field(default_factory=list)
    output_schema: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _skills_root():
    return SKILLS_ROOT


def load_skill_body(skill_slug: str) -> str:
    if not skill_slug:
        return ""
    path = _skills_root() / skill_slug / "SKILL.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]
    return text.strip()


def _load_skill_entry_object(skill_slug: str) -> SkillEntry | None:
    if not skill_slug:
        return None
    loader = SkillLoader()
    try:
        return loader.get(skill_slug)
    except Exception:
        return None


def _load_skill_entry(skill_slug: str) -> dict[str, Any]:
    entry = _load_skill_entry_object(skill_slug)
    return entry.to_dict() if entry is not None else {}


def load_skill_instruction_text(skill_slug: str) -> str:
    body = load_skill_body(skill_slug)
    entry = _load_skill_entry_object(skill_slug)
    if entry is None:
        return body
    loader = SkillLoader()
    references = loader.load_references(entry)
    pieces: list[str] = []
    if body:
        pieces.append(body)
    for relative_path, text in references.items():
        cleaned = str(text or "").strip()
        if cleaned:
            pieces.append(f"Reference ({relative_path}):\n{cleaned}")
    return "\n\n".join(pieces).strip()


def build_role_prompt_spec(role: str) -> RolePromptSpec:
    skill_slug = ROLE_SKILL_MAP.get(role, "")
    specs: dict[str, RolePromptSpec] = {
        "market_regime_analyst": RolePromptSpec(
            role=role,
            title="Market Regime Analyst",
            objective="Classify the current market regime and describe how it should constrain risk and execution.",
            required_outputs=[
                "summary",
                "score in [-1,1]",
                "confidence in [0,1]",
                "market_regime in bull|bear|sideways",
            ],
            constraints=[
                "Use market structure evidence only",
                "Return strict JSON matching summary, score, confidence, and market_regime",
            ],
            skill_slug=skill_slug,
        ),
        "market_analyst": RolePromptSpec(
            role=role,
            title="Market Analyst",
            objective="Analyze trend, momentum, support/resistance, and volatility context.",
            required_outputs=["summary", "score in [-1,1]", "confidence in [0,1]", "market_regime in bull|bear|sideways"],
            constraints=["Use technical evidence only", "Flag mixed signals", "Return strict JSON matching summary, score, confidence, and market_regime"],
            skill_slug=skill_slug,
        ),
        "news_analyst": RolePromptSpec(
            role=role,
            title="News Analyst",
            objective="Analyze company and macro news, timeliness, credibility, and catalyst impact.",
            required_outputs=["summary", "score in [-1,1]", "confidence in [0,1]", "market_regime in bull|bear|sideways"],
            constraints=["Separate facts from rumor", "Flag stale sources", "Return strict JSON matching summary, score, confidence, and market_regime"],
            skill_slug=skill_slug,
        ),
        "social_analyst": RolePromptSpec(
            role=role,
            title="Sentiment Analyst",
            objective=(
                "Analyze the LEVEL and RATE-OF-CHANGE of public/retail sentiment as a "
                "SHORT-HORIZON CONTRARIAN signal: extreme bullish consensus after a price "
                "run-up is a reversal risk (score negative), capitulation after a decline "
                "is a bounce setup (score positive)."
            ),
            required_outputs=["summary", "score in [-1,1]", "confidence in [0,1]", "market_regime in bull|bear|sideways"],
            constraints=[
                "Treat sentiment as mean-reverting on a 1-5 day horizon, not trend-following",
                "Weight the CHANGE in sentiment over the absolute level",
                "Avoid rumor amplification",
                "Return strict JSON matching summary, score, confidence, and market_regime",
            ],
            skill_slug=skill_slug,
        ),
        "fundamentals_analyst": RolePromptSpec(
            role=role,
            title="Fundamentals Analyst",
            objective="Analyze valuation, statements, margins, leverage, and cash flow quality.",
            required_outputs=["summary", "score in [-1,1]", "confidence in [0,1]", "market_regime in bull|bear|sideways"],
            constraints=["Anchor conclusions in metrics", "Flag stale filings", "Return strict JSON matching summary, score, confidence, and market_regime"],
            skill_slug=skill_slug,
        ),
        "risk_manager": RolePromptSpec(
            role=role,
            title="Risk Manager",
            objective="Translate uncertainty and disagreement into calibrated size guidance and execution constraints.",
            required_outputs=[
                "summary",
                "shared_state_updates.risk_uncertainty",
                "role_payload_updates object",
            ],
            constraints=[
                "Reflect missing verification in uncertainty and size, but do not default to blanket risk-off without concrete evidence failure",
                "Return strict JSON matching the final control-role contract",
            ],
            skill_slug=skill_slug,
        ),
        "portfolio_manager": RolePromptSpec(
            role=role,
            title="Portfolio Manager",
            objective="Synthesize analyst outputs into one final action contract.",
            required_outputs=[
                "summary",
                "shared_state_updates.proposed_action",
                "shared_state_updates.verification_verdict",
                "shared_state_updates.risk_uncertainty",
                "role_payload_updates.action_contract",
            ],
            constraints=["Resolve disagreement explicitly", "Do not hide uncertainty", "Return strict JSON matching the final control-role contract"],
            skill_slug=skill_slug,
        ),
        "verification_agent": RolePromptSpec(
            role=role,
            title="Verification Agent",
            objective="Try to break the proposed action and emit PASS, FAIL, or PARTIAL together with mandatory memory-driven verification directives.",
            required_outputs=[
                "summary",
                "shared_state_updates.verification_summary",
                "shared_state_updates.verification_verdict",
                "role_payload_updates.verification_directives.trailing_stop_recommended",
                "role_payload_updates.verification_directives.rebound_increase_recommended",
                "role_payload_updates.verification_directives.memory_thesis",
                "role_payload_updates.verification_directives.memory_capsule_patch",
                "role_payload_updates.verification_directives.capsule_verdict",
                "role_payload_updates.verification_directives.activation_evidence_refs",
                "role_payload_updates.verification_directives.invalidation_signal",
            ],
            constraints=[
                "Stress-test the proposal, but do not invent objections unsupported by evidence",
                "Prefer standardized verdicts",
                "Read historical alpha memory capsules before numeric evidence; they are experience memories, not comments",
                "Judge whether this setup matches or invalidates the active capsule pattern",
                "When evidence supports the proposal, say so directly instead of diluting it into generic caution",
                "memory_thesis must be a non-empty string; if no active capsule applies, say that explicitly",
                "invalidation_signal may be an empty string when no invalidation trigger exists",
                "Always return verification_directives; they are mandatory, not optional",
                "Return strict JSON matching the final control-role JSON contract",
            ],
            skill_slug=skill_slug,
        ),
        "callback_judge": RolePromptSpec(
            role=role,
            title="Callback Judge",
            objective="Judge whether a callback answer is sufficiently complete, decision-relevant, and evidence-bearing for the requesting stage.",
            required_outputs=[
                "sufficiency verdict",
                "missing items",
                "follow-up question",
            ],
            constraints=[
                "Be strict about missing triggers or missing evidence",
                "Return SUFFICIENT or INSUFFICIENT semantics clearly",
            ],
            skill_slug=skill_slug,
        ),
        "bull_research_agent": RolePromptSpec(
            role=role,
            title="Bull Research Agent",
            objective="Argue the strongest constructive long thesis using verified and callback-refined evidence.",
            required_outputs=["bull case", "key catalyst", "risk to thesis"],
            constraints=[
                "Make the upside case explicit",
                "Acknowledge the main failure mode",
            ],
            skill_slug=skill_slug,
        ),
        "bear_research_agent": RolePromptSpec(
            role=role,
            title="Bear Research Agent",
            objective="Argue the strongest defensive or bearish thesis using verified and callback-refined evidence.",
            required_outputs=["bear case", "key risk catalyst", "risk to thesis"],
            constraints=[
                "Make the downside case explicit",
                "Acknowledge the main squeeze risk",
            ],
            skill_slug=skill_slug,
        ),
        "risk_guardian": RolePromptSpec(
            role=role,
            title="Risk Guardian",
            objective="Argue for capital protection, smaller size, and stricter execution gating.",
            required_outputs=[
                "guardrail thesis",
                "failure scenario",
                "risk gate recommendation",
            ],
            constraints=[
                "Prioritize drawdown protection",
                "Be explicit about what should be blocked",
            ],
            skill_slug=skill_slug,
        ),
        "risk_opportunist": RolePromptSpec(
            role=role,
            title="Risk Opportunist",
            objective="Argue for selective risk-taking when evidence supports controlled exposure.",
            required_outputs=[
                "opportunity thesis",
                "trigger condition",
                "bounded size recommendation",
            ],
            constraints=["Keep size bounded", "State what evidence must hold"],
            skill_slug=skill_slug,
        ),
        "risk_moderator": RolePromptSpec(
            role=role,
            title="Risk Moderator",
            objective="Synthesize risk debate outputs into one bounded recommendation with explicit open risks.",
            required_outputs=[
                "summary",
                "shared_state_updates.risk_debate_resolution",
                "role_payload_updates object",
            ],
            constraints=[
                "Do not hide disagreement",
                "Return strict JSON matching the final control-role contract",
            ],
            skill_slug=skill_slug,
        ),
        "risk_chair": RolePromptSpec(
            role=role,
            title="Risk Chair",
            objective="Challenge the proposal on drawdown risk, evidence fragility, and whether more upstream research is required.",
            required_outputs=[
                "vote action",
                "risk objection",
                "size intent",
                "callback requirement",
            ],
            constraints=[
                "State clearly whether your objection is blocking or merely size-limiting",
                "Escalate only when the proposal is too aggressive for the actual evidence state",
            ],
            skill_slug=skill_slug,
        ),
        "execution_chair": RolePromptSpec(
            role=role,
            title="Execution Chair",
            objective="Challenge whether the proposal can be implemented through bounded open, add, trim, reduce, or hold actions.",
            required_outputs=[
                "vote action",
                "execution intent",
                "size band recommendation",
                "callback requirement",
            ],
            constraints=[
                "Prefer explicit execution pathways",
                "Respect current position state and continuation rules",
            ],
            skill_slug=skill_slug,
        ),
        "regime_chair": RolePromptSpec(
            role=role,
            title="Regime Chair",
            objective="Challenge whether the proposal fits the formal market regime and its downstream implications.",
            required_outputs=[
                "vote action",
                "regime fit assessment",
                "regime-conditioned size intent",
                "callback requirement",
            ],
            constraints=[
                "Use the explicit regime contract and regime evidence",
                "Call out regime mismatch directly",
            ],
            skill_slug=skill_slug,
        ),
        "verification_chair": RolePromptSpec(
            role=role,
            title="Verification Chair",
            objective="Challenge evidence integrity, trace quality, and verification conditions before the board authorizes action.",
            required_outputs=[
                "vote action",
                "verification condition",
                "objections",
                "callback requirement",
            ],
            constraints=[
                "Block expansion only when verification is materially weak or contradictory",
                "Name missing evidence explicitly and distinguish decisive gaps from non-decisive ones",
            ],
            skill_slug=skill_slug,
        ),
        "board_chair": RolePromptSpec(
            role=role,
            title="Board Chair",
            objective="Run the final management-board resolution protocol and produce the single governing action decision.",
            required_outputs=[
                "final board direction",
                "governance rationale",
                "size intent",
                "open risks",
            ],
            constraints=[
                "Resolve board votes explicitly",
                "Prefer the highest-conviction bounded decision supported by evidence; do not collapse to indecision when the edge remains clear",
            ],
            skill_slug=skill_slug,
        ),
        "board_moderator": RolePromptSpec(
            role=role,
            title="Board Moderator",
            objective="Run bounded board rounds, decide whether more callbacks are needed, and produce the final board-approved resolution under governance rules.",
            required_outputs=[
                "summary",
                "shared_state_updates.board_direction",
                "shared_state_updates.board_size_intent",
                "shared_state_updates.board_termination_decision",
                "role_payload_updates object",
            ],
            constraints=[
                "Say whether the board is terminating, escalating, or requesting more callbacks",
                "Preserve dissent in the final packet",
                "Prefer a decisive bounded resolution when further callbacks are unlikely to change the action",
                "Return strict JSON matching the final control-role contract",
            ],
            skill_slug=skill_slug,
        ),
        "board_secretary": RolePromptSpec(
            role=role,
            title="Board Secretary",
            objective="Maintain the unresolved dispute ledger, document escalation rules, and prepare the final board packet for the chair.",
            required_outputs=[
                "dispute ledger",
                "escalation rules",
                "board packet summary",
            ],
            constraints=[
                "List blocking disputes explicitly",
                "Do not resolve disputes silently",
            ],
            skill_slug=skill_slug,
        ),
        "allocator_agent": RolePromptSpec(
            role=role,
            title="Allocator Agent",
            objective="Choose exactly one candidate_id from an externally supplied candidate action set under hard constraints and structured preferences.",
            required_outputs=[
                "summary",
                "role_payload_updates.allocator_agent.allocator_decision.selected_action_id",
                "role_payload_updates.allocator_agent.allocator_decision.status",
                "role_payload_updates.allocator_agent.allocator_decision.rationale_tags",
                "role_payload_updates.allocator_agent.allocator_decision.soft_penalty_breakdown",
                "role_payload_updates.allocator_agent.allocator_decision.insufficiency_flags",
                "role_payload_updates.allocator_agent.allocator_decision.confidence",
            ],
            constraints=[
                "Do not invent new candidate IDs",
                "Do not modify hard constraints",
                "Return strict JSON matching the final control-role contract",
                "selected_action_id must exactly match one externally provided candidate_id or be empty when status is INFEASIBLE",
            ],
            skill_slug=skill_slug,
        ),
    }
    spec = specs.get(
        role, RolePromptSpec(role=role, title=role, objective="Perform assigned role.")
    )
    entry = _load_skill_entry(spec.skill_slug)
    spec.skill_text = load_skill_instruction_text(spec.skill_slug)
    spec.tool_order = list(entry.get("tool_order", []))
    spec.report_schema = list(entry.get("report_schema", []))
    spec.input_schema = list(entry.get("input_schema", []))
    spec.output_schema = list(entry.get("output_schema", []))
    spec.success_criteria = list(entry.get("success_criteria", []))
    return spec


def build_role_instruction(role: str) -> str:
    spec = build_role_prompt_spec(role)
    pieces = [
        f"Role: {spec.title}",
        f"Objective: {spec.objective}",
    ]
    if spec.required_outputs:
        pieces.append("Required outputs: " + "; ".join(spec.required_outputs))
    if spec.constraints:
        pieces.append("Constraints: " + "; ".join(spec.constraints))
    if spec.tool_order:
        pieces.append("Tool order: " + " -> ".join(spec.tool_order))
    if spec.report_schema:
        pieces.append("Report schema: " + "; ".join(spec.report_schema))
    if spec.input_schema:
        pieces.append("Input schema: " + "; ".join(spec.input_schema))
    if spec.output_schema:
        pieces.append("Output schema: " + "; ".join(spec.output_schema))
    if spec.success_criteria:
        pieces.append("Success criteria: " + "; ".join(spec.success_criteria))
    if spec.skill_text:
        pieces.append("Skill instructions:\n" + spec.skill_text)
    return "\n\n".join(pieces)


__all__ = [
    "ROLE_SKILL_MAP",
    "RolePromptSpec",
    "build_role_instruction",
    "build_role_prompt_spec",
    "load_skill_body",
    "load_skill_instruction_text",
]
