"""
Skill loader: reads SKILL.md files from the skills/ directory and returns
the instruction text to inject into agent system prompts.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- ... ---) and return the body."""
    text = text.strip()
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    return text[end + 3:].strip()


@lru_cache(maxsize=64)
def load_skill(slug: str) -> str:
    """
    Load instruction text for a skill by slug name.
    Returns empty string if skill not found (graceful degradation).
    """
    skill_file = _SKILLS_ROOT / slug / "SKILL.md"
    if not skill_file.exists():
        return ""
    try:
        return _strip_frontmatter(skill_file.read_text(encoding="utf-8"))
    except Exception:
        return ""


def skill_section(slug: str, *, header: str = "Skill instructions") -> str:
    """Return skill text formatted as a system prompt section, or empty string."""
    text = load_skill(slug)
    if not text:
        return ""
    return f"\n\n## {header}\n{text}"


__all__ = ["load_skill", "skill_section"]
