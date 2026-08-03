from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...paths import SKILLS_ROOT


SkillRunner = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class SkillEntry:
    name: str
    description: str
    path: str
    skill_type: str = "role"
    trigger_keywords: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    not_for: list[str] = field(default_factory=list)
    stage: str = "general"
    role_scope: list[str] = field(default_factory=list)
    tool_order: list[str] = field(default_factory=list)
    report_schema: list[str] = field(default_factory=list)
    input_schema: list[str] = field(default_factory=list)
    output_schema: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    reference_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
class SkillLoader:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else SKILLS_ROOT

    def discover(self) -> list[SkillEntry]:
        entries: list[SkillEntry] = []
        if not self.root.exists():
            return entries
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            entry = self._parse_skill_file(skill_file)
            if entry is not None:
                entries.append(entry)
        return entries
    def get(self, name: str) -> SkillEntry:
        for entry in self.discover():
            if entry.name == name:
                return entry
        raise KeyError(f"Unknown skill: {name}")

    def load_reference(self, name: str, relative_path: str) -> str:
        path = self.root / name / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Skill reference not found: {name}/{relative_path}"
            )
        return path.read_text(encoding="utf-8").strip()

    def load_references(self, entry: SkillEntry) -> dict[str, str]:
        loaded: dict[str, str] = {}
        for relative_path in entry.reference_paths:
            try:
                loaded[relative_path] = self.load_reference(entry.name, relative_path)
            except FileNotFoundError:
                continue
        return loaded

    def _parse_frontmatter(self, frontmatter: str) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        lines = frontmatter.splitlines()
        index = 0
        while index < len(lines):
            raw_line = lines[index].rstrip()
            stripped = raw_line.strip()
            if not stripped:
                index += 1
                continue
            if ":" not in raw_line:
                index += 1
                continue
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                items: list[str] = []
                index += 1
                while index < len(lines):
                    candidate = lines[index].rstrip()
                    candidate_stripped = candidate.strip()
                    if not candidate_stripped:
                        index += 1
                        continue
                    if not candidate.startswith((" ", "\t")):
                        break
                    if candidate_stripped.startswith("- "):
                        items.append(candidate_stripped[2:].strip())
                        index += 1
                        continue
                    break
                meta[key] = items
                continue
            if value in {">", ">-", "|", "|-"}:
                block_style = value[0]
                block_lines: list[str] = []
                index += 1
                while index < len(lines):
                    candidate = lines[index].rstrip("\n")
                    if not candidate.strip():
                        block_lines.append("")
                        index += 1
                        continue
                    if not candidate.startswith((" ", "\t")):
                        break
                    block_lines.append(candidate.lstrip())
                    index += 1
                if block_style == ">":
                    text = " ".join(part.strip() for part in block_lines if part.strip())
                else:
                    text = "\n".join(block_lines)
                meta[key] = text.strip()
                continue
            meta[key] = value.strip("'\"")
            index += 1
        return meta

    def _parse_skill_file(self, path: Path) -> SkillEntry | None:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return None
        parts = text.split("---", 2)
        if len(parts) < 3:
            return None
        frontmatter = parts[1]
        meta = self._parse_frontmatter(frontmatter)
        return SkillEntry(
            name=str(meta.get("name", path.parent.name)),
            description=str(meta.get("description", "")).strip(),
            path=str(path),
            skill_type=str(meta.get("skill_type", "role")),
            trigger_keywords=list(meta.get("trigger_keywords", [])),
            inputs=list(meta.get("inputs", [])),
            outputs=list(meta.get("outputs", [])),
            not_for=list(meta.get("not_for", [])),
            stage=str(meta.get("stage", "general")),
            role_scope=list(meta.get("role_scope", [])),
            tool_order=list(meta.get("tool_order", [])),
            report_schema=list(meta.get("report_schema", [])),
            input_schema=list(meta.get("input_schema", [])),
            output_schema=list(meta.get("output_schema", [])),
            success_criteria=list(meta.get("success_criteria", [])),
            reference_paths=list(meta.get("reference_paths", [])),
        )
__all__ = [
    "SkillEntry",
    "SkillLoader",
    ]
