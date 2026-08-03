from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .contracts import ToolManifestRecord

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifests" / "darwintrade_finance_tools.jsonl"


class ToolRegistry:
    def __init__(self, records: Iterable[ToolManifestRecord] | None = None) -> None:
        self._records: dict[str, ToolManifestRecord] = {}
        for record in records or []:
            self.register(record)

    def register(self, record: ToolManifestRecord) -> None:
        self._records[record.tool_id] = record

    def get(self, tool_id: str) -> ToolManifestRecord:
        return self._records[tool_id]

    def list(self) -> list[ToolManifestRecord]:
        return list(self._records.values())

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ToolRegistry":
        records: list[ToolManifestRecord] = []
        manifest_path = Path(path)
        if not manifest_path.exists():
            return cls()
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(ToolManifestRecord(**payload))
        return cls(records)


def load_default_registry() -> ToolRegistry:
    return ToolRegistry.from_jsonl(DEFAULT_MANIFEST)
