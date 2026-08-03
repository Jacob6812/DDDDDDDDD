from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any


_SAFE_PATH_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def safe_path_component(value: Any, *, fallback: str = "EMPTY") -> str:
    text = str(value or "").strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part and part not in {".", ".."}]
    collapsed = "_".join(parts)
    cleaned = _SAFE_PATH_COMPONENT_PATTERN.sub("_", collapsed).strip("._-")
    return cleaned[:120] or fallback


__all__ = ["safe_path_component", "to_jsonable"]
