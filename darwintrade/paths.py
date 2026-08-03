from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DOCS_ROOT = REPO_ROOT / "docs"
SKILLS_ROOT = REPO_ROOT / "skills"


__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "ARTIFACTS_ROOT",
    "DOCS_ROOT",
    "SKILLS_ROOT",
]
