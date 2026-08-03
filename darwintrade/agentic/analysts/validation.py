from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ReportValidationResult:
    role: str
    schema_fields: list[str]
    present_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    valid: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def validate_report_schema(
    summary: str, role: str, schema_fields: list[str]
) -> ReportValidationResult:
    present: list[str] = []
    missing: list[str] = []
    upper_lines = [
        line.strip().upper() for line in summary.splitlines() if line.strip()
    ]
    for schema_field in schema_fields:
        prefix = f"{schema_field}:"
        if any(line.startswith(prefix.upper()) for line in upper_lines):
            present.append(schema_field)
        else:
            missing.append(schema_field)
    return ReportValidationResult(
        role=role,
        schema_fields=schema_fields,
        present_fields=present,
        missing_fields=missing,
        valid=not missing,
    )
