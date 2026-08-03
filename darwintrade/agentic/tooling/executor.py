from __future__ import annotations

from typing import Any, Callable

from .contracts import EvidenceLedger, EvidenceRequest, ToolExecution, ToolPlan
from .registry import ToolRegistry
from .toolbox import DEFAULT_TOOLBOX

ToolCallable = Callable[[dict[str, Any]], dict[str, Any]]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        toolbox: dict[str, ToolCallable] | None = None,
    ) -> None:
        self.registry = registry
        self.toolbox = DEFAULT_TOOLBOX if toolbox is None else toolbox

    def execute_tool(
        self,
        tool_id: str,
        payload: dict[str, Any],
        *,
        step: int = 1,
    ) -> ToolExecution:
        record = self.registry.get(tool_id)
        key = record.implementation_key or f"{record.server}.{record.tool_name}"
        arguments = self._arguments_for(record.required_params, payload)
        missing = [param for param in record.required_params if param not in payload]
        if missing:
            return ToolExecution(
                step=step,
                tool_id=tool_id,
                arguments=arguments,
                output={},
                status="failed",
                error="missing_required_params: " + ", ".join(missing),
            )
        tool = self.toolbox[key]
        try:
            output = tool(arguments)
            if not isinstance(output, dict):
                raise TypeError(
                    f"Tool {tool_id} returned non-dict output: {type(output).__name__}"
                )
            if output.get("available") is False:
                raise RuntimeError(
                    str(output.get("reason") or output.get("error") or "tool unavailable")
                )
            if output.get("status") == "failed" or output.get("error"):
                raise RuntimeError(str(output.get("error") or "tool execution failed"))
            return ToolExecution(
                step=step,
                tool_id=tool_id,
                arguments=arguments,
                output=output,
                compressed_output=self._compress(output),
            )
        except Exception as exc:
            return ToolExecution(
                step=step,
                tool_id=tool_id,
                arguments=arguments,
                output={},
                status="failed",
                error=str(exc),
            )

    def execute_plan(
        self,
        request: EvidenceRequest,
        plan: ToolPlan,
        payload: dict[str, Any],
        *,
        start_step: int = 1,
    ) -> EvidenceLedger:
        executions: list[ToolExecution] = []
        for index, tool_id in enumerate(plan.selected_tool_ids, start=start_step):
            executions.append(self.execute_tool(tool_id, payload, step=index))
        return EvidenceLedger(
            request=request.to_dict(),
            plan=plan.to_dict(),
            executions=executions,
        )

    def _arguments_for(
        self, required_params: list[str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        boundary_params = {
            "as_of_time",
            "allowed_end_time",
            "source_timestamp",
            "release_time",
            "ingestion_time",
        }
        included = [
            key
            for key in dict.fromkeys([*required_params, *boundary_params])
            if key in payload
        ]
        return {key: payload[key] for key in included}

    def _compress(self, output: dict[str, Any]) -> dict[str, Any]:
        text = str(output)
        return {"summary": text[:600], "keys": sorted(output.keys())}
