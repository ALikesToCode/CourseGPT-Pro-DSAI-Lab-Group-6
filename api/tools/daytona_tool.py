"""LangChain tool for running commands in a Daytona sandbox.

The tool creates a fresh sandbox using the Daytona Python SDK (configured via
``DAYTONA_API_KEY`` and optional ``DAYTONA_ENDPOINT``/``DAYTONA_TARGET``),
executes a single shell command, captures the result, and tears down the
sandbox. It is intentionally simple to avoid long-lived sandboxes in the
agent flow.
"""

from __future__ import annotations

import shlex
from typing import List, Optional

from langchain.tools import tool


def _summarize_response(response) -> str:
    # Daytona responses typically expose `.result`; fall back to repr otherwise.
    if response is None:
        return "(no response)"
    return getattr(response, "result", None) or getattr(response, "message", None) or repr(response)


@tool
def daytona_run(
    command: Optional[str] = None,
    commands: Optional[List[str]] = None,
    api_url: str | None = None,
    target: str | None = None,
    workdir: str | None = None,
) -> str:
    """Run one or more shell commands inside a short-lived Daytona sandbox.

    Requires DAYTONA_API_KEY in the environment. Optionally override the API
    URL/target. Returns stdout/stderr from the command(s) or a failure message.
    - Use `command` for a single string.
    - Use `commands` for a list executed sequentially in the same sandbox.
    - Use `workdir` to run inside a specific directory.
    """

    try:
        from daytona import Daytona, DaytonaConfig
    except Exception as exc:  # noqa: BLE001
        return (
            "Daytona SDK is not installed. Add `daytona` to api/requirements.txt "
            "and ensure the environment has DAYTONA_API_KEY configured."
        )

    try:
        if not command and not commands:
            return "Provide `command` or `commands` to run inside Daytona."

        # Join multiple commands safely into a single shell invocation.
        combined_command = command or " && ".join(commands or [])
        if workdir:
            combined_command = f"cd {shlex.quote(workdir)} && {combined_command}"

        config_kwargs = {}
        if api_url:
            config_kwargs["api_url"] = api_url
        if target:
            config_kwargs["target"] = target

        client = Daytona(DaytonaConfig(**config_kwargs)) if config_kwargs else Daytona()
        sandbox = client.create()

        exec_response = sandbox.process.exec(combined_command)
        output = _summarize_response(exec_response)
    except Exception as exc:  # noqa: BLE001
        return f"Daytona run failed: {exc}"
    finally:
        try:
            if "sandbox" in locals():
                client.delete(sandbox)  # Clean up to avoid leaked sandboxes
        except Exception:
            pass

    return f"Sandbox command output: {output}"
