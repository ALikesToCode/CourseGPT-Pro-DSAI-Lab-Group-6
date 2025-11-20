from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Union


def normalize_list(value: Union[str, List[str], None]) -> List[str]:
    """
    Coerce router values into a list of clean strings so tool calls do not fail
    when the LLM sends a string instead of a list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        # Split on newlines and commas while trimming bullets/whitespace.
        raw_parts = value.replace("\r", "").split("\n")
        parts: List[str] = []
        for part in raw_parts:
            chunk = part.strip(" -[]")
            if not chunk:
                continue
            parts.extend([p.strip() for p in chunk.split(",") if p.strip()])
        return parts or [value.strip()]
    return [str(value).strip()]


def normalize_dict(value: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
    """
    Convert stringified dictionaries into real dicts; fall back to wrapping
    the value so downstream steps always receive a mapping.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {"value": value}
    return {"value": value}
