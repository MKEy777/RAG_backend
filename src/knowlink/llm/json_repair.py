from __future__ import annotations

import json
from typing import Any

from json_repair import repair_json


def parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = repair_json(text, return_objects=True)
    return repaired


def parse_json_array(text: str) -> list:
    result = parse_json(text)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                return v
        return [result]
    return []


def parse_json_object(text: str) -> dict:
    result = parse_json(text)
    if isinstance(result, dict):
        return result
    if isinstance(result, list) and result:
        return result[0] if isinstance(result[0], dict) else {}
    return {}
