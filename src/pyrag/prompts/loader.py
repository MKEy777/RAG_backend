from __future__ import annotations

import re
from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "config" / "prompts"
_cache: dict[str, dict] = {}


def load_prompt_file(name: str) -> dict:
    if name in _cache:
        return _cache[name]
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _cache[name] = data
    return data


def render(template: str, variables: dict) -> str:
    def replacer(m: re.Match) -> str:
        key = m.group(1).strip()
        return str(variables.get(key, m.group(0)))

    return re.sub(r"\{\{\.(\w+)\}\}", replacer, template)


def get_prompt(name: str, key: str, variables: dict | None = None) -> str:
    data = load_prompt_file(name)
    template = data.get(key, "")
    if variables:
        return render(template, variables)
    return template
