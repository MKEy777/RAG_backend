from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DocProfile:
    heading_count: int = 0
    form_feed_count: int = 0
    chapter_marker_count: int = 0
    uppercase_line_count: int = 0
    blank_burst_count: int = 0
    total_lines: int = 0
    dominant_heading_level: int = 0


_HEADING_RE = re.compile(r"^(#{1,6})\s+")
_CHAPTER_RE = re.compile(
    r"^(第[一二三四五六七八九十百千\d]+[章节篇部]|[A-Z][a-z]+\s+\d+|Chapter\s+\d+|Section\s+[\d.]+|\d+\.\d+\s+\S)",
    re.IGNORECASE,
)
_UPPER_RE = re.compile(r"^[A-Z][A-Z\s]{3,}$")


def analyze(text: str) -> DocProfile:
    lines = text.split("\n")
    profile = DocProfile(total_lines=len(lines))
    heading_levels: dict[int, int] = {}
    blank_run = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            if blank_run == 3:
                profile.blank_burst_count += 1
            continue
        blank_run = 0

        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            heading_levels[level] = heading_levels.get(level, 0) + 1
            profile.heading_count += 1
            continue

        if "\f" in line:
            profile.form_feed_count += 1

        if _CHAPTER_RE.match(stripped):
            profile.chapter_marker_count += 1

        if _UPPER_RE.match(stripped) and len(stripped) < 80:
            profile.uppercase_line_count += 1

    if heading_levels:
        profile.dominant_heading_level = max(heading_levels, key=heading_levels.get)

    return profile


def choose_strategy(profile: DocProfile) -> str:
    if profile.heading_count >= 3:
        return "heading"
    if (
        profile.form_feed_count >= 2
        or profile.chapter_marker_count >= 3
        or profile.uppercase_line_count >= 3
        or profile.blank_burst_count >= 3
    ):
        return "heuristic"
    return "recursive"
