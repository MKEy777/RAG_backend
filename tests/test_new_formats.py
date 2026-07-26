from __future__ import annotations

import importlib.util

import pytest

from knowlink.parsing.registry import parse_file

_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def test_image_file_parses_to_placeholder():
    result = parse_file(_PNG, "photo.png")
    assert len(result.images) == 1
    assert result.markdown.startswith("![photo.png](images/")
    assert result.images[0].mime == "image/png"


@pytest.mark.skipif(importlib.util.find_spec("markitdown") is None, reason="markitdown not installed")
def test_html_file_parses_via_markitdown():
    html = "<html><body><h1>标题</h1><p>正文内容。</p></body></html>".encode("utf-8")
    result = parse_file(html, "page.html")
    assert "正文内容" in result.markdown
