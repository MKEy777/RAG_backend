from __future__ import annotations

import re

from knowlink.chunking import chunk_document
from knowlink.chunking.heading import split_by_heading
from knowlink.chunking.recursive import split_recursive
from knowlink.chunking.parent_child import ParentChildPair
from knowlink.chunking.profiler import analyze, choose_strategy

_HEADING_ONLY = re.compile(r"(?:#{1,6}\s+\S[^\n]*\n*)+")


def _assert_no_heading_only(chunks):
    for c in chunks:
        assert not _HEADING_ONLY.fullmatch(c.text), f"heading-only chunk: {c.text!r}"


def test_profiler_heading():
    text = "# Title\n\n## Section 1\n\nContent\n\n## Section 2\n\nMore content\n\n## Section 3\n\nEnd"
    profile = analyze(text)
    assert profile.heading_count >= 3
    assert choose_strategy(profile) == "heading"


def test_profiler_recursive():
    text = "Just a plain paragraph with no headings at all. " * 20
    profile = analyze(text)
    assert choose_strategy(profile) == "recursive"


def test_split_by_heading():
    text = "# Title\n\nIntro text\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B"
    chunks = split_by_heading(text, chunk_size=512)
    assert len(chunks) >= 2
    assert any("Section A" in c.text for c in chunks)


def test_split_recursive():
    text = "Word " * 200
    chunks = split_recursive(text.strip(), chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 150


def test_chunk_document_parent_child():
    text = "# Doc\n\n" + ("Paragraph content here. " * 50 + "\n\n") * 5
    result = chunk_document(text, strategy="auto", enable_parent_child=True)
    assert len(result) > 0
    assert isinstance(result[0], ParentChildPair)
    for pair in result:
        assert pair.parent_text
        assert len(pair.children) > 0


def test_chunk_document_flat():
    text = "Simple text. " * 100
    result = chunk_document(text, strategy="recursive", enable_parent_child=False)
    assert len(result) > 0


def test_heading_not_isolated_long_paragraph():
    text = "# 总标题\n\n" + "这是一段很长的论述内容。" * 200
    chunks = split_by_heading(text, chunk_size=512)
    _assert_no_heading_only(chunks)
    assert chunks[0].text.startswith("# 总标题")
    for c in chunks:
        assert c.end - c.start == len(c.text)


def test_heading_not_isolated_before_code_fence():
    fence = "```python\n" + "x = 1\n" * 150 + "```"
    text = "# 代码节\n\n" + fence + "\n\n" + "尾部说明文字。" * 60
    chunks = split_by_heading(text, chunk_size=512)
    _assert_no_heading_only(chunks)
    assert chunks[0].text.startswith("# 代码节")


def test_heading_not_isolated_overflow_flush():
    para = "首段内容。" * 100
    text = "# 标题\n\n" + para + "\n\n" + "次段。" * 120
    chunks = split_by_heading(text, chunk_size=512)
    _assert_no_heading_only(chunks)


def test_whole_sections_merge_empty_heading_section():
    text = "# A\n\n## B\n\n" + "内容。" * 50
    sections = split_by_heading(text, chunk_size=512, whole_sections=True)
    _assert_no_heading_only(sections)
    assert len(sections) == 1
    assert sections[0].text.startswith("# A")
    assert sections[0].title_path == ["A", "B"]


def test_trailing_heading_only_merges_backward():
    text = "# A\n\n" + "内容。" * 30 + "\n\n# B"
    chunks = split_by_heading(text, chunk_size=512, whole_sections=True)
    _assert_no_heading_only(chunks)
    assert chunks[-1].text.endswith("# B")
