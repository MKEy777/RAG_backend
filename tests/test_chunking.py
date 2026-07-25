from __future__ import annotations

from pyrag.chunking import chunk_document
from pyrag.chunking.heading import split_by_heading
from pyrag.chunking.recursive import split_recursive
from pyrag.chunking.parent_child import ParentChildPair
from pyrag.chunking.profiler import analyze, choose_strategy


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
