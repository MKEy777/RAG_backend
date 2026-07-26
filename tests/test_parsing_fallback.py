from __future__ import annotations

from knowlink.parsing import registry as reg


def test_parser_exception_falls_back(monkeypatch):
    def boom(data, filename, **kwargs):
        raise RuntimeError("parse fail")

    monkeypatch.setitem(reg._PARSERS, "txt", boom)
    monkeypatch.setattr(reg, "_markitdown_fallback", lambda d, f: reg.ParseResult(markdown="fallback ok"))
    result = reg.parse_file(b"hello", "a.txt")
    assert result.markdown == "fallback ok"


def test_empty_result_falls_back(monkeypatch):
    def empty(data, filename, **kwargs):
        return reg.ParseResult(markdown="   ")

    monkeypatch.setitem(reg._PARSERS, "txt", empty)
    monkeypatch.setattr(reg, "_markitdown_fallback", lambda d, f: reg.ParseResult(markdown="fallback ok"))
    result = reg.parse_file(b"hello", "a.txt")
    assert result.markdown == "fallback ok"


def test_fallback_none_reraises(monkeypatch):
    def boom(data, filename, **kwargs):
        raise RuntimeError("parse fail")

    monkeypatch.setitem(reg._PARSERS, "txt", boom)
    monkeypatch.setattr(reg, "_markitdown_fallback", lambda d, f: None)
    try:
        reg.parse_file(b"hello", "a.txt")
        assert False, "should raise"
    except RuntimeError:
        pass
