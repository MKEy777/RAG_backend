from __future__ import annotations


def test_preview_parent_child_mode():
    from knowlink.api.routers.documents import ChunkPreviewBody, chunk_preview

    text = "# 章节一\n\n" + "正文内容。" * 50 + "\n\n# 章节二\n\n" + "另一段。" * 50
    out = chunk_preview(ChunkPreviewBody(text=text, strategy="heading"))
    assert out["mode"] == "parent_child"
    assert len(out["parents"]) == 2
    assert out["parents"][0]["children"]


def test_preview_flat_mode():
    from knowlink.api.routers.documents import ChunkPreviewBody, chunk_preview

    out = chunk_preview(ChunkPreviewBody(text="普通文本。" * 200, strategy="recursive", enable_parent_child=False))
    assert out["mode"] == "flat"
    assert len(out["chunks"]) >= 2
