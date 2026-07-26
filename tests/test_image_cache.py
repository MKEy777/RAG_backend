from __future__ import annotations

from knowlink.indexing import image_pipeline as ipm


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def vlm(self, image_base64, prompt, mime="image/jpeg", purpose="", model=None):
        self.calls += 1
        if not self.responses:
            raise RuntimeError("vlm down")
        return self.responses.pop(0)


def _img():
    return ipm.ImageRef(data=b"\x89PNG-fake-bytes", mime="image/png")


async def test_cache_hit_skips_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(ipm, "VLM_CACHE_DIR", tmp_path)
    fake = FakeClient(["识别出的文字"])
    monkeypatch.setattr(ipm, "get_llm_client", lambda: fake)

    r1 = await ipm.process_image_cascade(_img())
    assert r1["ocr_text"] == "识别出的文字"
    assert fake.calls == 1

    r2 = await ipm.process_image_cascade(_img())  # 同 hash 命中缓存
    assert r2 == r1
    assert fake.calls == 1


async def test_failure_returns_empty_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ipm, "VLM_CACHE_DIR", tmp_path)
    fake = FakeClient([])  # 每次都抛错
    monkeypatch.setattr(ipm, "get_llm_client", lambda: fake)
    r = await ipm.process_image_cascade(_img())
    assert r == {"ocr_text": "", "caption": ""}
    assert list(tmp_path.iterdir()) == []  # 失败不写缓存
