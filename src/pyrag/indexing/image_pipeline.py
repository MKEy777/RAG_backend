from __future__ import annotations

from ..llm.client import get_llm_client
from ..parsing.base import ImageRef

VLM_OCR_PROMPT = (
    "You are an OCR assistant. Extract all body text content from this document image "
    "and output in pure Markdown format. Ignore headers/footers. Use Markdown tables for tables. "
    "Use LaTeX ($/$$) for formulas. Maintain original reading order. "
    "Only output extracted content. If no text, reply 'No text content'."
)

VLM_CAPTION_PROMPT = (
    "Provide a brief and concise description of the main content of the image in Chinese."
)


async def process_image_cascade(
    image: ImageRef,
    use_ocr: bool = True,
    use_vlm: bool = True,
) -> dict:
    client = get_llm_client()
    result = {"ocr_text": "", "caption": ""}

    if use_ocr:
        try:
            ocr_text = await client.vlm(image.base64, VLM_OCR_PROMPT, mime=image.mime)
            if ocr_text and ocr_text.strip() != "No text content":
                result["ocr_text"] = ocr_text.strip()
        except Exception:
            pass

    if use_vlm and not result["ocr_text"]:
        try:
            caption = await client.vlm(image.base64, VLM_CAPTION_PROMPT, mime=image.mime)
            if caption:
                result["caption"] = caption.strip()
        except Exception:
            pass

    return result
