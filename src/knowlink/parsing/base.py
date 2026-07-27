from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class ImageRef:
    data: bytes
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mime: str = "image/jpeg"

    @property
    def base64(self) -> str:
        import base64

        return base64.b64encode(self.data).decode()


@dataclass
class ParseResult:
    markdown: str
    images: list[ImageRef] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ParserCapabilityError(Exception):
    def __init__(self, fmt: str, tool: str):
        self.fmt = fmt
        self.tool = tool
        super().__init__(
            f"Parsing '{fmt}' requires '{tool}' which is not installed. "
            f"Please install it or convert the file manually."
        )
