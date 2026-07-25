from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging import get_logger
from ..settings import settings

log = get_logger("ratelimit")

_window: dict[str, list[float]] = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health" or request.url.path == "/metrics":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_s = 60.0
        limit = settings.rate_limit_per_minute

        if client_ip not in _window:
            _window[client_ip] = []

        timestamps = _window[client_ip]
        cutoff = now - window_s
        _window[client_ip] = [t for t in timestamps if t > cutoff]
        timestamps = _window[client_ip]

        if len(timestamps) >= limit:
            log.warning("请求频率限制", ip=client_ip, count=len(timestamps))
            return Response(
                content='{"code":"RATE_LIMITED","message":"Too many requests"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        timestamps.append(now)
        return await call_next(request)
