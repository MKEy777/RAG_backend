from __future__ import annotations

import time
from enum import Enum

from ..logging import get_logger

log = get_logger("circuit_breaker")


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls

        self._state = State.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> State:
        if self._state == State.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout_s:
                self._state = State.HALF_OPEN
                self._half_open_calls = 0
                log.info("熔断器_半开", breaker=self.name)
        return self._state

    def allow_request(self) -> bool:
        state = self.state
        if state == State.CLOSED:
            return True
        if state == State.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        return False

    def record_success(self) -> None:
        if self._state == State.HALF_OPEN:
            log.info("熔断器_关闭", breaker=self.name)
        self._state = State.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == State.HALF_OPEN:
            self._state = State.OPEN
            log.warning("熔断器_重新打开", breaker=self.name, failures=self._failure_count)
        elif self._failure_count >= self.failure_threshold:
            self._state = State.OPEN
            log.warning(
                "熔断器_打开",
                breaker=self.name,
                failures=self._failure_count,
                recovery_in_s=self.recovery_timeout_s,
            )


class CircuitOpenError(Exception):
    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name
        super().__init__(f"Circuit breaker '{breaker_name}' is open")
