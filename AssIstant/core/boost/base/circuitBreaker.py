from enum import Enum
from typing import Optional
from time import monotonic

from AssIstant.common.logger import FormatLog


class CircuitState(Enum):
    OPEN        = "OPEN"
    CLOSE       = "CLOSE"
    HALF_OPEN   = "HALF_OPEN"


class CircuitBreaker(FormatLog):
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 60.0,
        name:str = "circuitBreaker"
    ):
        """
        熔断器
        :param failure_threshold:
        :param recovery_seconds:
        """
        self._threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        FormatLog.__init__(self, name)

        self._state = CircuitState.CLOSE
        self._failure_count = 0
        self._open_at: Optional[float] = None

    def allow(self) -> bool:
        if CircuitState.CLOSE == self._state:
            return True
        if CircuitState.OPEN == self._state:
            if monotonic() - self._open_at >= self._recovery_seconds:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self):
        self._failure_count = 0
        self._state = CircuitState.CLOSE

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count > self._threshold:
            self._state = CircuitState.OPEN
            self._open_at = monotonic()
            self.warning(f"open after {self._failure_count} failures")

    def state(self) -> str:
        return self._state.value