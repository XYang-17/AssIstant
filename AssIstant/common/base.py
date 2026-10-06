from dataclasses import dataclass

@dataclass
class State:
    success: int = 0
    total: int = 0
    latency: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def average_latency(self) -> float:
        return self.latency / self.total if self.total else 0.0