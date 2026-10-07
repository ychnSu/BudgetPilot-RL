from dataclasses import dataclass, field
import time


@dataclass
class Budget:
    tokens: int = 16000
    tool_calls: int = 20
    wall_seconds: int = 600
    max_turns: int = 24
    max_completion_tokens: int = 2048


@dataclass
class Usage:
    tokens: int = 0
    tool_calls: int = 0
    turns: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def remaining(self, limit: Budget):
        return {"tokens": max(0, limit.tokens - self.tokens),
                "tool_calls": max(0, limit.tool_calls - self.tool_calls),
                "seconds": round(max(0, limit.wall_seconds - self.elapsed), 2),
                "turns": max(0, limit.max_turns - self.turns)}

    def exhausted(self, limit: Budget):
        return any(value <= 0 for value in self.remaining(limit).values())
