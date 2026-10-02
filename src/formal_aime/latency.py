from __future__ import annotations

import json
import math
import threading
from collections import defaultdict, deque
from pathlib import Path


class RouteLatencyTracker:
    """Thread-safe rolling successful Worker-call latency observations by route."""

    def __init__(self, *, window_size: int = 64) -> None:
        if int(window_size) <= 0:
            raise ValueError("route latency window_size must be positive")
        self.window_size = int(window_size)
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self._lock = threading.Lock()

    def record(self, route: str, duration_s: float) -> None:
        route = str(route or "default").strip() or "default"
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("route latency duration must be finite and non-negative")
        with self._lock:
            self._samples[route].append(duration)

    def snapshot(self) -> dict[str, list[float]]:
        with self._lock:
            return {route: list(samples) for route, samples in sorted(self._samples.items())}


class RouteTokenTracker:
    """Thread-safe, optionally persistent Worker token observations by route."""

    def __init__(
        self,
        *,
        window_size: int = 64,
        path: str | Path | None = None,
    ) -> None:
        if int(window_size) <= 0:
            raise ValueError("route token window_size must be positive")
        self.window_size = int(window_size)
        self.path = Path(path) if path is not None else None
        self._samples: dict[str, deque[dict[str, int]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._lock = threading.Lock()
        self._load()

    def record(self, route: str, token_in: int, token_out: int) -> None:
        route = str(route or "default").strip() or "default"
        token_in = int(token_in)
        token_out = int(token_out)
        if token_in < 0 or token_out < 0:
            raise ValueError("route token usage must be non-negative")
        sample = {
            "token_in": token_in,
            "token_out": token_out,
            "total_tokens": token_in + token_out,
        }
        with self._lock:
            self._samples[route].append(sample)
            self._persist_locked()

    def snapshot(self) -> dict[str, list[dict[str, int]]]:
        with self._lock:
            return {
                route: [dict(sample) for sample in samples]
                for route, samples in sorted(self._samples.items())
            }

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"invalid route token state: {self.path}")
        routes = payload.get("routes", {})
        if not isinstance(routes, dict):
            raise ValueError(f"invalid route token routes: {self.path}")
        for route, raw_samples in routes.items():
            if not isinstance(raw_samples, list):
                raise ValueError(f"invalid route token samples for {route}")
            for raw in raw_samples[-self.window_size :]:
                if not isinstance(raw, dict):
                    raise ValueError(f"invalid route token sample for {route}")
                token_in = int(raw.get("token_in", 0))
                token_out = int(raw.get("token_out", 0))
                if token_in < 0 or token_out < 0:
                    raise ValueError(f"negative route token sample for {route}")
                self._samples[str(route)].append(
                    {
                        "token_in": token_in,
                        "token_out": token_out,
                        "total_tokens": token_in + token_out,
                    }
                )

    def _persist_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "window_size": self.window_size,
            "routes": {route: list(samples) for route, samples in sorted(self._samples.items())},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
