"""Keeps recently scored files in memory so the slider and the report need no second upload."""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd

from anomaly_auditor.engine import AnomalyEngine


@dataclass
class Run:
    engine: AnomalyEngine
    table: pd.DataFrame                     # every transaction, ranked, with scores and reasons
    filename: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)


class RunStore:
    """Small in-memory store: newest runs are kept, old ones expire. Safe to use from several threads."""

    def __init__(self, max_runs: int = 20, ttl_seconds: int = 3600):
        self.max_runs = max_runs
        self.ttl = ttl_seconds
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, run: Run) -> Run:
        with self._lock:
            self._expire()
            self._runs[run.run_id] = run
            while len(self._runs) > self.max_runs:
                self._runs.popitem(last=False)          # drop the oldest
        return run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            self._expire()
            return self._runs.get(run_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)

    def _expire(self) -> None:
        now = time.time()
        for run_id in [r for r, run in self._runs.items() if now - run.created > self.ttl]:
            del self._runs[run_id]