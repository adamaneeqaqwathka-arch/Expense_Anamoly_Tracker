"""API settings. Every value can be overridden with an environment variable (see from_env)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = 25                 # largest CSV accepted
    max_rows: int = 500_000                 # largest number of rows accepted
    max_runs: int = 20                      # scored files kept in memory
    run_ttl_seconds: int = 3600             # how long a scored file is kept
    max_table_rows: int = 5000              # most rows one response may carry
    cors_origins: tuple[str, ...] = ("*",)  # websites allowed to call the API from a browser

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.getenv("ANOMALY_CORS_ORIGINS", "*")
        return cls(
            max_upload_mb=int(os.getenv("ANOMALY_MAX_UPLOAD_MB", cls.max_upload_mb)),
            max_rows=int(os.getenv("ANOMALY_MAX_ROWS", cls.max_rows)),
            max_runs=int(os.getenv("ANOMALY_MAX_RUNS", cls.max_runs)),
            run_ttl_seconds=int(os.getenv("ANOMALY_RUN_TTL_SECONDS", cls.run_ttl_seconds)),
            max_table_rows=int(os.getenv("ANOMALY_MAX_TABLE_ROWS", cls.max_table_rows)),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )