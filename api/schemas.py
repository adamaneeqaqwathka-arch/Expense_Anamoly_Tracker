"""The shapes of what the API accepts and returns. FastAPI turns these into the /docs page."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    version: str


class ColumnDetection(BaseModel):
    columns: list[str]
    detected: dict[str, str | None] = Field(description="role -> column name the engine would use")
    required_roles: list[str]
    optional_roles: list[str]
    preview: list[dict[str, Any]] = Field(description="first rows of the file")
    total_rows: int


class ModelInfo(BaseModel):
    version: str
    method: str
    defaults: dict[str, Any]
    limits: dict[str, Any]
    roles: dict[str, list[str]]


class ScoreResponse(BaseModel):
    run_id: str = Field(description="use this to change the sensitivity, download the report or score one transaction")
    filename: str
    flag_rate: float
    total_rows: int
    flagged_count: int
    columns: dict[str, str] = Field(description="role -> column name that was used")
    summary: dict[str, Any] = Field(description="clusters, silhouette, features, cleaning report, cluster profiles")
    rows: list[dict[str, Any]] = Field(description="the most unusual transactions first")
    rows_returned: int
    charts: dict[str, Any] = Field(description="data for the score histogram and the flag breakdowns")


class TransactionIn(BaseModel):
    run_id: str = Field(description="a run from POST /score; its file defines what is normal")
    amount: float = Field(gt=0)
    date: datetime
    category: str | None = None
    merchant: str | None = None
    card: str | None = None


class TransactionOut(BaseModel):
    anomaly_score: float
    risk_percentile: float = Field(description="share of the file's transactions that score lower, 0-100")
    would_be_flagged: bool
    reason: str
    cluster: int


class ErrorOut(BaseModel):
    detail: str
