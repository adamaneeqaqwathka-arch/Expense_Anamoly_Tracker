"""A small client for the Expense Anomaly Auditor API. The Streamlit app uses it; so can any other Python code."""
from __future__ import annotations

from typing import Any

import requests


class ApiError(Exception):
    """The API refused the request or could not be reached. The message is safe to show to a user."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _detail(response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        return response.text or f"The API answered with status {response.status_code}."
    if isinstance(detail, list):            # FastAPI validation errors arrive as a list of problems
        return "; ".join(f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg', '')}"
                         for item in detail)
    return str(detail)


class AuditorClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", session: Any | None = None, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    # ---------------------------------------------------------------- status
    def health(self) -> dict:
        return self._json("get", "/health", timeout=5)

    def model_info(self) -> dict:
        return self._json("get", "/model/info")

    # ---------------------------------------------------------------- scoring
    def detect_columns(self, filename: str, content: bytes) -> dict:
        return self._json("post", "/columns/detect", files={"file": (filename, content, "text/csv")})

    def score(self, filename: str, content: bytes, flag_rate: float = 0.05,
              columns: dict[str, str | None] | None = None, limit: int = 1000) -> dict:
        import json
        data = {"flag_rate": str(flag_rate), "limit": str(limit)}
        if columns:
            data["columns"] = json.dumps(columns)
        return self._json("post", "/score", files={"file": (filename, content, "text/csv")}, data=data)

    def get_run(self, run_id: str, flag_rate: float, limit: int = 1000) -> dict:
        return self._json("get", f"/runs/{run_id}", params={"flag_rate": flag_rate, "limit": limit})

    def report(self, run_id: str, flag_rate: float, flagged_only: bool = True) -> bytes:
        return self._request("get", f"/runs/{run_id}/report.csv",
                             params={"flag_rate": flag_rate, "flagged_only": str(flagged_only).lower()}).content

    def score_transaction(self, run_id: str, amount: float, date: str, category: str | None = None,
                          merchant: str | None = None, card: str | None = None) -> dict:
        body = {"run_id": run_id, "amount": amount, "date": date, "category": category or None,
                "merchant": merchant or None, "card": card or None}
        return self._json("post", "/score/transaction", json=body)

    # ---------------------------------------------------------------- plumbing
    def _request(self, method: str, path: str, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = getattr(self.session, method)(self.base_url + path, **kwargs)
        except requests.exceptions.RequestException as err:
            raise ApiError(0, f"Cannot reach the API at {self.base_url}. Is it running? ({type(err).__name__})") from err
        if response.status_code >= 400:
            raise ApiError(response.status_code, _detail(response))
        return response

    def _json(self, method: str, path: str, **kwargs) -> dict:
        return self._request(method, path, **kwargs).json()
