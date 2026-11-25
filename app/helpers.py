"""Pure functions behind the Streamlit screen: easy to test without a browser."""
from __future__ import annotations

import itertools
from typing import Any

import pandas as pd
import plotly.graph_objects as go

NONE_CHOICE = "(none)"
ROLE_LABELS = {
    "amount": "Amount", "date": "Date", "category": "Category", "merchant": "Merchant",
    "card": "Card or customer ID", "txn_id": "Transaction ID", "label": "Fraud label (evaluation only)",
}
TABLE_LABELS = {
    "rank": "Rank", "source_row": "Row in file", "date": "Date", "amount": "Amount", "category": "Category",
    "merchant": "Merchant", "card": "Card", "anomaly_score": "Score", "risk_percentile": "Percentile",
    "reason": "Why it was flagged",
}
NAVY, GOLD, GREY = "#194169", "#C9A227", "#9FB3CC"


def overrides_from_choices(detected: dict[str, str | None], chosen: dict[str, str]) -> dict[str, str | None] | None:
    """Only the roles the user changed from the automatic guess. 'None' means the user cleared an optional column."""
    changes: dict[str, str | None] = {}
    for role, choice in chosen.items():
        value = None if choice == NONE_CHOICE else choice
        if value != detected.get(role):
            changes[role] = value
    return changes or None


def flagged_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """The flagged transactions, with friendly column names."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame[frame["flagged"]]
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d %H:%M")
    keep = [c for c in TABLE_LABELS if c in frame.columns]
    out = frame[keep].rename(columns=TABLE_LABELS)
    for col in ("Score", "Percentile"):
        if col in out:
            out[col] = out[col].round(2)
    return out.reset_index(drop=True)


def flag_threshold(rows: list[dict[str, Any]]) -> float | None:
    """The lowest score that is still flagged, if the returned rows include the end of the flagged group."""
    flagged = [r["anomaly_score"] for r in rows if r.get("flagged")]
    return min(flagged) if flagged else None


def _layout(fig: go.Figure, title: str, x: str, y: str) -> go.Figure:
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, margin={"l": 10, "r": 10, "t": 50, "b": 10},
                      height=360, showlegend=False, plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#E8EEF6")
    return fig


def score_histogram(charts: dict[str, Any], threshold: float | None = None) -> go.Figure:
    hist = charts["score_histogram"]
    edges, counts = hist["bin_edges"], hist["counts"]
    mids = [(a + b) / 2 for a, b in itertools.pairwise(edges)]
    fig = go.Figure(go.Bar(x=mids, y=counts, marker_color=GREY))
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color=GOLD,
                      annotation_text="flag threshold", annotation_position="top right")
    return _layout(fig, "How unusual are the transactions?", "Anomaly score (higher = more unusual)",
                   "Transactions")


def flagged_by_category(charts: dict[str, Any]) -> go.Figure:
    frame = pd.DataFrame(charts["flagged_by_category"])
    frame = frame.sort_values("flagged")
    fig = go.Figure(go.Bar(x=frame["flagged"], y=frame["category"], orientation="h", marker_color=NAVY))
    return _layout(fig, "Flagged transactions by category", "Flagged transactions", "")


def flagged_over_time(charts: dict[str, Any]) -> go.Figure:
    frame = pd.DataFrame(charts["flagged_by_month"])
    fig = go.Figure(go.Bar(x=frame["month"], y=frame["flagged"], marker_color=NAVY))
    return _layout(fig, "Flagged transactions per month", "", "Flagged transactions")


def cluster_table(summary: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(summary["cluster_profile"]).rename(columns={
        "cluster": "Cluster", "transactions": "Transactions", "share": "Share", "median_amount": "Median amount",
        "top_category": "Top category", "night_share": "Night share", "rare_pattern": "Rare pattern"})
    for col in ("Share", "Night share"):
        if col in frame:
            frame[col] = (frame[col] * 100).round(1).astype(str) + "%"
    return frame


def cleaning_lines(summary: dict[str, Any]) -> list[str]:
    report = summary.get("cleaning", {})
    lines = [f"{report.get('rows_in', 0):,} rows in, {report.get('rows_out', 0):,} used."]
    lines += [f"Removed {count:,} for: {reason}." for reason, count in report.get("removed", {}).items() if count]
    lines += report.get("notes", [])
    return lines