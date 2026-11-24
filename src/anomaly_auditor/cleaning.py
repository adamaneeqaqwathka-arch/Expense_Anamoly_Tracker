"""Step 2b: make the data analysis-ready. These are the wrangling steps from the notebook.

After cleaning, columns have canonical names: amount, date, category, merchant, card, txn_id, label.
The original row numbers are kept in the index so every result can be traced back to the input file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import ColumnMap, SchemaError

CANONICAL = ("amount", "date", "category", "merchant", "card", "txn_id", "label")


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    removed: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def removed_total(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> dict:
        return {"rows_in": self.rows_in, "rows_out": self.rows_out, "removed": dict(self.removed),
                "notes": list(self.notes)}


def clean(df: pd.DataFrame, cmap: ColumnMap) -> tuple[pd.DataFrame, CleaningReport]:
    """Return (clean data, report). The input frame is not modified."""
    report = CleaningReport(rows_in=len(df))
    out = df.copy()
    out.index.name = "source_row"

    # 1. canonical names, leftover index columns out
    out = out.drop(columns=[c for c in out.columns if str(c).startswith("Unnamed:")])
    rename = {col: role for role, col in cmap.used().items() if col != role}
    clash = [role for role in rename.values() if role in out.columns and role not in rename]
    if clash:                                   # an unrelated column already uses a canonical name
        out = out.rename(columns={c: f"{c}_original" for c in clash})
    out = out.rename(columns=rename)

    # 2. types
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if not pd.api.types.is_numeric_dtype(out["amount"]):
        out["amount"] = pd.to_numeric(out["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                                      errors="coerce")
        report.notes.append("Amount column was text; currency symbols and separators were stripped.")
    out["amount"] = out["amount"].astype(float)

    # 3. text columns: trim spaces, fill gaps, drop a meaningless prefix
    if "category" not in out.columns:
        out["category"] = "Unknown"
        report.notes.append("No category column; all transactions are treated as one category.")
    for col in ("category", "merchant", "card", "txn_id"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    out["category"] = out["category"].fillna("Unknown").replace("", "Unknown")
    if "merchant" in out.columns:
        out["merchant"] = out["merchant"].fillna("Unknown").replace("", "Unknown")
        if out["merchant"].str.startswith("fraud_").mean() > 0.9:
            out["merchant"] = out["merchant"].str.replace(r"^fraud_", "", regex=True)
            report.notes.append("Removed the 'fraud_' prefix from merchant names.")

    # 4. rows that cannot be scored
    before = len(out)
    out = out.dropna(subset=["amount", "date"])
    report.removed["missing amount or date"] = before - len(out)

    # 5. duplicates (by transaction id when there is one, otherwise identical rows)
    before = len(out)
    out = out[~out["txn_id"].duplicated()] if "txn_id" in out.columns else out[~out.duplicated()]
    report.removed["duplicates"] = before - len(out)

    # 6. impossible values (refunds and zero amounts cannot be log-transformed)
    before = len(out)
    out = out[out["amount"] > 0]
    report.removed["amount <= 0"] = before - len(out)
    before = len(out)
    out = out[out["date"] <= pd.Timestamp.now() + pd.Timedelta(days=1)]
    report.removed["future dates"] = before - len(out)

    if len(out) == 0:
        raise SchemaError("No usable rows remain after cleaning.")
    report.rows_out = len(out)
    return out, report