"""Step 2a: find out which column plays which role, and check the data is usable.

Roles: amount and date are required. category, merchant, card, txn_id and label are optional.
Detection matches normalised column names (lower case, letters and digits only) against known aliases.
The user can always override the guess.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

ROLES = ("amount", "date", "category", "merchant", "card", "txn_id", "label")
REQUIRED = ("amount", "date")

ALIASES: dict[str, tuple[str, ...]] = {
    "amount": ("amt", "amount", "transactionamount", "transactionamt", "value", "total", "price", "cost", "debit"),
    "date": ("transdatetranstime", "transactiondate", "transactiondatetime", "datetime", "timestamp", "date",
             "time", "transdate", "postingdate", "transactiontime"),
    "category": ("category", "merchantcategory", "expensecategory", "type", "mcc", "categoryname"),
    "merchant": ("merchant", "merchantname", "vendor", "payee", "store", "description"),
    "card": ("ccnum", "cardid", "cardnumber", "card", "accountid", "account", "customerid", "customer",
             "employeeid", "employee", "userid", "user", "nameorig"),
    "txn_id": ("transnum", "transactionid", "txnid", "id", "transid", "referenceid"),
    "label": ("isfraud", "fraud", "class", "label", "isanomaly", "anomaly"),
}
# make aliases comparable with normalised names
ALIASES = {role: tuple(_a.lower() for _a in names) for role, names in ALIASES.items()}


class SchemaError(ValueError):
    """The uploaded data cannot be used; the message says what to fix."""


@dataclass
class ColumnMap:
    amount: str
    date: str
    category: str | None = None
    merchant: str | None = None
    card: str | None = None
    txn_id: str | None = None
    label: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)

    def used(self) -> dict[str, str]:
        return {role: col for role, col in self.as_dict().items() if col}


def _normalise(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Guess the column for each role. A column is never used for two roles; earlier roles win."""
    by_norm = {}
    for col in df.columns:
        by_norm.setdefault(_normalise(col), col)
    taken: set[str] = set()
    found: dict[str, str | None] = {}
    for role in ROLES:
        found[role] = None
        for alias in ALIASES[role]:
            col = by_norm.get(alias)
            if col is not None and col not in taken:
                found[role] = col
                taken.add(col)
                break
    # fall back to column types when names gave no answer
    if found["date"] is None:
        for col in df.columns:
            if col not in taken and pd.api.types.is_datetime64_any_dtype(df[col]):
                found["date"] = col
                taken.add(col)
                break
    return found


def resolve_columns(df: pd.DataFrame, overrides: dict[str, str | None] | None = None) -> ColumnMap:
    """Detect the columns, apply the user's overrides, and validate the result."""
    if df is None or len(df.columns) == 0:
        raise SchemaError("The file has no columns.")
    mapping = detect_columns(df)
    for role, col in (overrides or {}).items():
        if role not in ROLES:
            raise SchemaError(f"Unknown role '{role}'. Valid roles: {', '.join(ROLES)}.")
        mapping[role] = col or None
    missing = [role for role in REQUIRED if not mapping[role]]
    if missing:
        raise SchemaError(
            f"Could not find the {' and '.join(missing)} column. Columns in the file: {list(df.columns)}. "
            "Tell the engine which column to use, for example overrides={'amount': 'Total'}."
        )
    cmap = ColumnMap(**mapping)
    validate(df, cmap)
    return cmap


def validate(df: pd.DataFrame, cmap: ColumnMap, min_parse_rate: float = 0.9) -> None:
    """Check the mapped columns exist and that amounts and dates can mostly be read."""
    absent = [col for col in cmap.used().values() if col not in df.columns]
    if absent:
        raise SchemaError(f"Mapped columns not found in the file: {absent}.")
    values = [c for c in cmap.used().values()]
    if len(set(values)) != len(values):
        raise SchemaError("The same column is mapped to two roles.")
    if len(df) == 0:
        raise SchemaError("The file has no rows.")

    amount = df[cmap.amount]
    if not pd.api.types.is_numeric_dtype(amount):
        amount = pd.to_numeric(amount.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
    if amount.notna().mean() < min_parse_rate:
        raise SchemaError(f"Column '{cmap.amount}' does not look like an amount: "
                          f"only {amount.notna().mean():.0%} of its values are numbers.")

    dates = pd.to_datetime(df[cmap.date], errors="coerce")
    if dates.notna().mean() < min_parse_rate:
        raise SchemaError(f"Column '{cmap.date}' does not look like a date: "
                          f"only {dates.notna().mean():.0%} of its values can be read as dates.")