"""Shared test data: a synthetic expense file with known anomalies hidden in it."""
import numpy as np
import pandas as pd
import pytest


def make_transactions(n: int = 3000, seed: int = 0, n_anomalies: int = 30):
    """Return (raw dataframe, set of row numbers that are injected anomalies).

    Normal rows: category-specific amounts, daytime hours, a few dozen cards.
    Anomalies: amounts 8 to 15 times what is normal for their category.
    """
    rng = np.random.default_rng(seed)
    categories = {"groceries": (4.0, 0.5), "fuel": (3.5, 0.4), "dining": (3.2, 0.6),
                  "travel": (5.5, 0.7), "software": (4.5, 0.5)}
    names = list(categories)
    cat = rng.choice(names, n, p=[0.35, 0.2, 0.25, 0.08, 0.12])
    amount = np.array([rng.lognormal(*categories[c]) for c in cat]).round(2)

    start = pd.Timestamp("2025-01-01")
    day = rng.integers(0, 180, n)
    hour = rng.choice(np.arange(24), n, p=np.r_[np.full(7, 0.002), np.full(15, 0.0653), np.full(2, 0.0105)] / 1.0 /
                      (np.r_[np.full(7, 0.002), np.full(15, 0.0653), np.full(2, 0.0105)].sum()))
    when = start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(
        rng.integers(0, 3600, n), unit="s")

    anomaly_rows = set(rng.choice(n, n_anomalies, replace=False).tolist())
    for i in anomaly_rows:
        amount[i] = round(amount[i] * rng.uniform(8, 15), 2)

    df = pd.DataFrame({
        "Txn ID": [f"T{i:06d}" for i in range(n)],
        "Transaction Date": when.strftime("%Y-%m-%d %H:%M:%S"),
        "Amount": amount,
        "Category": cat,
        "Merchant": [f"{c[:3]}_vendor_{rng.integers(1, 9)}" for c in cat],
        "Card ID": rng.integers(1, 41, n),
    })
    return df, anomaly_rows


@pytest.fixture
def data():
    return make_transactions()


@pytest.fixture
def raw(data):
    return data[0]


@pytest.fixture
def anomaly_rows(data):
    return data[1]