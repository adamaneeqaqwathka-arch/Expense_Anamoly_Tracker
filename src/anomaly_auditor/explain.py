"""Step 2e: a short, plain-language reason for each flagged transaction.

Each candidate reason has a strength (how many standard deviations unusual it is). The two strongest are shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FALLBACK = "overall pattern differs from typical transactions"


def reasons(clean: pd.DataFrame, full: pd.DataFrame, cluster_share: np.ndarray, small_cut: float,
            positions) -> dict[int, str]:
    """Return {row position: reason text} for the given positions in `clean`.

    clean: the cleaned transactions (canonical column names)
    full: every candidate feature for those transactions (FeatureBuilder.full)
    cluster_share: share of all transactions that sit in each row's cluster
    """
    amount = clean["amount"].to_numpy()
    category = clean["category"].astype(str).to_numpy()
    has_category = bool((clean["category"] != "Unknown").any())
    merchant = clean["merchant"].astype(str).to_numpy() if "merchant" in clean.columns else None
    hour = clean["date"].dt.hour.to_numpy()

    spread = full.std().replace(0, np.nan)
    z = ((full - full.mean()) / spread).fillna(0)          # how unusual each feature is within this batch
    amt_z = full["amt_cat_z"].to_numpy()
    rarity_z = z["merchant_rarity"].to_numpy() if "merchant_rarity" in z else None
    gap_z = z["log_gap"].to_numpy() if "log_gap" in z else None
    first = full["is_first_txn"].to_numpy() if "is_first_txn" in full else None
    night = full["is_night"].to_numpy()
    night_z = z["is_night"].to_numpy()

    out: dict[int, str] = {}
    for pos in positions:
        found: list[tuple[float, str]] = []
        if abs(amt_z[pos]) >= 2:
            where = f"'{category[pos]}'" if has_category else "this file"
            direction = "high" if amt_z[pos] > 0 else "low"
            found.append((abs(amt_z[pos]),
                          f"amount {amount[pos]:,.2f} is unusually {direction} for {where} (z = {amt_z[pos]:+.1f})"))
        if rarity_z is not None and rarity_z[pos] >= 2:
            found.append((rarity_z[pos], f"merchant '{merchant[pos]}' is rare in this file"))
        if gap_z is not None and abs(gap_z[pos]) >= 2 and not (first is not None and first[pos] == 1):
            text = ("very soon after the previous transaction on this card" if gap_z[pos] < 0
                    else "long silence since the previous transaction on this card")
            found.append((abs(gap_z[pos]), text))
        if night[pos] == 1 and night_z[pos] >= 1.5:
            found.append((night_z[pos], f"made at night ({hour[pos]:02d}:00), when few transactions happen"))
        if cluster_share[pos] < small_cut:
            found.append((2.5, f"belongs to a rare spending pattern ({cluster_share[pos]:.1%} of transactions)"))
        found.sort(key=lambda item: -item[0])
        out[int(pos)] = "; ".join(text for _, text in found[:2]) or FALLBACK
    return out