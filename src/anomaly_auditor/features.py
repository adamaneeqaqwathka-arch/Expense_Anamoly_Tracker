"""Step 2c: turn clean transactions into numbers describing spending behaviour.

FeatureBuilder learns its statistics on the data it is fitted on (category averages, merchant frequencies,
which columns are useful) and applies exactly the same recipe to any later data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SchemaError

# Computed for explanations but never given to the model: a card's first transaction in a file is not suspicious,
# the file just starts there, and a rare 0/1 flag would dominate the distances after scaling.
NOT_FOR_MODEL = {"is_first_txn"}
NUMERIC_ORDER = ["amt_cat_z", "log_amt", "merchant_rarity", "hour_sin", "hour_cos",
                 "is_weekend", "is_night", "log_gap", "is_first_txn"]


class FeatureBuilder:
    def __init__(self, max_onehot: int = 15, corr_cut: float = 0.9, const_cut: float = 0.995):
        self.max_onehot = max_onehot
        self.corr_cut = corr_cut
        self.const_cut = const_cut

    # ---------- learning ----------
    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        log_amt = np.log1p(df["amount"])
        self.global_mean_ = float(log_amt.mean())
        self.global_std_ = float(log_amt.std()) if log_amt.std() > 0 else 1.0

        stats = log_amt.groupby(df["category"]).agg(["mean", "std"])
        stats["std"] = stats["std"].where(stats["std"] > 0, self.global_std_).fillna(self.global_std_)
        self.cat_stats_ = stats
        self.cat_levels_ = list(df["category"].value_counts().head(self.max_onehot).index)

        self.has_merchant_ = "merchant" in df.columns and df["merchant"].nunique() > 1
        if self.has_merchant_:
            self.merchant_share_ = df["merchant"].value_counts(normalize=True)
            self.max_rarity_ = float(-np.log10(1.0 / (len(df) + 1)))
        self.has_card_ = "card" in df.columns and df["card"].nunique() > 1
        self.gap_median_ = 0.0
        if self.has_card_:
            gap = self._gap_hours(df).dropna()
            self.gap_median_ = float(gap.median()) if len(gap) else 0.0

        full = self._build_all(df)
        keep = [c for c in full.columns if c not in NOT_FOR_MODEL
                and full[c].round(6).value_counts(normalize=True).iloc[0] <= self.const_cut]  # near-constant out
        numeric = [c for c in NUMERIC_ORDER if c in keep]
        corr = full[numeric].corr().abs()
        redundant = {b for i, a in enumerate(numeric) for b in numeric[i + 1:] if corr.loc[a, b] > self.corr_cut}
        self.columns_ = [c for c in full.columns if c in keep and c not in redundant]
        self.dropped_ = {"near_constant": sorted(set(full.columns) - set(keep) - NOT_FOR_MODEL),
                         "redundant": sorted(redundant)}
        self.neutral_ = full[self.columns_].mean()     # used when new data lacks an optional column
        if not any(c in NUMERIC_ORDER for c in self.columns_):
            raise SchemaError("The amounts and times show no variation to learn from "
                              "(all transactions look identical).")
        return self

    # ---------- applying ----------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        full = self._build_all(df)
        out = full.reindex(columns=self.columns_)
        return out.fillna(self.neutral_)               # a missing optional column counts as "typical", not as 0

    def full(self, df: pd.DataFrame) -> pd.DataFrame:
        """Every candidate feature, including those dropped from the model (used to explain flags)."""
        return self._build_all(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ---------- internals ----------
    def _gap_hours(self, df: pd.DataFrame) -> pd.Series:
        order = df.sort_values(["card", "date"])
        gap = order.groupby("card")["date"].diff().dt.total_seconds() / 3600
        return gap.reindex(df.index)

    def _build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        log_amt = np.log1p(df["amount"])
        mean = df["category"].map(self.cat_stats_["mean"]).fillna(self.global_mean_)
        std = df["category"].map(self.cat_stats_["std"]).fillna(self.global_std_)
        f["amt_cat_z"] = (log_amt - mean) / std
        f["log_amt"] = log_amt

        if self.has_merchant_ and "merchant" in df.columns:
            share = df["merchant"].map(self.merchant_share_)
            rarity = -np.log10(share)
            f["merchant_rarity"] = rarity.fillna(self.max_rarity_)       # a merchant never seen is the rarest

        hour = df["date"].dt.hour
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)                     # the clock is circular
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        f["is_weekend"] = (df["date"].dt.day_of_week >= 5).astype(int)
        f["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)

        if self.has_card_ and "card" in df.columns:
            gap = self._gap_hours(df)
            f["is_first_txn"] = gap.isna().astype(int)
            f["log_gap"] = np.log1p(gap.fillna(self.gap_median_))

        for level in self.cat_levels_:
            f[f"cat_{level}"] = (df["category"] == level).astype(int)
        return f.astype(float)