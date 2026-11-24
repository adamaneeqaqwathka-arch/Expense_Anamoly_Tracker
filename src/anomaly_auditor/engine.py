"""Step 2d: the AnomalyEngine. Raw CSV data in, ranked table of unusual transactions out.

Method: build behaviour features, scale them, cluster with K-Means, and score every transaction by how far it
sits from the centre of its cluster (normalised inside the cluster). Tiny clusters are not treated as normal
behaviour: their members are measured against the nearest large cluster and get extra weight, because K-Means
tends to put a group of anomalies into a cluster of their own. A second signal, the amount's deviation from
what is typical for its category, is added so that large expenses cannot hide inside a cluster.
The model is fitted on the data being audited, so "normal" means normal for that file.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import explain
from .cleaning import CleaningReport, clean
from .features import FeatureBuilder
from .schema import ColumnMap, resolve_columns


class EngineError(ValueError):
    """The engine cannot do what was asked; the message says why."""


class AnomalyEngine:
    MAX_FLAG_RATE = 0.10          # reasons are written for the top 10%, so flags are limited to that

    def __init__(self, k: int | None = None, k_range: tuple[int, int] = (3, 10),
                 small_cluster_cut: float = 0.02, small_weight: float = 2.0, min_rows: int = 200,
                 flag_rate: float = 0.05, amount_weight: float = 1.0, amount_z_floor: float = 1.5,
                 random_state: int = 42):
        if not 0 < flag_rate <= self.MAX_FLAG_RATE:
            raise EngineError(f"flag_rate must be between 0 and {self.MAX_FLAG_RATE}.")
        self.k = k
        self.k_range = k_range
        self.small_cluster_cut = small_cluster_cut
        self.small_weight = small_weight
        self.min_rows = min_rows
        self.flag_rate = flag_rate
        self.amount_weight = amount_weight
        self.amount_z_floor = amount_z_floor
        self.random_state = random_state
        self.fitted_ = False

    # ------------------------------------------------------------------ preparing
    def prepare(self, raw: pd.DataFrame, overrides: dict | None = None) -> tuple[pd.DataFrame, ColumnMap, CleaningReport]:
        """Detect columns, validate, and clean a raw upload."""
        cmap = resolve_columns(raw, overrides)
        cleaned, report = clean(raw, cmap)
        self.columns_, self.cleaning_report_ = cmap, report
        return cleaned, cmap, report

    # ------------------------------------------------------------------ learning
    def fit(self, cleaned: pd.DataFrame) -> AnomalyEngine:
        """Learn what normal looks like from cleaned transactions."""
        if len(cleaned) < self.min_rows:
            raise EngineError(f"Need at least {self.min_rows} usable transactions to learn what is normal; "
                              f"got {len(cleaned)}.")
        self.builder_ = FeatureBuilder().fit(cleaned)
        X = self.builder_.transform(cleaned)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self._scale(X)

        self.k_, self.tuning_ = self._choose_k(Xs)
        self.kmeans_ = KMeans(n_clusters=self.k_, n_init=10, random_state=self.random_state).fit(Xs)
        labels = self.kmeans_.labels_
        dist = self.kmeans_.transform(Xs).min(axis=1)

        by_cluster = pd.DataFrame({"c": labels, "d": dist}).groupby("c")["d"].agg(["mean", "std"])
        by_cluster["std"] = by_cluster["std"].replace(0, np.nan).fillna(1.0)
        self.ref_ = by_cluster
        self.cluster_share_ = pd.Series(labels).value_counts(normalize=True)
        large = self.cluster_share_[self.cluster_share_ >= self.small_cluster_cut].index
        self.major_clusters_ = np.array(sorted(large if len(large) else [self.cluster_share_.idxmax()]))
        self.silhouette_ = float(silhouette_score(Xs, labels, sample_size=min(5000, len(Xs)),
                                                  random_state=self.random_state))
        self.profile_ = self._profile(cleaned, labels)
        self.n_fit_ = len(cleaned)
        self.fitted_ = True
        return self

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(X)

    def _choose_k(self, Xs: np.ndarray) -> tuple[int, pd.DataFrame | None]:
        if self.k is not None:
            return int(self.k), None
        kmax = min(self.k_range[1], len(Xs) // 30)          # keep clusters big enough to mean something
        kmin = min(self.k_range[0], kmax)
        if kmax < 2:
            raise EngineError("Too few transactions to form clusters.")
        if kmin == kmax:
            return kmax, None
        rng = np.random.default_rng(self.random_state)
        sample = Xs[rng.choice(len(Xs), size=min(len(Xs), 20_000), replace=False)]
        rows = []
        for k in range(kmin, kmax + 1):
            km = KMeans(n_clusters=k, n_init=3, random_state=self.random_state).fit(sample)
            sil = silhouette_score(sample, km.labels_, sample_size=min(3000, len(sample)),
                                   random_state=self.random_state)
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        tuning = pd.DataFrame(rows)
        return int(tuning.loc[tuning["silhouette"].idxmax(), "k"]), tuning

    def _profile(self, cleaned: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        frame = cleaned.assign(cluster=labels)
        grouped = frame.groupby("cluster")
        profile = pd.DataFrame({
            "transactions": grouped.size(),
            "share": grouped.size() / len(frame),
            "median_amount": grouped["amount"].median(),
            "top_category": grouped["category"].agg(lambda s: s.value_counts().index[0]),
            "night_share": grouped["date"].agg(lambda s: ((s.dt.hour >= 22) | (s.dt.hour < 6)).mean()),
        })
        profile["rare_pattern"] = profile["share"] < self.small_cluster_cut
        return profile.round(3)

    # ------------------------------------------------------------------ scoring
    def score(self, cleaned: pd.DataFrame, flag_rate: float | None = None) -> pd.DataFrame:
        """Score cleaned transactions. Returns them ranked, most unusual first, with reasons for the top 10%."""
        self._check_fitted()
        full = self.builder_.full(cleaned)
        Xs = self._scale(self.builder_.transform(cleaned))
        D = self.kmeans_.transform(Xs)                       # distance of every row to every centre
        labels = D.argmin(axis=1)
        major = self.major_clusters_
        to_major = D[:, major]
        nearest = major[to_major.argmin(axis=1)]             # the nearest cluster that counts as normal behaviour
        dist = to_major.min(axis=1)

        mean = self.ref_["mean"].reindex(nearest).to_numpy()
        std = self.ref_["std"].reindex(nearest).to_numpy()
        share = self.cluster_share_.reindex(labels).fillna(0.0).to_numpy()
        behaviour = (dist - mean) / std + self.small_weight * (share < self.small_cluster_cut)
        # Second signal: how far the amount is above (or below) what is typical for its category. K-Means alone
        # can hide a group of large expenses inside a cluster of their own, so this keeps amounts visible.
        amount_dev = np.maximum(np.abs(full["amt_cat_z"].to_numpy()) - self.amount_z_floor, 0.0)
        score = behaviour + self.amount_weight * amount_dev

        n = len(cleaned)
        order = np.argsort(-score, kind="stable")
        n_reason = max(1, int(np.ceil(self.MAX_FLAG_RATE * n)))
        texts = explain.reasons(cleaned, full, share, self.small_cluster_cut, order[:n_reason])
        reason = np.array([texts.get(int(pos), "") for pos in range(n)], dtype=object)

        percentile = pd.Series(score).rank(pct=True).to_numpy() * 100
        table = cleaned.assign(cluster=labels, distance=dist, anomaly_score=score,
                               risk_percentile=percentile, reason=reason).iloc[order]
        table.insert(0, "rank", np.arange(1, n + 1))
        return self.flag(table, flag_rate)

    def flag(self, table: pd.DataFrame, flag_rate: float | None = None, top_n: int | None = None) -> pd.DataFrame:
        """Mark the most unusual transactions. Change the rate without re-scoring (this is the sensitivity slider)."""
        rate = self.flag_rate if flag_rate is None else flag_rate
        if not 0 < rate <= self.MAX_FLAG_RATE:
            raise EngineError(f"flag_rate must be between 0 and {self.MAX_FLAG_RATE}.")
        n_flag = int(top_n) if top_n is not None else max(1, round(rate * len(table)))
        out = table.copy()
        out["flagged"] = out["rank"] <= n_flag
        return out

    def fit_score(self, raw: pd.DataFrame, overrides: dict | None = None, flag_rate: float | None = None) -> pd.DataFrame:
        """Raw upload in, ranked table out: detect columns, clean, learn normal, score."""
        cleaned, _, _ = self.prepare(raw, overrides)
        self.fit(cleaned)
        return self.score(cleaned, flag_rate)

    # ------------------------------------------------------------------ reporting and storage
    def summary(self) -> dict:
        self._check_fitted()
        return {
            "transactions_used": self.n_fit_,
            "clusters": self.k_,
            "silhouette": round(self.silhouette_, 3),
            "features": list(self.builder_.columns_),
            "features_dropped": self.builder_.dropped_,
            "columns": self.columns_.used() if hasattr(self, "columns_") else {},
            "cleaning": self.cleaning_report_.as_dict() if hasattr(self, "cleaning_report_") else {},
            "cluster_profile": self.profile_.reset_index().to_dict(orient="records"),
        }

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> AnomalyEngine:
        engine = joblib.load(path)
        if not isinstance(engine, cls):
            raise EngineError(f"{path} does not contain an AnomalyEngine.")
        return engine

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise EngineError("The engine has not learned anything yet. Call fit() or fit_score() first.")