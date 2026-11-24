import numpy as np
import pytest

from anomaly_auditor.engine import AnomalyEngine, EngineError
from conftest import make_transactions


@pytest.fixture
def scored(raw):
    return AnomalyEngine().fit_score(raw)


def test_output_is_ranked_most_unusual_first(scored, raw):
    assert len(scored) == len(raw)
    assert scored["anomaly_score"].is_monotonic_decreasing
    assert scored["rank"].tolist() == list(range(1, len(scored) + 1))
    for col in ("cluster", "distance", "risk_percentile", "flagged", "reason"):
        assert col in scored.columns


@pytest.mark.parametrize("seed, n_anomalies", [(0, 30), (1, 30), (2, 30), (3, 30), (4, 60), (5, 60), (6, 15)])
def test_injected_anomalies_are_caught(seed, n_anomalies):
    """Includes 60 anomalies (2% of rows), enough for K-Means to give them a cluster of their own."""
    raw, anomaly_rows = make_transactions(seed=seed, n_anomalies=n_anomalies)
    scored = AnomalyEngine().fit_score(raw)
    top5 = set(scored.index[: int(0.05 * len(scored))])
    recall = len(top5 & anomaly_rows) / len(anomaly_rows)
    assert recall >= 0.75, f"only {recall:.0%} of injected anomalies are in the top 5%"


def test_flag_rate_controls_how_many_are_flagged(scored):
    engine = AnomalyEngine()
    assert scored["flagged"].sum() == round(0.05 * len(scored))
    assert engine.flag(scored, 0.02)["flagged"].sum() == round(0.02 * len(scored))
    assert engine.flag(scored, top_n=7)["flagged"].sum() == 7


def test_flag_rate_above_the_limit_is_refused(scored):
    with pytest.raises(EngineError):
        AnomalyEngine().flag(scored, 0.5)
    with pytest.raises(EngineError):
        AnomalyEngine(flag_rate=0)


def test_every_flagged_row_has_a_reason(scored):
    assert (scored.loc[scored["flagged"], "reason"].str.len() > 0).all()


def test_too_few_rows_gives_clear_error(raw):
    with pytest.raises(EngineError, match="at least"):
        AnomalyEngine().fit_score(raw.head(50))


def test_scoring_before_fitting_is_refused(raw):
    with pytest.raises(EngineError, match="not learned"):
        AnomalyEngine().score(raw)


def test_save_and_load_give_identical_scores(raw, tmp_path):
    engine = AnomalyEngine()
    cleaned, _, _ = engine.prepare(raw)
    engine.fit(cleaned)
    before = engine.score(cleaned)
    path = engine.save(tmp_path / "engine.joblib")
    after = AnomalyEngine.load(path).score(cleaned)
    assert np.allclose(before["anomaly_score"].to_numpy(), after["anomaly_score"].to_numpy())
    assert before.index.tolist() == after.index.tolist()


def test_new_data_can_be_scored_with_a_fitted_engine(raw):
    engine = AnomalyEngine()
    cleaned, _, _ = engine.prepare(raw)
    engine.fit(cleaned.iloc[:2000])
    out = engine.score(cleaned.iloc[2000:])
    assert len(out) == len(cleaned) - 2000
    assert not out["anomaly_score"].isna().any()


def test_same_input_gives_same_output(raw):
    a = AnomalyEngine().fit_score(raw)
    b = AnomalyEngine().fit_score(raw)
    assert a.index.tolist() == b.index.tolist()


def test_works_with_only_amount_and_date(raw):
    out = AnomalyEngine().fit_score(raw[["Amount", "Transaction Date"]])
    assert len(out) == len(raw)


def test_fixed_k_is_respected(raw):
    engine = AnomalyEngine(k=4)
    engine.fit_score(raw)
    assert engine.k_ == 4 and engine.tuning_ is None


def test_summary_contents(raw):
    engine = AnomalyEngine()
    engine.fit_score(raw)
    info = engine.summary()
    assert info["clusters"] == engine.k_
    assert -1 <= info["silhouette"] <= 1
    assert len(info["cluster_profile"]) == engine.k_
    assert info["columns"]["amount"] == "Amount"