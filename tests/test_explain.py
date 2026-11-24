import pytest

from anomaly_auditor.engine import AnomalyEngine


@pytest.fixture
def scored(raw):
    return AnomalyEngine().fit_score(raw)


def test_huge_amounts_are_explained_as_amounts(scored, anomaly_rows):
    caught = scored[scored["flagged"] & scored.index.isin(anomaly_rows)]
    assert len(caught) > 0
    share = caught["reason"].str.contains("amount").mean()
    assert share >= 0.8


def test_reason_mentions_the_category(scored):
    row = scored[scored["reason"].str.contains("unusually high")].iloc[0]
    assert row["category"] in row["reason"]


def test_at_most_two_reasons_per_row(scored):
    assert (scored["reason"].str.count(";") <= 1).all()


def test_rows_outside_the_top_ten_percent_have_no_reason(scored):
    tail = scored.iloc[int(0.2 * len(scored)):]
    assert (tail["reason"] == "").all()