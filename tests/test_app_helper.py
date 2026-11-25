import plotly.graph_objects as go
from app import helpers

DETECTED = {"amount": "Amount", "date": "Date", "category": "Cat", "merchant": None}


def test_only_changed_roles_become_overrides():
    chosen = {"amount": "Amount", "date": "Date", "category": "Cat", "merchant": helpers.NONE_CHOICE}
    assert helpers.overrides_from_choices(DETECTED, chosen) is None
    chosen["amount"] = "Total"
    chosen["merchant"] = "Vendor"
    assert helpers.overrides_from_choices(DETECTED, chosen) == {"amount": "Total", "merchant": "Vendor"}


def test_clearing_an_optional_column_is_an_override_to_none():
    chosen = {"amount": "Amount", "date": "Date", "category": helpers.NONE_CHOICE, "merchant": helpers.NONE_CHOICE}
    assert helpers.overrides_from_choices(DETECTED, chosen) == {"category": None}


ROWS = [
    {"rank": 1, "source_row": 5, "date": "2025-01-02T10:00:00.000", "amount": 900.0, "category": "fuel",
     "merchant": "m", "card": "1", "anomaly_score": 12.346, "risk_percentile": 100.0, "flagged": True,
     "reason": "amount is unusually high"},
    {"rank": 2, "source_row": 9, "date": "2025-01-03T11:30:00.000", "amount": 80.0, "category": "fuel",
     "merchant": "m", "card": "2", "anomaly_score": 3.0, "risk_percentile": 99.9, "flagged": True, "reason": "x"},
    {"rank": 3, "source_row": 1, "date": "2025-01-04T09:00:00.000", "amount": 20.0, "category": "fuel",
     "merchant": "m", "card": "3", "anomaly_score": 1.0, "risk_percentile": 90.0, "flagged": False, "reason": ""},
]


def test_flagged_table_keeps_only_flagged_rows_with_friendly_names():
    table = helpers.flagged_table(ROWS)
    assert len(table) == 2
    assert list(table.columns)[:3] == ["Rank", "Row in file", "Date"]
    assert table.loc[0, "Date"] == "2025-01-02 10:00" and table.loc[0, "Score"] == 12.35
    assert helpers.flagged_table([]).empty


def test_flag_threshold_is_the_lowest_flagged_score():
    assert helpers.flag_threshold(ROWS) == 3.0
    assert helpers.flag_threshold([{"anomaly_score": 1, "flagged": False}]) is None


CHARTS = {"score_histogram": {"bin_edges": [0, 1, 2, 3], "counts": [5, 3, 1], "clipped_above": 3},
          "flagged_by_category": [{"category": "a", "flagged": 5, "total": 50},
                                  {"category": "b", "flagged": 2, "total": 40}],
          "flagged_by_month": [{"month": "2025-01", "flagged": 4, "total": 90},
                               {"month": "2025-02", "flagged": 3, "total": 80}]}


def test_figures_carry_the_data():
    hist = helpers.score_histogram(CHARTS, threshold=2.0)
    assert isinstance(hist, go.Figure) and list(hist.data[0].y) == [5, 3, 1]
    assert list(hist.data[0].x) == [0.5, 1.5, 2.5]
    assert list(helpers.flagged_by_category(CHARTS).data[0].x) == [2, 5]          # sorted so the biggest is on top
    assert list(helpers.flagged_over_time(CHARTS).data[0].y) == [4, 3]


def test_cluster_table_and_cleaning_lines():
    summary = {"cluster_profile": [{"cluster": 0, "transactions": 90, "share": 0.9, "median_amount": 40.0,
                                    "top_category": "fuel", "night_share": 0.02, "rare_pattern": False}],
               "cleaning": {"rows_in": 100, "rows_out": 96, "removed": {"duplicates": 4, "future dates": 0},
                            "notes": ["No category column; all transactions are treated as one category."]}}
    table = helpers.cluster_table(summary)
    assert table.loc[0, "Share"] == "90.0%" and table.loc[0, "Top category"] == "fuel"
    lines = helpers.cleaning_lines(summary)
    assert lines[0] == "100 rows in, 96 used." and any("duplicates" in line for line in lines)
    assert not any("future dates" in line for line in lines)                       # zero counts are not shown