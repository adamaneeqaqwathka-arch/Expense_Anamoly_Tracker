import pandas as pd
import pytest

from anomaly_auditor.schema import SchemaError, detect_columns, resolve_columns


def test_detects_common_column_names(raw):
    cmap = resolve_columns(raw)
    assert (cmap.amount, cmap.date, cmap.category) == ("Amount", "Transaction Date", "Category")
    assert (cmap.merchant, cmap.card, cmap.txn_id) == ("Merchant", "Card ID", "Txn ID")


def test_detects_kaggle_style_names():
    df = pd.DataFrame({"trans_date_trans_time": ["2024-01-01 10:00:00"], "cc_num": [1], "merchant": ["a"],
                       "category": ["food"], "amt": [5.0], "trans_num": ["x"], "is_fraud": [0]})
    found = detect_columns(df)
    assert found["amount"] == "amt" and found["date"] == "trans_date_trans_time"
    assert found["card"] == "cc_num" and found["label"] == "is_fraud" and found["txn_id"] == "trans_num"


def test_a_column_is_never_used_twice():
    df = pd.DataFrame({"date": ["2024-01-01"], "amount": [1.0], "type": ["x"], "category": ["y"]})
    found = detect_columns(df)
    assert found["category"] != found["amount"]
    assert len({v for v in found.values() if v}) == len([v for v in found.values() if v])


def test_override_wins_over_detection(raw):
    renamed = raw.rename(columns={"Amount": "Total Spend"})
    assert resolve_columns(renamed, overrides={"amount": "Total Spend"}).amount == "Total Spend"


def test_missing_required_column_gives_clear_error(raw):
    with pytest.raises(SchemaError, match="amount"):
        resolve_columns(raw.drop(columns=["Amount"]))


def test_unreadable_dates_are_rejected(raw):
    bad = raw.assign(**{"Transaction Date": "not a date"})
    with pytest.raises(SchemaError, match="date"):
        resolve_columns(bad)


def test_text_amounts_that_are_not_numbers_are_rejected(raw):
    bad = raw.assign(Amount="abc")
    with pytest.raises(SchemaError, match="amount"):
        resolve_columns(bad)


def test_unknown_role_in_override(raw):
    with pytest.raises(SchemaError, match="Unknown role"):
        resolve_columns(raw, overrides={"colour": "Amount"})


def test_empty_file():
    with pytest.raises(SchemaError):
        resolve_columns(pd.DataFrame())