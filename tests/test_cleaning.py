import numpy as np
import pandas as pd

from anomaly_auditor.cleaning import clean
from anomaly_auditor.schema import resolve_columns


def run(df):
    return clean(df, resolve_columns(df))


def test_canonical_names_and_original_index_kept(raw):
    out, _ = run(raw)
    assert {"amount", "date", "category", "merchant", "card", "txn_id"} <= set(out.columns)
    assert out.index.name == "source_row"
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_input_is_not_modified(raw):
    before = raw.copy()
    run(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_duplicates_by_transaction_id_are_removed(raw):
    doubled = pd.concat([raw, raw.iloc[:10]], ignore_index=True)
    out, report = run(doubled)
    assert report.removed["duplicates"] == 10
    assert len(out) == len(raw)


def test_non_positive_amounts_and_bad_dates_are_removed(raw):
    bad = raw.copy()
    bad.loc[0, "Amount"] = -5
    bad.loc[1, "Amount"] = 0
    bad.loc[2, "Transaction Date"] = "garbage"
    bad.loc[3, "Transaction Date"] = "2099-01-01 00:00:00"
    out, report = run(bad)
    assert report.removed["amount <= 0"] == 2
    assert report.removed["missing amount or date"] == 1
    assert report.removed["future dates"] == 1
    assert report.rows_in - report.rows_out == 4
    assert len(out) == len(raw) - 4


def test_currency_text_is_converted(raw):
    text = raw.assign(Amount=raw["Amount"].map(lambda v: f"${v:,.2f}"))
    out, report = run(text)
    assert np.allclose(out["amount"].to_numpy(), raw["Amount"].to_numpy())
    assert any("text" in note for note in report.notes)


def test_missing_category_becomes_unknown_and_missing_column_is_created(raw):
    gaps = raw.copy()
    gaps.loc[:4, "Category"] = None
    out, _ = run(gaps)
    assert (out["category"].iloc[:5] == "Unknown").all()
    out2, report = run(raw.drop(columns=["Category"]))
    assert (out2["category"] == "Unknown").all()
    assert any("category" in note.lower() for note in report.notes)


def test_whitespace_is_trimmed(raw):
    padded = raw.assign(Category="  " + raw["Category"] + " ")
    out, _ = run(padded)
    assert set(out["category"]) == set(raw["Category"])