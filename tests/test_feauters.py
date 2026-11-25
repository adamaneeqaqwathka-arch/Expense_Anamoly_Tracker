import numpy as np
import pandas as pd
import pytest

from anomaly_auditor.cleaning import clean
from anomaly_auditor.features import FeatureBuilder
from anomaly_auditor.schema import SchemaError, resolve_columns


@pytest.fixture
def cleaned(raw):
    return clean(raw, resolve_columns(raw))[0]


def test_features_have_no_missing_values(cleaned):
    X = FeatureBuilder().fit_transform(cleaned)
    assert not X.isna().any().any()
    assert len(X) == len(cleaned)
    assert "amt_cat_z" in X.columns


def test_amount_is_scored_relative_to_its_category(cleaned):
    X = FeatureBuilder().fit_transform(cleaned)
    per_category_mean = X["amt_cat_z"].groupby(cleaned["category"]).mean()
    assert np.allclose(per_category_mean, 0, atol=1e-6)


def test_same_columns_for_new_data_with_unseen_values(cleaned):
    builder = FeatureBuilder().fit(cleaned)
    new = cleaned.iloc[:50].copy()
    new["category"] = "never_seen"
    new["merchant"] = "brand_new_merchant"
    out = builder.transform(new)
    assert list(out.columns) == builder.columns_
    assert not out.isna().any().any()


def test_works_without_optional_columns(raw):
    minimal = raw[["Amount", "Transaction Date"]]
    cleaned = clean(minimal, resolve_columns(minimal))[0]
    X = FeatureBuilder().fit_transform(cleaned)
    assert "merchant_rarity" not in X.columns and "log_gap" not in X.columns
    assert "amt_cat_z" in X.columns


def test_constant_data_is_rejected(raw):
    flat = raw[["Amount", "Transaction Date"]].assign(Amount=10.0, **{"Transaction Date": "2025-03-03 12:00:00"})
    cleaned = clean(flat, resolve_columns(flat))[0]
    with pytest.raises(SchemaError):
        FeatureBuilder().fit(cleaned)


def test_deterministic(cleaned):
    a = FeatureBuilder().fit_transform(cleaned)
    b = FeatureBuilder().fit_transform(cleaned)
    pd.testing.assert_frame_equal(a, b)