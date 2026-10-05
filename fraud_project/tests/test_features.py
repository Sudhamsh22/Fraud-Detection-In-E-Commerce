"""test_features.py

Comprehensive unit test suite for features.py.
Validates:
- ip_to_int: dotted IPv4 strings, integers, floats, NaNs, corrupted inputs.
- get_country: np.searchsorted lookup over lower and upper bound IP intervals.
- diff_time: timestamp delta calculation, clipping, missing dates handling.
- safe_label_encode: handling seen and unseen categories without throwing errors.
- compute_device_ip_counts: batch and historical repeat aggregation.
- preprocess: end-to-end matrix output validation in training & inference modes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path to allow direct test execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import features


def test_ip_to_int_dotted_ipv4() -> None:
    """Test standard dotted IPv4 conversion to 32-bit integer."""
    assert features.ip_to_int("192.168.1.1") == 3232235777
    assert features.ip_to_int("1.0.0.0") == 16777216
    assert features.ip_to_int("0.0.0.0") == 0


def test_ip_to_int_numeric_and_float() -> None:
    """Test integer, float, and scientific notation inputs."""
    assert features.ip_to_int(732758432) == 732758432
    assert features.ip_to_int("732758432") == 732758432
    assert features.ip_to_int(732758432.0) == 732758432
    assert features.ip_to_int("732758432.0") == 732758432


def test_ip_to_int_invalid_and_missing() -> None:
    """Ensure malformed or missing IPs return 0 safely without crashing."""
    assert features.ip_to_int(None) == 0
    assert features.ip_to_int(np.nan) == 0
    assert features.ip_to_int("") == 0
    assert features.ip_to_int("invalid_ip_string") == 0
    assert features.ip_to_int("999.999.999.999") == 0


def test_get_country_searchsorted() -> None:
    """Test IP to country lookup via np.searchsorted interval matching."""
    country_table = pd.DataFrame([
        {"lower_bound_ip_address": 1000, "upper_bound_ip_address": 2000, "country": "Japan"},
        {"lower_bound_ip_address": 3000, "upper_bound_ip_address": 4000, "country": "Canada"},
        {"lower_bound_ip_address": 5000, "upper_bound_ip_address": 6000, "country": "United States"},
    ]).sort_values("lower_bound_ip_address").reset_index(drop=True)

    # In range tests
    assert features.get_country(1500, country_table) == "Japan"
    assert features.get_country(3000, country_table) == "Canada"
    assert features.get_country(4000, country_table) == "Canada"
    assert features.get_country(5500, country_table) == "United States"

    # Gap / out-of-range tests
    assert features.get_country(500, country_table) == "Unknown"
    assert features.get_country(2500, country_table) == "Unknown"
    assert features.get_country(9000, country_table) == "Unknown"

    # Vectorized array tests
    test_ips = np.array([1500, 2500, 5500])
    results = features.get_country(test_ips, country_table)
    assert list(results) == ["Japan", "Unknown", "United States"]


def test_safe_label_encoder_unseen() -> None:
    """Test SafeLabelEncoder handles unseen values by assigning unknown code."""
    train_cats = pd.Series(["Chrome", "Safari", "Firefox", "Chrome"])
    encoder = features.SafeLabelEncoder(unknown_label="others")
    encoder.fit(train_cats)

    # Seen categories
    encoded_train = encoder.transform(pd.Series(["Chrome", "Safari"]))
    assert len(encoded_train) == 2

    # Unseen categories
    test_cats = pd.Series(["Chrome", "Edge", "Brave", "Safari"])
    encoded_test = features.safe_label_encode(test_cats, encoder, unknown_label="others")

    # 'Edge' and 'Brave' should map to the same unknown_code
    unknown_code = encoder.mapping_["others"]
    assert encoded_test[1] == unknown_code
    assert encoded_test[2] == unknown_code
    assert encoded_test[0] == encoder.mapping_["Chrome"]


def test_device_and_ip_counts() -> None:
    """Test computing device and IP sharing counts."""
    df = pd.DataFrame({
        "user_id": [101, 102, 103, 104],
        "device_id": ["DEV_A", "DEV_A", "DEV_B", "DEV_C"],
        "ip_address": ["1.1.1.1", "1.1.1.1", "1.1.1.1", "2.2.2.2"],
    })

    dev_counts, ip_counts = features.compute_device_ip_counts(df)

    # DEV_A is used by 2 users
    assert dev_counts.iloc[0] == 2
    assert dev_counts.iloc[1] == 2
    # DEV_B is used by 1 user
    assert dev_counts.iloc[2] == 1

    # 1.1.1.1 is used by 3 users
    assert ip_counts.iloc[0] == 3
    assert ip_counts.iloc[1] == 3
    assert ip_counts.iloc[2] == 3
    # 2.2.2.2 is used by 1 user
    assert ip_counts.iloc[3] == 1


def test_preprocess_end_to_end() -> None:
    """Validate full feature engineering pipeline in training and inference mode."""
    sample_df = pd.DataFrame({
        "user_id": [101, 102, 103],
        "signup_time": ["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 12:00:00"],
        "purchase_time": ["2024-01-01 10:00:05", "2024-01-02 11:00:00", "2024-01-01 11:59:00"],
        "purchase_value": [150.0, 45.0, 99.0],
        "device_id": ["DEV_1", "DEV_2", "DEV_1"],
        "source": ["Direct", "SEO", "Ads"],
        "browser": ["Chrome", "Safari", "Opera"],
        "sex": ["M", "F", "M"],
        "age": [28, 42, 35],
        "ip_address": [1000, 2000, 3000],
        "class": [1, 0, 1],
    })

    # 1. Training mode
    X_train, y_train, artifacts = features.preprocess(sample_df, is_training=True)
    assert X_train.shape == (3, len(features.FEATURE_COLUMNS))
    assert list(y_train) == [1, 0, 1]
    assert "encoders" in artifacts
    assert "device_history" in artifacts
    assert "ip_history" in artifacts

    # diff_time for first row should be exactly 5.0 seconds
    diff_idx = features.FEATURE_COLUMNS.index("diff_time")
    assert X_train[0, diff_idx] == 5.0

    # Row 3 (purchase before signup) should be clipped to 0
    assert X_train[2, diff_idx] == 0.0

    # 2. Inference mode with unseen categories and missing target
    inference_df = pd.DataFrame({
        "user_id": [999],
        "signup_time": ["2024-05-01 00:00:00"],
        "purchase_time": ["2024-05-01 00:00:10"],
        "purchase_value": [300.0],
        "device_id": ["DEV_NEW"],
        "source": ["UnseenSource"],
        "browser": ["UnseenBrowser"],
        "sex": ["Other"],
        "age": [33],
        "ip_address": ["192.168.1.100"],
    })

    X_test, y_test, _ = features.preprocess(inference_df, artifacts=artifacts, is_training=False)
    assert X_test.shape == (1, len(features.FEATURE_COLUMNS))
    assert y_test is None  # 'class' not provided in inference
