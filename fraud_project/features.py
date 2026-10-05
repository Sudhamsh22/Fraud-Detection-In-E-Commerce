"""features.py

Shared feature engineering and preprocessing pipeline for E-Commerce Fraud Detection.
Used identically in BOTH model training (main.py) and production serving (predict.py).

Features engineered:
- diff_time: Time difference between purchase_time and signup_time in seconds.
- ip_to_int: Safe parsing of dotted quad or numeric IP strings/floats to integers.
- get_country: Geolocation IP-range resolution using np.searchsorted.
- num_used_device: Device sharing frequency (unique users per device_id).
- num_ip_repeat: IP address sharing frequency (unique users per ip_address).
- safe_label_encode: Robust categorical encoding supporting unseen categories.
- Behavioral flags: is_instant_purchase, is_device_shared, is_ip_shared.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("features")

# Standard feature order guaranteed across training and inference
FEATURE_COLUMNS: List[str] = [
    "purchase_value",
    "age",
    "diff_time",
    "diff_time_hours",
    "num_used_device",
    "num_ip_repeat",
    "is_instant_purchase",
    "is_device_shared",
    "is_ip_shared",
    "purchase_hour",
    "purchase_dayofweek",
    "purchase_month",
    "signup_hour",
    "signup_dayofweek",
    "source_enc",
    "browser_enc",
    "sex_enc",
    "country_enc",
]


def ip_to_int(ip: Union[str, int, float, None]) -> int:
    """Safely convert an IP address (dotted string, float, or integer) to an integer.

    Handles:
        - Dotted quad IPv4: "192.168.1.1" -> 3232235777
        - Float / scientific notation: 7.327584e+08 or "732758432.0" -> 732758432
        - Standard integer: 732758432 -> 732758432
        - Invalid / missing values: None, NaN, corrupt text -> 0

    Args:
        ip: Raw IP address representation.

    Returns:
        Integer representation of IPv4 address or 0 on failure.
    """
    if ip is None or pd.isna(ip):
        return 0

    # If it's already an integer
    if isinstance(ip, (int, np.integer)):
        return max(0, int(ip))

    # If it's a float
    if isinstance(ip, (float, np.floating)):
        try:
            return max(0, int(ip))
        except (ValueError, OverflowError):
            return 0

    ip_str = str(ip).strip()
    if not ip_str or ip_str.lower() in ("nan", "none", "null"):
        return 0

    # Try dotted IPv4 format
    if "." in ip_str and not ip_str.replace(".", "", 1).isdigit():
        try:
            return int(ipaddress.IPv4Address(ip_str))
        except (ValueError, ipaddress.AddressValueError):
            pass

    # Try numeric string (e.g. "732758432" or "732758432.0")
    try:
        float_val = float(ip_str)
        return max(0, int(float_val))
    except (ValueError, OverflowError):
        return 0


def load_country_lookup(country_path: str) -> pd.DataFrame:
    """Load and sort the IP-to-Country range lookup table.

    Args:
        country_path: Path to IpAddress_to_Country.csv

    Returns:
        Sorted DataFrame ready for searchsorted binary search.
    """
    df = pd.read_csv(country_path)
    required = ["lower_bound_ip_address", "upper_bound_ip_address", "country"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Country lookup file missing required column: {col}")

    df["lower_bound_ip_address"] = df["lower_bound_ip_address"].astype(np.int64)
    df["upper_bound_ip_address"] = df["upper_bound_ip_address"].astype(np.int64)
    df = df.sort_values("lower_bound_ip_address").reset_index(drop=True)
    return df


def get_country(
    ip_ints: Union[int, float, List[int], np.ndarray, pd.Series],
    country_df: pd.DataFrame,
) -> Union[str, np.ndarray]:
    """Resolve country for integer IP addresses using np.searchsorted on range intervals.

    Given a sorted table with [lower_bound_ip_address, upper_bound_ip_address]:
    Binary search locates the index where ip >= lower_bound.
    If ip <= upper_bound at that index, return the matching country, else 'Unknown'.

    Args:
        ip_ints: Single integer or 1D array-like of integer IP addresses.
        country_df: Sorted country lookup DataFrame.

    Returns:
        Country name string (for scalar input) or numpy array of strings (for array input).
    """
    is_scalar = np.isscalar(ip_ints)
    ips = np.atleast_1d(np.asarray(ip_ints, dtype=np.int64))

    if country_df.empty or len(ips) == 0:
        res = np.full(ips.shape, "Unknown", dtype=object)
        return res[0] if is_scalar else res

    lowers = country_df["lower_bound_ip_address"].values
    uppers = country_df["upper_bound_ip_address"].values
    countries = country_df["country"].values

    # Find position where lower_bound <= ip
    # side='right' returns index such that lowers[idx-1] <= ip < lowers[idx]
    indices = np.searchsorted(lowers, ips, side="right") - 1

    valid_mask = (indices >= 0) & (indices < len(lowers))
    in_range_mask = np.zeros_like(valid_mask, dtype=bool)
    in_range_mask[valid_mask] = ips[valid_mask] <= uppers[indices[valid_mask]]

    result = np.full(ips.shape, "Unknown", dtype=object)
    result[in_range_mask] = countries[indices[in_range_mask]]

    return result[0] if is_scalar else result


class SafeLabelEncoder:
    """Label encoder that handles unseen categories gracefully during inference.

    Maps all unseen categories to a dedicated 'others' bucket rather than raising
    ValueError.
    """

    def __init__(self, unknown_label: str = "others", top_n: Optional[int] = None) -> None:
        self.unknown_label: str = unknown_label
        self.top_n: Optional[int] = top_n
        self.classes_: List[str] = []
        self.mapping_: Dict[str, int] = {}
        self.unknown_code_: int = 0

    def fit(self, values: pd.Series | List[Any]) -> "SafeLabelEncoder":
        """Fit encoder on training series, optionally keeping only top N categories."""
        series = pd.Series(values).astype(str).fillna(self.unknown_label)

        if self.top_n is not None and self.top_n > 0:
            top_cats = series.value_counts().nlargest(self.top_n).index.tolist()
            if self.unknown_label not in top_cats:
                top_cats.append(self.unknown_label)
            unique_classes = sorted(list(set(top_cats)))
        else:
            unique_classes = sorted(list(set(series.unique()) | {self.unknown_label}))

        self.classes_ = unique_classes
        self.mapping_ = {cls_name: idx for idx, cls_name in enumerate(self.classes_)}
        self.unknown_code_ = self.mapping_[self.unknown_label]
        return self

    def transform(self, values: pd.Series | List[Any]) -> np.ndarray:
        """Transform values to integer codes, mapping unseen to unknown_label."""
        series = pd.Series(values).astype(str).fillna(self.unknown_label)
        return series.map(lambda x: self.mapping_.get(x, self.unknown_code_)).values

    def fit_transform(self, values: pd.Series | List[Any]) -> np.ndarray:
        """Fit encoder and return transformed values."""
        return self.fit(values).transform(values)


def safe_label_encode(
    values: pd.Series | List[Any],
    encoder: SafeLabelEncoder,
    unknown_label: str = "others",
) -> np.ndarray:
    """Safe label encoding wrapper matching specification interface.

    Args:
        values: Series or list of categorical values.
        encoder: SafeLabelEncoder instance.
        unknown_label: Token representing unknown/unseen category.

    Returns:
        Numpy array of integer encoded labels.
    """
    if encoder.unknown_label != unknown_label and unknown_label in encoder.mapping_:
        # Map with specified unknown label
        return pd.Series(values).astype(str).map(
            lambda x: encoder.mapping_.get(x, encoder.mapping_.get(unknown_label, 0))
        ).values
    return encoder.transform(values)


def compute_device_ip_counts(
    df: pd.DataFrame,
    device_history: Optional[Dict[str, int]] = None,
    ip_history: Optional[Dict[str, int]] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Compute num_used_device and num_ip_repeat for transactions.

    Calculates unique users per device/IP within current batch, and combines
    with historical frequency tables to detect repeat offenders across sessions.

    Args:
        df: Input DataFrame with 'device_id', 'ip_address', 'user_id'.
        device_history: Optional historical mapping from device_id -> user count.
        ip_history: Optional historical mapping from ip_address (int/str) -> user count.

    Returns:
        Tuple of (num_used_device, num_ip_repeat) as pandas Series.
    """
    # Batch internal counts
    if "user_id" in df.columns:
        batch_dev = df.groupby("device_id")["user_id"].transform("nunique")
        batch_ip = df.groupby("ip_address")["user_id"].transform("nunique")
    else:
        # If user_id missing, count occurrences
        batch_dev = df.groupby("device_id")["device_id"].transform("count")
        batch_ip = df.groupby("ip_address")["ip_address"].transform("count")

    # Combine with historical frequency tables if provided
    if device_history:
        hist_dev = df["device_id"].map(device_history).fillna(1)
        num_used_device = np.maximum(batch_dev.values, hist_dev.values)
    else:
        num_used_device = batch_dev.values

    if ip_history:
        hist_ip = df["ip_address"].map(ip_history).fillna(1)
        num_ip_repeat = np.maximum(batch_ip.values, hist_ip.values)
    else:
        num_ip_repeat = batch_ip.values

    return pd.Series(num_used_device, index=df.index), pd.Series(num_ip_repeat, index=df.index)


def preprocess(
    df: pd.DataFrame,
    artifacts: Optional[Dict[str, Any]] = None,
    country_df: Optional[pd.DataFrame] = None,
    is_training: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Master preprocessing pipeline shared by BOTH training and inference.

    Features generated:
        - diff_time: purchase_time - signup_time (in seconds)
        - Geolocation: ip_to_int -> country via np.searchsorted
        - num_used_device, num_ip_repeat (device & IP sharing counts)
        - Behavioral indicator flags: is_instant_purchase, is_device_shared, is_ip_shared
        - Time decomposition: hour, day of week, month
        - Categorical encoding with SafeLabelEncoder

    Args:
        df: Raw DataFrame containing e-commerce transaction data.
        artifacts: Pre-fitted encoders, frequency maps, etc. Required when is_training=False.
        country_df: Loaded IpAddress_to_Country DataFrame.
        is_training: If True, fits new encoders and frequency tables.

    Returns:
        Tuple of:
            - X: Model-ready feature matrix (np.ndarray of shape [N, num_features])
            - y: Target array (np.ndarray) if 'class' column exists, else None
            - artifacts: Dictionary containing encoders and mappings for future serving
    """
    df_clean = df.copy()

    # 1. Safe datetime parsing & diff_time calculation
    signup_dt = pd.to_datetime(df_clean["signup_time"], errors="coerce")
    purchase_dt = pd.to_datetime(df_clean["purchase_time"], errors="coerce")

    # Handle missing or invalid timestamps
    median_signup = signup_dt.dropna().median() if not signup_dt.dropna().empty else pd.Timestamp("2024-01-01")
    signup_dt = signup_dt.fillna(median_signup)
    purchase_dt = purchase_dt.fillna(signup_dt)

    diff_seconds = (purchase_dt - signup_dt).dt.total_seconds().clip(lower=0)
    df_clean["diff_time"] = diff_seconds.fillna(diff_seconds.median() if not diff_seconds.empty else 0)
    df_clean["diff_time_hours"] = df_clean["diff_time"] / 3600.0

    # Temporal decompositions
    df_clean["purchase_hour"] = purchase_dt.dt.hour.fillna(12).astype(int)
    df_clean["purchase_dayofweek"] = purchase_dt.dt.dayofweek.fillna(0).astype(int)
    df_clean["purchase_month"] = purchase_dt.dt.month.fillna(1).astype(int)
    df_clean["signup_hour"] = signup_dt.dt.hour.fillna(12).astype(int)
    df_clean["signup_dayofweek"] = signup_dt.dt.dayofweek.fillna(0).astype(int)

    # 2. IP to Int & Country Lookup via searchsorted
    raw_ips = df_clean["ip_address"]
    ip_ints = np.array([ip_to_int(ip) for ip in raw_ips], dtype=np.int64)
    df_clean["ip_int"] = ip_ints

    if country_df is not None:
        countries = get_country(ip_ints, country_df)
        df_clean["country"] = countries
    elif artifacts and "country_df" in artifacts:
        countries = get_country(ip_ints, artifacts["country_df"])
        df_clean["country"] = countries
    else:
        df_clean["country"] = "Unknown"

    # 3. Device & IP repetition counts
    dev_history = artifacts.get("device_history") if artifacts else None
    ip_history = artifacts.get("ip_history") if artifacts else None

    num_dev, num_ip = compute_device_ip_counts(
        df_clean, device_history=dev_history, ip_history=ip_history
    )
    df_clean["num_used_device"] = num_dev
    df_clean["num_ip_repeat"] = num_ip

    # 4. Behavioral indicator flags
    df_clean["is_instant_purchase"] = (df_clean["diff_time"] <= 120).astype(int)
    df_clean["is_device_shared"] = (df_clean["num_used_device"] > 1).astype(int)
    df_clean["is_ip_shared"] = (df_clean["num_ip_repeat"] > 1).astype(int)

    # Clean numeric features
    df_clean["purchase_value"] = pd.to_numeric(df_clean.get("purchase_value", 50.0), errors="coerce").fillna(50.0)
    df_clean["age"] = pd.to_numeric(df_clean.get("age", 30), errors="coerce").fillna(30.0).clip(18, 100)

    # 5. Categorical Encoding
    cat_columns = ["source", "browser", "sex", "country"]
    encoders: Dict[str, SafeLabelEncoder] = {}

    if is_training:
        artifacts = artifacts or {}

        # Build device and IP historical frequency dictionaries
        if "device_id" in df_clean.columns and "user_id" in df_clean.columns:
            artifacts["device_history"] = df_clean.groupby("device_id")["user_id"].nunique().to_dict()
        if "ip_address" in df_clean.columns and "user_id" in df_clean.columns:
            artifacts["ip_history"] = df_clean.groupby("ip_address")["user_id"].nunique().to_dict()

        for cat in cat_columns:
            vals = df_clean.get(cat, pd.Series(["others"] * len(df_clean)))
            top_n = 20 if cat == "country" else None
            enc = SafeLabelEncoder(unknown_label="others", top_n=top_n).fit(vals)
            encoders[cat] = enc
            df_clean[f"{cat}_enc"] = enc.transform(vals)

        artifacts["encoders"] = encoders
        artifacts["feature_columns"] = FEATURE_COLUMNS
        if country_df is not None:
            artifacts["country_df"] = country_df
    else:
        if not artifacts or "encoders" not in artifacts:
            raise ValueError("Pre-fitted 'encoders' must be supplied in artifacts when is_training=False.")
        encoders = artifacts["encoders"]
        for cat in cat_columns:
            vals = df_clean.get(cat, pd.Series(["others"] * len(df_clean)))
            enc = encoders.get(cat, SafeLabelEncoder().fit(vals))
            df_clean[f"{cat}_enc"] = safe_label_encode(vals, enc, unknown_label="others")

    # 6. Assemble output feature matrix
    X_df = df_clean[FEATURE_COLUMNS].copy()
    X = X_df.values.astype(np.float32)

    # Target extraction if available
    y = None
    if "class" in df_clean.columns:
        y = pd.to_numeric(df_clean["class"], errors="coerce").fillna(0).astype(int).values

    return X, y, artifacts
