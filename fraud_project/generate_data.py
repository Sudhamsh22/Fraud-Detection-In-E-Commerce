"""generate_data.py

Synthetic dataset generator for E-Commerce Fraud Detection.
Produces realistic offline datasets matching Kaggle "Fraud E-Commerce":
1. data/Fraud_Data.csv (~50k rows, ~3% fraud rate)
   - Features: user_id, signup_time, purchase_time, purchase_value,
               device_id, source, browser, sex, age, ip_address, class
   - Fraud patterns: near-instantaneous checkout (signup-to-purchase diff_time < 2 min),
                     device sharing (bot farms / account takeovers),
                     IP repetition (proxies / VPN nodes).
2. data/IpAddress_to_Country.csv (IP ranges mapped to countries for searchsorted lookup).
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_data")


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible synthetic generation."""
    np.random.seed(seed)
    random.seed(seed)


def generate_ip_country_ranges(output_path: str) -> pd.DataFrame:
    """Generate realistic IP-to-Country range lookup table.

    Table columns:
        lower_bound_ip_address: uint32 integer representation of start IP
        upper_bound_ip_address: uint32 integer representation of end IP
        country: Country name

    Args:
        output_path: File path to save IpAddress_to_Country.csv

    Returns:
        pd.DataFrame containing sorted IP ranges.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Generating synthetic IpAddress_to_Country table...")

    countries = [
        ("United States", 0.35),
        ("China", 0.15),
        ("United Kingdom", 0.08),
        ("Germany", 0.06),
        ("Canada", 0.05),
        ("Japan", 0.05),
        ("Brazil", 0.04),
        ("India", 0.04),
        ("France", 0.03),
        ("Australia", 0.03),
        ("Netherlands", 0.02),
        ("Russia", 0.02),
        ("Nigeria", 0.02),
        ("Vietnam", 0.02),
        ("Mexico", 0.02),
        ("South Korea", 0.02),
    ]

    country_names = [c[0] for c in countries]
    country_weights = [c[1] for c in countries]

    ranges = []
    current_ip = 16777216  # 1.0.0.0

    # Create 5,000 non-overlapping contiguous blocks
    num_blocks = 5000
    for _ in range(num_blocks):
        block_size = int(np.random.choice([65536, 131072, 262144, 524288], p=[0.4, 0.3, 0.2, 0.1]))
        country = str(np.random.choice(country_names, p=country_weights))
        lower_ip = current_ip
        upper_ip = current_ip + block_size - 1

        # Avoid exceeding 32-bit IPv4 boundary (4294967295)
        if upper_ip >= 4200000000:
            break

        ranges.append({
            "lower_bound_ip_address": lower_ip,
            "upper_bound_ip_address": upper_ip,
            "country": country,
        })
        # Gap between blocks (unallocated space)
        gap = int(np.random.choice([0, 256, 1024, 4096], p=[0.7, 0.15, 0.1, 0.05]))
        current_ip = upper_ip + 1 + gap

    df_country = pd.DataFrame(ranges).sort_values("lower_bound_ip_address").reset_index(drop=True)
    df_country.to_csv(output_path, index=False)
    logger.info("Saved %d IP range records to %s", len(df_country), output_path)
    return df_country


def generate_fraud_dataset(
    output_path: str,
    country_df: pd.DataFrame,
    num_records: int = 50000,
    fraud_rate: float = 0.035,
) -> pd.DataFrame:
    """Generate synthetic e-commerce transaction dataset with realistic fraud patterns.

    Args:
        output_path: Destination path for Fraud_Data.csv
        country_df: Pre-generated IP to country DataFrame
        num_records: Total count of rows to generate (default 50,000)
        fraud_rate: Fraction of fraudulent transactions (default ~3.5%)

    Returns:
        Generated pandas DataFrame
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Generating %d transactions with %.2f%% fraud rate...", num_records, fraud_rate * 100)

    num_fraud = int(num_records * fraud_rate)
    num_legit = num_records - num_fraud

    start_date = datetime(2024, 1, 1, 0, 0, 0)
    end_date = datetime(2024, 11, 30, 23, 59, 59)
    total_seconds = int((end_date - start_date).total_seconds())

    # --- Legitimate Transactions ---
    legit_users = list(range(100001, 100001 + num_legit))
    legit_signup_offsets = np.random.randint(0, total_seconds - 86400 * 30, size=num_legit)
    legit_signups = [start_date + timedelta(seconds=int(s)) for s in legit_signup_offsets]

    # Diff time for legit: log-normal distribution from 2 hours to 60 days
    # Log-normal with mean ~ 10 days (864,000 seconds)
    legit_diff_seconds = np.random.lognormal(mean=13.0, sigma=1.2, size=num_legit)
    # Clip between 3600 seconds (1 hour) and 90 days
    legit_diff_seconds = np.clip(legit_diff_seconds, 3600, 86400 * 90)
    legit_purchases = [
        s + timedelta(seconds=int(d)) for s, d in zip(legit_signups, legit_diff_seconds)
    ]

    legit_values = np.round(np.random.gamma(shape=3.5, scale=12.0, size=num_legit) + 9, 2)
    legit_values = np.clip(legit_values, 9.0, 350.0)

    # Legitimate devices: mostly 1 transaction per device, few natural returns
    legit_devices = [f"DEV_LEGIT_{i:06d}" for i in range(1, int(num_legit * 0.96) + 1)]
    # pad with random repeats for legitimate returning users
    legit_device_assignments = list(np.random.choice(legit_devices, size=num_legit))

    # Legitimate IPs: pick uniformly from valid IP ranges in country table
    sample_indices = np.random.choice(len(country_df), size=num_legit)
    low_ips = country_df["lower_bound_ip_address"].values[sample_indices]
    high_ips = country_df["upper_bound_ip_address"].values[sample_indices]
    legit_ips = np.random.randint(low_ips, high_ips + 1, dtype=np.int64)

    sources = ["SEO", "Ads", "Direct"]
    source_p = [0.38, 0.42, 0.20]
    legit_source = np.random.choice(sources, size=num_legit, p=source_p)

    browsers = ["Chrome", "Safari", "Firefox", "IE", "Opera"]
    browser_p = [0.50, 0.26, 0.14, 0.07, 0.03]
    legit_browser = np.random.choice(browsers, size=num_legit, p=browser_p)

    legit_sex = np.random.choice(["M", "F"], size=num_legit, p=[0.53, 0.47])
    legit_age = np.clip(np.random.normal(loc=34, scale=10, size=num_legit).astype(int), 18, 75)

    df_legit = pd.DataFrame({
        "user_id": legit_users,
        "signup_time": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in legit_signups],
        "purchase_time": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in legit_purchases],
        "purchase_value": legit_values,
        "device_id": legit_device_assignments,
        "source": legit_source,
        "browser": legit_browser,
        "sex": legit_sex,
        "age": legit_age,
        "ip_address": legit_ips,
        "class": 0,
    })

    # --- Fraudulent Transactions ---
    fraud_users = list(range(200001, 200001 + num_fraud))
    fraud_signup_offsets = np.random.randint(0, total_seconds - 3600, size=num_fraud)
    fraud_signups = [start_date + timedelta(seconds=int(s)) for s in fraud_signup_offsets]

    # Fraud pattern 1: Extremely short time between signup and purchase (bot scripts / carders)
    # 75% between 1 second and 60 seconds; 25% between 1 minute and 10 minutes
    instant_diffs = np.random.randint(1, 60, size=int(num_fraud * 0.75))
    rapid_diffs = np.random.randint(60, 600, size=num_fraud - len(instant_diffs))
    fraud_diff_seconds = np.concatenate([instant_diffs, rapid_diffs])
    np.random.shuffle(fraud_diff_seconds)

    fraud_purchases = [
        s + timedelta(seconds=int(d)) for s, d in zip(fraud_signups, fraud_diff_seconds)
    ]

    # Fraud pattern 2: Higher purchase value or round values (expensive electronics, gift cards)
    fraud_values = np.round(
        np.random.choice(
            [49.0, 99.0, 149.0, 199.0, 249.0, 299.0, 399.0],
            size=num_fraud,
            p=[0.1, 0.2, 0.25, 0.2, 0.12, 0.08, 0.05],
        ) + np.random.uniform(-0.5, 0.5, size=num_fraud),
        2,
    )

    # Fraud pattern 3: Massive device reuse (Device farms / bot networks)
    # 80 fraud devices shared among ~1,750 fraud instances
    fraud_device_pool = [f"DEV_FARM_{i:04d}" for i in range(1, 81)]
    fraud_devices = np.random.choice(fraud_device_pool, size=num_fraud)

    # Fraud pattern 4: Shared IP addresses (Tor exits, VPN subnets, proxy servers)
    # 60 IP addresses shared among multiple fraud cases
    sample_fraud_ranges = np.random.choice(len(country_df), size=60)
    fraud_ip_pool = country_df["lower_bound_ip_address"].values[sample_fraud_ranges] + 15
    fraud_ips = np.random.choice(fraud_ip_pool, size=num_fraud)

    # Fraud source: Direct visits are much higher in automated attacks
    fraud_source = np.random.choice(sources, size=num_fraud, p=[0.25, 0.30, 0.45])
    fraud_browser = np.random.choice(browsers, size=num_fraud, p=[0.58, 0.18, 0.15, 0.07, 0.02])
    fraud_sex = np.random.choice(["M", "F"], size=num_fraud, p=[0.55, 0.45])
    fraud_age = np.clip(np.random.normal(loc=31, scale=8, size=num_fraud).astype(int), 18, 65)

    df_fraud = pd.DataFrame({
        "user_id": fraud_users,
        "signup_time": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in fraud_signups],
        "purchase_time": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in fraud_purchases],
        "purchase_value": fraud_values,
        "device_id": fraud_devices,
        "source": fraud_source,
        "browser": fraud_browser,
        "sex": fraud_sex,
        "age": fraud_age,
        "ip_address": fraud_ips,
        "class": 1,
    })

    # Combine and shuffle
    df_combined = pd.concat([df_legit, df_fraud], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_combined.to_csv(output_path, index=False)
    logger.info(
        "Saved %d records (%d legit, %d fraud; %.2f%% fraud) to %s",
        len(df_combined),
        num_legit,
        num_fraud,
        (num_fraud / len(df_combined)) * 100,
        output_path,
    )
    return df_combined


def main() -> None:
    """Entry point for standalone data generation."""
    set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    country_file = os.path.join(data_dir, "IpAddress_to_Country.csv")
    fraud_file = os.path.join(data_dir, "Fraud_Data.csv")

    country_df = generate_ip_country_ranges(country_file)
    generate_fraud_dataset(fraud_file, country_df, num_records=50000, fraud_rate=0.035)
    logger.info("Dataset generation complete. Files are ready under %s", data_dir)


if __name__ == "__main__":
    main()
