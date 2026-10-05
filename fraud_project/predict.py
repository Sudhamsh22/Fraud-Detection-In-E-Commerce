"""predict.py

Production Flask Web Application for E-Commerce Fraud Detection:
- Dark Modern Dashboard with crimson/red cyber accents and responsive glassmorphism.
- CSV Upload, live dataset preview, and "Detect Fraud" batch inference.
- Shared feature preprocessing via features.preprocess().
- Probabilistic prediction with calibrated threshold.
- Row-level Explainability Engine:
    * Short signup-to-purchase time (< 2 min)
    * Device sharing (bot-farm signatures)
    * IP address repetition (proxy/VPN detection)
    * High-risk geolocation / transaction anomalies
- Download Enriched Audit Report (CSV) with risk tiers & reasons.
- Dedicated Model Metrics & Visualizations page (/metrics).
- Robust file validation and defensive error handling (never crashes on corrupt input).
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

import features

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict_service")

# Initialize Flask application with explicit relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["SECRET_KEY"] = "fraud_guard_ecommerce_secret_key_2026"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload limit

# Global model cache
MODEL_REGISTRY: Dict[str, Any] = {
    "model": None,
    "dnn_model": None,
    "scaler": None,
    "artifacts": None,
    "threshold": 0.35,
    "model_name": "Random Forest",
    "metrics": {},
    "is_loaded": False,
}

# In-memory store for recently processed report
LATEST_REPORT: Optional[pd.DataFrame] = None


def load_model_artifacts() -> None:
    """Load serialized models, scaler, encoders, and config from disk."""
    models_dir = os.path.join(BASE_DIR, "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    artifacts_path = os.path.join(models_dir, "artifacts.joblib")
    config_path = os.path.join(models_dir, "threshold_config.json")
    metrics_path = os.path.join(models_dir, "metrics_summary.json")
    dnn_path = os.path.join(models_dir, "dnn_model.keras")

    if not os.path.exists(best_model_path) or not os.path.exists(artifacts_path):
        logger.warning(
            "Model artifacts not found in %s. Run 'python main.py' to train models.",
            models_dir,
        )
        MODEL_REGISTRY["is_loaded"] = False
        return

    try:
        MODEL_REGISTRY["model"] = joblib.load(best_model_path)
        MODEL_REGISTRY["scaler"] = joblib.load(scaler_path)
        MODEL_REGISTRY["artifacts"] = joblib.load(artifacts_path)

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                MODEL_REGISTRY["threshold"] = cfg.get("optimal_threshold", 0.35)
                MODEL_REGISTRY["model_name"] = cfg.get("best_model_name", "Random Forest")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                MODEL_REGISTRY["metrics"] = json.load(f)

        if os.path.exists(dnn_path):
            try:
                from tensorflow import keras
                MODEL_REGISTRY["dnn_model"] = keras.models.load_model(dnn_path)
            except Exception as e:
                logger.warning("Could not load Keras DNN model: %s", e)

        MODEL_REGISTRY["is_loaded"] = True
        logger.info(
            "Models successfully loaded: %s (calibrated threshold: %.3f)",
            MODEL_REGISTRY["model_name"],
            MODEL_REGISTRY["threshold"],
        )
    except Exception as e:
        logger.error("Failed to load model artifacts: %s", e, exc_info=True)
        MODEL_REGISTRY["is_loaded"] = False


def generate_explainability_reasons(
    row: pd.Series,
    prob: float,
    threshold: float,
) -> Tuple[List[str], str]:
    """Generate human-readable risk factors and explainability rationale for a transaction.

    Args:
        row: Row of processed DataFrame with features.
        prob: Fraud probability (0.0 to 1.0).
        threshold: Decision threshold.

    Returns:
        Tuple of (list_of_reasons, risk_tier).
    """
    reasons: List[str] = []

    # 1. Rapid checkout
    diff_sec = float(row.get("diff_time", 9999))
    if diff_sec <= 30:
        reasons.append(f"⚡ Instant Checkout: Order submitted within {int(diff_sec)} seconds of registration (Bot pattern).")
    elif diff_sec <= 180:
        reasons.append(f"⚡ Rapid Checkout: Order completed in {int(diff_sec)}s after signup.")

    # 2. Shared device
    dev_users = int(row.get("num_used_device", 1))
    if dev_users >= 5:
        reasons.append(f"📱 Massive Device Sharing: Hardware signature linked to {dev_users} accounts (Device farm).")
    elif dev_users > 1:
        reasons.append(f"📱 Multi-Account Device: Device shared across {dev_users} user profiles.")

    # 3. Shared IP
    ip_users = int(row.get("num_ip_repeat", 1))
    if ip_users >= 5:
        reasons.append(f"🌐 High IP Velocity: Network IP observed across {ip_users} accounts (Suspicious Proxy/VPN).")
    elif ip_users > 1:
        reasons.append(f"🌐 Repeated IP: Network IP used by {ip_users} distinct accounts.")

    # 4. High Transaction Value
    val = float(row.get("purchase_value", 0))
    if val >= 250:
        reasons.append(f"💳 High Value Order: ${val:.2f} significantly above typical cart average.")

    # 5. Geolocation / Country
    country = str(row.get("country", "Unknown"))
    if country in ("Unknown", "others"):
        reasons.append("📍 Unverified Geolocation: IP address lacks registered geographical allocation.")

    # Risk level classification
    if prob >= 0.75:
        risk_level = "CRITICAL"
        if not reasons:
            reasons.append(f"🚨 High Risk Score ({prob * 100:.1f}%): Multiple statistical anomaly vectors triggered.")
    elif prob >= threshold:
        risk_level = "HIGH"
        if not reasons:
            reasons.append(f"⚠️ Elevated Risk ({prob * 100:.1f}%): Exceeded calibrated decision threshold ({threshold * 100:.1f}%).")
    elif prob >= threshold * 0.7:
        risk_level = "MEDIUM"
        if not reasons:
            reasons.append("🔍 Moderate Risk: Borderline anomaly indicators detected.")
    else:
        risk_level = "LOW"
        if not reasons:
            reasons.append("✅ Normal Behavior: Standard consumer browsing & payment metrics.")

    return reasons, risk_level


@app.route("/", methods=["GET"])
def index() -> str:
    """Render main fraud dashboard with upload controls and status."""
    return render_template(
        "index.html",
        model_loaded=MODEL_REGISTRY["is_loaded"],
        model_name=MODEL_REGISTRY["model_name"],
        threshold=MODEL_REGISTRY["threshold"],
        results=None,
        summary=None,
    )


@app.route("/detect", methods=["POST"])
def detect() -> Any:
    """Handle CSV upload, validate schema, run inference, and return risk report."""
    global LATEST_REPORT

    if not MODEL_REGISTRY["is_loaded"]:
        load_model_artifacts()
        if not MODEL_REGISTRY["is_loaded"]:
            flash(
                "Trained models not found! Please run 'python main.py' first to train models.",
                "danger",
            )
            return redirect(url_for("index"))

    if "file" not in request.files:
        flash("No file part provided in request.", "warning")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected for upload.", "warning")
        return redirect(url_for("index"))

    if not file.filename.lower().endswith(".csv"):
        flash("Invalid file format. Please upload a standard .CSV file.", "danger")
        return redirect(url_for("index"))

    try:
        df_raw = pd.read_csv(file)
    except Exception as e:
        flash(f"Could not parse CSV file: {str(e)}", "danger")
        return redirect(url_for("index"))

    if df_raw.empty:
        flash("Uploaded CSV is empty.", "warning")
        return redirect(url_for("index"))

    # Required columns validation
    required_cols = [
        "user_id", "signup_time", "purchase_time", "purchase_value",
        "device_id", "source", "browser", "sex", "age", "ip_address",
    ]
    missing_cols = [c for c in required_cols if c not in df_raw.columns]
    if missing_cols:
        flash(
            f"CSV is missing required column(s): {', '.join(missing_cols)}. Please check dataset schema.",
            "danger",
        )
        return redirect(url_for("index"))

    try:
        # Preprocess using shared feature pipeline
        X_raw, _, _ = features.preprocess(
            df_raw,
            artifacts=MODEL_REGISTRY["artifacts"],
            country_df=MODEL_REGISTRY["artifacts"].get("country_df"),
            is_training=False,
        )

        scaler = MODEL_REGISTRY["scaler"]
        X_scaled = scaler.transform(X_raw)

        # Inference using selected model
        model = MODEL_REGISTRY["model"]
        threshold = float(MODEL_REGISTRY["threshold"])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        elif MODEL_REGISTRY["dnn_model"] is not None:
            probabilities = MODEL_REGISTRY["dnn_model"].predict(X_scaled, verbose=0).ravel()
        else:
            probabilities = model.predict(X_scaled).astype(float)

        # Build detailed results per row
        results: List[Dict[str, Any]] = []
        df_report = df_raw.copy()
        df_report["fraud_probability"] = np.round(probabilities, 4)
        df_report["predicted_class"] = (probabilities >= threshold).astype(int)

        reasons_col: List[str] = []
        risk_levels_col: List[str] = []

        # Copy temp features for reason evaluation
        df_features = pd.DataFrame(X_raw, columns=features.FEATURE_COLUMNS)
        if "country" not in df_features.columns and "country" in df_report.columns:
            df_features["country"] = df_report["country"]

        for idx in range(len(df_raw)):
            prob = float(probabilities[idx])
            feat_row = df_features.iloc[idx]
            reasons, risk_level = generate_explainability_reasons(feat_row, prob, threshold)

            reasons_col.append("; ".join(reasons))
            risk_levels_col.append(risk_level)

            res_item = {
                "index": idx + 1,
                "user_id": int(df_raw.iloc[idx].get("user_id", idx + 1)),
                "purchase_value": float(df_raw.iloc[idx].get("purchase_value", 0.0)),
                "diff_time_sec": int(feat_row.get("diff_time", 0)),
                "device_id": str(df_raw.iloc[idx].get("device_id", "Unknown")),
                "ip_address": str(df_raw.iloc[idx].get("ip_address", "Unknown")),
                "probability": round(prob * 100, 1),
                "is_fraud": prob >= threshold,
                "risk_level": risk_level,
                "reasons": reasons,
            }
            results.append(res_item)

        df_report["risk_level"] = risk_levels_col
        df_report["risk_reasons"] = reasons_col
        LATEST_REPORT = df_report

        total_rows = len(results)
        fraud_count = sum(1 for r in results if r["is_fraud"])
        legit_count = total_rows - fraud_count
        fraud_rate = round((fraud_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        avg_risk = round(float(np.mean(probabilities)) * 100, 1)

        summary = {
            "total_transactions": total_rows,
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_rate": fraud_rate,
            "avg_risk": avg_risk,
            "threshold_used": round(threshold * 100, 1),
        }

        return render_template(
            "index.html",
            model_loaded=MODEL_REGISTRY["is_loaded"],
            model_name=MODEL_REGISTRY["model_name"],
            threshold=MODEL_REGISTRY["threshold"],
            results=results,
            summary=summary,
        )

    except Exception as e:
        logger.error("Error during fraud prediction: %s", e, exc_info=True)
        flash(f"Inference processing failed: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/download-report", methods=["GET"])
def download_report() -> Any:
    """Export the evaluated dataset with probabilities and risk explanations as CSV."""
    global LATEST_REPORT
    if LATEST_REPORT is None or LATEST_REPORT.empty:
        flash("No active audit report available to download. Please run fraud detection first.", "warning")
        return redirect(url_for("index"))

    csv_buffer = io.StringIO()
    LATEST_REPORT.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))

    return send_file(
        csv_bytes,
        mimetype="text/csv",
        as_attachment=True,
        download_name="fraud_detection_audit_report.csv",
    )


@app.route("/sample-csv", methods=["GET"])
def sample_csv() -> Any:
    """Generate and serve a realistic 25-row sample CSV for immediate demo testing."""
    sample_data = [
        # Legitimate transactions
        {
            "user_id": 10001, "signup_time": "2024-03-01 10:15:00",
            "purchase_time": "2024-03-05 14:22:10", "purchase_value": 45.50,
            "device_id": "DEV_USER_001", "source": "SEO", "browser": "Chrome",
            "sex": "M", "age": 32, "ip_address": 732758432,
        },
        {
            "user_id": 10002, "signup_time": "2024-03-02 09:00:00",
            "purchase_time": "2024-03-12 18:45:00", "purchase_value": 89.99,
            "device_id": "DEV_USER_002", "source": "Ads", "browser": "Safari",
            "sex": "F", "age": 28, "ip_address": 841285912,
        },
        {
            "user_id": 10003, "signup_time": "2024-03-03 11:30:00",
            "purchase_time": "2024-03-08 09:12:00", "purchase_value": 24.00,
            "device_id": "DEV_USER_003", "source": "Direct", "browser": "Firefox",
            "sex": "M", "age": 45, "ip_address": 912345678,
        },
        # Fraud transaction 1: Instant purchase (2 seconds after signup)
        {
            "user_id": 20001, "signup_time": "2024-03-10 12:00:00",
            "purchase_time": "2024-03-10 12:00:02", "purchase_value": 299.99,
            "device_id": "DEV_FARM_0001", "source": "Direct", "browser": "Chrome",
            "sex": "M", "age": 24, "ip_address": 1056789123,
        },
        # Fraud transaction 2: Shared device & instant purchase
        {
            "user_id": 20002, "signup_time": "2024-03-10 12:00:15",
            "purchase_time": "2024-03-10 12:00:22", "purchase_value": 249.50,
            "device_id": "DEV_FARM_0001", "source": "Direct", "browser": "Chrome",
            "sex": "F", "age": 29, "ip_address": 1056789123,
        },
        # Fraud transaction 3: Shared device third user
        {
            "user_id": 20003, "signup_time": "2024-03-10 12:01:00",
            "purchase_time": "2024-03-10 12:01:08", "purchase_value": 199.00,
            "device_id": "DEV_FARM_0001", "source": "Direct", "browser": "Chrome",
            "sex": "M", "age": 31, "ip_address": 1056789123,
        },
    ]

    # Fill additional varied sample rows
    for i in range(4, 23):
        is_fraud = (i % 7 == 0)
        dt_signup = pd.Timestamp("2024-04-01") + pd.Timedelta(days=i, hours=i % 12)
        diff_s = 5 if is_fraud else (3600 * 24 * (i % 8 + 1))
        dt_purch = dt_signup + pd.Timedelta(seconds=diff_s)
        sample_data.append({
            "user_id": 30000 + i,
            "signup_time": dt_signup.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_time": dt_purch.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_value": 320.0 if is_fraud else round(18.0 + (i * 4.5), 2),
            "device_id": "DEV_FARM_0002" if is_fraud else f"DEV_SAMPLE_{i:04d}",
            "source": "Direct" if is_fraud else ["SEO", "Ads", "Direct"][i % 3],
            "browser": ["Chrome", "Safari", "Firefox"][i % 3],
            "sex": "M" if i % 2 == 0 else "F",
            "age": 25 + (i % 35),
            "ip_address": 1150000000 + (100 if is_fraud else i * 65536),
        })

    sample_df = pd.DataFrame(sample_data)
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))

    return send_file(
        csv_bytes,
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_fraud_test_transactions.csv",
    )


@app.route("/metrics", methods=["GET"])
def metrics() -> str:
    """Render comprehensive model performance, validation benchmarks, and visual artifacts."""
    if not MODEL_REGISTRY["is_loaded"]:
        load_model_artifacts()

    metrics_data = MODEL_REGISTRY.get("metrics", {})
    best_meta = metrics_data.get("_meta", {})

    # Available static charts
    static_dir = os.path.join(BASE_DIR, "static")
    charts = {
        "roc_curve": os.path.exists(os.path.join(static_dir, "combined_roc_curve.png")),
        "pr_curve": os.path.exists(os.path.join(static_dir, "combined_pr_curve.png")),
        "confusion_matrix": os.path.exists(os.path.join(static_dir, "confusion_matrix_best.png")),
        "shap_summary": os.path.exists(os.path.join(static_dir, "shap_summary.png")),
        "class_balance": os.path.exists(os.path.join(static_dir, "eda_class_balance.png")),
        "diff_time": os.path.exists(os.path.join(static_dir, "eda_diff_time.png")),
        "device_sharing": os.path.exists(os.path.join(static_dir, "eda_device_sharing.png")),
        "correlation": os.path.exists(os.path.join(static_dir, "eda_correlation.png")),
    }

    return render_template(
        "metrics.html",
        metrics_data=metrics_data,
        best_meta=best_meta,
        charts=charts,
        model_name=MODEL_REGISTRY["model_name"],
        threshold=MODEL_REGISTRY["threshold"],
    )


def main() -> None:
    """Start local Flask server."""
    load_model_artifacts()
    logger.info("Starting Fraud Detection Web App on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
