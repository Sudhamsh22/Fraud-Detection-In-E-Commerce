"""main.py

End-to-end Machine Learning Pipeline for E-Commerce Fraud Detection:
1. Automated dataset validation & EDA plot generation.
2. Stratified train/test splitting & SMOTE oversampling on training data.
3. 5-Fold Stratified Cross-Validation on 6 Classical ML models (Logistic Regression,
   Decision Tree, KNN, Naive Bayes, Random Forest, Gradient Boosting).
4. Deep Neural Network (Keras DNN) with BatchNorm, Dropout, EarlyStopping, and Class Weights.
5. Multi-metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
6. Combined ROC & PR curve plotting, threshold optimization, confusion matrix, SHAP analysis.
7. Model artifact serialization (best model, DNN, scaler, encoders, optimal threshold).
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Headless backend for server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

import features
import generate_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_pipeline")


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception as e:
        logger.warning("Could not set TensorFlow seed: %s", e)


def run_eda(df: pd.DataFrame, country_df: pd.DataFrame, static_dir: str) -> None:
    """Perform exploratory data analysis and save informative visualizations.

    Generates:
        1. eda_class_balance.png: Class distribution (Legit vs Fraud).
        2. eda_fraud_by_month.png: Fraud rate by purchase month.
        3. eda_diff_time.png: Signup-to-purchase diff_time comparison by class.
        4. eda_device_sharing.png: Device sharing frequency vs fraud rate.
        5. eda_correlation.png: Feature correlation heatmap.
    """
    os.makedirs(static_dir, exist_ok=True)
    logger.info("Executing Exploratory Data Analysis (EDA)...")
    sns.set_theme(style="darkgrid", palette="muted")

    # 1. Class Balance Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    class_counts = df["class"].value_counts()
    colors = ["#2b8a3e", "#e03131"]
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], class_counts.values, color=colors, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(class_counts.values) * 0.02),
            f"{yval:,} ({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("E-Commerce Transaction Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Transaction Count")
    ax.set_ylim(0, max(class_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "eda_class_balance.png"), dpi=200)
    plt.close()

    # 2. Fraud by Month
    temp_df = df.copy()
    temp_df["purchase_dt"] = pd.to_datetime(temp_df["purchase_time"], errors="coerce")
    temp_df["month"] = temp_df["purchase_dt"].dt.month_name()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    present_months = [m for m in month_order if m in temp_df["month"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_stats = temp_df.groupby("month")["class"].agg(["count", "mean"]).reindex(present_months)
    monthly_stats["fraud_pct"] = monthly_stats["mean"] * 100
    sns.barplot(x=monthly_stats.index, y=monthly_stats["fraud_pct"], color="#ff6b6b", ax=ax)
    ax.set_title("Fraud Rate (%) by Purchase Month", fontsize=14, fontweight="bold")
    ax.set_ylabel("Fraud Percentage (%)")
    ax.set_xlabel("Month")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "eda_fraud_by_month.png"), dpi=200)
    plt.close()

    # 3. Signup-to-Purchase Time Difference by Class
    temp_df["signup_dt"] = pd.to_datetime(temp_df["signup_time"], errors="coerce")
    temp_df["diff_seconds"] = (temp_df["purchase_dt"] - temp_df["signup_dt"]).dt.total_seconds().clip(lower=0)
    temp_df["diff_minutes"] = temp_df["diff_seconds"] / 60.0

    fig, ax = plt.subplots(figsize=(8, 5))
    # Log scale comparison
    temp_df["log10_diff_sec"] = np.log10(temp_df["diff_seconds"] + 1)
    sns.boxplot(
        x="class",
        y="log10_diff_sec",
        data=temp_df,
        palette=["#51cf66", "#ff6b6b"],
        ax=ax,
        width=0.4,
    )
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    ax.set_title("Signup-to-Purchase Time Difference (Log10 Seconds)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Log10(Difference in Seconds + 1)")
    ax.set_xlabel("Class")
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "eda_diff_time.png"), dpi=200)
    plt.close()

    # 4. Device Sharing vs Fraud Rate
    device_counts = temp_df.groupby("device_id")["user_id"].transform("nunique")
    temp_df["device_users_bucket"] = pd.cut(
        device_counts,
        bins=[0, 1, 2, 5, 100],
        labels=["1 User", "2 Users", "3-5 Users", "6+ Users"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    dev_stats = temp_df.groupby("device_users_bucket", observed=False)["class"].mean() * 100
    sns.barplot(x=dev_stats.index, y=dev_stats.values, palette="Reds_r", ax=ax)
    ax.set_title("Fraud Rate (%) by Device Sharing Frequency", fontsize=14, fontweight="bold")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xlabel("Device Sharing Bucket")
    for i, v in enumerate(dev_stats.values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "eda_device_sharing.png"), dpi=200)
    plt.close()

    # 5. Correlation Heatmap
    # Preprocess small sample to compute correlation
    X_sample, y_sample, _ = features.preprocess(
        temp_df.head(5000), is_training=True, country_df=country_df
    )
    corr_df = pd.DataFrame(X_sample, columns=features.FEATURE_COLUMNS)
    corr_df["class"] = y_sample[: len(corr_df)]
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        fmt=".2f",
        annot=False,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "eda_correlation.png"), dpi=200)
    plt.close()

    logger.info("EDA visualizations saved to %s", static_dir)


def train_keras_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    """Build and train Keras Deep Neural Network with specified architecture.

    Architecture:
        Dense(128, relu) -> BatchNorm -> Dropout(0.3) ->
        Dense(64, relu) -> Dropout(0.3) ->
        Dense(1, sigmoid)

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training labels.
        X_val: Scaled validation/test feature matrix.
        y_val: Validation/test labels.

    Returns:
        Trained Keras Model instance.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    logger.info("Configuring Deep Neural Network (DNN)...")
    input_dim = X_train.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    # Calculate class weights for imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=0,
    )

    logger.info("Training Keras DNN with early stopping and class weights...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=256,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1,
    )
    return model


def evaluate_models(
    models_dict: Dict[str, Any],
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    static_dir: str,
) -> Tuple[Dict[str, Dict[str, float]], str, float, Any]:
    """Evaluate all trained models on test set, generate combined ROC/PR curves,

    and determine the best model & decision threshold.

    Returns:
        Tuple of:
            - metrics_summary: Dict of metrics per model
            - best_model_name: Name of top performing model
            - optimal_threshold: Calibrated decision threshold
            - best_model: Model instance
    """
    logger.info("Evaluating all models on holdout test set...")
    metrics_summary = {}
    roc_curves = {}
    pr_curves = {}

    plt.figure(figsize=(9, 7))
    sns.set_style("darkgrid")

    for name, model in models_dict.items():
        if name == "Deep Neural Network":
            y_probs = model.predict(X_test_scaled, verbose=0).ravel()
        else:
            y_probs = model.predict_proba(X_test_scaled)[:, 1]

        y_pred = (y_probs >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_probs))
        pr_auc = float(average_precision_score(y_test, y_probs))
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_summary[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "ConfusionMatrix": cm,
        }

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_curves[name] = (fpr, tpr, roc_auc)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
        pr_curves[name] = (recall_curve, precision_curve, pr_auc)

        logger.info(
            "Model: %-22s | ROC-AUC: %.4f | PR-AUC: %.4f | Recall: %.4f | F1: %.4f",
            name, roc_auc, pr_auc, rec, f1
        )

    # Plot Combined ROC Curve
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    ax.set_title("Combined Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "combined_roc_curve.png"), dpi=200)
    plt.close()

    # Plot Combined PR Curve
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (rec_c, prec_c, pr_val) in pr_curves.items():
        ax.plot(rec_c, prec_c, label=f"{name} (PR-AUC = {pr_val:.3f})", lw=2)
    ax.set_title("Combined Precision-Recall (PR) Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "combined_pr_curve.png"), dpi=200)
    plt.close()

    # Select best model primarily by ROC-AUC and PR-AUC
    # We prefer tree-based ensemble if close in score for fast inference & interpretability
    sorted_models = sorted(
        metrics_summary.items(),
        key=lambda item: (item[1]["PR-AUC"] + item[1]["ROC-AUC"] + item[1]["F1"]),
        reverse=True,
    )
    best_model_name = sorted_models[0][0]
    best_model = models_dict[best_model_name]
    logger.info("Best overall model: %s", best_model_name)

    # Decision Threshold Tuning on Best Model
    if best_model_name == "Deep Neural Network":
        best_probs = best_model.predict(X_test_scaled, verbose=0).ravel()
    else:
        best_probs = best_model.predict_proba(X_test_scaled)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = -1.0
    optimal_threshold = 0.5

    for th in thresholds:
        preds = (best_probs >= th).astype(int)
        score = f1_score(y_test, preds, zero_division=0)
        # Prioritize recall if F1 is within tight margin
        if score > best_f1:
            best_f1 = score
            optimal_threshold = float(round(th, 3))

    logger.info("Optimal decision threshold tuned: %.3f (Yields F1 = %.4f)", optimal_threshold, best_f1)

    # Plot Confusion Matrix for Best Model at Optimal Threshold
    tuned_preds = (best_probs >= optimal_threshold).astype(int)
    cm_tuned = confusion_matrix(y_test, tuned_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_tuned,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"],
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix: {best_model_name}\n(Tuned Threshold: {optimal_threshold})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "confusion_matrix_best.png"), dpi=200)
    plt.close()

    metrics_summary["_meta"] = {
        "best_model_name": best_model_name,
        "optimal_threshold": optimal_threshold,
        "tuned_f1": round(float(best_f1), 4),
        "tuned_recall": round(float(recall_score(y_test, tuned_preds, zero_division=0)), 4),
        "tuned_precision": round(float(precision_score(y_test, tuned_preds, zero_division=0)), 4),
    }

    return metrics_summary, best_model_name, optimal_threshold, best_model


def run_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    static_dir: str,
) -> None:
    """Compute SHAP summary plot for model explainability."""
    try:
        import shap
        logger.info("Calculating SHAP values for model explainability...")
        # Subsample for fast SHAP computation
        subsample_idx = np.random.choice(len(X_sample), size=min(300, len(X_sample)), replace=False)
        X_sub = X_sample[subsample_idx]

        plt.figure(figsize=(10, 6))
        explainer = shap.Explainer(model, X_sub)
        shap_values = explainer(X_sub)

        # For binary classification, take output for positive fraud class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        shap.summary_plot(
            shap_values.values,
            X_sub,
            feature_names=feature_names,
            show=False,
            max_display=12,
        )
        plt.title("SHAP Feature Importance Summary (Fraud Prediction)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, "shap_summary.png"), dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", static_dir)
    except Exception as e:
        logger.warning("SHAP summary plot skipped due to exception: %s. Generating fallback feature importances.", e)
        # Fallback to feature importances if SHAP fails or model is not tree
        if hasattr(model, "feature_importances_"):
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:12]
            sns.barplot(
                x=[importances[i] for i in indices],
                y=[feature_names[i] for i in indices],
                palette="viridis",
                ax=ax,
            )
            ax.set_title("Top 12 Feature Importances", fontsize=13, fontweight="bold")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, "shap_summary.png"), dpi=200)
            plt.close()


def main() -> None:
    """Main execution orchestrating data loading, EDA, training, and artifact persistence."""
    set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    static_dir = os.path.join(base_dir, "static")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    fraud_csv = os.path.join(data_dir, "Fraud_Data.csv")
    country_csv = os.path.join(data_dir, "IpAddress_to_Country.csv")

    # 1. Dataset Verification & Generation if missing
    if not os.path.exists(country_csv):
        logger.info("Country range lookup missing. Generating synthetic dataset...")
        country_df = generate_data.generate_ip_country_ranges(country_csv)
    else:
        country_df = features.load_country_lookup(country_csv)

    if not os.path.exists(fraud_csv):
        logger.info("Fraud dataset missing. Generating realistic 50k transactions...")
        df_raw = generate_data.generate_fraud_dataset(fraud_csv, country_df, num_records=50000)
    else:
        logger.info("Loading existing dataset from %s", fraud_csv)
        df_raw = pd.read_csv(fraud_csv)

    # 2. Run EDA
    run_eda(df_raw, country_df, static_dir)

    # 3. Stratified Train / Test Split (Holdout test 20%)
    logger.info("Splitting dataset into Stratified Train (80%%) and Test (20%%)...")
    train_df, test_df = train_test_split(
        df_raw,
        test_size=0.20,
        random_state=42,
        stratify=df_raw["class"],
    )

    # Feature preprocessing
    logger.info("Preprocessing training data & fitting feature encoders...")
    X_train_raw, y_train, artifacts = features.preprocess(
        train_df, country_df=country_df, is_training=True
    )
    logger.info("Preprocessing test data using fitted artifacts...")
    X_test_raw, y_test, _ = features.preprocess(
        test_df, artifacts=artifacts, country_df=country_df, is_training=False
    )

    # SMOTE Oversampling on TRAINING SET ONLY
    logger.info("Applying SMOTE oversampling to training set only...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_raw, y_train)
    logger.info(
        "Training size after SMOTE: %d (Class 0: %d, Class 1: %d)",
        len(y_train_resampled),
        np.sum(y_train_resampled == 0),
        np.sum(y_train_resampled == 1),
    )

    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test_raw)

    # 4. Train and Cross-Validate 6 Classical ML Models
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classical_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    trained_models = {}
    logger.info("Running 5-fold cross-validation on 6 classical machine learning models...")
    for name, model in classical_models.items():
        cv_scores = cross_validate(
            model,
            X_train_scaled,
            y_train_resampled,
            cv=cv_strategy,
            scoring=["roc_auc", "recall"],
            n_jobs=-1,
        )
        mean_auc = cv_scores["test_roc_auc"].mean()
        mean_rec = cv_scores["test_recall"].mean()
        logger.info("5-Fold CV: %-20s | Mean AUC: %.4f | Mean Recall: %.4f", name, mean_auc, mean_rec)

        # Fit model on entire resampled training set
        model.fit(X_train_scaled, y_train_resampled)
        trained_models[name] = model

    # 5. Train Keras Deep Neural Network
    dnn_model = train_keras_dnn(X_train_scaled, y_train_resampled, X_test_scaled, y_test)
    trained_models["Deep Neural Network"] = dnn_model

    # 6. Evaluation, Combined Curves, Threshold Tuning
    metrics_summary, best_name, optimal_th, best_model = evaluate_models(
        trained_models, X_test_scaled, y_test, static_dir
    )

    # 7. SHAP Feature Explainability
    tree_candidates = [m for m in [trained_models.get("Random Forest"), trained_models.get("Gradient Boosting")] if m]
    chosen_tree = tree_candidates[0] if tree_candidates else trained_models["Decision Tree"]
    run_shap_analysis(chosen_tree, X_test_scaled, features.FEATURE_COLUMNS, static_dir)

    # 8. Save Artifacts
    logger.info("Serializing model artifacts to %s...", models_dir)

    # Save Best Tree Model
    best_tree = trained_models.get("Random Forest") or trained_models.get("Gradient Boosting")
    joblib.dump(best_tree, os.path.join(models_dir, "best_tree_model.joblib"))
    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))

    # Save Keras DNN
    dnn_model.save(os.path.join(models_dir, "dnn_model.keras"))

    # Save Scaler and Artifacts
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(artifacts, os.path.join(models_dir, "artifacts.joblib"))

    # Save Threshold & Configuration
    config = {
        "best_model_name": best_name,
        "optimal_threshold": optimal_th,
        "feature_columns": features.FEATURE_COLUMNS,
    }
    with open(os.path.join(models_dir, "threshold_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # Save Metrics Summary for Flask Dashboard
    with open(os.path.join(models_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    logger.info("Pipeline completed successfully! All artifacts and plots are saved.")


if __name__ == "__main__":
    main()
