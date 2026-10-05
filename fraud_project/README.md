# 🛡️ FraudGuard AI: E-Commerce Fraud Detection System

A complete, production-grade, local fraud-detection system for e-commerce transactions built strictly with open-source Python libraries.

- **100% Offline & Local**: No external APIs, no cloud services, no API keys required.
- **Explainable ML**: Combines 6 Classical Machine Learning models + a Deep Neural Network (Keras DNN) with SHAP value interpretability and rule-based velocity triggers.
- **Production-Ready Web Dashboard**: Dark theme with crimson red cyber accents, drag-and-drop CSV batch upload, real-time risk scoring, and downloadable enriched audit reports.

---

## 🏗️ Project Architecture & Directory Structure

```text
fraud_project/
├── data/
│   ├── Fraud_Data.csv              # Kaggle E-commerce transaction dataset
│   └── IpAddress_to_Country.csv    # IP range to country mapping table
├── models/
│   ├── best_model.joblib           # Serialized champion tree ensemble
│   ├── best_tree_model.joblib      # Champion tree model
│   ├── dnn_model.keras             # Trained Keras Deep Neural Network
│   ├── scaler.joblib               # StandardScaler fitted on training features
│   ├── artifacts.joblib            # Label encoders, country lookup, frequency dictionaries
│   ├── threshold_config.json       # Calibrated decision threshold & feature list
│   └── metrics_summary.json        # Evaluation metrics for dashboard leaderboard
├── static/
│   ├── css/
│   │   └── styles.css              # Dark theme, glassmorphism, responsive grid, red accents
│   ├── js/
│   │   └── main.js                 # Drag & drop upload, table search & risk filtering
│   ├── combined_roc_curve.png      # All 7 models on single ROC plot
│   ├── combined_pr_curve.png       # Precision-Recall curve
│   ├── confusion_matrix_best.png   # Calibrated confusion matrix
│   ├── shap_summary.png            # SHAP feature importance plot
│   ├── eda_class_balance.png       # EDA: Class balance
│   ├── eda_fraud_by_month.png      # EDA: Monthly fraud rate
│   ├── eda_diff_time.png           # EDA: Signup-to-purchase time delta
│   ├── eda_device_sharing.png      # EDA: Device sharing frequency vs fraud
│   └── eda_correlation.png         # EDA: Feature correlation heatmap
├── templates/
│   ├── base.html                   # Master layout with navigation and footer
│   ├── index.html                  # Main dashboard (upload, preview, risk breakdown)
│   └── metrics.html                # Model leaderboard, diagnostic curves & EDA gallery
├── tests/
│   └── test_features.py            # Unit test suite for feature engineering
├── test_features.py                # Root test discovery wrapper
├── features.py                     # Unified preprocessing (shared by training & serving)
├── generate_data.py                # Synthetic data generator (~50k rows, ~3% fraud)
├── main.py                         # EDA + SMOTE + 5-Fold CV + DNN + Evaluation + Persistence
├── predict.py                      # Flask web application & inference engine
└── requirements.txt                # Pinned dependencies
```

---

## ⚙️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Data Processing** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `imbalanced-learn` (SMOTE) |
| **Deep Learning** | `TensorFlow` / `Keras` (Dense-BatchNorm-Dropout architecture) |
| **Explainability** | `shap` (Shapley Additive exPlanations) |
| **Model Persistence** | `joblib`, native Keras `.keras` |
| **Web Application** | `Flask` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Unit Testing** | `pytest` |

---

## 🚀 Setup & Execution Guide

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
cd "c:\Users\sivas\Desktop\FD in E-Commerce\fraud_project"
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
cd "c:/Users/sivas/Desktop/FD in E-Commerce/fraud_project"
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Unit Tests

Execute the unit test suite covering `diff_time`, `ip_to_int`, `get_country`, `safe_label_encode`, and full pipeline `preprocess()`:

```bash
pytest -v
```

### 4. (Optional) Generate Realistic Synthetic Data

If you do not have the Kaggle dataset files already, you can generate 50,000 realistic transactions with ~3.5% fraud rate:

```bash
python generate_data.py
```
*(Note: `main.py` will automatically run this step if the files are not present in `data/`)*.

### 5. Train Models, Run EDA & Benchmarks

Run the complete machine learning pipeline:

```bash
python main.py
```

**What `main.py` performs:**
1. Loads or generates transaction dataset & IP range lookup tables.
2. Generates 5 detailed EDA charts saved directly to `static/`.
3. Performs a stratified 80/20 train/test split.
4. Applies **SMOTE** on the training set only (preventing data leakage).
5. Trains and runs **5-fold Stratified Cross-Validation** (scored by ROC-AUC and Recall) across:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors (KNN)
   - Naive Bayes
   - Random Forest
   - Gradient Boosting
6. Builds and trains a **Keras Deep Neural Network**:
   `Input -> Dense(128, relu) -> BatchNormalization -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(1, sigmoid)`
   with `StandardScaler`, `EarlyStopping`, and balanced `class_weight`.
7. Evaluates all 7 models on the holdout test set (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
8. Plots combined ROC and Precision-Recall curves.
9. **Calibrates Decision Threshold**: Sweeps decision thresholds to find the boundary that balances high recall with strong precision rather than an arbitrary 0.5 cut-off.
10. Generates SHAP feature importance plot and exports model artifacts to `models/`.

### 6. Launch the Fraud Detection Web Dashboard

```bash
python predict.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## 🔍 Feature Engineering (`features.py`)

Both training (`main.py`) and real-time inference (`predict.py`) strictly utilize the exact same feature engineering module:

1. **`diff_time = purchase_time - signup_time` (seconds)**:
   - Safely parses irregular datetime formats with median fallback and clips negative deltas.
   - Fraudulent transactions cluster dramatically within 1–120 seconds after signup (bot automation).
2. **`ip_to_int(ip)`**:
   - Parses dotted IPv4 strings, integers, floats, scientific notations, and corrupt strings safely to 32-bit unsigned integers.
3. **`get_country(ip_ints)` with `np.searchsorted`**:
   - High-speed $O(\log N)$ binary search across thousands of contiguous IP range intervals from `IpAddress_to_Country.csv`.
4. **`num_used_device` and `num_ip_repeat`**:
   - Measures multi-accounting: how many distinct `user_id` values share the same `device_id` or `ip_address`.
   - Combines batch-internal counts with historical frequency dictionaries to detect returning bot farms.
5. **Behavioral Flags**:
   - `is_instant_purchase`: checkout in $\le 120$ seconds.
   - `is_device_shared`: hardware footprint used by $\ge 2$ users.
   - `is_ip_shared`: network IP reused by $\ge 2$ users.
6. **`SafeLabelEncoder`**:
   - Gracefully handles unseen categories in production by encoding them to `'others'` without throwing errors.

---

## 🎯 Decision Threshold Optimization

In fraud detection, positive cases are rare (~3%). An arbitrary 0.5 threshold yields high precision but misses a large proportion of fraud (low recall).

`main.py` evaluates the Precision-Recall trade-off across thresholds from 0.10 to 0.90 to find the calibrated threshold that maximizes the F1 score:
- **Default 0.5 Threshold**: Often misses subtle device sharing or fast checkouts.
- **Calibrated Threshold (~0.28 - 0.38)**: Catches up to 85%+ of fraud cases while keeping false alarms low.

---

## 🖥️ Web Dashboard Capabilities

- **Live Drag-and-Drop CSV Upload**: Accepts batches up to 16 MB with automatic column validation.
- **Sample Test Data Download**: Built-in 25-row sample dataset ready for one-click testing.
- **Real-Time Anomaly Scoring**: Displays fraud probability (0–100%), calibrated risk tiers (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), and visual risk gauges.
- **Row-Level Explainability**:
  - `⚡ Rapid Checkout: Order completed in 5s after registration (Bot signature).`
  - `📱 Massive Device Sharing: Hardware signature linked to 14 accounts (Device farm).`
  - `🌐 High IP Velocity: Network IP observed across 8 accounts (Proxy/VPN suspect).`
  - `💳 High Value Order: $299.99 (elevated risk category).`
- **Download Audit Report (CSV)**: Export the complete analyzed dataset enriched with model risk scores, risk levels, and explainability reasons.
- **Model Metrics Page (`/metrics`)**: Displays the 7-model leaderboard, combined ROC/PR curves, confusion matrix, SHAP explainability plot, and EDA distributions.
