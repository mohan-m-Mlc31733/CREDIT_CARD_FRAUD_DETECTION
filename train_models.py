"""
train_models.py
===============
Trains three credit card fraud classifiers (XGBoost, Random Forest, Logistic Regression)
on fraudTrain.csv and evaluates on fraudTest.csv.
Persists trained models and manifest.json into backend/models/.
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

try:
    import xgboost as xgb
except ImportError:
    sys.exit("XGBoost is required. Install it with: pip install xgboost")

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
PROJECT   = os.path.dirname(BASE)


def find_csv(name):
    """Look for the Kaggle CSV in $FRAUD_DATA_DIR, data/, the project root, or ~/Downloads."""
    candidates = [
        os.path.join(os.environ["FRAUD_DATA_DIR"], name) if os.environ.get("FRAUD_DATA_DIR") else "",
        os.path.join(PROJECT, "data", name),
        os.path.join(PROJECT, name),
        os.path.join(os.path.expanduser("~"), "Downloads", name),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"{name} not found. Download it from Kaggle (kartik2112/fraud-detection) "
        f"and place it in the project's data/ folder."
    )


TRAIN_CSV = find_csv("fraudTrain.csv")
TEST_CSV  = find_csv("fraudTest.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Approximate age from dob
    dob = pd.to_datetime(df["dob"], errors="coerce")
    ref_date = dt.max()
    df["age"] = ((ref_date - dob).dt.days / 365.25).fillna(40).astype(int)

    # Distance between cardholder and merchant (Haversine approx in km)
    lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
    lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    df["distance_km"] = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    df["amt_log"] = np.log1p(df["amt"])
    return df


def prepare_Xy(df: pd.DataFrame, cat_encoder=None, state_encoder=None):
    df = engineer_features(df)

    if cat_encoder is None:
        cat_encoder = LabelEncoder()
        df["category_enc"] = cat_encoder.fit_transform(df["category"].astype(str))
    else:
        known_cats = set(cat_encoder.classes_)
        df["category_enc"] = df["category"].apply(
            lambda c: cat_encoder.transform([c])[0] if c in known_cats else 0
        )

    if state_encoder is None:
        state_encoder = LabelEncoder()
        df["state_enc"] = state_encoder.fit_transform(df["state"].astype(str))
    else:
        known_states = set(state_encoder.classes_)
        df["state_enc"] = df["state"].apply(
            lambda s: state_encoder.transform([s])[0] if s in known_states else 0
        )

    features = [
        "amt", "amt_log", "category_enc", "hour", "day_of_week",
        "is_weekend", "age", "city_pop", "distance_km", "state_enc",
    ]

    X = df[features].fillna(0).values
    y = df["is_fraud"].values if "is_fraud" in df.columns else None
    return X, y, cat_encoder, state_encoder, features


def evaluate(name, model, X_test, y_test, scaler=None):
    X_eval = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":   round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":    round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc":     round(float(average_precision_score(y_test, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"  {name:22s} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | "
          f"Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | "
          f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    return metrics


def main():
    t0 = time.time()
    print("=" * 65)
    print("  FRAUDSHIELD MODEL TRAINING PIPELINE")
    print("=" * 65)

    print("[1/5] Loading fraud dataset...")
    df_raw = pd.read_csv(TRAIN_CSV)
    total_train = len(df_raw)
    print(f"      Train raw: {total_train:,} transactions")

    # Smart sampling for fast training while preserving ALL fraud attacks:
    df_fraud = df_raw[df_raw["is_fraud"] == 1]
    df_legit = df_raw[df_raw["is_fraud"] == 0].sample(n=min(80000, len(df_raw) - len(df_fraud)), random_state=42)
    df_train = pd.concat([df_fraud, df_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"      Optimized training subset: {len(df_train):,} rows ({len(df_fraud):,} fraud + {len(df_legit):,} legitimate)")

    print("[2/5] Loading test evaluation dataset...")
    df_test_raw = pd.read_csv(TEST_CSV)
    df_test_fraud = df_test_raw[df_test_raw["is_fraud"] == 1]
    df_test_legit = df_test_raw[df_test_raw["is_fraud"] == 0].sample(n=min(25000, len(df_test_raw) - len(df_test_fraud)), random_state=42)
    df_test = pd.concat([df_test_fraud, df_test_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"      Test evaluation subset: {len(df_test):,} rows ({len(df_test_fraud):,} fraud + {len(df_test_legit):,} legitimate)")

    print("[3/5] Engineering domain risk features...")
    X_train, y_train, cat_enc, state_enc, feature_names = prepare_Xy(df_train)
    X_test, y_test, _, _, _ = prepare_Xy(df_test, cat_enc, state_enc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_weight = float(neg / pos) if pos > 0 else 1.0

    all_metrics = {}

    print("\n[4/5] Training Classifiers:")
    # --- 1. XGBoost ---
    print("  >> Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    all_metrics["xgboost"] = evaluate("XGBoost", xgb_model, X_test, y_test)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost.joblib"))

    # --- 2. Random Forest ---
    print("  >> Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=70,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    all_metrics["rf"] = evaluate("Random Forest", rf_model, X_test, y_test)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest.joblib"))

    # --- 3. Logistic Regression ---
    print("  >> Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    lr_model.fit(X_train_scaled, y_train)
    all_metrics["lr"] = evaluate("Logistic Regression", lr_model, X_test, y_test, scaler=scaler)
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression.joblib"))

    print("\n[5/5] Exporting metadata and encoders...")
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(cat_enc, os.path.join(MODEL_DIR, "category_encoder.joblib"))
    joblib.dump(state_enc, os.path.join(MODEL_DIR, "state_encoder.joblib"))

    cat_rates = {}
    if "category" in df_raw.columns and "is_fraud" in df_raw.columns:
        cat_grp = df_raw.groupby("category")["is_fraud"].agg(["sum", "count"])
        for cat, row in cat_grp.iterrows():
            cat_rates[str(cat)] = round(float(row["sum"] / row["count"] * 100), 2)

    manifest = {
        "feature_names": feature_names,
        "categories": cat_enc.classes_.tolist(),
        "states": state_enc.classes_.tolist(),
        "train_rows": total_train,
        "test_rows": len(df_test_raw),
        "fraud_rate_pct": round(float(df_raw["is_fraud"].mean() * 100), 3),
        "category_fraud_rates": cat_rates,
        "metrics": all_metrics,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(MODEL_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[SUCCESS] All 3 models trained and saved in {elapsed:.1f}s!")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
