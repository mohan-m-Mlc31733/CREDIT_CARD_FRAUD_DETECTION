# FraudShield Production ML Backend

High-performance Python Flask REST API backend with trained machine learning models for real-time credit card fraud detection.

---

## Architecture Overview

```
CREDIT CARD FRAUD DETECTION/
├── backend/
│   ├── app.py                 # Flask REST API server & static file host
│   ├── train_models.py        # ML training pipeline on fraudTrain.csv & fraudTest.csv
│   ├── requirements.txt       # Dependencies (flask, xgboost, scikit-learn, etc.)
│   └── models/
│       ├── xgboost.joblib             # Trained XGBoost Classifier (ROC-AUC 0.9972)
│       ├── random_forest.joblib       # Trained Random Forest (ROC-AUC 0.9959)
│       ├── logistic_regression.joblib # Trained Logistic Regression (ROC-AUC 0.8755)
│       ├── scaler.joblib              # Feature standardizer
│       ├── category_encoder.joblib    # 14 merchant categories encoder
│       ├── state_encoder.joblib       # US state encoder
│       └── manifest.json              # Trained evaluation metrics & metadata
```

---

## Trained Model Metrics (Evaluated on Test Dataset)

| Classifier Model | Test Accuracy | Precision | Fraud Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost (Hist Gradient Boosting)** | **98.4%** | **85.2%** | **96.1%** | **90.3%** | **0.9972** | **0.9789** |
| **Random Forest (Balanced Trees)** | **98.6%** | **88.7%** | **93.9%** | **91.2%** | **0.9959** | **0.9717** |
| **Logistic Regression (L-BFGS)** | **94.4%** | **62.5%** | **73.8%** | **67.7%** | **0.8755** | **0.6580** |

---

## API Reference

### 1. User Authentication
* **Endpoint:** `POST /api/login`
* **Body:**
  ```json
  { "username": "admin", "password": "admin123" }
  ```
* **Response:**
  ```json
  {
    "success": true,
    "token": "GisqaQ8NVMi-ZZMDdks792wv3HO8uIq39272RDTKyn4",
    "user": "admin",
    "message": "Welcome, admin!"
  }
  ```

### 2. Live Transaction Fraud Scan
* **Endpoint:** `POST /api/scan`
* **Headers:** `Authorization: Bearer <token>` or `X-User: admin`
* **Body:**
  ```json
  {
    "model": "xgboost",
    "amt": 945.50,
    "category": "shopping_net",
    "channel": "cnp_online",
    "card_type": "visa",
    "cvv": "mismatch",
    "three_ds": "failed",
    "velocity": 8,
    "hour": 3,
    "day": "weekend",
    "state": "TX",
    "routing": "offshore",
    "merch_risk": "flagged",
    "age": "adult",
    "threshold": 0.196
  }
  ```
* **Response (Sample):**
  ```json
  {
    "success": true,
    "verdict": "FLAGGED AS FRAUD",
    "is_fraud": true,
    "probability": 99.8,
    "threshold": 0.196,
    "threshold_label": "19.6%",
    "model": "XGBOOST",
    "latency_ms": 3.84,
    "metrics": {
      "accuracy": "98.4%",
      "precision": "85.2%",
      "recall": "96.1%",
      "f1": "90.3%",
      "roc_auc": "0.9972",
      "pr_auc": "0.9789"
    },
    "shap": [
      { "feature": "Transaction Amount ($945.50)", "val": 7.879, "pct": 100, "positive": true },
      { "feature": "3D Secure Status (Challenge Failed)", "val": 2.85, "pct": 36, "positive": true },
      { "feature": "CVV Verification (Mismatch Code N)", "val": 2.45, "pct": 31, "positive": true }
    ]
  }
  ```

### 3. Model Manifest & Benchmarks
* **Endpoint:** `GET /api/models`
* Returns all evaluation metrics, ROC-AUC, PR-AUC, and feature metadata.

### 4. Dataset Macro Statistics
* **Endpoint:** `GET /api/stats`
* Returns category fraud percentages, class balance ratios, and macro velocity distributions.

### 5. Health Check
* **Endpoint:** `GET /api/health`

---

## How to Run

### Option 1: Double-click or terminal
```cmd
run.bat
```
*(macOS/Linux: `./run.sh`)*

### Option 2: Python command
```bash
python backend/app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.
