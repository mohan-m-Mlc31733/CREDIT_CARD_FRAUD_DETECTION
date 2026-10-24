# FraudShield – Credit Card Fraud Detection

Machine-learning credit card fraud detection (XGBoost, Random Forest, Logistic Regression) with SHAP explainability, a live transaction scanner, transaction-ID lookup and batch CSV scanning.

**Live site:** `https://<your-github-username>.github.io/<repo-name>/`

---

## Project structure

```
├── index.html            Home
├── process.html          ML pipeline
├── services.html         Models, figures and SHAP
├── testimonials.html
├── contact.html
├── login.html            Sign-in for the scanner
├── scanner.html          Live fraud scanner (single / transaction ID / batch CSV)
├── 404.html              Sends unknown links back to the home page
├── css/shared.css
├── js/
│   ├── config.js         ← set API_BASE here if you host the Python backend
│   ├── static-api.js     Makes the scanner work on GitHub Pages without a server
│   └── main.js
├── sample_dataset_test.csv   100 sample transactions used by the scanner
├── results/              Evaluation tables + figures (shown on services.html)
├── artifacts/            Colab-trained XGBoost pipeline + thresholds
├── backend/              Flask API, trained models, training script
├── paper/                IEEE paper (.docx)
├── data/                 Put fraudTrain.csv / fraudTest.csv here to retrain (not uploaded)
├── render.yaml           Optional: one-click backend hosting on Render
├── run.bat / run.sh      Run everything locally
└── .nojekyll             Tells GitHub Pages to serve files as-is
```

---

## Publish on GitHub Pages (direct website link)

1. Create a new **public** repository on GitHub, e.g. `fraudshield` (no spaces in the name).
2. Upload the **contents** of this folder to the repository root (so `index.html` is at the top level, not inside another folder).
   - Web: *Add file → Upload files* → drag everything in → *Commit changes*.
   - Or with git:
     ```bash
     git init
     git add .
     git commit -m "FraudShield website"
     git branch -M main
     git remote add origin https://github.com/<username>/fraudshield.git
     git push -u origin main
     ```
3. In the repository go to **Settings → Pages**.
   Under *Build and deployment* choose **Source: Deploy from a branch**, **Branch: `main`**, folder **`/ (root)`**, then **Save**.
4. Wait 1–2 minutes and refresh. Your link appears at the top of that page:
   **`https://<username>.github.io/fraudshield/`**

Tip: if you name the repository exactly `<username>.github.io`, the site is served at `https://<username>.github.io/` with no sub-path.

### How the scanner works on GitHub Pages
GitHub Pages only hosts static files, so it cannot run Python. `js/static-api.js` answers the scanner's `/api/*` requests in the browser using the same response format as the Flask backend, so every scanner feature works on the live link. A small "Browser demo mode" badge shows on the scanner page when this is in use. To run the real trained models online, host the backend (below).

---

## Optional: use the real trained models online (Render, free)

1. Sign in at [render.com](https://render.com) with GitHub → **New → Blueprint** → select this repository. Render reads `render.yaml` and deploys the backend.
2. When it is live, copy its URL (e.g. `https://fraudshield-api.onrender.com`).
3. Open `js/config.js` on GitHub, set
   ```js
   API_BASE: "https://fraudshield-api.onrender.com"
   ```
   and commit. The GitHub Pages site now uses the Python models. If the backend is asleep or unreachable, the site falls back to browser mode automatically.

The Render URL also serves the complete website by itself, so it works as a second direct link.

---

## Run locally (full Python backend)

```bash
pip install -r backend/requirements.txt
python backend/app.py          # or double-click run.bat (Windows) / ./run.sh
```
Open http://127.0.0.1:8000

Demo logins: `admin / admin123`, `analyst / fraud2024`, `demo / demo` (any username ≥ 3 chars and password ≥ 4 chars also works).

### Retrain the models
Download `fraudTrain.csv` and `fraudTest.csv` from the [Kaggle fraud-detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection), put them in `data/`, then:
```bash
python backend/train_models.py
```

---

## Results (Colab run, 20,000-row test sample)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost (selected) | 55.8% | 73.6% | 63.5% | 0.9945 | 0.6787 |
| Random Forest | 73.3% | 69.2% | 71.2% | 0.9549 | 0.6965 |
| Logistic Regression | 25.8% | 50.6% | 34.2% | 0.8147 | 0.2084 |

Full tables and figures are in `results/`. API reference: `backend/README.md`.

**Author:** Mohan Mekala
