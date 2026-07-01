"""
Retrain the 5-fold IW ensemble classifier.

Requires the training data files that are NOT included in this repo:
  models/roberta_ft_embeddings.npz   — (166, 768) FT RoBERTa CLS vectors
  models/token_pause_features.npz     — train/test pause features (optional)
  models/test_features_iw.pkl         — (71, 768) test RoBERTa vectors

These are stored on the CCI training server at:
  /mnt/afs/L202500480/train/processed/roberta_ft_embeddings.npz
  /mnt/afs/L202500480/train/processed_v9/token_pause/token_pause_features.npz
  /mnt/afs/L202500480/train/processed_v9/test_features_iw.pkl

Outputs (overwrites models/classifier/):
  iw_fold_models.pkl   — 5 LogisticRegression models
  iw_fold_scalers.pkl  — 5 StandardScalers (one per fold)
  config.json          — performance metadata
"""
import numpy as np, pickle, json, csv
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
CLASSIFIER_DIR = MODEL_DIR / "classifier"

# ── Load data ──
print("Loading training data...")
ft = np.load(MODEL_DIR / "roberta_ft_embeddings.npz")
X_train = ft["embeddings"].astype(np.float64)
y_train = ft["labels"].astype(int)

with open(MODEL_DIR / "test_features_iw.pkl", "rb") as f:
    test_data = pickle.load(f)
X_test = np.stack([s["roberta"] for s in test_data]).astype(np.float64)

# Labels (if CSV available)
labels_csv = MODEL_DIR.parent / "data" / "task1.csv"
if labels_csv.exists():
    with open(labels_csv) as f:
        labels_map = {}
        for row in csv.DictReader(f):
            dx = row["Dx"].strip().strip('"')
            labels_map[row["ID"]] = 1 if dx in ("ProbableAD", "AD") else 0
    y_test = np.array([labels_map.get(s["file_id"], 0) for s in test_data])
else:
    y_test = np.zeros(len(test_data))  # no labels for eval

print(f"Train: {len(y_train)} (AD={y_train.sum()})  Test: {len(y_test)}")

# ── 5-fold IW CV ──
C_dom, C_AD = 0.20, 0.15
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
fold_models = []
fold_scalers = []

print(f"\n5-fold IW CV (C_dom={C_dom}, C_AD={C_AD})...")
for fold_i, (tr_idx, val_idx) in enumerate(cv.split(range(len(y_train)), y_train)):
    X_tr_raw = X_train[tr_idx]
    X_val_raw = X_train[val_idx]
    y_tr = y_train[tr_idx]
    y_val = y_train[val_idx]

    fs = StandardScaler()
    X_tr_s = fs.fit_transform(X_tr_raw)
    X_val_s = fs.transform(X_val_raw)
    X_te_s = fs.transform(X_test)
    fold_scalers.append(fs)

    # Domain classifier (balanced)
    n_bal = min(len(X_tr_s), len(X_test))
    rng = np.random.RandomState(42 + fold_i)
    bal_idx = rng.choice(len(X_tr_s), n_bal, replace=False)

    ds = StandardScaler()
    X_dom = ds.fit_transform(np.vstack([X_tr_s[bal_idx], X_te_s]))
    y_dom = np.concatenate([np.ones(n_bal), np.zeros(len(X_te_s))])
    dom = LogisticRegression(penalty="l2", C=C_dom, solver="lbfgs",
                             max_iter=5000, tol=1e-4, random_state=42)
    dom.fit(X_dom, y_dom)

    # IW weights
    X_tr_dom = ds.transform(X_tr_s)
    eta = np.clip(dom.predict_proba(X_tr_dom)[:, 1], 1e-8, 1 - 1e-8)
    w = (1.0 - eta) / eta
    w = w / w.mean()
    w = np.clip(w, 0.1, 10.0)

    # AD classifier
    clf = LogisticRegression(penalty="l2", C=C_AD, solver="lbfgs",
                             max_iter=5000, tol=1e-4, random_state=42)
    clf.fit(X_tr_s, y_tr, sample_weight=w)
    fold_models.append(clf)

    prob_val = clf.predict_proba(X_val_s)[:, 1]
    print(f"  Fold {fold_i+1}: val_acc={accuracy_score(y_val, prob_val > 0.5):.4f} "
          f"val_f1={f1_score(y_val, prob_val > 0.5):.4f}")

# ── Ensemble evaluation ──
test_probs = np.zeros(len(y_test))
for clf_i, scaler_i in zip(fold_models, fold_scalers):
    X_te_s = scaler_i.transform(X_test)
    test_probs += clf_i.predict_proba(X_te_s)[:, 1]
test_probs /= len(fold_models)
test_pred = (test_probs > 0.5).astype(int)

if y_test.sum() > 0:
    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_probs)
    print(f"\nEnsemble: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

# ── Save ──
print(f"\nSaving to {CLASSIFIER_DIR}...")
CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
with open(CLASSIFIER_DIR / "iw_fold_models.pkl", "wb") as f:
    pickle.dump(fold_models, f)
with open(CLASSIFIER_DIR / "iw_fold_scalers.pkl", "wb") as f:
    pickle.dump(fold_scalers, f)

config = {
    "method": "IW",
    "features": "5-fold IW ensemble, RoBERTa(768d)",
    "C_dom": C_dom, "C_AD": C_AD,
    "test_acc": float(acc) if y_test.sum() > 0 else None,
    "test_f1": float(f1) if y_test.sum() > 0 else None,
    "test_auc": float(auc) if y_test.sum() > 0 else None,
}
with open(CLASSIFIER_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Done.")
