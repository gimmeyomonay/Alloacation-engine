"""
Phase 2 — XGBoost model training pipeline.

Steps:
  1. Load labelled outcome data (CSV or generate synthetic)
  2. Train / evaluate XGBoost classifier
  3. Report AUC, log loss, calibration
  4. Save model to models/xgb_recovery.pkl

Usage:
  # Train on synthetic data (quick start)
  python -m training.train --synthetic --n 5000

  # Train on real outcome CSV
  python -m training.train --input training/outcomes.csv

  # Full options
  python -m training.train --input outcomes.csv --out models/xgb_recovery.pkl --tune
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, log_loss, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from allocation_engine.features import FEATURE_NAMES

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["did_pay_after_visit"].values.astype(int)
    return X, y


def generate_synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from training.synthetic_outcomes import generate_outcome_dataset
    from allocation_engine.features import FEATURE_NAMES
    rows = generate_outcome_dataset(n=n, seed=seed)
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=np.float32)
    y = np.array([r["did_pay_after_visit"] for r in rows], dtype=int)
    return X, y


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    tune: bool = False,
) -> "xgb.XGBClassifier":
    """Train and return a calibrated XGBoost classifier."""
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 5,
        "scale_pos_weight": pos_weight,
        "eval_metric":      "logloss",
        "use_label_encoder": False,
        "random_state":     42,
        "n_jobs":           -1,
    }

    if tune:
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = {
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.01, 0.05, 0.10],
            "n_estimators":     [100, 200, 300, 400],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [3, 5, 7],
        }
        base = xgb.XGBClassifier(
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        search = RandomizedSearchCV(
            base, param_dist, n_iter=30, cv=3,
            scoring="roc_auc", random_state=42, n_jobs=-1, verbose=1,
        )
        search.fit(X, y)
        print(f"Best params: {search.best_params_}")
        params.update(search.best_params_)

    clf = xgb.XGBClassifier(**params)

    # Isotonic calibration for well-calibrated probabilities
    calibrated = CalibratedClassifierCV(clf, cv=5, method="isotonic")
    calibrated.fit(X, y)
    return calibrated


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate on held-out test split. Returns metrics dict."""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    model.fit(X_tr, y_tr) if hasattr(model, "fit") else None

    # For the calibrated model we need to re-fit on train split
    # Re-train on train split for clean evaluation
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )
    cal = CalibratedClassifierCV(clf, cv=5, method="isotonic")
    cal.fit(X_tr, y_tr)

    probs = cal.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc      = roc_auc_score(y_te, probs)
    ll       = log_loss(y_te, probs)
    report   = classification_report(y_te, preds, output_dict=True)

    print(f"\n{'='*50}")
    print(f"  AUC:      {auc:.4f}  (target > 0.70)")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Precision (paid=1): {report['1']['precision']:.3f}")
    print(f"  Recall    (paid=1): {report['1']['recall']:.3f}")
    print(f"  F1        (paid=1): {report['1']['f1-score']:.3f}")
    print(f"{'='*50}\n")

    # Cross-validated AUC on full dataset
    clf2 = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )
    cv_aucs = cross_val_score(clf2, X, y, cv=5, scoring="roc_auc")
    print(f"  5-fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    return {
        "auc":       round(auc, 4),
        "log_loss":  round(ll, 4),
        "cv_auc":    round(float(cv_aucs.mean()), 4),
        "cv_auc_std": round(float(cv_aucs.std()), 4),
        "n_train":   len(X_tr),
        "n_test":    len(X_te),
        "pos_rate":  round(float(y.mean()), 3),
    }


def save_model(model, path: str) -> None:
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model) -> None:
    """Print top features from the underlying XGBoost estimator."""
    try:
        # Navigate through CalibratedClassifierCV wrappers
        estimators = model.calibrated_classifiers_
        xgb_clf = estimators[0].estimator
        imp = xgb_clf.feature_importances_
        pairs = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
        print("\nFeature importances (gain):")
        for name, score in pairs:
            bar = "█" * int(score * 40)
            print(f"  {name:<40} {score:.4f}  {bar}")
    except Exception:
        print("(Feature importance not available for this model type)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost recovery model")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--input",     type=str)
    parser.add_argument("--n",    type=int, default=5000, help="Synthetic records (default 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="models/xgb_recovery.pkl")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    args = parser.parse_args()

    print("Loading data...")
    if args.input:
        X, y = load_data(args.input)
    else:
        print(f"Generating {args.n} synthetic records (seed={args.seed})...")
        X, y = generate_synthetic(args.n, args.seed)

    print(f"Dataset: {len(X)} rows  |  paid rate: {y.mean():.1%}")
    print(f"Features: {len(FEATURE_NAMES)}")

    print("\nEvaluating model...")
    metrics = evaluate(None, X, y)

    if metrics["auc"] < 0.65:
        print("WARNING: AUC below 0.65 — check data quality before deploying.")

    print("\nTraining final model on full dataset...")
    model = train_xgboost(X, y, tune=args.tune)

    print_feature_importance(model)
    save_model(model, args.out)
    save_metrics(metrics, args.out.replace(".pkl", "_metrics.json"))

    print(f"\nDone. Deploy with: XGBoostModel('{args.out}')")
