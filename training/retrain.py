"""
Phase 2 — Retraining pipeline from feedback log.

Loads feedback_log.jsonl, builds a training dataset from stored feature
snapshots and outcome labels, retrains XGBoost, evaluates AUC, and saves a
new versioned model.

Usage:
    python -m training.retrain
    python -m training.retrain --min-records 200 --out models/xgb_recovery.pkl
    python -m training.retrain --dry-run      # evaluate only, don't save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from allocation_engine.features import FEATURE_NAMES
from allocation_engine.feedback import load_feedback_log
from allocation_engine.model_registry import ModelRegistry
from training.train import train_xgboost, evaluate, save_model, save_metrics, print_feature_importance

_DEFAULT_MIN_RECORDS = 100


def build_dataset_from_feedback(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert feedback log records into (X, y) arrays for training.

    Each record must have:
      - features_at_time: dict with the 14 feature values logged at visit time
      - did_pay_after_visit: bool label
    """
    X_rows, y_rows = [], []
    skipped = 0

    for rec in records:
        feat = rec.get("features_at_time", {})
        if not feat:
            skipped += 1
            continue

        row = []
        valid = True
        for name in FEATURE_NAMES:
            if name not in feat:
                skipped += 1
                valid = False
                break
            row.append(float(feat[name]))

        if valid:
            X_rows.append(row)
            y_rows.append(int(rec["did_pay_after_visit"]))

    if skipped:
        print(f"  Skipped {skipped} records with missing features.")

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=int),
    )


def run_retrain(
    log_path: Path | None = None,
    out_path: str = "models/xgb_recovery.pkl",
    min_records: int = _DEFAULT_MIN_RECORDS,
    dry_run: bool = False,
) -> dict:
    """
    Main retraining entry point.

    Returns a dict with {"auc", "n_records", "version_id", "model_path"}.
    """
    registry = ModelRegistry()

    print("Loading feedback log...")
    records = load_feedback_log(log_path)
    print(f"  Found {len(records)} feedback records.")

    if len(records) < min_records:
        raise ValueError(
            f"Only {len(records)} records in feedback log — "
            f"need at least {min_records} to retrain. "
            f"Use --min-records to lower the threshold, or collect more outcomes."
        )

    print("Building training dataset from feedback features...")
    X, y = build_dataset_from_feedback(records)
    n = len(X)

    if n < min_records:
        raise ValueError(
            f"Only {n} usable records after filtering — need {min_records}. "
            f"Ensure features_at_time is logged in each outcome record."
        )

    print(f"  Dataset: {n} records  |  paid rate: {y.mean():.1%}")

    print("\nEvaluating on hold-out split...")
    metrics = evaluate(None, X, y)
    auc = metrics["auc"]

    if auc < 0.60:
        print(f"\nWARNING: AUC = {auc:.4f} — below 0.60. Data quality may be low.")
        print("Proceeding, but review results before deploying.")

    if dry_run:
        print("\n[Dry run] Skipping model save and registry update.")
        return {"auc": auc, "n_records": n, "version_id": None, "model_path": None}

    print("\nTraining final model on full dataset...")
    model = train_xgboost(X, y)
    print_feature_importance(model)

    save_model(model, out_path)
    save_metrics(metrics, out_path.replace(".pkl", "_metrics.json"))

    version_id = registry.register(
        model_path=out_path,
        auc=auc,
        n_records=n,
        make_active=True,
        notes=f"Retrained from {n} feedback records",
    )

    print(f"\nRegistered as {version_id} (active).")
    print(f"Deploy with: XGBoostModel('{out_path}')")

    return {
        "auc":        auc,
        "n_records":  n,
        "version_id": version_id,
        "model_path": out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain XGBoost from feedback log")
    parser.add_argument("--log",         type=str, default=None, help="Path to feedback_log.jsonl")
    parser.add_argument("--out",         type=str, default="models/xgb_recovery.pkl")
    parser.add_argument("--min-records", type=int, default=_DEFAULT_MIN_RECORDS)
    parser.add_argument("--dry-run",     action="store_true", help="Evaluate only — don't save or register")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None

    result = run_retrain(
        log_path=log_path,
        out_path=args.out,
        min_records=args.min_records,
        dry_run=args.dry_run,
    )

    print("\nResult:", result)
