"""Probability models — heuristic (Phase 1/2) and ML stub (Phase 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# (dpd_lo, dpd_hi_inclusive, base_P)
_DPD_BUCKETS = [
    (1,   7,   0.85),
    (8,   30,  0.65),
    (31,  60,  0.45),
    (61,  90,  0.28),
    (91,  180, 0.12),
    (181, 9999, 0.05),
]

_REASON_SEVERITY: dict[str, float] = {
    "TCI":      0.95,
    "IDV_TEMP": 0.80,
    "IDV_LONG": 0.50,
    "SRI":      0.45,
    "TNC":      0.60,
    "MSD":      0.30,
    "LGL":      0.25,
    "ABS":      0.15,
    "WLD":      0.10,
    "SBL":      0.05,
    "STF":      0.05,
}

# Provision percentages per bucket — used by scoring.py for urgency boost
DPD_BUCKET_PROVISION: list[tuple[int, int, float]] = [
    (1,   7,   0.01),
    (8,   30,  0.05),
    (31,  60,  0.30),
    (61,  90,  0.50),
    (91,  180, 0.80),
    (181, 9999, 1.00),
]


def get_base_P(dpd: int) -> float:
    for lo, hi, base_p in _DPD_BUCKETS:
        if lo <= dpd <= hi:
            return base_p
    return 0.05


def get_provision_pct(dpd: int) -> float:
    for lo, hi, prov in DPD_BUCKET_PROVISION:
        if lo <= dpd <= hi:
            return prov
    return 1.00


def get_severity(reason_code: str) -> float:
    return _REASON_SEVERITY.get(reason_code, 0.50)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ProbabilityModel(ABC):
    @abstractmethod
    def predict(self, dpd: int, reason_code: str, **kwargs) -> float:
        """Returns P(recovery within 90 days), clamped to [0.02, 1.0]."""


# ---------------------------------------------------------------------------
# Phase 1/2 — heuristic model
# ---------------------------------------------------------------------------

class HeuristicModel(ProbabilityModel):
    """Provisioning-norm base probability × reason-code severity multiplier."""

    def predict(self, dpd: int, reason_code: str, **kwargs) -> float:
        base_p = get_base_P(dpd)
        severity = get_severity(reason_code)
        return min(1.0, max(0.02, base_p * severity))


# ---------------------------------------------------------------------------
# Phase 2 — XGBoost model (trained, not a stub)
# ---------------------------------------------------------------------------

class XGBoostModel(ProbabilityModel):
    """
    XGBoost classifier trained on historical visit outcomes.

    Expects a model saved by training/train.py (joblib pickle of a
    CalibratedClassifierCV wrapping XGBClassifier).

    Usage:
        model = XGBoostModel("models/xgb_recovery.pkl")
        engine = AllocationEngine(prob_model=model)
    """

    def __init__(self, model_path: str):
        try:
            import joblib
            self._model = joblib.load(model_path)
            self._model_path = model_path
        except Exception as exc:
            raise RuntimeError(
                f"Could not load XGBoost model from {model_path}: {exc}\n"
                f"Train first with: python -m training.train --synthetic"
            ) from exc

    def predict(self, dpd: int, reason_code: str, customer=None, **kwargs) -> float:
        """
        Predict P(recovery) using the full Phase 2 feature vector.

        Pass `customer` (a Customer instance) for the full feature set.
        Falls back to a simple DPD-only feature vector if customer is None.
        """
        import numpy as np
        from datetime import date

        if customer is not None:
            from .features import extract_features
            features = extract_features(customer, date.today())
        else:
            # Minimal fallback — only DPD available, pad remaining features
            from .features import FEATURE_NAMES
            features = [float(dpd)] + [0.0] * (len(FEATURE_NAMES) - 1)

        X = np.array([features], dtype=np.float32)
        prob = float(self._model.predict_proba(X)[0][1])
        return min(1.0, max(0.02, prob))

    def predict_batch(self, customers: list, today=None) -> list[float]:
        """
        Batch predict for a list of Customer objects.
        More efficient than calling predict() one-by-one.
        """
        import numpy as np
        from datetime import date
        from .features import extract_features_batch

        if today is None:
            today = date.today()

        X = np.array(extract_features_batch(customers, today), dtype=np.float32)
        probs = self._model.predict_proba(X)[:, 1]
        return [min(1.0, max(0.02, float(p))) for p in probs]


# ---------------------------------------------------------------------------
# Phase 3 — Online/Bandit model (future stub)
# ---------------------------------------------------------------------------

class MLModel(ProbabilityModel):
    """
    Legacy stub name — now an alias for XGBoostModel.
    Kept for backwards compatibility.
    """

    def __init__(self, model_path: str):
        self._inner = XGBoostModel(model_path)

    def predict(self, dpd: int, reason_code: str, **kwargs) -> float:
        return self._inner.predict(dpd, reason_code, **kwargs)
