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
# Phase 3 — ML model stub
# ---------------------------------------------------------------------------

class MLModel(ProbabilityModel):
    """
    XGBoost model trained on historical outcomes.
    Swap in once 6-month outcome data is available.
    """

    def __init__(self, model_path: str):
        # Lazy import so scikit/xgboost aren't required until this class is used
        try:
            import joblib
            self._model = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(f"Could not load ML model from {model_path}: {exc}") from exc

    def predict(
        self,
        dpd: int,
        reason_code: str,
        osp: float = 0.0,
        ptp_reliability: float = 0.5,
        days_since_contact: int = 0,
        mob: int = 0,
        prior_writeoff: bool = False,
        bureau_score: float = 0.0,
        **kwargs,
    ) -> float:
        import numpy as np

        features = np.array([[
            dpd,
            osp,
            ptp_reliability,
            days_since_contact,
            mob,
            int(prior_writeoff),
            bureau_score,
        ]])
        prob = float(self._model.predict_proba(features)[0][1])
        return min(1.0, max(0.02, prob))
