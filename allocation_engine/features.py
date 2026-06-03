"""
Phase 2 — Feature engineering layer.

Derives the full ML feature vector from a Customer record.
CRITICAL: This exact logic must be used for BOTH training and serving —
          never derive features differently in each place.

Feature vector (14 features, all numeric):
  [0]  dpd
  [1]  days_since_last_payment      (999 if never paid)
  [2]  payment_to_emi_ratio         (capped at 2.0; 0 if emi_amount == 0)
  [3]  payment_frequency            (payments per day over 90-day window)
  [4]  delinquency_depth            (composite: dpd bucket + missed EMIs)
  [5]  emi_amount                   (log-scaled)
  [6]  number_of_missed_installments
  [7]  payment_count_last_90_days
  [8]  avg_payment_amount           (log-scaled)
  [9]  delinquency_cycle_count
  [10] times_rolled_back_to_current
  [11] months_since_first_delinquency
  [12] contact_attempt_count        (maps to contact_attempts field)
  [13] days_since_last_contact      (999 if never contacted)
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

from .models import Customer

# Feature names — used for training DataFrames and model explainability
FEATURE_NAMES = [
    "dpd",
    "days_since_last_payment",
    "payment_to_emi_ratio",
    "payment_frequency",
    "delinquency_depth",
    "emi_amount_log",
    "number_of_missed_installments",
    "payment_count_last_90_days",
    "avg_payment_amount_log",
    "delinquency_cycle_count",
    "times_rolled_back_to_current",
    "months_since_first_delinquency",
    "contact_attempt_count",
    "days_since_last_contact",
]

# Contactability encoding
_CONTACT_STATUS_MAP = {
    "reachable":   0,
    "unreachable": 1,
    "refused":     2,
}


def _safe_log(x: float) -> float:
    return math.log1p(max(0.0, x))


def _days_since(d: Optional[date], today: date, default: int = 999) -> int:
    if d is None:
        return default
    return max(0, (today - d).days)


def _delinquency_depth(dpd: int, missed_installments: int) -> float:
    """
    Composite delinquency depth:
    - DPD bucket index (0–5) normalised to [0,1]
    - Missed installments normalised (capped at 24 months)
    Combined with equal weight.
    """
    bucket_thresholds = [7, 30, 60, 90, 180]
    bucket_idx = sum(1 for t in bucket_thresholds if dpd > t)  # 0..5
    dpd_norm = bucket_idx / 5.0
    emi_norm = min(missed_installments, 24) / 24.0
    return round((dpd_norm + emi_norm) / 2.0, 4)


def extract_features(customer: Customer, today: Optional[date] = None) -> list[float]:
    """
    Compute the 14-element feature vector for one customer.
    Returns a plain Python list of floats (compatible with XGBoost and sklearn).
    """
    if today is None:
        today = date.today()

    c = customer

    days_since_payment = _days_since(c.last_paid_date, today, default=999)

    if c.emi_amount > 0:
        payment_to_emi = min(c.last_paid_amount / c.emi_amount, 2.0)
    else:
        payment_to_emi = 0.0

    payment_frequency = c.payment_count_last_90_days / 90.0

    depth = _delinquency_depth(c.dpd, c.number_of_missed_installments)

    days_since_contact = _days_since(c.last_contact_date, today, default=999)

    return [
        float(c.dpd),
        float(days_since_payment),
        float(payment_to_emi),
        float(payment_frequency),
        float(depth),
        _safe_log(c.emi_amount),
        float(c.number_of_missed_installments),
        float(c.payment_count_last_90_days),
        _safe_log(c.avg_payment_amount),
        float(c.delinquency_cycle_count),
        float(c.times_rolled_back_to_current),
        float(c.months_since_first_delinquency),
        float(c.contact_attempts),
        float(days_since_contact),
    ]


def extract_features_batch(
    customers: list[Customer],
    today: Optional[date] = None,
) -> list[list[float]]:
    """Extract feature vectors for a list of customers. Returns list of lists."""
    if today is None:
        today = date.today()
    return [extract_features(c, today) for c in customers]


def features_to_dict(customer: Customer, today: Optional[date] = None) -> dict:
    """Return features as a named dict — useful for logging and debugging."""
    vec = extract_features(customer, today)
    return dict(zip(FEATURE_NAMES, vec))
