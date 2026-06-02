"""
Per-customer value scoring:
  V_i, urgency_boost, V'_i (adjusted), interaction_time.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .models import Customer
from .probability import get_base_P, get_provision_pct, ProbabilityModel

# DPD bucket boundaries (lower bounds) — used to find "next bucket"
_BUCKET_LOWER_BOUNDS = [1, 8, 31, 61, 91, 181]


def _next_bucket_boundary(dpd: int) -> Optional[int]:
    """Return the DPD value of the next bucket boundary, or None if already in last bucket."""
    for boundary in _BUCKET_LOWER_BOUNDS:
        if dpd < boundary:
            return boundary
    return None


def compute_urgency_boost(dpd: int, due_date: date, amount: float, today: date) -> float:
    """
    If the account will cross a DPD bucket boundary within 3 days, return the
    incremental provisioning cost as an urgency boost.
    """
    next_boundary = _next_bucket_boundary(dpd)
    if next_boundary is None:
        return 0.0

    days_to_boundary = next_boundary - dpd
    if days_to_boundary > 3:
        return 0.0

    current_prov = get_provision_pct(dpd)
    next_prov = get_provision_pct(next_boundary)
    delta_provision = next_prov - current_prov
    return delta_provision * amount


def compute_repeat_penalty_decay(last_visit_date: Optional[date], today: date,
                                  decay_days: int = 30) -> float:
    """Returns the decay multiplier [0.0, 1.0] for the repeat-visit penalty."""
    if last_visit_date is None:
        days_since = 999
    else:
        days_since = (today - last_visit_date).days
    return max(0.0, 1.0 - days_since / decay_days)


def compute_ptp_multiplier(ptp_given: int, ptp_kept: int, contact_attempts: int) -> float:
    """Laplace-smoothed PTP reliability, with unresponsive-customer penalty."""
    if ptp_given == 0 and contact_attempts >= 3:
        return 0.40  # unresponsive
    return (ptp_kept + 1) / (ptp_given + 2)


def compute_V_adj(
    V_i: float,
    urgency_boost: float,
    last_visit_date: Optional[date],
    today: date,
    ptp_given: int,
    ptp_kept: int,
    contact_attempts: int,
    repeat_penalty_coeff: float = 0.35,
    decay_days: int = 30,
) -> float:
    decay = compute_repeat_penalty_decay(last_visit_date, today, decay_days)
    ptp_mult = compute_ptp_multiplier(ptp_given, ptp_kept, contact_attempts)
    return (V_i * (1 - repeat_penalty_coeff * decay) * ptp_mult) + urgency_boost


def compute_interaction_time(reason_code: str, dpd: int) -> float:
    """Returns estimated interaction time in minutes."""
    if dpd <= 7:
        return 20.0
    if reason_code in ("LGL", "ABS", "MSD"):
        return 60.0
    if reason_code in ("IDV_LONG", "WLD"):
        return 45.0
    return 30.0


def score_customer(
    customer: Customer,
    prob_model: ProbabilityModel,
    today: date,
    repeat_penalty_coeff: float = 0.35,
    decay_days: int = 30,
) -> dict:
    """
    Compute all scoring fields for a single customer.

    Returns a dict with keys:
        probability, V_i, urgency_boost, V_adj, interaction_min
    """
    probability = prob_model.predict(customer.dpd, customer.reason_code)
    V_i = customer.amount * probability

    urgency_boost = compute_urgency_boost(
        customer.dpd, customer.due_date, customer.amount, today
    )

    V_adj = compute_V_adj(
        V_i=V_i,
        urgency_boost=urgency_boost,
        last_visit_date=customer.last_visit_date,
        today=today,
        ptp_given=customer.ptp_given,
        ptp_kept=customer.ptp_kept,
        contact_attempts=customer.contact_attempts,
        repeat_penalty_coeff=repeat_penalty_coeff,
        decay_days=decay_days,
    )

    interaction_min = compute_interaction_time(customer.reason_code, customer.dpd)

    return {
        "probability":    probability,
        "V_i":            V_i,
        "urgency_boost":  urgency_boost,
        "V_adj":          V_adj,
        "interaction_min": interaction_min,
    }
