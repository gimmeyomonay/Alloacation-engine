"""
5-day watch list and mid-day replan.
"""

from __future__ import annotations

from datetime import date

from .models import Customer, VisitPlan, VisitOutcome, WatchItem
from .probability import get_base_P, get_severity, get_provision_pct

# DPD bucket lower-bounds (used to find crossing within horizon)
_BUCKET_LOWER_BOUNDS = [1, 8, 31, 61, 91, 181]


def _next_bucket_boundary(dpd: int) -> int | None:
    for b in _BUCKET_LOWER_BOUNDS:
        if dpd < b:
            return b
    return None


def _projected_urgency(projected_dpd: int, amount: float) -> float:
    boundary = _next_bucket_boundary(projected_dpd)
    if boundary is None:
        return 0.0
    days_to = boundary - projected_dpd
    if days_to > 3:
        return 0.0
    current_prov = get_provision_pct(projected_dpd)
    next_prov = get_provision_pct(boundary)
    return (next_prov - current_prov) * amount


def project_dpd_trajectory(
    customer: Customer,
    horizon_days: int = 5,
) -> list[dict]:
    """
    Return a list of per-day projections for days 1..horizon_days.
    Each entry: {day, projected_dpd, projected_P, projected_V, projected_urgency}
    """
    projections = []
    for day_k in range(1, horizon_days + 1):
        projected_dpd = customer.dpd + day_k
        projected_P = get_base_P(projected_dpd) * get_severity(customer.reason_code)
        projected_P = min(1.0, max(0.02, projected_P))
        projected_V = customer.amount * projected_P
        projected_urgency = _projected_urgency(projected_dpd, customer.amount)
        projections.append({
            "day":               day_k,
            "projected_dpd":     projected_dpd,
            "projected_P":       projected_P,
            "projected_V":       projected_V,
            "projected_urgency": projected_urgency,
        })
    return projections


def build_watch_list(
    customers: list[Customer],
    unselected_indices: list[int],
    horizon_days: int = 5,
) -> list[WatchItem]:
    """
    Customers not selected today who will cross a DPD bucket boundary within
    horizon_days. Ranked by (projected_urgency + projected_V) descending.
    """
    items: list[WatchItem] = []

    for idx in unselected_indices:
        c = customers[idx]
        boundary = _next_bucket_boundary(c.dpd)
        if boundary is None:
            continue

        days_to_boundary = boundary - c.dpd
        if days_to_boundary > horizon_days:
            continue

        # Use projection at the crossing day
        crossing_day = days_to_boundary
        proj = project_dpd_trajectory(c, horizon_days=crossing_day)[-1]

        score = proj["projected_urgency"] + proj["projected_V"]
        items.append(WatchItem(
            customer_id=c.customer_id,
            name=c.name,
            osp=c.osp,
            current_dpd=c.dpd,
            days_to_boundary=days_to_boundary,
            projected_dpd=proj["projected_dpd"],
            projected_V=proj["projected_V"],
            projected_urgency=proj["projected_urgency"],
            score=score,
        ))

    items.sort(key=lambda w: w.score, reverse=True)
    return items


# ---------------------------------------------------------------------------
# Mid-day replan
# ---------------------------------------------------------------------------

def replan(
    remaining_budget_minutes: float,
    completed_outcomes: list[VisitOutcome],
    remaining_customers: list[Customer],
    engine,                   # AllocationEngine — avoids circular import
) -> VisitPlan:
    """
    Re-run the allocation pipeline on remaining customers with updated states.

    Outcome effects:
      - "recovered"          → remove from pool (already done by caller passing remaining_customers)
      - "ptp_given"          → no structural change; engine re-scores naturally
      - "confirmed_abscond"  → flip reason_code to ABS; engine will re-score
      - "no_contact"         → no change

    TRIGGER: call when XYZ posts a visit-complete event.
    Until XYZ integration is ready, this is called once at day start only.
    """
    outcome_map = {o.customer_id: o for o in completed_outcomes}

    updated: list[Customer] = []
    for c in remaining_customers:
        outcome = outcome_map.get(c.customer_id)
        if outcome is None:
            updated.append(c)
            continue
        if outcome.outcome == "recovered":
            continue  # drop from pool
        if outcome.outcome == "confirmed_abscond":
            from dataclasses import replace
            c = replace(c, reason_code="ABS", is_mandatory=True)
        updated.append(c)

    # Override budget for this replan run
    original_budget = engine.config.daily_budget_minutes
    engine.config.daily_budget_minutes = remaining_budget_minutes
    plan = engine.run(updated)
    engine.config.daily_budget_minutes = original_budget
    return plan
