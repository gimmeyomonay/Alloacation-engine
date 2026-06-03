"""
Generate a labelled synthetic outcome dataset for Phase 2 XGBoost training.

Each row is one historical customer visit with:
  - All 14 Phase 2 features
  - did_pay_after_visit label (0 / 1)

Recovery probability is governed by ground-truth rules that reflect
realistic collection dynamics — the model must learn these from data.

Usage:
  python -m training.synthetic_outcomes --n 5000 --out training/outcomes.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from allocation_engine.features import FEATURE_NAMES, extract_features
from allocation_engine.models import Customer

# ── Realistic reason code severity (matches HeuristicModel) ─────────────────
_SEVERITY = {
    "TCI": 0.95, "IDV_TEMP": 0.80, "TNC": 0.60, "IDV_LONG": 0.50,
    "SRI": 0.45, "MSD": 0.30, "LGL": 0.25, "ABS": 0.15, "WLD": 0.10,
    "SBL": 0.05, "STF": 0.05,
}

REASON_CODES   = list(_SEVERITY.keys())
REASON_WEIGHTS = [0.30, 0.18, 0.15, 0.08, 0.04, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02]

ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
LOAN_PRODUCTS = ["GL", "IL", "SHL", "SBL", "MSE", "LAP"]


def _ground_truth_prob(c: Customer, today: date) -> float:
    """
    Compute the true recovery probability used to generate labels.
    This is intentionally richer than the heuristic model so XGBoost
    learns non-linear patterns from the Phase 2 features.
    """
    # Base: DPD bucket
    dpd = c.dpd
    if dpd <= 7:       base = 0.82
    elif dpd <= 30:    base = 0.62
    elif dpd <= 60:    base = 0.42
    elif dpd <= 90:    base = 0.26
    elif dpd <= 180:   base = 0.11
    else:              base = 0.04

    # Reason code severity
    sev = _SEVERITY.get(c.reason_code, 0.50)
    p = base * sev

    # Payment behaviour boost/penalty
    if c.payment_count_last_90_days >= 3:
        p *= 1.30
    elif c.payment_count_last_90_days == 0 and c.contact_attempts >= 3:
        p *= 0.55

    # EMI ratio
    if c.emi_amount > 0:
        ratio = c.last_paid_amount / c.emi_amount
        if ratio >= 0.8:
            p *= 1.20
        elif ratio == 0:
            p *= 0.70

    # Stability
    if c.delinquency_cycle_count >= 3:
        p *= 0.75
    if c.times_rolled_back_to_current >= 2:
        p *= 0.85

    # Contactability
    if c.contact_status == "unreachable":
        p *= 0.40
    elif c.contact_status == "refused":
        p *= 0.30

    # PTP reliability
    if c.ptp_given > 0:
        ptp_rel = (c.ptp_kept + 1) / (c.ptp_given + 2)
        p *= (0.5 + ptp_rel)

    # Recent payment recency
    if c.last_paid_date:
        days_since = (today - c.last_paid_date).days
        if days_since <= 14:
            p *= 1.25
        elif days_since >= 180:
            p *= 0.80

    return min(0.97, max(0.02, p))


def _random_customer(rng: random.Random, np_rng: np.random.Generator, today: date, idx: int) -> Customer:
    dpd = int(np_rng.choice(
        [rng.randint(1, 7), rng.randint(8, 30), rng.randint(31, 60),
         rng.randint(61, 90), rng.randint(91, 180), rng.randint(181, 365)],
        p=[0.18, 0.26, 0.22, 0.16, 0.11, 0.07],
    ))
    osp = float(np.clip(np_rng.lognormal(math.log(45000), 0.85), 5000, 200000))
    rc  = rng.choices(REASON_CODES, weights=REASON_WEIGHTS)[0]

    emi = round(float(np.clip(np_rng.lognormal(math.log(osp / 12), 0.3), 500, 30000)), 2)
    missed = min(int(dpd // 30) + rng.randint(0, 2), 24)

    pay_count = max(0, int(np_rng.poisson(max(0, 3 - dpd / 60))))
    last_paid = osp * rng.uniform(0.05, 0.80) if pay_count > 0 else 0.0
    avg_paid  = last_paid * rng.uniform(0.7, 1.3)
    total_30  = last_paid * rng.uniform(0, 1.5) if pay_count > 0 else 0.0

    lp_days = rng.randint(1, min(dpd + 1, 365)) if pay_count > 0 else None
    last_paid_date = (today - timedelta(days=lp_days)) if lp_days else None

    delinq_cycles = rng.randint(0, min(3, dpd // 60 + 1))
    rolled_back   = rng.randint(0, min(2, delinq_cycles))
    months_del    = max(1, dpd // 30 + rng.randint(0, 6))

    contact_status = rng.choices(
        ["reachable", "unreachable", "refused"],
        weights=[0.70, 0.20, 0.10],
    )[0]
    contact_attempts = rng.randint(0, 8)
    lc_days = rng.randint(0, 30) if contact_attempts > 0 else None
    last_contact_date = (today - timedelta(days=lc_days)) if lc_days else None

    ptp_given  = rng.randint(0, 4)
    ptp_broken = rng.randint(0, ptp_given) if ptp_given else 0
    ptp_kept   = max(0, ptp_given - ptp_broken - rng.randint(0, max(0, ptp_given - ptp_broken)))

    return Customer(
        customer_id=f"H{idx:06d}",
        name=f"Customer {idx}",
        osp=round(osp, 2),
        dpd=dpd,
        due_date=today - timedelta(days=dpd),
        lat=None, lon=None,
        zone_id=rng.choice(ZONES),
        reason_code=rc,
        is_ots=rng.random() < 0.07,
        settlement_amount=round(osp * rng.uniform(0.5, 0.85), 2),
        last_visit_date=None,
        ptp_given=ptp_given, ptp_kept=ptp_kept, ptp_broken=ptp_broken,
        contact_attempts=contact_attempts,
        is_mandatory=False, is_msd_zone=(rc == "MSD"),
        loan_product=rng.choice(LOAN_PRODUCTS),
        # Phase 2
        emi_amount=emi,
        number_of_missed_installments=missed,
        next_due_amount=round(emi, 2),
        last_paid_amount=round(last_paid, 2),
        last_paid_date=last_paid_date,
        payment_count_last_90_days=pay_count,
        total_paid_last_30_days=round(total_30, 2),
        avg_payment_amount=round(avg_paid, 2),
        delinquency_cycle_count=delinq_cycles,
        times_rolled_back_to_current=rolled_back,
        months_since_first_delinquency=months_del,
        contact_status=contact_status,
        last_contact_date=last_contact_date,
    )


def generate_outcome_dataset(
    n: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate n labelled historical visit records.
    Returns list of dicts: features + label.
    """
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    today  = date(2026, 1, 1)   # fixed reference date for reproducibility

    rows = []
    for i in range(n):
        c     = _random_customer(rng, np_rng, today, i)
        p     = _ground_truth_prob(c, today)
        label = int(rng.random() < p)
        feats = extract_features(c, today)
        row   = dict(zip(FEATURE_NAMES, feats))
        row["did_pay_after_visit"] = label
        rows.append(row)

    return rows


def save_csv(rows: list[dict], path: str) -> None:
    cols = FEATURE_NAMES + ["did_pay_after_visit"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="training/outcomes.csv")
    args = parser.parse_args()

    rows = generate_outcome_dataset(n=args.n, seed=args.seed)
    paid = sum(r["did_pay_after_visit"] for r in rows)
    print(f"Generated {len(rows)} records  |  paid={paid} ({paid/len(rows)*100:.1f}%)")
    save_csv(rows, args.out)
