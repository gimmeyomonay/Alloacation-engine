"""Synthetic portfolio generator for allocation engine testing."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Zone definitions — 5 geographic clusters simulating a mid-size Indian city
# ---------------------------------------------------------------------------
ZONES = {
    "Z1": (12.9716, 77.5946),   # Bangalore city centre
    "Z2": (12.9352, 77.6245),   # South-east
    "Z3": (13.0100, 77.5500),   # North-west
    "Z4": (12.9850, 77.7200),   # East
    "Z5": (12.9200, 77.5500),   # South-west
}

REASON_CODES = ["IDV_TEMP", "IDV_LONG", "TNC", "WLD", "TCI", "ABS", "MSD", "SRI", "LGL", "SBL", "STF"]

# Realistic distribution weights (must sum to 1.0)
REASON_WEIGHTS = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01]

LOAN_PRODUCTS = ["GL", "IL", "SHL", "SBL", "MSE", "LAP"]
LOAN_PRODUCT_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]

FIRST_NAMES = [
    "Ravi", "Priya", "Suresh", "Anjali", "Mohan", "Lakshmi", "Vijay", "Meena",
    "Arun", "Sunita", "Ramesh", "Kavitha", "Sanjay", "Deepa", "Kiran", "Usha",
    "Amit", "Rekha", "Rahul", "Geeta", "Ashok", "Nirmala", "Prakash", "Savitha",
    "Ganesh", "Padma", "Sunil", "Radha", "Manoj", "Sarala",
]
LAST_NAMES = [
    "Kumar", "Sharma", "Reddy", "Naik", "Patil", "Hegde", "Rao", "Gowda",
    "Joshi", "Iyer", "Pillai", "Menon", "Shetty", "Nair", "Krishnan",
]

# DPD bucket boundaries (upper-exclusive)
DPD_BUCKETS = [
    (1,   7,   0.85, 0.01),
    (8,   30,  0.65, 0.05),
    (31,  60,  0.45, 0.30),
    (61,  90,  0.28, 0.50),
    (91,  180, 0.12, 0.80),
    (181, 365, 0.05, 1.00),
]

# Weight each bucket so the portfolio looks realistic
DPD_BUCKET_WEIGHTS = [0.15, 0.25, 0.20, 0.18, 0.14, 0.08]


def _random_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _sample_dpd(rng: random.Random) -> int:
    bucket = rng.choices(DPD_BUCKETS, weights=DPD_BUCKET_WEIGHTS, k=1)[0]
    lo, hi = bucket[0], bucket[1]
    return rng.randint(lo, hi)


def _gps_for_zone(rng: random.Random, zone_id: str, missing: bool) -> tuple[Optional[float], Optional[float]]:
    if missing:
        return None, None
    clat, clon = ZONES[zone_id]
    # ~0.02 degree spread ≈ 2 km radius
    lat = clat + rng.gauss(0, 0.015)
    lon = clon + rng.gauss(0, 0.015)
    return round(lat, 6), round(lon, 6)


def _osp_lognormal(rng: random.Random, np_rng: np.random.Generator) -> float:
    """Log-normal OSP in ₹5,000–₹2,00,000 range."""
    # mean=ln(40000), sigma=0.9 gives a good spread
    val = np_rng.lognormal(mean=math.log(40_000), sigma=0.9)
    val = float(np.clip(val, 5_000, 2_00_000))
    return round(val, 2)


def _due_date_from_dpd(today: date, dpd: int) -> date:
    return today - timedelta(days=dpd)


def _last_visit(rng: random.Random, today: date, has_recent: bool) -> Optional[date]:
    if not has_recent:
        # 50% chance of never visited, otherwise a random old date
        if rng.random() < 0.50:
            return None
        days_ago = rng.randint(31, 180)
    else:
        days_ago = rng.randint(1, 30)
    return today - timedelta(days=days_ago)


def generate_synthetic_portfolio(n: int = 70, seed: int = 42) -> list[dict]:
    """
    Returns a list of customer dicts matching the engine's data schema.

    Parameters
    ----------
    n    : number of customers (recommended 60–80)
    seed : random seed for reproducibility
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    today = date.today()

    zone_ids = list(ZONES.keys())

    # Pre-compute per-customer flags to hit target ratios
    mandatory_flags  = [i < round(n * 0.15) for i in range(n)]
    msd_flags        = [i < round(n * 0.10) for i in range(n)]
    ptp_broken_flags = [i < round(n * 0.20) for i in range(n)]
    recent_visit_flags = [i < round(n * 0.30) for i in range(n)]
    gps_missing_flags  = [i < round(n * 0.075) for i in range(n)]

    rng.shuffle(mandatory_flags)
    rng.shuffle(msd_flags)
    rng.shuffle(ptp_broken_flags)
    rng.shuffle(recent_visit_flags)
    rng.shuffle(gps_missing_flags)

    customers = []

    for i in range(n):
        cid = f"CUST{1000 + i:04d}"
        name = _random_name(rng)
        dpd = _sample_dpd(rng)
        osp = _osp_lognormal(rng, np_rng)
        is_msd = msd_flags[i]
        is_mandatory = mandatory_flags[i]

        # Force MSD accounts to use MSD reason code
        if is_msd:
            reason_code = "MSD"
        else:
            reason_code = rng.choices(
                [rc for rc in REASON_CODES if rc != "MSD"],
                weights=[w for rc, w in zip(REASON_CODES, REASON_WEIGHTS) if rc != "MSD"],
                k=1,
            )[0]

        # Mandatory accounts: bias toward WLD or ABS
        if is_mandatory and not is_msd:
            reason_code = rng.choices(["WLD", "ABS", reason_code], weights=[0.4, 0.3, 0.3], k=1)[0]

        zone_id = rng.choice(zone_ids)
        lat, lon = _gps_for_zone(rng, zone_id, gps_missing_flags[i])

        due_date = _due_date_from_dpd(today, dpd)

        is_ots = rng.random() < 0.08  # ~8% OTS
        settlement_amount = round(osp * rng.uniform(0.50, 0.85), 2) if is_ots else 0.0

        has_recent = recent_visit_flags[i]
        last_visit_date = _last_visit(rng, today, has_recent)

        # PTP history
        ptp_given = rng.randint(0, 5)
        if ptp_broken_flags[i] and ptp_given > 0:
            ptp_broken = rng.randint(1, min(3, ptp_given))
        else:
            ptp_broken = 0
        ptp_kept = max(0, ptp_given - ptp_broken - rng.randint(0, max(0, ptp_given - ptp_broken)))

        contact_attempts = rng.randint(0, 8)
        loan_product = rng.choices(LOAN_PRODUCTS, weights=LOAN_PRODUCT_WEIGHTS, k=1)[0]

        customers.append({
            "customer_id":       cid,
            "name":              name,
            "osp":               osp,
            "dpd":               dpd,
            "due_date":          due_date,
            "lat":               lat,
            "lon":               lon,
            "zone_id":           zone_id,
            "reason_code":       reason_code,
            "is_ots":            is_ots,
            "settlement_amount": settlement_amount,
            "last_visit_date":   last_visit_date,
            "ptp_given":         ptp_given,
            "ptp_kept":          ptp_kept,
            "ptp_broken":        ptp_broken,
            "contact_attempts":  contact_attempts,
            "is_mandatory":      is_mandatory,
            "is_msd_zone":       is_msd,
            "loan_product":      loan_product,
        })

    return customers


# ---------------------------------------------------------------------------
# Quick sanity print
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import collections

    portfolio = generate_synthetic_portfolio(n=70, seed=42)
    print(f"Generated {len(portfolio)} customers\n")

    reason_dist = collections.Counter(c["reason_code"] for c in portfolio)
    print("Reason code distribution:")
    for code, cnt in sorted(reason_dist.items(), key=lambda x: -x[1]):
        print(f"  {code:<10} {cnt:>3}  ({cnt/len(portfolio)*100:.1f}%)")

    print(f"\nMandatory:        {sum(c['is_mandatory'] for c in portfolio)}")
    print(f"MSD zone:         {sum(c['is_msd_zone'] for c in portfolio)}")
    print(f"OTS:              {sum(c['is_ots'] for c in portfolio)}")
    print(f"GPS missing:      {sum(c['lat'] is None for c in portfolio)}")
    print(f"Recent visit:     {sum(c['last_visit_date'] is not None and (date.today()-c['last_visit_date']).days <= 30 for c in portfolio)}")
    print(f"PTP broken >= 1:  {sum(c['ptp_broken'] >= 1 for c in portfolio)}")

    osps = [c["osp"] for c in portfolio]
    print(f"\nOSP  min: Rs{min(osps):,.0f}  max: Rs{max(osps):,.0f}  mean: Rs{sum(osps)/len(osps):,.0f}")

    dpds = [c["dpd"] for c in portfolio]
    print(f"DPD  min: {min(dpds)}  max: {max(dpds)}  mean: {sum(dpds)/len(dpds):.1f}")
