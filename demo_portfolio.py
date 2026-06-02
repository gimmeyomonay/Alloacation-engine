"""
Generate a demo-friendly CSV portfolio that shows all engine features:
- Enough scoreable accounts so ranked visits populate
- Realistic mix of clusters, outliers, watch list entries
- Low enough mandatory % to leave budget for ranked visits
"""

import csv
import random
import math
import numpy as np
from datetime import date, timedelta

random.seed(99)
np.random.seed(99)

TODAY = date.today()

ZONES = {
    "Z1": (12.9716, 77.5946),
    "Z2": (12.9352, 77.6245),
    "Z3": (13.0100, 77.5500),
    "Z4": (12.9850, 77.7200),
    "Z5": (12.9200, 77.5500),
}

REASON_CODES   = ["IDV_TEMP","IDV_LONG","TNC","WLD","TCI","ABS","MSD","SRI","LGL","SBL","STF"]
REASON_WEIGHTS = [0.30, 0.18, 0.15, 0.08, 0.10, 0.04, 0.04, 0.04, 0.03, 0.02, 0.02]
LOAN_PRODUCTS  = ["GL","IL","SHL","SBL","MSE","LAP"]
LOAN_WEIGHTS   = [0.35,0.25,0.15,0.10,0.10,0.05]

FIRST_NAMES = ["Ravi","Priya","Suresh","Anjali","Mohan","Lakshmi","Vijay","Meena",
               "Arun","Sunita","Ramesh","Kavitha","Sanjay","Deepa","Kiran","Usha",
               "Amit","Rekha","Rahul","Geeta","Ashok","Nirmala","Prakash","Savitha",
               "Ganesh","Padma","Sunil","Radha","Manoj","Sarala","Divya","Arjun",
               "Pooja","Vikram","Sneha","Rohit","Anita","Nikhil","Swati","Rajesh"]
LAST_NAMES  = ["Kumar","Sharma","Reddy","Naik","Patil","Hegde","Rao","Gowda",
               "Joshi","Iyer","Pillai","Menon","Shetty","Nair","Krishnan"]

# DPD buckets — weighted for a realistic mid-cycle portfolio
DPD_BUCKETS = [
    (1,   7,  0.20),   # fresh bounces
    (8,   30, 0.28),   # early
    (31,  60, 0.22),   # mid
    (61,  90, 0.15),   # late
    (91,  180,0.10),   # NPA edge
    (181, 300,0.05),   # deep NPA
]

def sample_dpd():
    bucket = random.choices(DPD_BUCKETS, weights=[b[2] for b in DPD_BUCKETS])[0]
    return random.randint(bucket[0], bucket[1])

def osp_lognormal():
    val = np.random.lognormal(mean=math.log(45000), sigma=0.85)
    return round(float(np.clip(val, 5000, 200000)), 2)

def gps(zone_id, missing=False):
    if missing:
        return "", ""
    clat, clon = ZONES[zone_id]
    return round(clat + random.gauss(0, 0.012), 6), round(clon + random.gauss(0, 0.012), 6)

rows = []
n = 120

# Pre-assign flags — low mandatory (6%) so ranked visits populate well
n_mandatory  = round(n * 0.06)
n_msd        = round(n * 0.08)
n_ots        = round(n * 0.10)
n_recent     = round(n * 0.35)
n_gps_miss   = round(n * 0.06)
n_ptp_broken = round(n * 0.18)

flags = list(range(n))
mandatory_set  = set(random.sample(flags, n_mandatory))
msd_set        = set(random.sample(flags, n_msd))
ots_set        = set(random.sample(flags, n_ots))
recent_set     = set(random.sample(flags, n_recent))
gps_miss_set   = set(random.sample(flags, n_gps_miss))
ptp_broken_set = set(random.sample(flags, n_ptp_broken))

used_names = set()
def unique_name():
    for _ in range(100):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name not in used_names:
            used_names.add(name)
            return name
    return f"Customer {len(used_names)}"

for i in range(n):
    cid  = f"CUST{2000+i:04d}"
    name = unique_name()
    dpd  = sample_dpd()
    osp  = osp_lognormal()

    is_msd       = i in msd_set
    is_mandatory = i in mandatory_set
    is_ots       = i in ots_set

    # Reason code
    if is_msd:
        reason_code = "MSD"
    elif is_mandatory:
        reason_code = random.choices(["WLD","ABS","IDV_LONG"], weights=[0.4,0.35,0.25])[0]
    else:
        non_msd_codes   = [rc for rc in REASON_CODES if rc != "MSD"]
        non_msd_weights = [w  for rc, w in zip(REASON_CODES, REASON_WEIGHTS) if rc != "MSD"]
        reason_code = random.choices(non_msd_codes, weights=non_msd_weights)[0]

    zone_id = random.choice(list(ZONES.keys()))
    lat, lon = gps(zone_id, missing=(i in gps_miss_set))

    due_date = TODAY - timedelta(days=dpd)
    settlement_amount = round(osp * random.uniform(0.50, 0.85), 2) if is_ots else 0.0

    if i in recent_set:
        last_visit_date = TODAY - timedelta(days=random.randint(1, 29))
    elif random.random() < 0.45:
        last_visit_date = None
    else:
        last_visit_date = TODAY - timedelta(days=random.randint(30, 180))

    ptp_given = random.randint(0, 6)
    if i in ptp_broken_set and ptp_given > 0:
        ptp_broken = random.randint(1, min(3, ptp_given))
    else:
        ptp_broken = 0
    ptp_kept = max(0, ptp_given - ptp_broken - random.randint(0, max(0, ptp_given - ptp_broken)))
    contact_attempts = random.randint(0, 10)
    loan_product = random.choices(LOAN_PRODUCTS, weights=LOAN_WEIGHTS)[0]

    rows.append({
        "customer_id":       cid,
        "name":              name,
        "osp":               osp,
        "dpd":               dpd,
        "due_date":          due_date.isoformat(),
        "lat":               lat,
        "lon":               lon,
        "zone_id":           zone_id,
        "reason_code":       reason_code,
        "is_ots":            is_ots,
        "settlement_amount": settlement_amount,
        "last_visit_date":   last_visit_date.isoformat() if last_visit_date else "",
        "ptp_given":         ptp_given,
        "ptp_kept":          ptp_kept,
        "ptp_broken":        ptp_broken,
        "contact_attempts":  contact_attempts,
        "is_mandatory":      is_mandatory,
        "is_msd_zone":       is_msd,
        "loan_product":      loan_product,
    })

# Write CSV
fields = list(rows[0].keys())
with open("demo_portfolio.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {n} customers to demo_portfolio.csv")
print(f"Mandatory: {n_mandatory}  MSD: {n_msd}  OTS: {n_ots}  GPS missing: {n_gps_miss}")
import collections
rc_dist = collections.Counter(r["reason_code"] for r in rows)
for rc, cnt in sorted(rc_dist.items(), key=lambda x: -x[1]):
    print(f"  {rc:<10} {cnt:>3}  ({cnt/n*100:.0f}%)")
