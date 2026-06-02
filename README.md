# Collection Agent Allocation Engine

A Python engine that takes a daily customer portfolio snapshot and outputs an optimised visit plan for one field collection agent — maximising recovery per unit of agent time.

---

## What it does

- Scores each customer by expected recovery value, urgency, and visit history
- Clusters customers geographically using value-weighted DBSCAN
- Sequences visits within each cluster using a nearest-neighbour TSP
- Greedily fills an 8-hour daily budget (max 70 accounts) with the highest-efficiency clusters
- Flags mandatory visits (willful defaulters, absconders, nominee OD cases)
- Routes escalation cases (mass default zones, repeat PTP breakers) to a supervisor queue
- Projects a 5-day watch list of accounts approaching DPD bucket boundaries
- Supports mid-day replan when visit outcomes are reported

---

## Project structure

```
allocation_engine/
├── engine.py        # Main pipeline — orchestrates all stages
├── models.py        # Customer, VisitPlan, CustomerVisit dataclasses
├── config.py        # EngineConfig — all tunable parameters
├── probability.py   # HeuristicModel (Phase 1/2), MLModel stub (Phase 3)
├── scoring.py       # V_i, V_adj, urgency boost, interaction time
├── routing.py       # Haversine travel time, TSP, optional Maps API
├── clustering.py    # Value-weighted DBSCAN, zone centroid fallback
├── selection.py     # Greedy selection, outlier absorption, rationale labels
├── horizon.py       # 5-day watch list, mid-day replan
├── data_gen.py      # Synthetic portfolio generator
└── main.py          # CLI entry point
```

---

## Quickstart

**Install dependencies:**
```bash
pip install scikit-learn numpy pandas requests rich
```

**Run on synthetic data (70 customers):**
```bash
python -m allocation_engine.main --synthetic --n 70
```

**Run on a real CSV portfolio:**
```bash
python -m allocation_engine.main --input portfolio.csv --date 2026-06-02
```

**Export plan to JSON:**
```bash
python -m allocation_engine.main --synthetic --output plan.json
```

**Mid-day replan:**
```bash
python -m allocation_engine.main --replan --input portfolio.csv --remaining-budget 210 --outcomes outcomes.json
```

**Enable Google Maps API for real driving times:**
```bash
export GOOGLE_MAPS_API_KEY=your_key_here
python -m allocation_engine.main --synthetic --use-maps-api
```

---

## Input schema

Each customer row requires these fields:

| Field | Type | Description |
|---|---|---|
| `customer_id` | str | Unique ID |
| `name` | str | Customer name |
| `osp` | float | Outstanding principal (Rs) |
| `dpd` | int | Days past due |
| `due_date` | date | Loan due date |
| `lat` / `lon` | float | GPS coordinates (None if unavailable) |
| `zone_id` | str | Branch/zone fallback if GPS missing |
| `reason_code` | str | IC4 code: IDV_TEMP, IDV_LONG, TCI, MSD, SBL, STF, WLD, LGL, TNC, ABS, SRI |
| `is_ots` | bool | OTS negotiated — use settlement_amount |
| `settlement_amount` | float | OTS amount |
| `last_visit_date` | date | None if never visited |
| `ptp_given` | int | Total PTPs given historically |
| `ptp_kept` | int | Total PTPs honoured |
| `ptp_broken` | int | Total PTPs broken |
| `contact_attempts` | int | Total contact attempts |
| `is_mandatory` | bool | Pre-pinned visit |
| `is_msd_zone` | bool | Mass default zone flag |
| `loan_product` | str | GL, IL, SHL, SBL, MSE, LAP |

---

## Configuration

All parameters are tunable via `EngineConfig` in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `daily_budget_minutes` | 480 | 8-hour working day |
| `acr_cap` | 70 | Max accounts per agent per day |
| `repeat_penalty_coeff` | 0.35 | Penalty for revisiting recent customers |
| `penalty_decay_days` | 30 | Days after which repeat penalty is zero |
| `eps_base_km` | 3.0 | DBSCAN clustering radius in km |
| `alpha` | 0.5 | Value expansion factor for clustering |
| `road_factor` | 1.35 | Haversine to road distance multiplier |
| `agent_speed_kmh` | 25.0 | Average agent travel speed |
| `outlier_absorb_delta` | 0.10 | Max efficiency drop to absorb an outlier |
| `horizon_days` | 5 | Watch list forward horizon |

---

## Probability model

The engine ships with a heuristic model (Phase 1/2) based on provisioning norms and reason code severity. A `MLModel` stub is available in `probability.py` — swap it in once 6 months of outcome data is available:

```python
from allocation_engine.probability import MLModel
from allocation_engine.engine import AllocationEngine

engine = AllocationEngine(prob_model=MLModel("path/to/model.pkl"))
```

---

## Dependencies

```
scikit-learn
numpy
pandas
requests
rich
```

No ML framework required until `MLModel` is activated.
