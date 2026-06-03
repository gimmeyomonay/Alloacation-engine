# Phase 2 Bug Fixes — Engine Correctness

**Branch:** `fix/phase2-engine-correctness`  
**Date:** 2026-06-03  
**PRD reference:** Sections 7 (Allocation Pipeline), 13 (Phase 2 Roadmap)

These bugs were identified by observing that the Streamlit map showed geographically adjacent customers with non-adjacent sequence numbers (e.g. visits #1 and #6 side-by-side), and that the clustering algorithm appeared to have no effect on the final plan.

---

## Bug 1 — Mandatory visits sequenced in pool order, not route order

**File:** `allocation_engine/engine.py` — `_compute_mandatory_travel`, `_mandatory_visits`

**Root cause:**  
`_compute_mandatory_travel` computed the nearest-neighbour TSP route for mandatory customers and calculated per-leg travel times, but then re-mapped the legs back to the *original pool order* (the order customers appeared in the input list). `_mandatory_visits` iterated in that same original order and assigned sequence numbers 1, 2, 3 … accordingly.

The result: sequence numbers on the map reflected which customer appeared first in the data, not which the agent would physically visit first. An agent following the numbered sequence would zigzag across the city.

**Fix:**  
`_compute_mandatory_travel` now returns both the travel legs *and* the TSP route order. `_mandatory_visits` iterates in TSP order, so sequence numbers 1, 2, 3 … match the geographic route.

---

## Bug 2 — Ranked visits re-sorted after cluster/TSP ordering

**File:** `allocation_engine/engine.py` — `_ranked_visits`

**Root cause:**  
After building ranked visits in the correct order (clusters sorted by cluster-level efficiency, customers within each cluster in TSP route order), a final `visits.sort(key=lambda v: v.efficiency, reverse=True)` re-sorted the entire list by *individual customer efficiency*. This interleaved customers from different clusters — a customer from cluster B with high individual efficiency could appear between two cluster A customers on the map, forcing the agent to leave and re-enter a geographic area.

The PRD (Section 7, Step 7–8) specifies that clusters should be selected and ranked as a whole unit; individual customer efficiency is not an ordering criterion within the final plan.

**Fix:**  
Removed the final sort. The list is already in the correct order: clusters in descending cluster-efficiency order, customers within each cluster in TSP route order.

---

## Bug 3 — DBSCAN producing clusters too large to fit in the daily budget

**Files:** `allocation_engine/clustering.py`, `allocation_engine/config.py`, `allocation_engine/engine.py`

**Root cause:**  
DBSCAN at `eps=1.5km` in Bengaluru's dense layout produced 4 clusters of 17–41 members each, with total visit times of 672–1650 minutes. The greedy selector rejected all of them (none fit in the 480-minute daily budget), so the final plan contained only standalone outlier visits — making the clustering algorithm appear to have no effect.

**Fix:**  
Added `max_cluster_visits: int = 12` to `EngineConfig` and a new `split_oversized_clusters()` function in `clustering.py`. After DBSCAN, any cluster exceeding `max_cluster_visits` is split into sequential sub-clusters along the TSP route. Each sub-cluster is compact enough to be selected by the greedy algorithm. The default of 12 leaves approximately 1.5–2 clusters visitable per day alongside mandatory visits.

---

## Bug 4 — Greedy selector consumed budget with outliers before trying smaller clusters

**File:** `allocation_engine/selection.py` — `greedy_select`

**Root cause:**  
When the highest-efficiency cluster didn't fit in the remaining budget, the selector fell back to the best outlier *without advancing the cluster pointer*. It kept comparing the same oversized cluster against successive outliers, consuming the budget one outlier at a time. By the time it advanced past the oversized cluster, smaller clusters that would have fit were no longer affordable.

**Fix:**  
The selector now advances past all clusters that cannot fit in the current remaining budget *before* comparing against the best outlier. This ensures the comparison is always between the best *fitting* cluster and the best outlier — matching the PRD's intent of preferring high-efficiency geographic clusters over standalone visits.

---

## Bug 5 — Travel matrix indexed by sc_customer local index instead of matrix position

**File:** `allocation_engine/engine.py` — `_ranked_visits`

**Root cause:**  
The travel time matrix stored in each cluster is a square matrix indexed 0..N-1, where N is the cluster member count. The code accessed it as `travel_mat[prev_customer_idx, curr_customer_idx]`, using the customer's local index within the full scoreable pool (which can be 0–103 for a 120-customer portfolio). For the original large clusters this worked by coincidence — cluster members happened to have small local indices that fell within the matrix bounds. After cluster splitting (Bug 3 fix), sub-clusters of 5–12 members have local indices up to 103, causing an `IndexError`.

An additional variant: `absorb_outliers` appends outlier customers to `member_indices` after the travel matrix was built, so absorbed outliers also had no valid matrix entry.

**Fix:**  
`_ranked_visits` now builds a `mat_pos` lookup (`sc_customer_local_index → matrix_position`) from the cluster's `member_indices`. Matrix lookups use `mat_pos` values (0..N-1). For absorbed outliers not present in `mat_pos`, travel time is computed directly via haversine.

---

## Also completed in this session — Phase 2 features

These were new additions, not bug fixes:

| Item | File(s) |
|------|---------|
| Batch prediction via `XGBoostModel.predict_batch()` wired into engine | `engine.py`, `scoring.py` |
| Model versioning registry | `allocation_engine/model_registry.py`, `models/registry.json` |
| Retraining pipeline from feedback log | `training/retrain.py` |
| Streamlit model toggle (Heuristic / XGBoost) with AUC display | `allocation_engine/app.py` |
| API `/model/info`, `/model/versions`, `/model/activate/{version_id}` | `allocation_engine/api.py` |
