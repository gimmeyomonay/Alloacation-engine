"""
Greedy visit-plan selection:
  - split mandatory / escalation / scoreable
  - rank clusters + outliers by efficiency
  - fill daily budget respecting ACR cap
  - outlier absorption check
  - build CustomerVisit records with rationale labels
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .models import Customer, CustomerVisit
from .routing import haversine_minutes


# ---------------------------------------------------------------------------
# Pool split
# ---------------------------------------------------------------------------

def split_pool(customers: list[Customer]) -> tuple[list[int], list[int], list[int]]:
    """
    Returns (mandatory_indices, escalation_indices, scoreable_indices).

    Escalation criteria (removed from scoreable pool):
      - is_msd_zone == True  → NCM/Vigilance queue
      - ptp_broken >= 3 AND contact_attempts >= 3 → supervisor queue
    Mandatory accounts bypass the scoreable pool entirely (pre-pinned).
    """
    mandatory, escalation, scoreable = [], [], []
    for i, c in enumerate(customers):
        if c.is_mandatory:
            mandatory.append(i)
        elif c.is_msd_zone or (c.ptp_broken >= 3 and c.contact_attempts >= 3):
            escalation.append(i)
        else:
            scoreable.append(i)
    return mandatory, escalation, scoreable


# ---------------------------------------------------------------------------
# Efficiency helpers
# ---------------------------------------------------------------------------

def cluster_efficiency(cluster: dict, V_adj: list[float]) -> float:
    V_C = sum(V_adj[i] for i in cluster["member_indices"])
    T_C = cluster["total_time_min"]
    return V_C / T_C if T_C > 0 else 0.0


def outlier_efficiency(idx: int, V_adj: list[float], interaction_times: list[float]) -> float:
    return V_adj[idx] / interaction_times[idx] if interaction_times[idx] > 0 else 0.0


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

def greedy_select(
    customers: list[Customer],
    scoreable_indices: list[int],
    clusters: list[dict],
    outlier_indices: list[int],
    V_adj: list[float],
    interaction_times: list[float],
    daily_budget_min: float = 480.0,
    acr_cap: int = 70,
    mandatory_time_min: float = 0.0,
    mandatory_count: int = 0,
) -> tuple[list[dict], list[int]]:
    """
    Greedily fill remaining budget with clusters and outliers from the scoreable pool.

    Only includes clusters/outliers whose members are all in scoreable_indices.

    Returns (selected_clusters, selected_outlier_indices).
    """
    scoreable_set = set(scoreable_indices)

    # Filter clusters to those fully within the scoreable pool
    eligible_clusters = [
        c for c in clusters
        if all(i in scoreable_set for i in c["member_indices"])
    ]
    eligible_outliers = [i for i in outlier_indices if i in scoreable_set]

    # Annotate with efficiency
    for cl in eligible_clusters:
        cl["efficiency"] = cluster_efficiency(cl, V_adj)

    # Sort descending by efficiency
    eligible_clusters.sort(key=lambda c: c["efficiency"], reverse=True)
    eligible_outliers.sort(key=lambda i: outlier_efficiency(i, V_adj, interaction_times), reverse=True)

    remaining_budget = daily_budget_min - mandatory_time_min
    remaining_cap = acr_cap - mandatory_count

    selected_clusters: list[dict] = []
    selected_outliers: list[int] = []

    # Track agent's current location to charge inter-cluster travel
    current_exit: tuple[float, float] | None = None

    def _cluster_centroid(cl: dict) -> tuple[float, float]:
        lats = [cl["coords"][k][0] for k in range(len(cl["coords"]))]
        lons = [cl["coords"][k][1] for k in range(len(cl["coords"]))]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def _customer_coord(local_idx: int) -> tuple[float, float]:
        from .routing import haversine_minutes
        from .clustering import compute_zone_centroids
        c = customers[local_idx]
        if c.lat is not None and c.lon is not None:
            return (c.lat, c.lon)
        zc = compute_zone_centroids(customers)
        return zc.get(c.zone_id, (0.0, 0.0))

    def _inter_travel(dest: tuple[float, float]) -> float:
        from .routing import haversine_minutes
        if current_exit is None:
            return 0.0
        return haversine_minutes(
            current_exit[0], current_exit[1],
            dest[0], dest[1],
        )

    cluster_ptr = 0
    outlier_ptr = 0

    while remaining_budget > 0 and remaining_cap > 0:
        # Advance past clusters that can no longer fit (budget only decreases)
        while cluster_ptr < len(eligible_clusters):
            cl = eligible_clusters[cluster_ptr]
            inter = _inter_travel(_cluster_centroid(cl))
            if cl["total_time_min"] + inter <= remaining_budget and len(cl["member_indices"]) <= remaining_cap:
                break
            cluster_ptr += 1

        next_cluster = eligible_clusters[cluster_ptr] if cluster_ptr < len(eligible_clusters) else None
        next_outlier_idx = eligible_outliers[outlier_ptr] if outlier_ptr < len(eligible_outliers) else None

        if next_cluster is None and next_outlier_idx is None:
            break

        cluster_eff = next_cluster["efficiency"] if next_cluster else -1
        outlier_eff = (
            outlier_efficiency(next_outlier_idx, V_adj, interaction_times)
            if next_outlier_idx is not None else -1
        )

        if cluster_eff >= outlier_eff and next_cluster is not None:
            cl = next_cluster
            centroid = _cluster_centroid(cl)
            inter = _inter_travel(centroid)
            cl["inter_travel_min"] = inter
            selected_clusters.append(cl)
            remaining_budget -= cl["total_time_min"] + inter
            remaining_cap -= len(cl["member_indices"])
            current_exit = centroid
            cluster_ptr += 1
        elif next_outlier_idx is not None:
            o_coord = _customer_coord(next_outlier_idx)
            o_inter = _inter_travel(o_coord)
            o_time = interaction_times[next_outlier_idx] + o_inter
            if o_time <= remaining_budget and remaining_cap >= 1:
                selected_outliers.append(next_outlier_idx)
                remaining_budget -= o_time
                remaining_cap -= 1
                current_exit = o_coord
            outlier_ptr += 1
        else:
            break

    return selected_clusters, selected_outliers


# ---------------------------------------------------------------------------
# Outlier absorption
# ---------------------------------------------------------------------------

def absorb_outliers(
    selected_clusters: list[dict],
    remaining_outliers: list[int],
    customers: list[Customer],
    V_adj: list[float],
    interaction_times: list[float],
    zone_centroids: dict[str, tuple[float, float]],
    speed_kmh: float = 25.0,
    road_factor: float = 1.35,
    absorb_delta: float = 0.10,
) -> list[int]:
    """
    For each unselected outlier, check if absorbing it into the nearest selected
    cluster degrades that cluster's efficiency by <= absorb_delta (10%).
    Returns list of outlier indices that were absorbed (mutates selected_clusters).
    """
    absorbed: list[int] = []

    def _coord(idx: int) -> tuple[float, float]:
        c = customers[idx]
        if c.lat is not None and c.lon is not None:
            return c.lat, c.lon
        return zone_centroids.get(c.zone_id, (0.0, 0.0))

    for o_idx in remaining_outliers:
        if not selected_clusters:
            break

        o_coord = _coord(o_idx)

        # Find nearest cluster by travel time from outlier to cluster centroid
        def _cluster_centroid(cl: dict) -> tuple[float, float]:
            lats = [cl["coords"][k][0] for k in range(len(cl["coords"]))]
            lons = [cl["coords"][k][1] for k in range(len(cl["coords"]))]
            return (sum(lats) / len(lats), sum(lons) / len(lons))

        nearest = min(
            selected_clusters,
            key=lambda cl: haversine_minutes(
                o_coord[0], o_coord[1],
                _cluster_centroid(cl)[0], _cluster_centroid(cl)[1],
                speed_kmh=speed_kmh, road_factor=road_factor,
            ),
        )

        travel_to_outlier = haversine_minutes(
            o_coord[0], o_coord[1],
            _cluster_centroid(nearest)[0], _cluster_centroid(nearest)[1],
            speed_kmh=speed_kmh, road_factor=road_factor,
        )

        V_C = sum(V_adj[i] for i in nearest["member_indices"])
        orig_eff = nearest["efficiency"]
        combined_eff = (V_C + V_adj[o_idx]) / (
            nearest["total_time_min"] + travel_to_outlier + interaction_times[o_idx]
        )

        if combined_eff >= orig_eff * (1 - absorb_delta):
            # Absorb
            nearest["member_indices"].append(o_idx)
            nearest["route"].append(o_idx)
            nearest["coords"].append(o_coord)
            nearest["travel_time_min"] += travel_to_outlier
            nearest["interaction_time_min"] += interaction_times[o_idx]
            nearest["total_time_min"] += travel_to_outlier + interaction_times[o_idx]
            nearest["efficiency"] = (V_C + V_adj[o_idx]) / nearest["total_time_min"]
            absorbed.append(o_idx)

    return absorbed


# ---------------------------------------------------------------------------
# Rationale labelling
# ---------------------------------------------------------------------------

def _rationale(
    customer: Customer,
    is_mandatory: bool,
    urgency_boost: float,
    cluster_size: Optional[int],
    is_absorbed_outlier: bool,
    is_standalone_outlier: bool,
    is_escalation: bool,
) -> str:
    if is_escalation:
        if customer.is_msd_zone:
            return "Escalation - mass default zone"
        return "Escalation - PTP broken 3+ times"
    if is_mandatory:
        if customer.reason_code in ("WLD", "ABS"):
            return "Mandatory - willful defaulter" if customer.reason_code == "WLD" else "Mandatory - abscond case"
        return "Mandatory - nominee OD case"
    if urgency_boost > 0:
        days = _days_to_boundary(customer.dpd)
        return f"Urgent - bucket boundary in {days} day(s)"
    if cluster_size and cluster_size > 1:
        if is_absorbed_outlier:
            return f"High-value outlier, absorbed into cluster"
        return f"High-efficiency cluster ({cluster_size} customers)"
    if customer.dpd <= 7:
        return "Fresh customer, high recovery value"
    if is_standalone_outlier:
        return "High-value standalone visit"
    return "Ranked visit"


def _days_to_boundary(dpd: int) -> int:
    for boundary in [8, 31, 61, 91, 181]:
        if dpd < boundary:
            return boundary - dpd
    return 0


# ---------------------------------------------------------------------------
# Assemble CustomerVisit records
# ---------------------------------------------------------------------------

def build_visit_records(
    customers: list[Customer],
    indices: list[int],
    scores: list[dict],       # aligned with `customers`
    cluster_id: Optional[int],
    cluster_size: Optional[int],
    visit_sequence_start: int,
    is_mandatory: bool = False,
    is_escalation: bool = False,
    absorbed_outlier_indices: Optional[set] = None,
) -> list[CustomerVisit]:
    if absorbed_outlier_indices is None:
        absorbed_outlier_indices = set()

    visits = []
    for seq_offset, idx in enumerate(indices):
        c = customers[idx]
        s = scores[idx]
        eff = s["V_adj"] / (s["interaction_min"]) if s["interaction_min"] > 0 else 0.0

        rationale = _rationale(
            customer=c,
            is_mandatory=is_mandatory,
            urgency_boost=s["urgency_boost"],
            cluster_size=cluster_size,
            is_absorbed_outlier=(idx in absorbed_outlier_indices),
            is_standalone_outlier=(cluster_id is None and not is_mandatory and not is_escalation),
            is_escalation=is_escalation,
        )

        visits.append(CustomerVisit(
            rank=0,  # filled by caller
            customer_id=c.customer_id,
            name=c.name,
            osp=c.osp,
            dpd=c.dpd,
            probability=s["probability"],
            V_i=s["V_i"],
            V_adj=s["V_adj"],
            urgency_boost=s["urgency_boost"],
            efficiency=eff,
            travel_minutes=0.0,   # per-leg travel filled by engine
            interaction_min=s["interaction_min"],
            cluster_id=cluster_id,
            visit_sequence=visit_sequence_start + seq_offset,
            rationale=rationale,
        ))
    return visits
