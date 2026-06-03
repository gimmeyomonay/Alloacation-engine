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

def _trim_cluster_by_ml(
    cl: dict,
    V_adj: list[float],
    interaction_times: list[float],
    budget: float,
    cap: int,
    inter_travel: float,
) -> dict:
    """
    When a cluster cannot be visited whole, trim it to the highest-efficiency
    members that fit within the remaining budget and ACR cap.

    Members are ranked by ML-derived individual efficiency (V_adj / interaction_time)
    — the model's signal of which customers are worth visiting most.  The TSP
    route order is then re-applied to the kept members so the agent still follows
    a geographic sequence.

    Returns a new cluster dict representing the trimmed visit set.
    """
    from .routing import haversine_matrix, nearest_neighbour_tsp, route_travel_time
    from .clustering import compute_zone_centroids

    members = cl["member_indices"]
    coords  = cl["coords"]

    # Rank members by ML efficiency descending
    ranked = sorted(
        members,
        key=lambda i: V_adj[i] / interaction_times[i] if interaction_times[i] > 0 else 0.0,
        reverse=True,
    )

    # Greedily add members until budget or cap is exhausted
    # (conservative: use interaction_time only; intra-cluster travel is
    #  re-computed on the final kept set)
    kept, time_used = [], inter_travel
    for idx in ranked:
        t = interaction_times[idx]
        if time_used + t <= budget and len(kept) < cap:
            kept.append(idx)
            time_used += t
        if len(kept) >= cap or time_used >= budget:
            break

    if not kept:
        return cl  # nothing fits — caller will skip this cluster

    # Re-run TSP on the kept members so the geographic route is preserved
    member_pos  = {idx: pos for pos, idx in enumerate(members)}
    kept_coords = [coords[member_pos[i]] for i in kept]
    if len(kept) == 1:
        sub_mat      = haversine_matrix(kept_coords)
        trimmed_route = kept[:]
        travel_min   = 0.0
    else:
        sub_mat      = haversine_matrix(kept_coords)
        local_route  = nearest_neighbour_tsp(list(range(len(kept))), sub_mat, two_opt=True)
        trimmed_route = [kept[j] for j in local_route]
        travel_min   = route_travel_time(local_route, sub_mat)

    interaction_min = sum(interaction_times[i] for i in kept)
    total_min       = travel_min + interaction_min
    V_kept          = sum(V_adj[i] for i in kept)

    return {
        **cl,
        "member_indices": kept,
        "route":          trimmed_route,
        "coords":         [kept_coords[kept.index(i)] for i in trimmed_route],
        "travel_matrix":  sub_mat,
        "travel_time_min":     travel_min,
        "interaction_time_min": interaction_min,
        "total_time_min":      total_min,
        "efficiency":          V_kept / total_min if total_min > 0 else 0.0,
        "_trimmed": True,
    }


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
    Greedily fill remaining budget with clusters and outliers.

    For each cluster, the ML-derived efficiency (V_adj / time) determines
    both the cluster ranking and, when a cluster is too large to visit whole,
    which members to keep (highest ML efficiency first).  The geographic TSP
    route is then re-applied to the kept members, so the agent always follows
    a sensible geographic sequence within each cluster.

    Returns (selected_clusters, selected_outlier_indices).
    """
    scoreable_set = set(scoreable_indices)

    eligible_clusters = [
        c for c in clusters
        if all(i in scoreable_set for i in c["member_indices"])
    ]
    eligible_outliers = [i for i in outlier_indices if i in scoreable_set]

    # Annotate clusters with ML-derived efficiency
    for cl in eligible_clusters:
        cl["efficiency"] = cluster_efficiency(cl, V_adj)

    eligible_clusters.sort(key=lambda c: c["efficiency"], reverse=True)
    eligible_outliers.sort(
        key=lambda i: outlier_efficiency(i, V_adj, interaction_times), reverse=True
    )

    remaining_budget = daily_budget_min - mandatory_time_min
    remaining_cap    = acr_cap - mandatory_count

    selected_clusters: list[dict] = []
    selected_outliers: list[int]  = []
    current_exit: tuple[float, float] | None = None

    def _centroid(cl: dict) -> tuple[float, float]:
        lats = [cl["coords"][k][0] for k in range(len(cl["coords"]))]
        lons = [cl["coords"][k][1] for k in range(len(cl["coords"]))]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def _coord(local_idx: int) -> tuple[float, float]:
        c = customers[local_idx]
        if c.lat is not None and c.lon is not None:
            return (c.lat, c.lon)
        from .clustering import compute_zone_centroids
        zc = compute_zone_centroids(customers)
        return zc.get(c.zone_id, (0.0, 0.0))

    def _inter(dest: tuple[float, float]) -> float:
        from .routing import haversine_minutes
        if current_exit is None:
            return 0.0
        return haversine_minutes(current_exit[0], current_exit[1], dest[0], dest[1])

    cluster_ptr = 0
    outlier_ptr = 0

    while remaining_budget > 0 and remaining_cap > 0:
        next_cl  = eligible_clusters[cluster_ptr] if cluster_ptr < len(eligible_clusters) else None
        next_out = eligible_outliers[outlier_ptr]  if outlier_ptr  < len(eligible_outliers)  else None

        if next_cl is None and next_out is None:
            break

        # Best cluster: either fits whole, or trim to what ML says is worth visiting
        best_cl = None
        best_cl_eff = -1.0
        if next_cl is not None:
            centroid = _centroid(next_cl)
            inter    = _inter(centroid)
            if (next_cl["total_time_min"] + inter <= remaining_budget
                    and len(next_cl["member_indices"]) <= remaining_cap):
                best_cl     = next_cl
                best_cl_eff = next_cl["efficiency"]
            else:
                # Too large to visit whole — trim by ML efficiency
                trimmed = _trim_cluster_by_ml(
                    next_cl, V_adj, interaction_times,
                    remaining_budget, remaining_cap, inter,
                )
                if len(trimmed["member_indices"]) > 0:
                    best_cl     = trimmed
                    best_cl_eff = trimmed["efficiency"]

        outlier_eff = (
            outlier_efficiency(next_out, V_adj, interaction_times)
            if next_out is not None else -1.0
        )

        if best_cl is not None and best_cl_eff >= outlier_eff:
            centroid = _centroid(best_cl)
            inter    = _inter(centroid)
            best_cl["inter_travel_min"] = inter
            selected_clusters.append(best_cl)
            remaining_budget -= best_cl["total_time_min"] + inter
            remaining_cap    -= len(best_cl["member_indices"])
            current_exit      = centroid
            cluster_ptr      += 1
        elif next_out is not None:
            o_coord = _coord(next_out)
            o_inter = _inter(o_coord)
            o_time  = interaction_times[next_out] + o_inter
            if o_time <= remaining_budget and remaining_cap >= 1:
                selected_outliers.append(next_out)
                remaining_budget -= o_time
                remaining_cap    -= 1
                current_exit      = o_coord
            outlier_ptr += 1
        else:
            cluster_ptr += 1  # cluster trimmed to 0, skip it

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
