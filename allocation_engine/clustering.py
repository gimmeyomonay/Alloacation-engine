"""
Value-weighted DBSCAN clustering + zone-centroid fallback for GPS-missing customers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN

from .models import Customer
from .routing import haversine_matrix, nearest_neighbour_tsp, route_travel_time


# ---------------------------------------------------------------------------
# Zone centroid fallback
# ---------------------------------------------------------------------------

def compute_zone_centroids(customers: list[Customer]) -> dict[str, tuple[float, float]]:
    """
    For each zone_id, compute the mean (lat, lon) of customers that have GPS.
    Zones with no GPS customers at all are excluded from the result.
    """
    zone_coords: dict[str, list[tuple[float, float]]] = {}
    for c in customers:
        if c.lat is not None and c.lon is not None:
            zone_coords.setdefault(c.zone_id, []).append((c.lat, c.lon))

    return {
        zone_id: (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
        )
        for zone_id, pts in zone_coords.items()
    }


def assign_coordinates(
    customers: list[Customer],
    zone_centroids: dict[str, tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[bool]]:
    """
    Return a coordinate list aligned with `customers`.
    Missing-GPS customers get their zone centroid.
    Returns (coords, synthetic_flags) where synthetic_flags[i]=True means the
    coordinate is a zone centroid, not a real GPS fix.
    """
    coords: list[tuple[float, float]] = []
    synthetic: list[bool] = []

    for c in customers:
        if c.lat is not None and c.lon is not None:
            coords.append((c.lat, c.lon))
            synthetic.append(False)
        elif c.zone_id in zone_centroids:
            coords.append(zone_centroids[c.zone_id])
            synthetic.append(True)
        else:
            # Zone has no GPS customers — will be handled as a whole-zone cluster
            coords.append((0.0, 0.0))
            synthetic.append(True)

    return coords, synthetic


# ---------------------------------------------------------------------------
# Value-weighted DBSCAN
# ---------------------------------------------------------------------------

def value_weighted_dbscan(
    customers: list[Customer],
    V_adj: list[float],
    eps_base_rad: float,
    alpha: float = 0.5,
    min_samples: int = 2,
) -> np.ndarray:
    """
    DBSCAN where each point has its own epsilon expanded by its relative value.

    sklearn DBSCAN only supports a single epsilon, so we approximate
    value-weighting by running standard DBSCAN at eps = eps_base_rad * (1 + alpha)
    (the maximum possible epsilon) and then pruning edges whose haversine distance
    exceeds the per-point epsilon.

    Returns cluster labels array aligned with `customers` (-1 = outlier).
    """
    n = len(customers)
    if n == 0:
        return np.array([], dtype=int)

    zone_centroids = compute_zone_centroids(customers)
    coords, _ = assign_coordinates(customers, zone_centroids)

    # Identify zones with no GPS at all — their customers form a forced cluster
    no_gps_zones: set[str] = set()
    for c in customers:
        if c.lat is None or c.lon is None:
            if c.zone_id not in zone_centroids:
                no_gps_zones.add(c.zone_id)

    max_V = max(V_adj) if max(V_adj) > 0 else 1.0
    eps_per_point = np.array([
        eps_base_rad * (1 + alpha * v / max_V) for v in V_adj
    ])

    # Run DBSCAN at the maximum eps to get candidate neighbours
    max_eps = float(eps_per_point.max())
    coords_rad = np.radians(coords)

    db = DBSCAN(
        eps=max_eps,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords_rad)

    labels = db.labels_.copy()

    # --- Per-point epsilon pruning ---
    # For each cluster, verify every member is within its own eps of at least
    # (min_samples - 1) other members.  Demote points that fail this test.
    _prune_by_individual_eps(labels, coords_rad, eps_per_point, min_samples)

    # --- Force-cluster GPS-missing customers whose zone has no GPS fix ---
    _assign_no_gps_zones(customers, labels, no_gps_zones)

    return labels


def _prune_by_individual_eps(
    labels: np.ndarray,
    coords_rad: np.ndarray,
    eps_per_point: np.ndarray,
    min_samples: int,
) -> None:
    """In-place: demote cluster members that violate their own epsilon."""
    from sklearn.metrics.pairwise import haversine_distances

    unique_clusters = set(labels) - {-1}
    for cid in unique_clusters:
        members = np.where(labels == cid)[0]
        if len(members) < min_samples:
            labels[members] = -1
            continue

        sub_coords = coords_rad[members]
        dist_mat = haversine_distances(sub_coords)  # in radians

        to_demote = []
        for local_i, global_i in enumerate(members):
            eps_i = eps_per_point[global_i]
            neighbours_within = np.sum(dist_mat[local_i] <= eps_i) - 1  # exclude self
            if neighbours_within < min_samples - 1:
                to_demote.append(global_i)

        for idx in to_demote:
            labels[idx] = -1


def _assign_no_gps_zones(
    customers: list[Customer],
    labels: np.ndarray,
    no_gps_zones: set[str],
) -> None:
    """In-place: assign a shared cluster label to entire no-GPS zones."""
    if not no_gps_zones:
        return

    next_label = int(labels.max()) + 1 if labels.max() >= 0 else 0
    for zone_id in no_gps_zones:
        zone_indices = [i for i, c in enumerate(customers) if c.zone_id == zone_id]
        for idx in zone_indices:
            labels[idx] = next_label
        next_label += 1


# ---------------------------------------------------------------------------
# Build cluster dicts with TSP-sequenced routes
# ---------------------------------------------------------------------------

def build_clusters(
    customers: list[Customer],
    labels: np.ndarray,
    interaction_times: list[float],
    speed_kmh: float = 25.0,
    road_factor: float = 1.35,
) -> tuple[list[dict], list[int]]:
    """
    For each cluster (label >= 0), compute the TSP route and total time.

    Returns
    -------
    clusters : list of dicts, each with keys:
                 cluster_id, member_indices, route (ordered indices),
                 travel_time_min, interaction_time_min, total_time_min
    outliers : list of customer indices with label == -1
    """
    zone_centroids = compute_zone_centroids(customers)
    coords, _ = assign_coordinates(customers, zone_centroids)

    unique_labels = sorted(set(labels) - {-1})
    clusters: list[dict] = []
    outliers: list[int] = [i for i, lbl in enumerate(labels) if lbl == -1]

    for cid in unique_labels:
        member_indices = [i for i, lbl in enumerate(labels) if lbl == cid]
        member_coords = [coords[i] for i in member_indices]

        mat = haversine_matrix(member_coords, speed_kmh=speed_kmh, road_factor=road_factor)
        local_route = nearest_neighbour_tsp(list(range(len(member_indices))), mat)
        global_route = [member_indices[j] for j in local_route]

        travel_min = route_travel_time(local_route, mat)
        interaction_min = sum(interaction_times[i] for i in member_indices)
        total_min = travel_min + interaction_min

        # mat_pos maps each route index → its row/col position in travel_matrix
        mat_pos = {global_idx: pos for pos, global_idx in enumerate(member_indices)}

        clusters.append({
            "cluster_id":          cid,
            "member_indices":      member_indices,
            "route":               global_route,
            "travel_time_min":     travel_min,
            "interaction_time_min": interaction_min,
            "total_time_min":      total_min,
            "coords":              member_coords,
            "travel_matrix":       mat,
            "mat_pos":             mat_pos,
        })

    return clusters, outliers


def split_oversized_clusters(
    clusters: list[dict],
    max_visits: int,
    interaction_times: list[float],
    speed_kmh: float,
    road_factor: float,
) -> list[dict]:
    """
    Split any cluster whose member count exceeds `max_visits` into sequential
    sub-clusters along the TSP route.  Each sub-cluster is capped at max_visits
    members and gets a new unique cluster_id.

    This ensures the greedy selector can actually fit clusters within the daily
    budget in dense urban areas (e.g. Bengaluru) where DBSCAN creates very large
    connected regions.
    """
    result: list[dict] = []
    next_id = max((cl["cluster_id"] for cl in clusters), default=-1) + 1

    for cl in clusters:
        route = cl["route"]
        if len(route) <= max_visits:
            result.append(cl)
            continue

        # Split route into chunks of max_visits
        mat = cl["travel_matrix"]
        member_indices = cl["member_indices"]
        all_coords = cl["coords"]

        for chunk_start in range(0, len(route), max_visits):
            chunk_route = route[chunk_start: chunk_start + max_visits]
            # member_indices and coords for this chunk
            chunk_members = list(chunk_route)  # route already holds local indices
            chunk_coords = [all_coords[member_indices.index(i)] for i in chunk_members]

            # Recompute travel matrix and TSP for the sub-cluster
            sub_mat = haversine_matrix(chunk_coords, speed_kmh=speed_kmh, road_factor=road_factor)
            local_route = nearest_neighbour_tsp(list(range(len(chunk_members))), sub_mat)
            sub_route = [chunk_members[j] for j in local_route]
            sub_coords = [chunk_coords[j] for j in local_route]

            travel_min = route_travel_time(local_route, sub_mat)
            interaction_min = sum(interaction_times[i] for i in chunk_members)
            total_min = travel_min + interaction_min

            cid = cl["cluster_id"] if chunk_start == 0 else next_id
            if chunk_start > 0:
                next_id += 1

            result.append({
                "cluster_id":           cid,
                "member_indices":       chunk_members,
                "route":                sub_route,
                "travel_time_min":      travel_min,
                "interaction_time_min": interaction_min,
                "total_time_min":       total_min,
                "coords":               sub_coords,
                "travel_matrix":        sub_mat,
            })

    return result
