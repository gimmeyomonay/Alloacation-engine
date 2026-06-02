"""Travel-time computation and nearest-neighbour TSP with optional 2-opt."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Haversine travel time
# ---------------------------------------------------------------------------

def haversine_minutes(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    speed_kmh: float = 25.0,
    road_factor: float = 1.35,
) -> float:
    """Straight-line haversine distance converted to driving minutes."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    dist_km = 2 * R * math.asin(math.sqrt(min(1.0, a)))
    road_dist = dist_km * road_factor
    return (road_dist / speed_kmh) * 60.0


def haversine_matrix(
    coords: list[tuple[float, float]],
    speed_kmh: float = 25.0,
    road_factor: float = 1.35,
) -> np.ndarray:
    """
    Build a pairwise travel-time matrix (minutes) for a list of (lat, lon) tuples.
    Returns an (n × n) numpy float array.
    """
    n = len(coords)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            t = haversine_minutes(
                coords[i][0], coords[i][1],
                coords[j][0], coords[j][1],
                speed_kmh=speed_kmh,
                road_factor=road_factor,
            )
            mat[i, j] = t
            mat[j, i] = t
    return mat


# ---------------------------------------------------------------------------
# Optional Maps API (stub — activated by EngineConfig.use_maps_api=True)
# ---------------------------------------------------------------------------

def maps_api_minutes(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    api_key: str,
) -> float:
    """
    Fetch driving duration via Google Distance Matrix API.
    Falls back to haversine on any error.
    """
    import requests

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins":       f"{lat1},{lon1}",
        "destinations":  f"{lat2},{lon2}",
        "mode":          "driving",
        "departure_time": "now",
        "key":           api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        element = data["rows"][0]["elements"][0]
        if element["status"] == "OK":
            duration = element.get("duration_in_traffic", element["duration"])
            return duration["value"] / 60.0
    except Exception:
        pass
    return haversine_minutes(lat1, lon1, lat2, lon2)


def travel_time(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    speed_kmh: float = 25.0,
    road_factor: float = 1.35,
    use_maps_api: bool = False,
    api_key: str = "",
) -> float:
    if use_maps_api and api_key:
        return maps_api_minutes(lat1, lon1, lat2, lon2, api_key)
    return haversine_minutes(lat1, lon1, lat2, lon2, speed_kmh, road_factor)


# ---------------------------------------------------------------------------
# Nearest-neighbour TSP + optional 2-opt improvement
# ---------------------------------------------------------------------------

def nearest_neighbour_tsp(
    indices: list[int],
    travel_matrix: np.ndarray,
    start_idx: Optional[int] = None,
    two_opt: bool = True,
) -> list[int]:
    """
    Greedy nearest-neighbour TSP over a subset of nodes.

    Parameters
    ----------
    indices      : customer indices (into travel_matrix) forming the cluster
    travel_matrix: full pairwise travel-time matrix
    start_idx    : matrix index to start from (None → use first in cluster)
    two_opt      : whether to run a single 2-opt improvement pass

    Returns ordered list of indices (same elements as `indices`).
    """
    if len(indices) <= 1:
        return list(indices)

    unvisited = list(indices)
    if start_idx is None or start_idx not in unvisited:
        start_idx = unvisited[0]

    route = [start_idx]
    unvisited.remove(start_idx)

    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda j: travel_matrix[current, j])
        route.append(nearest)
        unvisited.remove(nearest)

    if two_opt and len(route) >= 4:
        route = _two_opt(route, travel_matrix)

    return route


def _two_opt(route: list[int], mat: np.ndarray) -> list[int]:
    """Single pass of 2-opt improvements."""
    best = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if _route_cost(new_route, mat) < _route_cost(best, mat):
                    best = new_route
                    improved = True
    return best


def _route_cost(route: list[int], mat: np.ndarray) -> float:
    return sum(mat[route[k], route[k + 1]] for k in range(len(route) - 1))


def route_travel_time(route: list[int], mat: np.ndarray) -> float:
    """Total travel time (minutes) for an ordered route."""
    return _route_cost(route, mat)
