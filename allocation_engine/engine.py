"""
AllocationEngine — orchestrates the full pipeline end-to-end.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .config import EngineConfig
from .models import Customer, CustomerVisit, VisitPlan, WatchItem
from .probability import ProbabilityModel, HeuristicModel
from .scoring import score_customer
from .clustering import compute_zone_centroids, value_weighted_dbscan, build_clusters
from .selection import (
    split_pool, greedy_select, absorb_outliers,
    build_visit_records, cluster_efficiency, outlier_efficiency,
)
from .horizon import build_watch_list
from .routing import haversine_minutes


class AllocationEngine:
    def __init__(
        self,
        prob_model: Optional[ProbabilityModel] = None,
        config: Optional[EngineConfig] = None,
    ):
        self.prob_model = prob_model or HeuristicModel()
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        customers: list[Customer],
        today: Optional[date] = None,
    ) -> VisitPlan:
        today = today or date.today()
        cfg = self.config

        # ── Step 1: Score all customers ──────────────────────────────
        scores = [
            score_customer(c, self.prob_model, today,
                           cfg.repeat_penalty_coeff, cfg.penalty_decay_days)
            for c in customers
        ]
        V_adj_all     = [s["V_adj"]          for s in scores]
        interact_all  = [s["interaction_min"] for s in scores]

        # ── Step 2: Split pool ───────────────────────────────────────
        mandatory_idx, escalation_idx, scoreable_idx = split_pool(customers)

        # ── Step 3: Cluster scoreable customers ──────────────────────
        sc_customers   = [customers[i] for i in scoreable_idx]
        sc_V_adj       = [V_adj_all[i]    for i in scoreable_idx]
        sc_interact    = [interact_all[i] for i in scoreable_idx]

        labels = value_weighted_dbscan(
            sc_customers, sc_V_adj,
            cfg.eps_base_rad, cfg.alpha, cfg.min_cluster_size,
        )
        clusters, outliers_local = build_clusters(
            sc_customers, labels, sc_interact,
            cfg.agent_speed_kmh, cfg.road_factor,
        )

        # ── Step 4: Cluster mandatory customers ──────────────────────
        #
        # Rather than routing all mandatory accounts as one flat
        # nearest-neighbour chain, group them by geography first.
        # Each geographic cluster is visited with a TSP-optimal internal
        # route; clusters are then sequenced nearest-first.
        # Mandatory outliers (isolated accounts) are slotted in between.
        mandatory_customers = [customers[i] for i in mandatory_idx]
        mand_interact       = [interact_all[i] for i in mandatory_idx]
        mand_V_adj          = [V_adj_all[i]    for i in mandatory_idx]

        mand_clusters, mand_outliers_local = self._cluster_mandatory(
            mandatory_customers, mand_V_adj, mand_interact,
            cfg.eps_base_rad, cfg.alpha, cfg.min_cluster_size,
            cfg.agent_speed_kmh, cfg.road_factor,
        )

        # Compute total mandatory time (deducted from budget before greedy)
        mandatory_time = self._mandatory_total_time(
            mand_clusters, mand_outliers_local,
            mandatory_customers, mand_interact,
            cfg.agent_speed_kmh, cfg.road_factor,
        )

        # ── Step 5: Greedy select scoreable clusters + outliers ──────
        all_local = list(range(len(sc_customers)))
        sel_clusters, sel_outliers_local = greedy_select(
            sc_customers, all_local, clusters, outliers_local,
            sc_V_adj, sc_interact,
            cfg.daily_budget_minutes, cfg.acr_cap,
            mandatory_time, len(mandatory_idx),
        )

        # ── Step 6: Outlier absorption ───────────────────────────────
        sel_outlier_set = set(sel_outliers_local)
        zone_centroids  = compute_zone_centroids(sc_customers)
        remaining_outliers = [i for i in outliers_local if i not in sel_outlier_set]
        absorbed_local = absorb_outliers(
            sel_clusters, remaining_outliers,
            sc_customers, sc_V_adj, sc_interact,
            zone_centroids, cfg.agent_speed_kmh, cfg.road_factor,
            cfg.outlier_absorb_delta,
        )
        absorbed_set = set(absorbed_local)

        # ── Step 7: Build visit records ──────────────────────────────
        seq = 1

        # Mandatory visits — cluster-routed
        mandatory_visits = self._mandatory_visits_clustered(
            mand_clusters, mand_outliers_local,
            customers, mandatory_idx, scores,
            cfg.agent_speed_kmh, cfg.road_factor, seq,
        )
        seq += len(mandatory_visits)

        # Ranked visits (clusters + selected outliers)
        ranked_visits, seq = self._ranked_visits(
            sc_customers, sc_V_adj, sc_interact, scores, scoreable_idx,
            sel_clusters, sel_outliers_local, absorbed_set, seq,
        )

        # Escalation queue
        escalation_visits = self._escalation_visits(
            customers, escalation_idx, scores, seq
        )

        # ── Step 8: Watch list ───────────────────────────────────────
        selected_global = (
            set(mandatory_idx)
            | {scoreable_idx[i] for cl in sel_clusters for i in cl["member_indices"]}
            | {scoreable_idx[i] for i in sel_outliers_local}
            | {scoreable_idx[i] for i in absorbed_local}
        )
        unselected_idx = [i for i in range(len(customers)) if i not in selected_global]
        watch_list = build_watch_list(customers, unselected_idx, cfg.horizon_days)

        # ── Step 9: Assemble VisitPlan ───────────────────────────────
        all_in_plan = mandatory_visits + ranked_visits
        for rank, v in enumerate(all_in_plan, start=1):
            v.rank = rank

        planned_time      = sum(v.interaction_min + v.travel_minutes for v in all_in_plan)
        expected_recovery = sum(v.V_adj for v in all_in_plan)

        return VisitPlan(
            date=today,
            mandatory_visits=mandatory_visits,
            ranked_visits=ranked_visits,
            escalation_queue=escalation_visits,
            watch_list=watch_list,
            total_budget_min=cfg.daily_budget_minutes,
            planned_time_min=planned_time,
            expected_recovery=expected_recovery,
            customer_count=len(all_in_plan),
        )

    # ------------------------------------------------------------------
    # Mandatory clustering helpers
    # ------------------------------------------------------------------

    def _cluster_mandatory(
        self,
        mandatory_customers: list[Customer],
        mand_V_adj: list[float],
        mand_interact: list[float],
        eps_base_rad: float,
        alpha: float,
        min_cluster_size: int,
        speed_kmh: float,
        road_factor: float,
    ) -> tuple[list[dict], list[int]]:
        """
        Run DBSCAN + TSP on the mandatory pool exactly as for scoreable
        customers.  min_samples=2 so even a pair counts as a cluster.
        Returns (clusters, outlier_local_indices).
        """
        if not mandatory_customers:
            return [], []
        labels = value_weighted_dbscan(
            mandatory_customers, mand_V_adj,
            eps_base_rad, alpha,
            min_samples=2,
        )
        return build_clusters(
            mandatory_customers, labels, mand_interact,
            speed_kmh, road_factor,
        )

    def _mandatory_total_time(
        self,
        mand_clusters: list[dict],
        mand_outliers_local: list[int],
        mandatory_customers: list[Customer],
        mand_interact: list[float],
        speed_kmh: float,
        road_factor: float,
    ) -> float:
        """
        Estimate total time consumed by mandatory visits:
          - intra-cluster travel + interaction for each cluster
          - interaction time for standalone outlier mandatories
          - inter-stop travel (nearest-first across clusters + outliers)
        """
        if not mand_clusters and not mand_outliers_local:
            return 0.0

        from .clustering import compute_zone_centroids, assign_coordinates

        def _centroid(cl: dict) -> tuple[float, float]:
            lats = [cl["coords"][k][0] for k in range(len(cl["coords"]))]
            lons = [cl["coords"][k][1] for k in range(len(cl["coords"]))]
            return (sum(lats) / len(lats), sum(lons) / len(lons))

        total = sum(cl["total_time_min"] for cl in mand_clusters)
        total += sum(mand_interact[i] for i in mand_outliers_local)

        stops: list[tuple[float, float]] = [_centroid(cl) for cl in mand_clusters]
        zone_centroids = compute_zone_centroids(mandatory_customers)
        coords, _ = assign_coordinates(mandatory_customers, zone_centroids)
        for i in mand_outliers_local:
            stops.append(coords[i])

        if len(stops) > 1:
            current = stops[0]
            remaining = stops[1:]
            while remaining:
                nearest = min(
                    remaining,
                    key=lambda s: haversine_minutes(
                        current[0], current[1], s[0], s[1],
                        speed_kmh=speed_kmh, road_factor=road_factor,
                    ),
                )
                total += haversine_minutes(
                    current[0], current[1], nearest[0], nearest[1],
                    speed_kmh=speed_kmh, road_factor=road_factor,
                )
                current = nearest
                remaining.remove(nearest)

        return total

    def _mandatory_visits_clustered(
        self,
        mand_clusters: list[dict],
        mand_outliers_local: list[int],
        customers: list[Customer],
        mandatory_idx: list[int],
        scores: list[dict],
        speed_kmh: float,
        road_factor: float,
        seq_start: int,
    ) -> list[CustomerVisit]:
        """
        Build CustomerVisit records for all mandatory accounts.

        Clustered accounts are emitted in TSP route order within each
        cluster.  Clusters are sequenced nearest-first; isolated outlier
        mandatories are slotted in nearest-first among the clusters.
        """
        from .clustering import compute_zone_centroids, assign_coordinates

        mandatory_customers = [customers[i] for i in mandatory_idx]
        zone_centroids = compute_zone_centroids(mandatory_customers)
        coords, _ = assign_coordinates(mandatory_customers, zone_centroids)

        def _centroid(cl: dict) -> tuple[float, float]:
            lats = [cl["coords"][k][0] for k in range(len(cl["coords"]))]
            lons = [cl["coords"][k][1] for k in range(len(cl["coords"]))]
            return (sum(lats) / len(lats), sum(lons) / len(lons))

        stops = []
        for cl in mand_clusters:
            stops.append({"type": "cluster",  "centroid": _centroid(cl), "cluster": cl})
        for i in mand_outliers_local:
            stops.append({"type": "outlier",  "centroid": coords[i], "local_idx": i})

        if not stops:
            return []

        # Sequence stops nearest-first
        ordered = [stops[0]]
        remaining = stops[1:]
        current_pos = stops[0]["centroid"]
        while remaining:
            nearest = min(
                remaining,
                key=lambda s: haversine_minutes(
                    current_pos[0], current_pos[1],
                    s["centroid"][0], s["centroid"][1],
                    speed_kmh=speed_kmh, road_factor=road_factor,
                ),
            )
            ordered.append(nearest)
            remaining.remove(nearest)
            current_pos = nearest["centroid"]

        visits: list[CustomerVisit] = []
        seq = seq_start
        prev_exit: tuple[float, float] | None = None

        for stop in ordered:
            if stop["type"] == "cluster":
                cl         = stop["cluster"]
                route      = cl["route"]
                travel_mat = cl["travel_matrix"]
                mat_pos    = cl["mat_pos"]

                V_C = sum(scores[mandatory_idx[i]]["V_adj"] for i in cl["member_indices"])
                T_C = cl["total_time_min"]
                eff = V_C / T_C if T_C > 0 else 0.0

                for pos_in_route, local_idx in enumerate(route):
                    c = mandatory_customers[local_idx]
                    s = scores[mandatory_idx[local_idx]]

                    if pos_in_route == 0:
                        t_travel = (
                            haversine_minutes(
                                prev_exit[0], prev_exit[1],
                                coords[local_idx][0], coords[local_idx][1],
                                speed_kmh=speed_kmh, road_factor=road_factor,
                            ) if prev_exit is not None else 0.0
                        )
                    else:
                        prev_local = route[pos_in_route - 1]
                        t_travel = float(
                            travel_mat[mat_pos[prev_local], mat_pos[local_idx]]
                        )

                    total_time = s["interaction_min"] + t_travel
                    visit_eff  = s["V_adj"] / total_time if total_time > 0 else eff

                    visits.append(CustomerVisit(
                        rank=0,
                        customer_id=c.customer_id,
                        name=c.name,
                        osp=c.osp,
                        dpd=c.dpd,
                        probability=s["probability"],
                        V_i=s["V_i"],
                        V_adj=s["V_adj"],
                        urgency_boost=s["urgency_boost"],
                        efficiency=visit_eff,
                        travel_minutes=round(t_travel, 1),
                        interaction_min=s["interaction_min"],
                        cluster_id=cl["cluster_id"],
                        visit_sequence=seq,
                        rationale=self._mandatory_rationale(c),
                    ))
                    seq += 1

                prev_exit = coords[route[-1]]

            else:  # solo outlier mandatory
                local_idx = stop["local_idx"]
                c = mandatory_customers[local_idx]
                s = scores[mandatory_idx[local_idx]]

                t_travel = (
                    haversine_minutes(
                        prev_exit[0], prev_exit[1],
                        coords[local_idx][0], coords[local_idx][1],
                        speed_kmh=speed_kmh, road_factor=road_factor,
                    ) if prev_exit is not None else 0.0
                )

                total_time = s["interaction_min"] + t_travel
                eff        = s["V_adj"] / total_time if total_time > 0 else 0.0

                visits.append(CustomerVisit(
                    rank=0,
                    customer_id=c.customer_id,
                    name=c.name,
                    osp=c.osp,
                    dpd=c.dpd,
                    probability=s["probability"],
                    V_i=s["V_i"],
                    V_adj=s["V_adj"],
                    urgency_boost=s["urgency_boost"],
                    efficiency=eff,
                    travel_minutes=round(t_travel, 1),
                    interaction_min=s["interaction_min"],
                    cluster_id=None,
                    visit_sequence=seq,
                    rationale=self._mandatory_rationale(c),
                ))
                seq += 1
                prev_exit = coords[local_idx]

        return visits

    # ------------------------------------------------------------------
    # Remaining private helpers (unchanged from Phase 2)
    # ------------------------------------------------------------------

    def _mandatory_rationale(self, c: Customer) -> str:
        if c.reason_code == "WLD":
            return "Mandatory - willful defaulter"
        if c.reason_code == "ABS":
            return "Mandatory - abscond case"
        return "Mandatory - nominee OD case"

    def _ranked_visits(
        self,
        sc_customers: list[Customer],
        sc_V_adj: list[float],
        sc_interact: list[float],
        scores_all: list[dict],
        scoreable_idx: list[int],
        sel_clusters: list[dict],
        sel_outliers_local: list[int],
        absorbed_set: set[int],
        seq_start: int,
    ) -> tuple[list[CustomerVisit], int]:
        visits: list[CustomerVisit] = []
        seq = seq_start

        for cl in sorted(sel_clusters, key=lambda c: c.get("efficiency", 0.0), reverse=True):
            cid        = cl["cluster_id"]
            route      = cl["route"]
            travel_mat = cl["travel_matrix"]
            mat_pos    = cl["mat_pos"]
            size       = len(route)

            valid_members = [i for i in cl["member_indices"] if i < len(sc_V_adj)]
            V_C = sum(sc_V_adj[i] for i in valid_members)
            T_C = cl["total_time_min"]
            eff = V_C / T_C if T_C > 0 else 0.0

            for local_pos, local_idx in enumerate(route):
                if local_idx >= len(scoreable_idx):
                    continue
                c = sc_customers[local_idx]
                s = scores_all[scoreable_idx[local_idx]]

                if local_pos == 0:
                    t_travel = 0.0
                else:
                    prev_local = route[local_pos - 1]
                    if prev_local in mat_pos and local_idx in mat_pos:
                        t_travel = float(travel_mat[mat_pos[prev_local], mat_pos[local_idx]])
                    else:
                        # Absorbed outlier not in original travel matrix — fall back to haversine
                        prev_c = sc_customers[prev_local]
                        curr_c = sc_customers[local_idx]
                        t_travel = haversine_minutes(
                            prev_c.lat or 0.0, prev_c.lon or 0.0,
                            curr_c.lat or 0.0, curr_c.lon or 0.0,
                        )

                rationale = self._cluster_rationale(c, s, size, local_idx, absorbed_set)

                visits.append(CustomerVisit(
                    rank=0,
                    customer_id=c.customer_id,
                    name=c.name,
                    osp=c.osp,
                    dpd=c.dpd,
                    probability=s["probability"],
                    V_i=s["V_i"],
                    V_adj=s["V_adj"],
                    urgency_boost=s["urgency_boost"],
                    efficiency=eff,
                    travel_minutes=round(t_travel, 1),
                    interaction_min=s["interaction_min"],
                    cluster_id=cid,
                    visit_sequence=seq,
                    rationale=rationale,
                ))
                seq += 1

        for local_idx in sel_outliers_local:
            c = sc_customers[local_idx]
            s = scores_all[scoreable_idx[local_idx]]
            eff = outlier_efficiency(local_idx, sc_V_adj, sc_interact)
            visits.append(CustomerVisit(
                rank=0,
                customer_id=c.customer_id,
                name=c.name,
                osp=c.osp,
                dpd=c.dpd,
                probability=s["probability"],
                V_i=s["V_i"],
                V_adj=s["V_adj"],
                urgency_boost=s["urgency_boost"],
                efficiency=eff,
                travel_minutes=0.0,
                interaction_min=s["interaction_min"],
                cluster_id=None,
                visit_sequence=seq,
                rationale="High-value standalone visit",
            ))
            seq += 1

        visits.sort(key=lambda v: v.efficiency, reverse=True)
        return visits, seq

    def _cluster_rationale(
        self, c: Customer, s: dict, cluster_size: int,
        local_idx: int, absorbed_set: set[int],
    ) -> str:
        if local_idx in absorbed_set:
            return "High-value outlier, absorbed into cluster"
        if s["urgency_boost"] > 0:
            from .horizon import _next_bucket_boundary
            days = (_next_bucket_boundary(c.dpd) or c.dpd + 1) - c.dpd
            return f"Urgent - bucket boundary in {days} day(s)"
        if c.dpd <= 7:
            return "Fresh customer, high recovery value"
        return f"High-efficiency cluster ({cluster_size} customers)"

    def _escalation_visits(
        self,
        customers: list[Customer],
        escalation_idx: list[int],
        scores: list[dict],
        seq_start: int,
    ) -> list[CustomerVisit]:
        visits = []
        for offset, idx in enumerate(escalation_idx):
            c = customers[idx]
            s = scores[idx]
            if c.is_msd_zone:
                rationale = "Escalation - mass default zone"
            else:
                rationale = "Escalation - PTP broken 3+ times"
            visits.append(CustomerVisit(
                rank=0,
                customer_id=c.customer_id,
                name=c.name,
                osp=c.osp,
                dpd=c.dpd,
                probability=s["probability"],
                V_i=s["V_i"],
                V_adj=s["V_adj"],
                urgency_boost=s["urgency_boost"],
                efficiency=0.0,
                travel_minutes=0.0,
                interaction_min=s["interaction_min"],
                cluster_id=None,
                visit_sequence=seq_start + offset,
                rationale=rationale,
            ))
        return visits
