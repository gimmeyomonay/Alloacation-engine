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

        # ── Step 3: Cluster scoreable customers only ─────────────────
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

        # ── Step 4: Compute mandatory visit time (interaction + travel) ─
        mandatory_customers = [customers[i] for i in mandatory_idx]
        mandatory_travel_legs = self._compute_mandatory_travel(
            mandatory_customers, cfg.agent_speed_kmh, cfg.road_factor
        )
        mandatory_time = (
            sum(interact_all[i] for i in mandatory_idx)
            + sum(mandatory_travel_legs)
        )
        all_local = list(range(len(sc_customers)))
        sel_clusters, sel_outliers_local = greedy_select(
            sc_customers, all_local, clusters, outliers_local,
            sc_V_adj, sc_interact,
            cfg.daily_budget_minutes, cfg.acr_cap,
            mandatory_time, len(mandatory_idx),
        )

        # ── Step 5: Outlier absorption ───────────────────────────────
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

        # ── Step 6: Build visit records ──────────────────────────────
        seq = 1   # global visit sequence counter

        # Mandatory visits
        mandatory_visits = self._mandatory_visits(
            customers, mandatory_idx, scores, seq, mandatory_travel_legs
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

        # ── Step 7: Watch list ───────────────────────────────────────
        selected_global = (
            set(mandatory_idx)
            | {scoreable_idx[i] for cl in sel_clusters for i in cl["member_indices"]}
            | {scoreable_idx[i] for i in sel_outliers_local}
            | {scoreable_idx[i] for i in absorbed_local}
        )
        unselected_idx = [i for i in range(len(customers)) if i not in selected_global]
        watch_list = build_watch_list(customers, unselected_idx, cfg.horizon_days)

        # ── Step 8: Assemble VisitPlan ───────────────────────────────
        all_in_plan = mandatory_visits + ranked_visits
        for rank, v in enumerate(all_in_plan, start=1):
            v.rank = rank

        planned_time = sum(v.interaction_min + v.travel_minutes for v in all_in_plan)
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
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_mandatory_travel(
        self,
        mandatory_customers: list[Customer],
        speed_kmh: float,
        road_factor: float,
    ) -> list[float]:
        """
        Compute per-leg travel times for mandatory visits using nearest-neighbour
        ordering. Returns a list of travel times (minutes) aligned with
        mandatory_customers — index 0 is always 0.0 (no prior travel for first visit).
        """
        from .routing import haversine_minutes
        from .clustering import compute_zone_centroids, assign_coordinates
        if not mandatory_customers:
            return []
        zone_centroids = compute_zone_centroids(mandatory_customers)
        coords, _ = assign_coordinates(mandatory_customers, zone_centroids)
        n = len(coords)
        if n == 1:
            return [0.0]
        # Nearest-neighbour ordering starting from index 0
        unvisited = list(range(1, n))
        order = [0]
        while unvisited:
            current = order[-1]
            nearest = min(
                unvisited,
                key=lambda j: haversine_minutes(
                    coords[current][0], coords[current][1],
                    coords[j][0], coords[j][1],
                    speed_kmh=speed_kmh, road_factor=road_factor,
                )
            )
            order.append(nearest)
            unvisited.remove(nearest)
        # Build per-leg travel times in order sequence
        legs_by_order = [0.0]
        for k in range(1, len(order)):
            prev, curr = order[k - 1], order[k]
            legs_by_order.append(haversine_minutes(
                coords[prev][0], coords[prev][1],
                coords[curr][0], coords[curr][1],
                speed_kmh=speed_kmh, road_factor=road_factor,
            ))
        # Re-map back to original customer order
        legs = [0.0] * n
        for pos, orig_idx in enumerate(order):
            legs[orig_idx] = legs_by_order[pos]
        return legs

    def _mandatory_visits(
        self,
        customers: list[Customer],
        mandatory_idx: list[int],
        scores: list[dict],
        seq_start: int,
        travel_legs: list[float] | None = None,
    ) -> list[CustomerVisit]:
        if travel_legs is None:
            travel_legs = [0.0] * len(mandatory_idx)
        visits = []
        for offset, idx in enumerate(mandatory_idx):
            c = customers[idx]
            s = scores[idx]
            t_travel = travel_legs[offset] if offset < len(travel_legs) else 0.0
            total_time = s["interaction_min"] + t_travel
            eff = s["V_adj"] / total_time if total_time > 0 else 0.0
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
                visit_sequence=seq_start + offset,
                rationale=self._mandatory_rationale(c),
            ))
        return visits

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

        cfg = self.config

        # Sort clusters by efficiency descending for visit ordering
        for cl in sorted(sel_clusters, key=lambda c: c.get("efficiency", 0.0), reverse=True):
            cid = cl["cluster_id"]
            route = cl["route"]           # ordered local indices
            size  = len(route)
            travel_mat = cl["travel_matrix"]
            # Recompute efficiency using only members within sc_V_adj bounds
            # (absorbed outliers may have extended member_indices)
            valid_members = [i for i in cl["member_indices"] if i < len(sc_V_adj)]
            V_C = sum(sc_V_adj[i] for i in valid_members)
            T_C = cl["total_time_min"]
            eff = V_C / T_C if T_C > 0 else 0.0

            for local_pos, local_idx in enumerate(route):
                # Guard: skip absorbed outlier indices that exceed scoreable_idx
                if local_idx >= len(scoreable_idx):
                    continue
                c   = sc_customers[local_idx]
                s   = scores_all[scoreable_idx[local_idx]]

                # Travel time: first member gets 0 (depot start), rest get leg time
                if local_pos == 0:
                    t_travel = 0.0
                else:
                    prev_local = route[local_pos - 1]
                    t_travel = float(travel_mat[prev_local, local_idx])

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
                    travel_minutes=t_travel,
                    interaction_min=s["interaction_min"],
                    cluster_id=cid,
                    visit_sequence=seq,
                    rationale=rationale,
                ))
                seq += 1

        # Standalone selected outliers
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

        # Re-sort the full ranked list by efficiency descending
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
