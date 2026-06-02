"""Engine configuration — all tunable parameters in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    daily_budget_minutes:  float = 480.0
    acr_cap:               int   = 70
    repeat_penalty_coeff:  float = 0.35
    penalty_decay_days:    int   = 30
    eps_base_km:           float = 3.0       # DBSCAN base radius in km
    alpha:                 float = 0.5       # value expansion factor for epsilon
    min_cluster_size:      int   = 2
    road_factor:           float = 1.35      # haversine → road distance multiplier
    agent_speed_kmh:       float = 25.0
    outlier_absorb_delta:  float = 0.10      # max efficiency degradation to absorb outlier
    horizon_days:          int   = 5         # watch-list forward horizon
    use_maps_api:          bool  = False
    maps_api_key:          str   = field(default_factory=lambda: os.getenv("GOOGLE_MAPS_API_KEY", ""))

    @property
    def eps_base_rad(self) -> float:
        """DBSCAN epsilon in radians (required by sklearn haversine metric)."""
        return self.eps_base_km / 6371.0

    @property
    def daily_budget_hours(self) -> float:
        return self.daily_budget_minutes / 60.0
