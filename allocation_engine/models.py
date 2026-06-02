"""Core dataclasses for the allocation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Customer:
    customer_id:       str
    name:              str
    osp:               float
    dpd:               int
    due_date:          date
    lat:               Optional[float]
    lon:               Optional[float]
    zone_id:           str
    reason_code:       str
    is_ots:            bool
    settlement_amount: float
    last_visit_date:   Optional[date]
    ptp_given:         int
    ptp_kept:          int
    ptp_broken:        int
    contact_attempts:  int
    is_mandatory:      bool
    is_msd_zone:       bool
    loan_product:      str

    @classmethod
    def from_dict(cls, d: dict) -> "Customer":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def amount(self) -> float:
        return self.settlement_amount if self.is_ots else self.osp


@dataclass
class CustomerVisit:
    rank:            int
    customer_id:     str
    name:            str
    osp:             float
    dpd:             int
    probability:     float
    V_i:             float         # expected recovery
    V_adj:           float         # adjusted value after penalties/boost
    urgency_boost:   float
    efficiency:      float         # V_adj / total_time (per minute)
    travel_minutes:  float
    interaction_min: float
    cluster_id:      Optional[int]
    visit_sequence:  int
    rationale:       str


@dataclass
class WatchItem:
    customer_id:        str
    name:               str
    osp:                float
    current_dpd:        int
    days_to_boundary:   int         # days until next bucket crossing
    projected_dpd:      int
    projected_V:        float
    projected_urgency:  float
    score:              float       # projected_urgency + projected_V


@dataclass
class VisitPlan:
    date:              date
    mandatory_visits:  list[CustomerVisit] = field(default_factory=list)
    ranked_visits:     list[CustomerVisit] = field(default_factory=list)
    escalation_queue:  list[CustomerVisit] = field(default_factory=list)
    watch_list:        list[WatchItem]     = field(default_factory=list)
    total_budget_min:  float = 480.0
    planned_time_min:  float = 0.0
    expected_recovery: float = 0.0
    customer_count:    int   = 0


@dataclass
class VisitOutcome:
    """Passed to replan() after a mid-day visit is completed."""
    customer_id:        str
    outcome:            str     # "recovered" | "ptp_given" | "confirmed_abscond" | "no_contact"
    amount_recovered:   float = 0.0
    ptp_amount:         float = 0.0
