"""Core dataclasses for the allocation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Customer:
    # ── Phase 1 fields ───────────────────────────────────────────────────────
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

    # ── Phase 2 fields — EMI / obligation context ────────────────────────────
    emi_amount:                   float = 0.0
    number_of_missed_installments: int  = 0
    next_due_amount:              float = 0.0

    # ── Phase 2 fields — payment behaviour history ───────────────────────────
    last_paid_amount:             float = 0.0
    last_paid_date:               Optional[date] = None
    payment_count_last_90_days:   int   = 0
    total_paid_last_30_days:      float = 0.0
    avg_payment_amount:           float = 0.0

    # ── Phase 2 fields — repeat / stability indicators ───────────────────────
    delinquency_cycle_count:      int   = 0
    times_rolled_back_to_current: int   = 0
    months_since_first_delinquency: int = 0

    # ── Phase 2 fields — contactability ─────────────────────────────────────
    contact_status:               str   = "reachable"  # reachable|unreachable|refused
    last_contact_date:            Optional[date] = None

    # ── Phase 2 fields — ML outcome label (for training data) ────────────────
    did_pay_after_visit:          Optional[bool]  = None
    amount_recovered_after_visit: Optional[float] = None

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
    days_to_boundary:   int
    projected_dpd:      int
    projected_V:        float
    projected_urgency:  float
    score:              float


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
