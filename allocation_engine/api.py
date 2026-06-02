"""
FastAPI REST service for the Allocation Engine.

Endpoints:
  POST /predict          — score a single customer (probability + V_i)
  POST /predict/batch    — score a list of customers
  POST /plan             — run full allocation pipeline, return VisitPlan
  POST /outcome          — log a visit outcome to the feedback store
  GET  /feedback/summary — summary stats over the feedback log
  GET  /health           — liveness check

Run with:
  uvicorn allocation_engine.api:app --reload --port 8000
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import EngineConfig
from .engine import AllocationEngine
from .feedback import log_visit_outcome, feedback_summary
from .models import Customer
from .probability import HeuristicModel


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Collection Allocation Engine",
    description="ML-powered visit planning for field collection agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared engine instance (stateless — safe for concurrent requests)
_engine = AllocationEngine(prob_model=HeuristicModel(), config=EngineConfig())


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------

class CustomerIn(BaseModel):
    customer_id:       str
    name:              str
    osp:               float
    dpd:               int
    due_date:          date
    lat:               Optional[float] = None
    lon:               Optional[float] = None
    zone_id:           str
    reason_code:       str
    is_ots:            bool = False
    settlement_amount: float = 0.0
    last_visit_date:   Optional[date] = None
    ptp_given:         int = 0
    ptp_kept:          int = 0
    ptp_broken:        int = 0
    contact_attempts:  int = 0
    is_mandatory:      bool = False
    is_msd_zone:       bool = False
    loan_product:      str = "GL"

    def to_customer(self) -> Customer:
        return Customer(**self.model_dump())


class PredictRequest(BaseModel):
    customer: CustomerIn


class PredictResponse(BaseModel):
    customer_id: str
    probability: float
    V_i:         float
    V_adj:       float
    urgency_boost: float
    interaction_min: float


class BatchPredictRequest(BaseModel):
    customers: list[CustomerIn]


class PlanRequest(BaseModel):
    customers: list[CustomerIn]
    plan_date: Optional[date] = None
    config: Optional[dict] = Field(
        default=None,
        description="Optional EngineConfig overrides (e.g. daily_budget_minutes, eps_base_km)"
    )


class OutcomeRequest(BaseModel):
    customer_id:                  str
    agent_id:                     str
    visit_timestamp:              datetime
    action_type:                  str = "visit"
    features_at_time:             dict = {}
    did_pay_after_visit:          bool
    amount_recovered_after_visit: float = 0.0
    recovery_timestamp:           Optional[datetime] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Score a single customer — returns probability and value metrics."""
    from .scoring import score_customer

    c = req.customer.to_customer()
    s = score_customer(c, _engine.prob_model, date.today(),
                       _engine.config.repeat_penalty_coeff,
                       _engine.config.penalty_decay_days)
    return PredictResponse(
        customer_id=c.customer_id,
        probability=s["probability"],
        V_i=s["V_i"],
        V_adj=s["V_adj"],
        urgency_boost=s["urgency_boost"],
        interaction_min=s["interaction_min"],
    )


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    """Score a list of customers in one call."""
    from .scoring import score_customer

    today = date.today()
    results = []
    for cin in req.customers:
        c = cin.to_customer()
        s = score_customer(c, _engine.prob_model, today,
                           _engine.config.repeat_penalty_coeff,
                           _engine.config.penalty_decay_days)
        results.append({
            "customer_id":   c.customer_id,
            "probability":   s["probability"],
            "V_i":           s["V_i"],
            "V_adj":         s["V_adj"],
            "urgency_boost": s["urgency_boost"],
            "interaction_min": s["interaction_min"],
        })
    return {"predictions": results, "count": len(results)}


@app.post("/plan")
def plan(req: PlanRequest):
    """
    Run the full allocation pipeline.
    Returns a VisitPlan with mandatory, ranked, escalation, and watch list.
    """
    customers = [c.to_customer() for c in req.customers]
    plan_date = req.plan_date or date.today()

    # Apply any config overrides
    engine = _engine
    if req.config:
        from dataclasses import replace
        cfg = replace(_engine.config, **{
            k: v for k, v in req.config.items()
            if hasattr(_engine.config, k)
        })
        engine = AllocationEngine(prob_model=_engine.prob_model, config=cfg)

    try:
        visit_plan = engine.run(customers, today=plan_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return asdict(visit_plan)


@app.post("/outcome")
def record_outcome(req: OutcomeRequest):
    """Log a visit outcome to the feedback store (JSONL)."""
    try:
        log_visit_outcome(
            customer_id=req.customer_id,
            agent_id=req.agent_id,
            visit_timestamp=req.visit_timestamp,
            action_type=req.action_type,
            features_at_time=req.features_at_time,
            did_pay_after_visit=req.did_pay_after_visit,
            amount_recovered_after_visit=req.amount_recovered_after_visit,
            recovery_timestamp=req.recovery_timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "logged", "customer_id": req.customer_id}


@app.get("/feedback/summary")
def get_feedback_summary():
    """Return summary stats over the feedback log."""
    return feedback_summary()
