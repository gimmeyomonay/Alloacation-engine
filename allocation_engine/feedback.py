"""
Feedback / outcome logger.

Every visit outcome is appended to a JSONL file (one JSON object per line).
This file is the raw training data source for Phase 2 model retraining.

Schema logged per action:
  customer_id, agent_id, visit_timestamp, action_type,
  features_at_time, did_pay_after_visit,
  amount_recovered_after_visit, recovery_timestamp
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Default log path — override via FEEDBACK_LOG_PATH env var
DEFAULT_LOG_PATH = Path(__file__).parent.parent / "feedback_log.jsonl"


def _get_log_path() -> Path:
    env = os.getenv("FEEDBACK_LOG_PATH")
    return Path(env) if env else DEFAULT_LOG_PATH


def _serialise(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {obj!r}")


def log_visit_outcome(
    customer_id: str,
    agent_id: str,
    visit_timestamp: datetime,
    action_type: str,                        # "visit" | "call" | "digital"
    features_at_time: dict,                  # snapshot of customer features used for scoring
    did_pay_after_visit: bool,
    amount_recovered_after_visit: float = 0.0,
    recovery_timestamp: Optional[datetime] = None,
    log_path: Optional[Path] = None,
) -> None:
    """
    Append one visit outcome record to the JSONL feedback log.
    Thread-safe via line-append (each write is atomic on most OS).
    """
    record = {
        "customer_id":                  customer_id,
        "agent_id":                     agent_id,
        "visit_timestamp":              visit_timestamp.isoformat(),
        "action_type":                  action_type,
        "features_at_time":             features_at_time,
        "did_pay_after_visit":          did_pay_after_visit,
        "amount_recovered_after_visit": amount_recovered_after_visit,
        "recovery_timestamp":           recovery_timestamp.isoformat() if recovery_timestamp else None,
    }

    path = log_path or _get_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_serialise) + "\n")


def load_feedback_log(log_path: Optional[Path] = None) -> list[dict]:
    """Load all records from the JSONL feedback log. Returns [] if file doesn't exist."""
    path = log_path or _get_log_path()
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def feedback_summary(log_path: Optional[Path] = None) -> dict:
    """Return summary stats over the feedback log — useful for monitoring."""
    records = load_feedback_log(log_path)
    if not records:
        return {"total_records": 0}

    paid = sum(1 for r in records if r["did_pay_after_visit"])
    total_recovered = sum(r["amount_recovered_after_visit"] for r in records)

    return {
        "total_records":        len(records),
        "paid_count":           paid,
        "contact_to_recovery":  round(paid / len(records) * 100, 1),
        "total_recovered":      round(total_recovered, 2),
        "avg_recovered":        round(total_recovered / max(paid, 1), 2),
        "agents":               len({r["agent_id"] for r in records}),
        "date_range": {
            "from": min(r["visit_timestamp"] for r in records)[:10],
            "to":   max(r["visit_timestamp"] for r in records)[:10],
        },
    }
