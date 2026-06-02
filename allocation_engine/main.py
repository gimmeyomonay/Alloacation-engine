"""
CLI entry point for the Allocation Engine.

Usage examples:
  python -m allocation_engine.main --synthetic --n 70
  python -m allocation_engine.main --input portfolio.csv --date 2026-06-02
  python -m allocation_engine.main --synthetic --output plan.json
  python -m allocation_engine.main --synthetic --use-maps-api
  python -m allocation_engine.main --replan --remaining-budget 210 --outcomes outcomes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from dataclasses import asdict

# ── Try rich for pretty tables; fall back to plain text ──────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None  # type: ignore


def _print(msg: str = "") -> None:
    if _RICH:
        console.print(msg)
    else:
        print(msg)


# ── Import engine pieces ─────────────────────────────────────────────────────
from .config import EngineConfig
from .engine import AllocationEngine
from .models import Customer, VisitOutcome
from .probability import HeuristicModel
from .data_gen import generate_synthetic_portfolio


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[Customer]:
    import csv
    from datetime import date as _date

    def _parse(row: dict) -> dict:
        bool_fields  = {"is_ots", "is_mandatory", "is_msd_zone"}
        int_fields   = {"dpd", "ptp_given", "ptp_kept", "ptp_broken", "contact_attempts"}
        float_fields = {"osp", "settlement_amount", "lat", "lon"}
        date_fields  = {"due_date", "last_visit_date"}

        out = {}
        for k, v in row.items():
            v = v.strip()
            if k in bool_fields:
                out[k] = v.lower() in ("true", "1", "yes")
            elif k in int_fields:
                out[k] = int(v) if v else 0
            elif k in float_fields:
                out[k] = float(v) if v else None
            elif k in date_fields:
                out[k] = _date.fromisoformat(v) if v else None
            else:
                out[k] = v
        return out

    customers = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            customers.append(Customer.from_dict(_parse(row)))
    return customers


def _load_outcomes(path: str) -> list[VisitOutcome]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [VisitOutcome(**item) for item in data]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _date_serial(obj):
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {obj!r}")


def _export_json(plan, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, default=_date_serial, indent=2, ensure_ascii=False)
    _print(f"[green]Plan exported to {path}[/green]" if _RICH else f"Plan exported to {path}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _display_plan(plan) -> None:
    _print()

    if _RICH:
        # ── Summary ──────────────────────────────────────────────────────────
        summary = Table(title="Visit Plan Summary", box=box.SIMPLE_HEAVY, show_header=False)
        summary.add_column("Metric", style="bold cyan", width=30)
        summary.add_column("Value", justify="right")
        summary.add_row("Date",               str(plan.date))
        summary.add_row("Total budget",       f"{plan.total_budget_min:.0f} min")
        summary.add_row("Planned time",       f"{plan.planned_time_min:.0f} min  ({plan.planned_time_min/plan.total_budget_min*100:.0f}%)")
        summary.add_row("Accounts in plan",   str(plan.customer_count))
        summary.add_row("Mandatory visits",   str(len(plan.mandatory_visits)))
        summary.add_row("Ranked visits",      str(len(plan.ranked_visits)))
        summary.add_row("Escalation queue",   str(len(plan.escalation_queue)))
        summary.add_row("Watch list",         str(len(plan.watch_list)))
        summary.add_row("Expected recovery",  f"Rs{plan.expected_recovery:,.0f}")
        console.print(summary)

        # ── Mandatory ────────────────────────────────────────────────────────
        if plan.mandatory_visits:
            _print_visit_table("Mandatory Visits", plan.mandatory_visits)

        # ── Ranked ───────────────────────────────────────────────────────────
        if plan.ranked_visits:
            _print_visit_table("Ranked Visits", plan.ranked_visits)

        # ── Escalation ───────────────────────────────────────────────────────
        if plan.escalation_queue:
            _print_visit_table("Escalation Queue", plan.escalation_queue, show_efficiency=False)

        # ── Watch list ───────────────────────────────────────────────────────
        if plan.watch_list:
            wt = Table(title="Watch List (5-day horizon)", box=box.SIMPLE)
            for col in ("ID", "Name", "DPD", "Days to Boundary", "Proj. V (Rs)", "Urgency (Rs)", "Score"):
                wt.add_column(col)
            for w in plan.watch_list[:15]:
                wt.add_row(
                    w.customer_id, w.name, str(w.current_dpd),
                    str(w.days_to_boundary),
                    f"{w.projected_V:,.0f}", f"{w.projected_urgency:,.0f}",
                    f"{w.score:,.0f}",
                )
            console.print(wt)

    else:
        # ── Plain-text fallback ───────────────────────────────────────────────
        print(f"\n=== Visit Plan: {plan.date} ===")
        print(f"Budget:           {plan.planned_time_min:.0f}/{plan.total_budget_min:.0f} min")
        print(f"Accounts in plan: {plan.customer_count}")
        print(f"Mandatory:        {len(plan.mandatory_visits)}")
        print(f"Ranked:           {len(plan.ranked_visits)}")
        print(f"Escalation:       {len(plan.escalation_queue)}")
        print(f"Watch list:       {len(plan.watch_list)}")
        print(f"Expected recovery: Rs{plan.expected_recovery:,.0f}\n")

        _print_plain_table("MANDATORY", plan.mandatory_visits)
        _print_plain_table("RANKED", plan.ranked_visits)
        _print_plain_table("ESCALATION", plan.escalation_queue, show_efficiency=False)

        if plan.watch_list:
            print("\nWATCH LIST (top 10):")
            print(f"  {'ID':<12} {'Name':<20} {'DPD':>4} {'Days':>5} {'Score':>10}")
            for w in plan.watch_list[:10]:
                print(f"  {w.customer_id:<12} {w.name:<20} {w.current_dpd:>4} {w.days_to_boundary:>5} {w.score:>10,.0f}")


def _print_visit_table(title: str, visits, show_efficiency: bool = True) -> None:
    if _RICH:
        t = Table(title=title, box=box.SIMPLE)
        cols = ["#", "ID", "Name", "DPD", "RC", "OSP (Rs)", "V_adj (Rs)", "Travel", "Interact", "Cluster", "Rationale"]
        if show_efficiency:
            cols.insert(7, "Eff (Rs/min)")
        for col in cols:
            t.add_column(col)
        for v in visits:
            row = [
                str(v.visit_sequence), v.customer_id, v.name[:16],
                str(v.dpd), "", f"{v.osp:,.0f}", f"{v.V_adj:,.0f}",
            ]
            if show_efficiency:
                row.append(f"{v.efficiency:.1f}")
            row += [
                f"{v.travel_minutes:.0f}m", f"{v.interaction_min:.0f}m",
                str(v.cluster_id) if v.cluster_id is not None else "-",
                v.rationale[:40],
            ]
            t.add_row(*row)
        console.print(t)


def _print_plain_table(title: str, visits, show_efficiency: bool = True) -> None:
    if not visits:
        return
    print(f"\n{title} ({len(visits)} accounts):")
    hdr = f"  {'#':>3} {'ID':<12} {'Name':<20} {'DPD':>4} {'OSP':>10} {'V_adj':>10}"
    if show_efficiency:
        hdr += f" {'Eff':>8}"
    hdr += f"  Rationale"
    print(hdr)
    for v in visits:
        row = f"  {v.visit_sequence:>3} {v.customer_id:<12} {v.name:<20} {v.dpd:>4} {v.osp:>10,.0f} {v.V_adj:>10,.0f}"
        if show_efficiency:
            row += f" {v.efficiency:>8.1f}"
        row += f"  {v.rationale}"
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collection Agent Allocation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="Use synthetic portfolio")
    src.add_argument("--input", metavar="FILE", help="CSV portfolio file")
    src.add_argument("--replan", action="store_true", help="Run mid-day replan")

    p.add_argument("--n",               type=int,   default=70,   help="Synthetic portfolio size (default 70)")
    p.add_argument("--seed",            type=int,   default=42,   help="Synthetic data seed (default 42)")
    p.add_argument("--date",            type=str,   default=None, help="Plan date YYYY-MM-DD (default today)")
    p.add_argument("--output",          type=str,   default=None, help="Export plan to JSON file")
    p.add_argument("--use-maps-api",    action="store_true")
    p.add_argument("--remaining-budget",type=float, default=240.0,help="Remaining budget in minutes (replan)")
    p.add_argument("--outcomes",        type=str,   default=None, help="JSON file with VisitOutcome records (replan)")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    plan_date = date.fromisoformat(args.date) if args.date else date.today()
    cfg = EngineConfig(use_maps_api=args.use_maps_api)
    engine = AllocationEngine(prob_model=HeuristicModel(), config=cfg)

    # ── Load customers ───────────────────────────────────────────────────────
    if args.replan:
        if not args.input:
            _print("[red]--replan requires --input portfolio.csv[/red]" if _RICH
                   else "--replan requires --input portfolio.csv")
            sys.exit(1)
        customers = _load_csv(args.input)
        outcomes  = _load_outcomes(args.outcomes) if args.outcomes else []
        from .horizon import replan
        plan = replan(args.remaining_budget, outcomes, customers, engine)

    elif args.input:
        customers = _load_csv(args.input)
        plan = engine.run(customers, today=plan_date)

    else:
        # Default: synthetic
        raw = generate_synthetic_portfolio(n=args.n, seed=args.seed)
        customers = [Customer.from_dict(r) for r in raw]
        plan = engine.run(customers, today=plan_date)

    # ── Display ──────────────────────────────────────────────────────────────
    _display_plan(plan)

    # ── Export ───────────────────────────────────────────────────────────────
    if args.output:
        _export_json(plan, args.output)


if __name__ == "__main__":
    main()
