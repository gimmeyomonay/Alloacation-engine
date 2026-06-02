"""
Streamlit dashboard for the Allocation Engine.
Run with: streamlit run allocation_engine/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

from allocation_engine.config import EngineConfig
from allocation_engine.engine import AllocationEngine
from allocation_engine.models import Customer
from allocation_engine.probability import HeuristicModel
from allocation_engine.data_gen import generate_synthetic_portfolio

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Field Collection — Visit Planner",
    page_icon="🗺️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #7c3aed;
    }
    .tag-mandatory  { background:#dc2626; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
    .tag-escalation { background:#d97706; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
    .tag-ranked     { background:#16a34a; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
    .tag-watch      { background:#2563eb; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Engine Controls")

# Data source
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio("Choose input", ["Synthetic data", "Upload CSV"], index=0)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload portfolio CSV",
        type=["csv"],
        help="CSV must match the engine schema — download the demo CSV for reference"
    )
    # Template — headers + 5 example rows
    with open(os.path.join(os.path.dirname(__file__), "..", "portfolio_template.csv"), "rb") as f:
        st.sidebar.download_button(
            "⬇ Download blank template",
            data=f,
            file_name="portfolio_template.csv",
            mime="text/csv",
            help="Headers + 5 example rows showing the expected format",
        )
    # Full demo dataset — 120 realistic customers
    with open(os.path.join(os.path.dirname(__file__), "..", "demo_portfolio.csv"), "rb") as f:
        st.sidebar.download_button(
            "⬇ Download demo dataset (120 customers)",
            data=f,
            file_name="demo_portfolio.csv",
            mime="text/csv",
            help="Fully populated demo portfolio — upload this to see the engine in action",
        )

st.sidebar.markdown("---")
st.sidebar.subheader("Synthetic Data Settings")
n_customers = st.sidebar.slider("Portfolio size", 50, 200, 120, step=10,
                                 disabled=(data_source == "Upload CSV"))
seed        = st.sidebar.number_input("Random seed", value=99, step=1,
                                       disabled=(data_source == "Upload CSV"))

st.sidebar.markdown("---")
st.sidebar.subheader("Engine Config")
budget_hours   = st.sidebar.slider("Daily budget (hours)", 6, 12, 8)
acr_cap        = st.sidebar.slider("Max accounts (ACR cap)", 20, 100, 70)
eps_base_km    = st.sidebar.slider("Cluster radius (km)", 1.0, 10.0, 3.0, step=0.5)
mandatory_pct  = st.sidebar.slider("Mandatory account %", 0, 30, 6,
                                    help="Only applies to synthetic data",
                                    disabled=(data_source == "Upload CSV"))

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶ Run Engine", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state — cache last run
# ---------------------------------------------------------------------------
if "plan" not in st.session_state:
    st.session_state.plan = None
    st.session_state.customers = None

# ---------------------------------------------------------------------------
# Run engine
# ---------------------------------------------------------------------------
def run_engine(n, seed, budget_hours, acr_cap, eps_base_km, mandatory_pct):
    raw = generate_synthetic_portfolio(n=n, seed=int(seed))

    # Apply mandatory_pct override
    n_mandatory = round(n * mandatory_pct / 100)
    for i, r in enumerate(raw):
        r["is_mandatory"] = i < n_mandatory

    import random
    rng = random.Random(int(seed) + 1)
    rng.shuffle(raw)

    customers = [Customer.from_dict(r) for r in raw]

    cfg = EngineConfig(
        daily_budget_minutes=budget_hours * 60,
        acr_cap=acr_cap,
        eps_base_km=eps_base_km,
    )
    engine = AllocationEngine(prob_model=HeuristicModel(), config=cfg)
    plan   = engine.run(customers, today=date.today())
    return plan, customers


# Auto-run on first load
if st.session_state.plan is None or run_btn:
    if data_source == "Upload CSV" and uploaded_file is not None:
        with st.spinner("Loading CSV and running engine..."):
            from allocation_engine.main import _load_csv
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                shutil.copyfileobj(uploaded_file, tmp)
                tmp_path = tmp.name
            customers = _load_csv(tmp_path)
            cfg = EngineConfig(
                daily_budget_minutes=budget_hours * 60,
                acr_cap=acr_cap,
                eps_base_km=eps_base_km,
            )
            engine = AllocationEngine(prob_model=HeuristicModel(), config=cfg)
            plan   = engine.run(customers, today=date.today())
        st.session_state.plan      = plan
        st.session_state.customers = customers
    elif data_source == "Upload CSV" and uploaded_file is None:
        st.info("Upload a CSV file in the sidebar to get started, or switch to Synthetic data.")
        st.stop()
    else:
        with st.spinner("Running allocation engine..."):
            plan, customers = run_engine(
                n_customers, seed, budget_hours, acr_cap, eps_base_km, mandatory_pct
            )
        st.session_state.plan      = plan
        st.session_state.customers = customers

plan      = st.session_state.plan
customers = st.session_state.customers

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🗺️ Field Collection — Daily Visit Plan")
st.caption(f"Plan date: {plan.date}  |  Generated by Allocation Engine v1.0")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
utilisation = plan.planned_time_min / plan.total_budget_min * 100
k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Accounts in Plan",   plan.customer_count)
k2.metric("Expected Recovery",  f"₹{plan.expected_recovery:,.0f}")
k3.metric("Time Utilisation",   f"{utilisation:.0f}%",
          delta=f"{plan.planned_time_min:.0f}/{plan.total_budget_min:.0f} min")
k4.metric("Mandatory Visits",   len(plan.mandatory_visits))
k5.metric("Escalation Queue",   len(plan.escalation_queue))
k6.metric("Watch List",         len(plan.watch_list))

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Map", "📋 Visit Plan", "⚠️ Escalation", "👁️ Watch List", "📊 Analytics"
])

# ── helpers ──────────────────────────────────────────────────────────────────

def visits_to_df(visits, label):
    rows = []
    for v in visits:
        rows.append({
            "Seq":        v.visit_sequence,
            "ID":         v.customer_id,
            "Name":       v.name,
            "DPD":        v.dpd,
            "OSP (Rs)":   round(v.osp),
            "V_adj (Rs)": round(v.V_adj),
            "Prob":        f"{v.probability:.0%}",
            "Eff (Rs/min)": round(v.efficiency, 1),
            "Travel (min)": round(v.travel_minutes),
            "Interact (min)": round(v.interaction_min),
            "Cluster":    v.cluster_id if v.cluster_id is not None else "–",
            "Rationale":  v.rationale,
            "Type":       label,
        })
    return pd.DataFrame(rows)


# ── TAB 1: Map ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Customer Locations & Visit Plan")

    # Build map dataframe
    map_rows = []
    cust_map = {c.customer_id: c for c in customers}

    all_visits = (
        [(v, "Mandatory",  "#ef4444") for v in plan.mandatory_visits] +
        [(v, "Ranked",     "#22c55e") for v in plan.ranked_visits] +
        [(v, "Escalation", "#f59e0b") for v in plan.escalation_queue]
    )

    for v, vtype, color in all_visits:
        c = cust_map.get(v.customer_id)
        if c and c.lat is not None:
            map_rows.append({
                "lat":   c.lat,
                "lon":   c.lon,
                "name":  v.name,
                "type":  vtype,
                "dpd":   v.dpd,
                "V_adj": round(v.V_adj),
                "seq":   v.visit_sequence,
            })

    if map_rows:
        map_df = pd.DataFrame(map_rows)

        # Color map
        col1, col2 = st.columns([3, 1])
        with col1:
            st.map(map_df, latitude="lat", longitude="lon", size=80)
        with col2:
            st.markdown("**Legend**")
            st.markdown('<span class="tag-mandatory">Mandatory</span>', unsafe_allow_html=True)
            st.markdown('<span class="tag-ranked">Ranked</span>',     unsafe_allow_html=True)
            st.markdown('<span class="tag-escalation">Escalation</span>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"**{len(map_rows)}** customers plotted")
            no_gps = sum(1 for c in customers if c.lat is None)
            st.markdown(f"**{no_gps}** missing GPS (not shown)")
    else:
        st.info("No GPS coordinates available to plot.")

# ── TAB 2: Visit Plan ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Mandatory Visits")
    st.caption("Pre-pinned — bypass efficiency ranking")
    if plan.mandatory_visits:
        df_m = visits_to_df(plan.mandatory_visits, "Mandatory")
        st.dataframe(df_m.drop(columns=["Type"]), use_container_width=True, hide_index=True)
    else:
        st.info("No mandatory visits today.")

    st.markdown("---")
    st.subheader("Ranked Visits")
    st.caption("Sorted by efficiency (₹ recovered per minute)")
    if plan.ranked_visits:
        df_r = visits_to_df(plan.ranked_visits, "Ranked")
        st.dataframe(df_r.drop(columns=["Type"]), use_container_width=True, hide_index=True)
    else:
        st.info("No ranked visits — mandatory accounts have consumed the full daily budget.")
        st.markdown(
            f"**Tip:** Reduce mandatory % in the sidebar (currently "
            f"{mandatory_pct}%) to free up budget for ranked visits."
        )

# ── TAB 3: Escalation ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Escalation Queue")
    st.caption("MSD zone accounts and repeat PTP breakers — route to supervisor / NCM")
    if plan.escalation_queue:
        df_e = visits_to_df(plan.escalation_queue, "Escalation")
        st.dataframe(df_e.drop(columns=["Type", "Eff (Rs/min)"]), use_container_width=True, hide_index=True)
    else:
        st.success("No escalation cases today.")

# ── TAB 4: Watch List ────────────────────────────────────────────────────────
with tab4:
    st.subheader("5-Day Watch List")
    st.caption("Accounts not visited today that will cross a DPD bucket boundary within 5 days")
    if plan.watch_list:
        watch_rows = [{
            "ID":              w.customer_id,
            "Name":            w.name,
            "Current DPD":     w.current_dpd,
            "Days to Boundary": w.days_to_boundary,
            "Projected DPD":   w.projected_dpd,
            "Proj. V (Rs)":    round(w.projected_V),
            "Urgency (Rs)":    round(w.projected_urgency),
            "Score":           round(w.score),
        } for w in plan.watch_list]
        df_w = pd.DataFrame(watch_rows)
        st.dataframe(df_w, use_container_width=True, hide_index=True)

        # Bar chart of top watch list scores
        st.markdown("**Top 10 by Score**")
        chart_df = df_w.head(10).set_index("Name")["Score"]
        st.bar_chart(chart_df)
    else:
        st.success("No accounts approaching bucket boundaries in the next 5 days.")

# ── TAB 5: Analytics ─────────────────────────────────────────────────────────
with tab5:
    st.subheader("Portfolio Analytics")

    col1, col2 = st.columns(2)

    # DPD distribution
    with col1:
        st.markdown("**DPD Distribution**")
        dpd_vals = [c.dpd for c in customers]
        bins = [0, 7, 30, 60, 90, 180, 999]
        labels = ["1–7", "8–30", "31–60", "61–90", "91–180", "180+"]
        counts = pd.cut(dpd_vals, bins=bins, labels=labels).value_counts().sort_index()
        st.bar_chart(counts)

    # Reason code breakdown
    with col2:
        st.markdown("**Reason Code Breakdown**")
        rc_counts = pd.Series([c.reason_code for c in customers]).value_counts()
        st.bar_chart(rc_counts)

    col3, col4 = st.columns(2)

    # OSP distribution
    with col3:
        st.markdown("**OSP Distribution (Rs)**")
        osp_vals   = [c.osp for c in customers]
        osp_bins   = [0, 25000, 50000, 100000, 150000, 200000, float("inf")]
        osp_labels = ["<25k", "25-50k", "50-100k", "1-1.5L", "1.5-2L", ">2L"]
        osp_counts = (
            pd.cut(osp_vals, bins=osp_bins, labels=osp_labels, right=False)
            .value_counts()
            .reindex(osp_labels, fill_value=0)
        )
        st.bar_chart(osp_counts)

    # Plan composition
    with col4:
        st.markdown("**Plan Composition**")
        comp = pd.Series({
            "Mandatory":  len(plan.mandatory_visits),
            "Ranked":     len(plan.ranked_visits),
            "Escalation": len(plan.escalation_queue),
            "Unvisited":  len(customers) - plan.customer_count - len(plan.escalation_queue),
        })
        st.bar_chart(comp)

    # V_adj vs DPD scatter proxy (bar by bucket)
    st.markdown("**Average Adjusted Value (V_adj) by DPD Bucket**")
    all_plan_visits = plan.mandatory_visits + plan.ranked_visits
    if all_plan_visits:
        bucket_v = {}
        for v in all_plan_visits:
            for lo, hi, label in [(1,7,"1–7"),(8,30,"8–30"),(31,60,"31–60"),
                                   (61,90,"61–90"),(91,180,"91–180"),(181,999,"180+")]:
                if lo <= v.dpd <= hi:
                    bucket_v.setdefault(label, []).append(v.V_adj)
        avg_v = {k: round(sum(vs)/len(vs)) for k, vs in bucket_v.items()}
        st.bar_chart(pd.Series(avg_v))
