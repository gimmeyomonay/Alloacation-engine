"""
Streamlit dashboard for the Allocation Engine.
Run with: streamlit run allocation_engine/app.py
"""

import sys
import os
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    _git_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    _git_branch = "unknown"

from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

from allocation_engine.config import EngineConfig
from allocation_engine.engine import AllocationEngine
from allocation_engine.models import Customer
from allocation_engine.probability import HeuristicModel, XGBoostModel
from allocation_engine.model_registry import ModelRegistry
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
st.sidebar.info(f"🌿 branch: **{_git_branch}**")

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
st.sidebar.subheader("Probability Model")

_registry = ModelRegistry()
_active_model_info = _registry.get_active()
_xgb_available = _active_model_info is not None and os.path.exists(_active_model_info["model_path"])

model_choice = st.sidebar.radio(
    "Select model",
    ["Heuristic (rule-based)", "XGBoost (ML)"],
    index=1 if _xgb_available else 0,
    disabled=not _xgb_available,
    help="XGBoost uses the trained ML model. Falls back to heuristic if no model is found.",
)

if model_choice == "XGBoost (ML)" and _xgb_available:
    st.sidebar.markdown(
        f"<div style='background:#1e2a1e;border-left:3px solid #22c55e;"
        f"padding:8px 12px;border-radius:4px;font-size:12px;'>"
        f"<b style='color:#22c55e'>XGBoost active</b><br/>"
        f"Version: {_active_model_info['version_id']} &nbsp;|&nbsp; "
        f"AUC: {_active_model_info['auc']}<br/>"
        f"Trained: {_active_model_info['trained_date']}<br/>"
        f"Records: {_active_model_info['n_records']:,}</div>",
        unsafe_allow_html=True,
    )
elif not _xgb_available:
    st.sidebar.info("No trained XGBoost model found. Run `python -m training.train --synthetic` to train one.")

st.sidebar.markdown("---")
st.sidebar.subheader("Engine Config")
budget_hours   = st.sidebar.slider("Daily budget (hours)", 6, 12, 8)
acr_cap        = st.sidebar.slider("Max accounts (ACR cap)", 20, 100, 70)
eps_base_km    = st.sidebar.slider(
    "Cluster radius (km)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.5,
    help=(
        "Radius within which customers are grouped into a cluster. "
        "Lower = tighter, more clusters. Higher = looser, fewer larger clusters. "
        "For Bengaluru: 1.0-2.0km works well."
    ),
)
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
def _build_prob_model(model_choice: str, xgb_available: bool, model_info: dict | None):
    if model_choice == "XGBoost (ML)" and xgb_available and model_info:
        try:
            return XGBoostModel(model_info["model_path"])
        except Exception:
            st.warning("Failed to load XGBoost model — falling back to heuristic.")
    return HeuristicModel()


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
    prob_model = _build_prob_model(model_choice, _xgb_available, _active_model_info)
    engine = AllocationEngine(prob_model=prob_model, config=cfg)
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
            prob_model = _build_prob_model(model_choice, _xgb_available, _active_model_info)
            engine = AllocationEngine(prob_model=prob_model, config=cfg)
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
st.caption(f"Plan date: {plan.date}  |  Generated by Allocation Engine v1.0  |  branch: `{_git_branch}`")

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
    import pydeck as pdk
    cust_map = {c.customer_id: c for c in customers}

    # ── Colour palette for clusters (up to 12 clusters) ──────────────────
    CLUSTER_COLOURS = [
        [29, 158, 117],   # teal
        [55, 138, 221],   # blue
        [216, 90, 48],    # orange-red
        [159, 63, 190],   # purple
        [186, 117, 23],   # amber
        [59, 109, 17],    # dark green
        [220, 38, 38],    # red
        [14, 165, 233],   # sky blue
        [168, 85, 247],   # violet
        [234, 88, 12],    # orange
        [20, 184, 166],   # cyan
        [132, 204, 22],   # lime
    ]
    MANDATORY_COLOUR  = [239, 68, 68, 220]
    ESCALATION_COLOUR = [245, 158, 11, 200]
    OUTLIER_COLOUR    = [240, 240, 240, 200]

    # ── Build point layer data ────────────────────────────────────────────
    point_rows = []
    arc_rows   = []
    cluster_colour_map: dict = {}

    # Ranked visits — colour by cluster
    for v in plan.ranked_visits:
        c = cust_map.get(v.customer_id)
        if not c or c.lat is None:
            continue
        if v.cluster_id is not None:
            if v.cluster_id not in cluster_colour_map:
                idx = len(cluster_colour_map) % len(CLUSTER_COLOURS)
                cluster_colour_map[v.cluster_id] = CLUSTER_COLOURS[idx]
            colour = cluster_colour_map[v.cluster_id] + [220]
        else:
            colour = OUTLIER_COLOUR
        point_rows.append({
            "lat":       c.lat,
            "lon":       c.lon,
            "colour":    colour,
            "label":     f"#{v.visit_sequence} {v.name}",
            "dpd":       v.dpd,
            "osp":       f"Rs{v.osp:,.0f}",
            "v_adj":     f"Rs{v.V_adj:,.0f}",
            "prob":      f"{v.probability:.0%}",
            "eff":       f"{v.efficiency:.1f} Rs/min",
            "cluster":   f"Cluster {v.cluster_id}" if v.cluster_id is not None else "Outlier",
            "rationale": v.rationale,
            "seq_label":  str(v.visit_sequence),
            "radius":    max(300, min(800, int(v.V_adj / 50))),
        })

    # Mandatory visits — always red
    for v in plan.mandatory_visits:
        c = cust_map.get(v.customer_id)
        if not c or c.lat is None:
            continue
        point_rows.append({
            "lat":       c.lat,
            "lon":       c.lon,
            "colour":    MANDATORY_COLOUR,
            "label":     f"#{v.visit_sequence} {v.name} [MANDATORY]",
            "seq_label": str(v.visit_sequence),
            "dpd":       v.dpd,
            "osp":       f"Rs{v.osp:,.0f}",
            "v_adj":     f"Rs{v.V_adj:,.0f}",
            "prob":      f"{v.probability:.0%}",
            "eff":       f"{v.efficiency:.1f} Rs/min",
            "cluster":   "Mandatory",
            "rationale": v.rationale,
            "radius":    max(400, min(900, int(v.V_adj / 40))),
        })

    # Escalation visits — amber
    for v in plan.escalation_queue:
        c = cust_map.get(v.customer_id)
        if not c or c.lat is None:
            continue
        point_rows.append({
            "lat":       c.lat,
            "lon":       c.lon,
            "colour":    ESCALATION_COLOUR,
            "label":     f"{v.name} [ESCALATION]",
            "seq_label": "",
            "dpd":       v.dpd,
            "osp":       f"Rs{v.osp:,.0f}",
            "v_adj":     f"Rs{v.V_adj:,.0f}",
            "prob":      f"{v.probability:.0%}",
            "eff":       "-",
            "cluster":   "Escalation",
            "rationale": v.rationale,
            "radius":    300,
        })

    # ── Arc layer — TSP sequence lines within each cluster ────────────────
    from collections import defaultdict
    cluster_visits: dict = defaultdict(list)
    for v in plan.ranked_visits:
        if v.cluster_id is not None:
            c = cust_map.get(v.customer_id)
            if c and c.lat is not None:
                cluster_visits[v.cluster_id].append(
                    (v.visit_sequence, c.lat, c.lon, v.cluster_id)
                )
    for cid, members in cluster_visits.items():
        members.sort(key=lambda x: x[0])
        colour = cluster_colour_map.get(cid, CLUSTER_COLOURS[0])
        for i in range(len(members) - 1):
            _, lat1, lon1, _ = members[i]
            _, lat2, lon2, _ = members[i + 1]
            arc_rows.append({
                "source_lat": lat1, "source_lon": lon1,
                "target_lat": lat2, "target_lon": lon2,
                "colour":     colour + [160],
            })

    # ── Render ────────────────────────────────────────────────────────────
    if point_rows:
        points_df = pd.DataFrame(point_rows)
        arcs_df   = pd.DataFrame(arc_rows) if arc_rows else pd.DataFrame()

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=points_df,
            get_position=["lon", "lat"],
            get_fill_color="colour",
            get_radius="radius",
            radius_min_pixels=6,
            radius_max_pixels=24,
            pickable=True,
            opacity=1.0,
            stroked=True,
            get_line_color=[255, 255, 255, 180],
            line_width_min_pixels=1,
        )
        text_layer = pdk.Layer(
            "TextLayer",
            data=points_df[points_df["cluster"] != "Escalation"],
            get_position=["lon", "lat"],
            get_text="seq_label",
            get_size=12,
            get_color=[255, 255, 255, 220],
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
            pickable=False,
        )
        layers = [scatter_layer, text_layer]

        if not arcs_df.empty:
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=arcs_df,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_source_color="colour",
                get_target_color="colour",
                get_width=2,
                pickable=False,
                auto_highlight=False,
            )
            layers.append(arc_layer)

        centre_lat = points_df["lat"].mean()
        centre_lon = points_df["lon"].mean()
        view_state = pdk.ViewState(latitude=centre_lat, longitude=centre_lon, zoom=11, pitch=0, bearing=0)

        tooltip = {
            "html": """
                <div style='font-family:sans-serif;font-size:13px;padding:6px;'>
                  <b>{label}</b><br/>
                  DPD: {dpd} &nbsp;|&nbsp; {cluster}<br/>
                  OSP: {osp} &nbsp;|&nbsp; V_adj: {v_adj}<br/>
                  P(recovery): {prob} &nbsp;|&nbsp; Efficiency: {eff}<br/>
                  <i>{rationale}</i>
                </div>
            """,
            "style": {
                "backgroundColor": "#1e1e2e",
                "color": "white",
                "border": "1px solid #444",
                "borderRadius": "6px",
            },
        }

        col_map, col_legend = st.columns([3, 1])
        with col_map:
            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                ),
                use_container_width=True,
                height=520,
            )

        with col_legend:
            st.markdown("**Visit types**")
            st.markdown(
                '<span class="tag-mandatory">● Mandatory</span><br/>'
                '<span class="tag-ranked">● Ranked</span><br/>'
                '<span class="tag-escalation">● Escalation</span>',
                unsafe_allow_html=True,
            )
            st.markdown("&nbsp;")

            if cluster_colour_map:
                st.markdown("**Clusters**")
                cl_stats: dict = defaultdict(lambda: {"count": 0, "recovery": 0.0, "eff": []})
                for v in plan.ranked_visits:
                    if v.cluster_id is not None:
                        cl_stats[v.cluster_id]["count"]    += 1
                        cl_stats[v.cluster_id]["recovery"] += v.V_adj
                        cl_stats[v.cluster_id]["eff"].append(v.efficiency)
                for cid in sorted(cl_stats.keys()):
                    stat    = cl_stats[cid]
                    colour  = cluster_colour_map.get(cid, [150, 150, 150])
                    hex_col = "#{:02x}{:02x}{:02x}".format(*colour[:3])
                    avg_eff = sum(stat["eff"]) / len(stat["eff"]) if stat["eff"] else 0
                    st.markdown(
                        f'<div style="border-left:4px solid {hex_col};'
                        f'padding:6px 10px;margin-bottom:6px;'
                        f'background:rgba(255,255,255,0.04);border-radius:0 4px 4px 0;">'
                        f'<b style="color:{hex_col}">Cluster {cid}</b><br/>'
                        f'<span style="font-size:12px;color:#aaa;">'
                        f'{stat["count"]} customers &middot; '
                        f'Rs{stat["recovery"]:,.0f} &middot; '
                        f'{avg_eff:.1f} Rs/min</span></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            total_gps = sum(1 for c in customers if c.lat is not None)
            no_gps    = sum(1 for c in customers if c.lat is None)
            st.markdown(f"**{total_gps}** customers plotted")
            if no_gps:
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
