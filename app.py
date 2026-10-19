from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from model import Economics, generate_instance, solve_profit_curve

st.set_page_config(page_title="Chennai Dark Store Optimizer", page_icon="📍", layout="wide")
st.title("Chennai Dark Store Location Optimizer")
st.caption("Capacitated facility-location MIP · Distance-constrained demand allocation · Synthetic Chennai demonstration data")
st.info("This prototype generates reproducible synthetic customer demand and candidate sites around Chennai hubs. It is not real customer data or a surveyed list of available properties.", icon="ℹ️")

with st.sidebar:
    st.header("Scenario inputs")
    with st.form("scenario_inputs"):
        customers_count = st.number_input("Customer locations", min_value=500, max_value=10000, value=10000, step=500)
        candidates_count = st.number_input("Candidate dark store sites", min_value=5, max_value=100, value=50, step=5)
        max_distance_km = st.slider("Maximum store-to-customer distance (km)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        max_stores_for_map = st.selectbox("Store limit for the detailed map", options=[2, 5, 10, 20, 25], index=1)
        st.subheader("Economic assumptions (₹, per day unless stated)")
        avg_order_value = st.number_input("Average order value", min_value=100, max_value=3000, value=500, step=50)
        variable_product_cost = st.number_input("Product + pick/pack cost per order", min_value=0, max_value=2500, value=350, step=25)
        delivery_cost_per_km = st.number_input("Delivery cost per modeled road km", min_value=0.0, max_value=200.0, value=18.0, step=1.0)
        fixed_store_cost = st.number_input("Fixed operating cost per store/day", min_value=0, max_value=200000, value=15000, step=1000)
        capacity = st.number_input("Capacity per store (orders/day)", min_value=100, max_value=10000, value=1800, step=100)
        mean_demand = st.number_input("Expected orders/customer/day", min_value=0.01, max_value=5.0, value=0.35, step=0.05, format="%.2f")
        solver_seconds = st.slider("Solver time limit per scenario (seconds)", min_value=3, max_value=60, value=20)
        run = st.form_submit_button("Run optimization", type="primary", use_container_width=True)

    st.caption("The profit curve always evaluates the assignment limits: 2, 5, 10, 20 and 25 sites, clipped to the candidate count.")

assumptions = Economics(
    average_order_value=float(avg_order_value),
    variable_product_cost=float(variable_product_cost),
    delivery_cost_per_km=float(delivery_cost_per_km),
    fixed_store_cost_per_day=float(fixed_store_cost),
    store_capacity_per_day=float(capacity),
    orders_per_customer_per_day=float(mean_demand),
)

if run:
    customers, candidates = generate_instance(int(customers_count), int(candidates_count), float(mean_demand))
    caps = [min(k, int(candidates_count)) for k in (2, 5, 10, 20, 25)]
    status = st.status("Building the Chennai instance and solving the mixed-integer models…", expanded=True)
    try:
        def show_scenario_progress(step, total, limit):
            status.update(
                label=f"Solved store limit ≤ {limit} ({step} of {total} scenarios)",
                state="running",
                expanded=True,
            )

        curve, scenario_solutions = solve_profit_curve(
            customers,
            candidates,
            caps,
            float(max_distance_km),
            assumptions,
            time_limit_seconds=float(solver_seconds),
            return_solutions=True,
            progress_callback=show_scenario_progress,
        )
        detailed = scenario_solutions[min(int(max_stores_for_map), int(candidates_count))]
        st.session_state["result"] = {
            "customers": customers,
            "candidates": candidates,
            "curve": curve,
            "detailed": detailed,
            "economics": assumptions,
            "max_distance_km": float(max_distance_km),
            "map_cap": int(max_stores_for_map),
            "candidate_count": int(candidates_count),
            "customer_count": int(customers_count),
        }
        status.update(label="Optimization complete", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Optimization could not finish", state="error", expanded=True)
        st.exception(exc)

result = st.session_state.get("result")
if result is None:
    st.markdown("### How to use this app")
    st.write("Choose the customer count, candidate count, maximum service distance and operating assumptions in the sidebar, then select **Run optimization**. The model solves a maximum-profit plan for each store-count cap and draws the selected cap on the map.")
    st.markdown("**Default experiment:** 10,000 customer locations, 50 candidate sites, 2 km maximum modeled road distance, and caps of 2, 5, 10, 20 and 25 stores.")
    st.stop()

# Keep results in sync with the solved inputs; a new run updates this snapshot.
customers = result["customers"]
candidates = result["candidates"]
curve = result["curve"]
detailed = result["detailed"]
metrics = detailed["metrics"]

if not pd.isna(metrics["optimality_gap"]) and metrics["optimality_gap"] > 0.05:
    st.warning(
        f"The selected plan is feasible, but the solver stopped with a {100 * metrics['optimality_gap']:.1f}% MIP gap. "
        "Increase the per-scenario time limit in the sidebar to give it longer to improve the proof of optimality."
    )

st.subheader(f"Detailed solution: at most {result['map_cap']} stores")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Expected profit / day", f"₹{metrics['profit_per_day']:,.0f}")
k2.metric("Stores opened", f"{metrics['opened_stores']} / {result['map_cap']}")
k3.metric("Demand served", f"{metrics['demand_served_pct']:.1f}%", f"{metrics['served_orders_per_day']:,.0f} orders/day")
k4.metric("Avg. modeled distance", f"{metrics['average_distance_km']:.2f} km", f"max assigned {metrics['max_assigned_distance_km']:.2f} km")

left, right = st.columns([1.05, 1.55], gap="large")
with left:
    st.markdown("#### Operating KPIs")
    kpi_rows = [
        ("Customer locations", f"{result['customer_count']:,}"),
        ("Candidate locations", f"{result['candidate_count']:,}"),
        ("Expected demand", f"{metrics['total_demand']:,.0f} orders/day"),
        ("Orders served", f"{metrics['served_orders_per_day']:,.0f} orders/day"),
        ("Orders unserved", f"{metrics['unserved_orders_per_day']:,.0f} orders/day"),
        ("Customers receiving service", f"{metrics['customers_served']:,}"),
        ("Contribution before delivery", f"₹{metrics['contribution_before_delivery']:,.0f}/day"),
        ("Delivery cost", f"₹{metrics['delivery_cost_per_day']:,.0f}/day"),
        ("Fixed store cost", f"₹{metrics['fixed_cost_per_day']:,.0f}/day"),
        ("Maximum allowed distance", f"{result['max_distance_km']:.1f} km"),
        ("MILP optimality gap", "n/a" if pd.isna(metrics["optimality_gap"]) else f"{100 * metrics['optimality_gap']:.2f}%"),
    ]
    st.dataframe(pd.DataFrame(kpi_rows, columns=["KPI", "Result"]), hide_index=True, use_container_width=True)
    st.markdown("#### Selected dark stores")
    if detailed["stores"].empty:
        st.warning("No store is opened under these assumptions. Try a higher order contribution, lower fixed cost, or a wider service distance.")
    else:
        stores_display = detailed["stores"].copy()
        stores_display["orders_per_day"] = stores_display["orders_per_day"].round(1)
        stores_display["utilization_pct"] = stores_display["utilization_pct"].round(1)
        st.dataframe(stores_display[["candidate_id", "orders_per_day", "capacity_per_day", "utilization_pct", "lat", "lon"]], hide_index=True, use_container_width=True)

with right:
    st.markdown("#### Profit versus allowed store count")
    chart = px.line(
        curve,
        x="Allowed stores",
        y="Profit per day (₹)",
        markers=True,
        hover_data={"Stores opened": True, "Orders served/day": ":,.0f", "Demand served (%)": ":.1f", "Average distance (km)": ":.2f", "Optimality gap": ":.3f"},
        labels={"Allowed stores": "Maximum stores allowed", "Profit per day (₹)": "Expected profit (₹/day)"},
        title="Independent profit-maximizing solve at each cap",
    )
    chart.add_hline(y=0, line_dash="dash", line_color="#8b95a5")
    chart.update_layout(height=385, margin=dict(l=15, r=15, t=60, b=15), hovermode="x unified")
    st.plotly_chart(chart, use_container_width=True)
    curve_display = curve.copy()
    curve_display["Profit per day (₹)"] = curve_display["Profit per day (₹)"].map(lambda x: f"₹{x:,.0f}")
    curve_display["Orders served/day"] = curve_display["Orders served/day"].map(lambda x: f"{x:,.0f}")
    curve_display["Demand served (%)"] = curve_display["Demand served (%)"].map(lambda x: f"{x:.1f}%")
    curve_display["Average distance (km)"] = curve_display["Average distance (km)"].map(lambda x: f"{x:.2f}")
    curve_display["Optimality gap"] = curve_display["Optimality gap"].map(lambda x: "n/a" if pd.isna(x) else f"{100 * x:.1f}%")
    st.dataframe(curve_display[["Store limit", "Stores opened", "Profit per day (₹)", "Orders served/day", "Demand served (%)", "Average distance (km)", "Optimality gap"]], hide_index=True, use_container_width=True)

st.markdown("#### Customer-to-store assignment map")
assignments = detailed["assignments"]
served_idx = set(assignments["customer_index"].astype(int).unique()) if not assignments.empty else set()
customer_map = customers.copy()
customer_map["served"] = [1 if i in served_idx else 0 for i in range(len(customers))]
customer_map["color_r"] = np.where(customer_map["served"].eq(1), 31, 180)
customer_map["color_g"] = np.where(customer_map["served"].eq(1), 120, 180)
customer_map["color_b"] = np.where(customer_map["served"].eq(1), 190, 180)
customer_map["radius_m"] = 28 + np.minimum(customer_map["demand"].to_numpy() * 32, 80)

candidate_map = candidates.copy()
stores = detailed["stores"].copy()
if not stores.empty:
    stores["label"] = stores["candidate_id"] + " · " + stores["orders_per_day"].map(lambda v: f"{v:,.0f} orders/day")

layers = [
    pdk.Layer("ScatterplotLayer", data=candidate_map, get_position="[lon, lat]", get_radius=70, get_fill_color=[110, 116, 128, 100], pickable=True, auto_highlight=True),
]
if not assignments.empty:
    # Draw a readable sample of the highest-volume connections; every customer point remains on the map.
    arcs = assignments.nlargest(min(1800, len(assignments)), "orders_per_day").copy()
    layers.append(pdk.Layer("ArcLayer", data=arcs, get_source_position="[customer_lon, customer_lat]", get_target_position="[store_lon, store_lat]", get_source_color=[245, 145, 48, 80], get_target_color=[220, 38, 38, 150], get_width=1.2, pickable=True, auto_highlight=True))
layers.append(pdk.Layer("ScatterplotLayer", data=customer_map, get_position="[lon, lat]", get_radius="radius_m", get_fill_color="[color_r, color_g, color_b, 180]", radius_min_pixels=2, radius_max_pixels=9, pickable=True, auto_highlight=True))
if not stores.empty:
    layers.append(pdk.Layer("ScatterplotLayer", data=stores, get_position="[lon, lat]", get_radius=260, get_fill_color=[220, 38, 38, 235], get_line_color=[255, 255, 255, 255], line_width_min_pixels=2, stroked=True, pickable=True, auto_highlight=True))

view = pdk.ViewState(latitude=13.02, longitude=80.21, zoom=10.0, pitch=0)
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={"html": "<b>{customer_id}</b><br/>Demand: {demand} orders/day<br/>Store: {candidate_id}<br/>Flow: {orders_per_day} orders/day<br/>Distance: {distance_km} km", "style": {"backgroundColor": "#172033", "color": "white"}},
)
st.pydeck_chart(deck, use_container_width=True, height=620)
st.caption("Blue dots are demand locations, gray dots are candidate sites, red markers are selected stores, and orange links show up to 1,800 highest-volume assignments. The distance limit uses a 1.25× road-circuity estimate applied to straight-line Haversine distance.")

if not assignments.empty:
    csv = assignments[["customer_id", "candidate_id", "orders_per_day", "distance_km", "customer_lat", "customer_lon", "store_lat", "store_lon"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download detailed assignment CSV", csv, file_name="dark_store_assignments.csv", mime="text/csv")

with st.expander("Optimization model and assumptions"):
    st.markdown(r"""
**Model:** capacitated maximum-profit facility location. The binary decision (y_j) indicates whether candidate site (j) opens. Continuous (x_{ij}) is the expected daily order flow allocated from customer location (i) to site (j). Aggregate flow can be split across nearby stores; individual orders would be routed to one store in operations.

\[
\max \sum_{(i,j):d_{ij}\le R}(A-C_p-c_d d_{ij})x_{ij} - F\sum_j y_j
\]

subject to

\[
\sum_j x_{ij}\le D_i\quad\forall i,\qquad
\sum_i x_{ij}\le Q_jy_j\quad\forall j,\qquad
\sum_j y_j\le K,\qquad x_{ij}\ge0,\;y_j\in\{0,1\}.
\]

Here (A) is average order value, (C_p) is product plus pick/pack cost, (c_d) is delivery cost per modeled road km, (F) is daily fixed cost, (Q_j) is store capacity, (R) is the maximum service distance, and (D_i) is expected daily demand. Arcs outside (R) are removed, so no customer demand can be assigned beyond the specified distance. A HiGHS mixed-integer solver chooses the store sites and flow allocations.

**Assumptions:** the demonstration customers and sites are synthetic points sampled around Chennai hubs; expected orders are heterogeneous and scaled to the chosen average; demand can remain unserved; a store’s capacity is daily; costs are treated as daily; distance is straight-line Haversine distance multiplied by 1.25 as an approximate road-distance factor. Replace these assumptions with operational data before using it for a real estate decision.
""")

with st.expander("Solver status"):
    st.write(metrics["optimizer_status"])
    if not pd.isna(metrics["optimality_gap"]):
        st.write(f"Reported relative MIP gap: {100 * metrics['optimality_gap']:.3f}%")
