"""Synthetic Chennai instance generator and capacitated dark-store MIP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# Plausible Chennai-area demand hubs (latitude, longitude, demand weight).
# Coordinates are used to create a reproducible synthetic instance, not as surveyed sites.
CHENNAI_HUBS = [
    (13.115, 80.234, 0.055),  # Perambur
    (13.085, 80.210, 0.085),  # Anna Nagar
    (13.041, 80.233, 0.100),  # T. Nagar
    (13.034, 80.269, 0.070),  # Mylapore
    (13.006, 80.257, 0.075),  # Adyar
    (13.006, 80.220, 0.065),  # Guindy
    (12.981, 80.220, 0.070),  # Velachery
    (13.036, 80.157, 0.055),  # Porur
    (13.114, 80.154, 0.050),  # Ambattur
    (13.051, 80.212, 0.065),  # Vadapalani
    (12.939, 80.236, 0.055),  # Thoraipakkam
    (12.901, 80.227, 0.045),  # Sholinganallur
    (12.924, 80.100, 0.040),  # Tambaram
    (13.058, 80.277, 0.045),  # Triplicane
    (12.967, 80.149, 0.025),  # Pallavaram
]
ROAD_CIRCUITY_FACTOR = 1.25


@dataclass(frozen=True)
class Economics:
    average_order_value: float = 500.0
    variable_product_cost: float = 350.0
    delivery_cost_per_km: float = 18.0
    fixed_store_cost_per_day: float = 15000.0
    store_capacity_per_day: float = 1800.0
    orders_per_customer_per_day: float = 0.35

    @property
    def contribution_per_order(self) -> float:
        return self.average_order_value - self.variable_product_cost


def generate_instance(
    customers_count: int,
    candidates_count: int,
    orders_per_customer_per_day: float,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate stable, demand-weighted points around Chennai urban hubs."""
    rng = np.random.default_rng(seed)
    hubs = np.asarray(CHENNAI_HUBS, dtype=float)
    weights = hubs[:, 2] / hubs[:, 2].sum()

    def points(n: int, prefix: str, spread: float) -> pd.DataFrame:
        hub_idx = rng.choice(len(hubs), size=n, p=weights)
        lat = hubs[hub_idx, 0] + rng.normal(0, spread, n)
        # Longitude degrees are shorter than latitude degrees at Chennai's latitude.
        lon = hubs[hub_idx, 1] + rng.normal(0, spread * 1.18, n)
        return pd.DataFrame({
            f"{prefix}_id": [f"{prefix[0].upper()}{i + 1:05d}" for i in range(n)],
            "lat": np.clip(lat, 12.86, 13.17),
            "lon": np.clip(lon, 80.08, 80.31),
        })

    customers = points(customers_count, "customer", 0.0065)
    candidates = points(candidates_count, "candidate", 0.0080)
    # Lognormal heterogeneity gives locations unequal expected daily order volume.
    raw_demand = rng.lognormal(mean=0.0, sigma=0.48, size=customers_count)
    raw_demand *= orders_per_customer_per_day / raw_demand.mean()
    customers["demand"] = raw_demand
    return customers, candidates


def haversine_matrix(customers: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    """Return customer-by-candidate straight-line distances in km."""
    lat1 = np.radians(customers["lat"].to_numpy())[:, None]
    lon1 = np.radians(customers["lon"].to_numpy())[:, None]
    lat2 = np.radians(candidates["lat"].to_numpy())[None, :]
    lon2 = np.radians(candidates["lon"].to_numpy())[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def solve_location_model(
    customers: pd.DataFrame,
    candidates: pd.DataFrame,
    max_store_count: int,
    max_distance_km: float,
    economics: Economics,
    time_limit_seconds: float = 20.0,
    mip_relative_gap: float = 0.03,
) -> dict:
    """Maximize daily contribution after delivery and fixed store costs.

    x[i,j] is expected daily order flow from customer location i to store j.
    It is continuous because customer points aggregate expected orders; y[j] is binary.
    Haversine distance is multiplied by a road-circuity proxy before coverage/cost checks.
    """
    n_i, n_j = len(customers), len(candidates)
    demand = customers["demand"].to_numpy(dtype=float)
    road_km = haversine_matrix(customers, candidates) * ROAD_CIRCUITY_FACTOR
    feasible_i, feasible_j = np.where(road_km <= max_distance_km + 1e-9)
    n_edges = len(feasible_i)
    if n_edges == 0:
        return _empty_solution(customers, candidates, economics, max_store_count, max_distance_km)

    distance = road_km[feasible_i, feasible_j]
    unit_margin = economics.contribution_per_order - economics.delivery_cost_per_km * distance

    # Variable order: all eligible expected order flows, followed by binary open/close y[j].
    c = np.r_[-unit_margin, np.full(n_j, economics.fixed_store_cost_per_day)]
    lower = np.zeros(n_edges + n_j)
    upper = np.r_[demand[feasible_i], np.ones(n_j)]
    integrality = np.r_[np.zeros(n_edges, dtype=np.uint8), np.ones(n_j, dtype=np.uint8)]

    # Demand rows: sum_j x[i,j] <= demand[i].
    # Capacity rows: sum_i x[i,j] - capacity*y[j] <= 0.
    # One final row: sum_j y[j] <= max_store_count.
    edge_cols = np.arange(n_edges)
    rows = np.r_[
        feasible_i,
        n_i + feasible_j,
        n_i + np.arange(n_j),
        np.full(n_j, n_i + n_j),
    ]
    cols = np.r_[
        edge_cols,
        edge_cols,
        n_edges + np.arange(n_j),
        n_edges + np.arange(n_j),
    ]
    vals = np.r_[
        np.ones(n_edges),
        np.ones(n_edges),
        np.full(n_j, -economics.store_capacity_per_day),
        np.ones(n_j),
    ]
    matrix = coo_matrix((vals, (rows, cols)), shape=(n_i + n_j + 1, n_edges + n_j)).tocsr()
    row_upper = np.r_[demand, np.zeros(n_j), min(max_store_count, n_j)]

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.full(len(row_upper), -np.inf), row_upper),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": mip_relative_gap},
    )
    if result.x is None:
        raise RuntimeError(f"The optimizer did not return a feasible solution (status {result.status}: {result.message}).")

    flow = np.maximum(result.x[:n_edges], 0.0)
    opened = np.flatnonzero(result.x[n_edges:] > 0.5)
    active = flow > 1e-7
    assignments = pd.DataFrame({
        "customer_id": customers.iloc[feasible_i[active]]["customer_id"].to_numpy(),
        "candidate_id": candidates.iloc[feasible_j[active]]["candidate_id"].to_numpy(),
        "customer_index": feasible_i[active],
        "candidate_index": feasible_j[active],
        "customer_lat": customers.iloc[feasible_i[active]]["lat"].to_numpy(),
        "customer_lon": customers.iloc[feasible_i[active]]["lon"].to_numpy(),
        "store_lat": candidates.iloc[feasible_j[active]]["lat"].to_numpy(),
        "store_lon": candidates.iloc[feasible_j[active]]["lon"].to_numpy(),
        "orders_per_day": flow[active],
        "distance_km": distance[active],
    })
    if not assignments.empty:
        assignments = assignments[assignments["orders_per_day"] > 1e-6].copy()

    total_demand = float(demand.sum())
    served = float(assignments["orders_per_day"].sum()) if not assignments.empty else 0.0
    total_contribution = served * economics.contribution_per_order
    delivery_cost = float((assignments["orders_per_day"] * assignments["distance_km"] * economics.delivery_cost_per_km).sum()) if not assignments.empty else 0.0
    fixed_cost = len(opened) * economics.fixed_store_cost_per_day
    profit = total_contribution - delivery_cost - fixed_cost
    weighted_distance = (
        float((assignments["orders_per_day"] * assignments["distance_km"]).sum() / served) if served > 0 else 0.0
    )
    served_customers = int(assignments.loc[assignments["orders_per_day"] > 1e-6, "customer_index"].nunique()) if not assignments.empty else 0

    store_rows = []
    for j in opened:
        volume = float(assignments.loc[assignments["candidate_index"] == j, "orders_per_day"].sum()) if not assignments.empty else 0.0
        store_rows.append({
            "candidate_id": candidates.iloc[j]["candidate_id"],
            "candidate_index": int(j),
            "lat": float(candidates.iloc[j]["lat"]),
            "lon": float(candidates.iloc[j]["lon"]),
            "orders_per_day": volume,
            "capacity_per_day": economics.store_capacity_per_day,
            "utilization_pct": 100.0 * volume / economics.store_capacity_per_day,
        })
    stores = pd.DataFrame(store_rows, columns=["candidate_id", "candidate_index", "lat", "lon", "orders_per_day", "capacity_per_day", "utilization_pct"])
    metrics = {
        "profit_per_day": profit,
        "opened_stores": len(opened),
        "max_store_limit": min(max_store_count, n_j),
        "total_demand": total_demand,
        "served_orders_per_day": served,
        "unserved_orders_per_day": max(0.0, total_demand - served),
        "demand_served_pct": 100.0 * served / total_demand if total_demand else 0.0,
        "customers_served": served_customers,
        "average_distance_km": weighted_distance,
        "max_assigned_distance_km": float(assignments["distance_km"].max()) if not assignments.empty else 0.0,
        "contribution_before_delivery": total_contribution,
        "delivery_cost_per_day": delivery_cost,
        "fixed_cost_per_day": fixed_cost,
        "optimizer_status": str(result.message),
        "optimality_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else np.nan,
    }
    return {"metrics": metrics, "assignments": assignments, "stores": stores, "road_km": road_km}


def _empty_solution(customers, candidates, economics, max_store_count, max_distance_km):
    assignments = pd.DataFrame(columns=["customer_id", "candidate_id", "customer_index", "candidate_index", "customer_lat", "customer_lon", "store_lat", "store_lon", "orders_per_day", "distance_km"])
    stores = pd.DataFrame(columns=["candidate_id", "candidate_index", "lat", "lon", "orders_per_day", "capacity_per_day", "utilization_pct"])
    metrics = {
        "profit_per_day": 0.0, "opened_stores": 0, "max_store_limit": min(max_store_count, len(candidates)),
        "total_demand": float(customers["demand"].sum()), "served_orders_per_day": 0.0,
        "unserved_orders_per_day": float(customers["demand"].sum()), "demand_served_pct": 0.0,
        "customers_served": 0, "average_distance_km": 0.0, "max_assigned_distance_km": 0.0,
        "contribution_before_delivery": 0.0, "delivery_cost_per_day": 0.0, "fixed_cost_per_day": 0.0,
        "optimizer_status": f"No customer-candidate pairs within {max_distance_km:.1f} km.", "optimality_gap": np.nan,
    }
    return {"metrics": metrics, "assignments": assignments, "stores": stores, "road_km": np.empty((len(customers), len(candidates)))}


def solve_profit_curve(
    customers,
    candidates,
    limits,
    max_distance_km,
    economics,
    time_limit_seconds: float = 20.0,
    return_solutions: bool = False,
    progress_callback=None,
):
    """Solve one independent profit-maximizing model for each allowed store limit."""
    rows = []
    solutions = {}
    for step, limit in enumerate(limits, start=1):
        # Several displayed caps can collapse to the same candidate count.
        if limit not in solutions:
            solutions[limit] = solve_location_model(
                customers,
                candidates,
                limit,
                max_distance_km,
                economics,
                time_limit_seconds=time_limit_seconds,
            )
        solution = solutions[limit]
        m = solution["metrics"]
        rows.append({
            "Store limit": f"≤ {limit}",
            "Allowed stores": limit,
            "Stores opened": m["opened_stores"],
            "Profit per day (₹)": m["profit_per_day"],
            "Orders served/day": m["served_orders_per_day"],
            "Demand served (%)": m["demand_served_pct"],
            "Average distance (km)": m["average_distance_km"],
            "Optimality gap": m["optimality_gap"],
        })
        if progress_callback is not None:
            progress_callback(step, len(limits), limit)
    curve = pd.DataFrame(rows)
    return (curve, solutions) if return_solutions else curve
