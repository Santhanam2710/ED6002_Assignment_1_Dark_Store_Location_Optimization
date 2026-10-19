# Chennai Dark Store Location Optimizer

A deployable Streamlit prototype for the dark-store location assignment. It implements a capacitated maximum-profit facility-location MILP with a maximum service-distance constraint, KPI table, profit curve for store-count limits, interactive Chennai map, and assignment CSV export.

## Files

- `app.py` — Streamlit interface, KPI reporting, profit chart, interactive map, and downloads.
- `model.py` — synthetic Chennai instance generation, Haversine distance, sparse MILP formulation, and scenario optimizer.
- `requirements.txt` — Python dependencies.

## Run locally

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The first time you run an optimization, submit the sidebar form. The default case is 10,000 synthetic customer locations, 50 candidate sites, a 2 km maximum modeled road distance, and profit-maximizing limits of at most 2, 5, 10, 20, and 25 stores.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `model.py`, and `requirements.txt` (the README is optional).
2. In Streamlit Community Cloud, create an app from that repository and set the app file to `app.py`.
3. Deploy. No API key or external data file is required. Basemap tiles use CARTO; if tiles are unavailable, the points and assignment layers still render.

## Optimization formulation

Let (y_j\in\{0,1\}) indicate whether candidate site (j) is opened. Let (x_{ij}\ge0) represent expected daily orders allocated from customer (i) to site (j). Candidate-customer arcs whose modeled road distance exceeds (R) are omitted.

\[
\max \sum_{(i,j):d_{ij}\le R}(A-C_p-c_d d_{ij})x_{ij}-F\sum_j y_j
\]

subject to customer demand, store capacity, and the store-count limit:

\[
\sum_jx_{ij}\le D_i,\quad \sum_ix_{ij}\le Q_jy_j,\quad \sum_jy_j\le K.
\]

The binary store decisions and continuous allocation variables form a mixed-integer linear program, solved with SciPy's HiGHS-backed `milp` interface. Each point on the profit curve is an independent maximum-profit solve. Since the objective maximizes profit and the store count is only an upper bound, the optimizer may open fewer than the allowed number.

## Assumptions and limitations

- Customer and candidate coordinates are synthetic and sampled around named Chennai demand hubs; they are not actual addresses or available properties.
- Demand is expected daily order flow, so continuous allocations represent aggregate shares over many orders. A production dispatch system would assign each individual order to one store.
- Profit is average order value minus product/pick-pack cost and modeled delivery cost, minus daily fixed store costs. Taxes, promotions, inventory, labor constraints, demand growth, traffic and road networks, zoning, and opening costs are not separately modeled.
- Distance is Haversine distance multiplied by 1.25 to approximate road distance. A routing API or road network should replace this approximation for a real deployment.
- HiGHS has a configurable time limit per scenario (20 seconds by default) and a 3% relative MIP gap target. The app reports the solver's status and achieved gap; a time-limited feasible solution may not be globally optimal.

## Input notes

The assignment's three required inputs are candidate-site count, customer count, and maximum store-customer distance. Economic assumptions are exposed in the sidebar so the profit calculation is reviewable and tunable rather than hidden in code.

## Optional model tests

The deployment dependencies do not include a test runner. To run the included small-instance regression checks locally, install `pytest` and run `python -m pytest -q tests` from this folder.
