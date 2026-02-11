# -*- coding: utf-8 -*-
"""
part3_stochastic.py — Part 3: Stochastic MECWLP (Original, z=binary)
随机MECWLP模型 — 原始版

两阶段随机规划: 建设决策(场景无关) + 运营决策(场景相关)
Two-stage stochastic programming: build decisions (scenario-independent)
+ operational decisions (per scenario).

保守公式: 所有场景下约束均需成立。
Conservative: constraints enforced for ALL scenarios.

独立运行: python part3_stochastic.py
"""

import xpress as xp
import time
import sys
from datetime import datetime

xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

from shared_data import load_all

TIME_LIMIT = -600  # 10 min


# =============================================================================
# Logging
# =============================================================================
class Tee:
    def __init__(self, filepath, original):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = original
    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
        self._file.flush()
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()

_tee = Tee("part3_output.txt", sys.stdout)
sys.stdout = _tee


def log(msg=""):
    print(msg)


# =============================================================================
# Load data
# =============================================================================
D = load_all()

Candidates    = D["Candidates"]
PostalAreas   = D["PostalAreas"]
Products      = D["Products"]
Times         = D["Times"]
Suppliers     = D["Suppliers"]
ScenariosSet  = D["ScenariosSet"]
SuppliersByProduct = D["SuppliersByProduct"]
AggDemandScenarios = D["AggDemandScenarios"]
CostSupplierCandidate = D["CostSupplierCandidate"]
CostCandidateCustomer = D["CostCandidateCustomer"]
f, g, u, s = D["f"], D["g"], D["u"], D["s"]
Candidates_df = D["Candidates_df"]
prob_s        = D["prob_s"]
NUM_SCENARIOS = D["NUM_SCENARIOS"]


# =============================================================================
# Build and solve
# =============================================================================
log(f"Part 3: Stochastic MECWLP (Original) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Scenarios: {NUM_SCENARIOS}")
log(f"\n{'='*60}")
log("Part 3: Stochastic MECWLP (z=binary)")
log(f"{'='*60}")

prob = xp.problem("Part3_Stochastic_MECWLP")
xp.setOutputEnabled(True)

# --- First-stage variables (scenario-independent) ---
y = {(j, t): xp.var(vartype=xp.binary, name=f"y_{j}_{t}")
     for j in Candidates for t in Times}
z = {(j, t): xp.var(vartype=xp.binary, name=f"z_{j}_{t}")
     for j in Candidates for t in Times}

# --- Second-stage variables (per scenario) ---
x = {(j, area, p, t, sc): xp.var(lb=0, ub=1, name=f"x_{j}_{area}_{p}_{t}_{sc}")
     for j in Candidates for area in PostalAreas
     for p in Products for t in Times for sc in ScenariosSet}
w = {(k, j, t, sc): xp.var(lb=0, name=f"w_{k}_{j}_{t}_{sc}")
     for k in Suppliers for j in Candidates
     for t in Times for sc in ScenariosSet}

prob.addVariable(*y.values(), *z.values(), *x.values(), *w.values())

nvar = len(y) + len(z) + len(x) + len(w)
log(f"Variables: y={len(y)}, z={len(z)}, x={len(x)}, w={len(w)}, total={nvar}")

# --- Objective ---
# Setup + operating (scenario-independent) + expected transport
prob.setObjective(
    xp.Sum(f[j] * y[j, t] for j in Candidates for t in Times)
    + xp.Sum(g[j] * z[j, t] for j in Candidates for t in Times)
    + xp.Sum(
        prob_s * CostCandidateCustomer[(j, area)]
        * AggDemandScenarios.get((area, p, t, sc), 0)
        * x[j, area, p, t, sc]
        for j in Candidates for area in PostalAreas
        for p in Products for t in Times for sc in ScenariosSet
    )
    + xp.Sum(
        prob_s * CostSupplierCandidate[(k, j)] * w[k, j, t, sc]
        for k in Suppliers for j in Candidates
        for t in Times for sc in ScenariosSet
    ),
    sense=xp.minimize,
)

# --- Constraints ---
ncon = 0

# (C1) Cumulative open (scenario-independent)
for j in Candidates:
    for t in Times:
        prob.addConstraint(z[j, t] == xp.Sum(y[j, tau] for tau in range(1, t + 1)))
        ncon += 1

# (C2) Build at most once (scenario-independent)
for j in Candidates:
    prob.addConstraint(xp.Sum(y[j, t] for t in Times) <= 1)
    ncon += 1

# Per-scenario constraints
for sc in ScenariosSet:
    # (C3) Demand satisfaction
    for area in PostalAreas:
        for p in Products:
            for t in Times:
                prob.addConstraint(xp.Sum(x[j, area, p, t, sc] for j in Candidates) == 1)
                ncon += 1

    # (C5) Warehouse capacity
    for j in Candidates:
        for t in Times:
            prob.addConstraint(
                xp.Sum(
                    AggDemandScenarios.get((area, p, t, sc), 0) * x[j, area, p, t, sc]
                    for area in PostalAreas for p in Products
                ) <= u[j] * z[j, t]
            )
            ncon += 1

    # (C6) Flow balance per product
    for j in Candidates:
        for p in Products:
            for t in Times:
                prob.addConstraint(
                    xp.Sum(w[k, j, t, sc] for k in SuppliersByProduct[p])
                    == xp.Sum(
                        AggDemandScenarios.get((area, p, t, sc), 0) * x[j, area, p, t, sc]
                        for area in PostalAreas
                    )
                )
                ncon += 1

    # (C7) Supplier capacity
    for k in Suppliers:
        for t in Times:
            prob.addConstraint(xp.Sum(w[k, j, t, sc] for j in Candidates) <= s[k])
            ncon += 1

log(f"Constraints: {ncon}")

# --- Solve ---
prob.controls.maxtime = TIME_LIMIT
log(f"Solving (time limit = {abs(TIME_LIMIT)}s)...")
t0 = time.time()
prob.solve()
solve_time = time.time() - t0

# --- Results ---
sol = prob.attributes.solstatus
status_map = {
    xp.SolStatus.OPTIMAL: "OPTIMAL",
    xp.SolStatus.FEASIBLE: "FEASIBLE",
    xp.SolStatus.INFEASIBLE: "INFEASIBLE",
    xp.SolStatus.UNBOUNDED: "UNBOUNDED",
}
log(f"\nStatus: {status_map.get(sol, 'NO SOLUTION')}")
log(f"Solve time: {solve_time:.2f}s")

if sol in (xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
    obj_val = prob.attributes.objval
    best_bound = prob.attributes.bestbound
    mip_gap = abs(obj_val - best_bound) / (1e-10 + abs(obj_val))
    log(f"Objective: £{obj_val:,.2f}")
    log(f"Best bound: £{best_bound:,.2f}")
    log(f"MIP gap: {mip_gap * 100:.2f}%")

    # --- Batch extract all solution values (avoid slow per-variable API calls) ---
    y_keys = list(y.keys())
    y_vals = prob.getSolution(list(y[k] for k in y_keys))
    y_sol = dict(zip(y_keys, y_vals))

    z_keys = list(z.keys())
    z_vals = prob.getSolution(list(z[k] for k in z_keys))
    z_sol = dict(zip(z_keys, z_vals))

    x_keys = list(x.keys())
    x_vals = prob.getSolution(list(x[k] for k in x_keys))
    x_sol = dict(zip(x_keys, x_vals))

    w_keys = list(w.keys())
    w_vals = prob.getSolution(list(w[k] for k in w_keys))
    w_sol = dict(zip(w_keys, w_vals))

    # Extract open warehouses
    warehouses = []
    open_wh = set()
    open_jt = set()
    setup_val = 0.0
    operating_val = 0.0
    for j in Candidates:
        for t in Times:
            if y_sol[(j, t)] > 0.5:
                warehouses.append((j, t))
                setup_val += f[j]
            if z_sol[(j, t)] > 0.5:
                open_wh.add(j)
                open_jt.add((j, t))
                operating_val += g[j]

    # Expected transport costs (only open warehouses)
    exp_transport_down = sum(
        prob_s * CostCandidateCustomer[(j, area)]
        * AggDemandScenarios.get((area, p, t, sc), 0)
        * x_sol[(j, area, p, t, sc)]
        for j in open_wh for area in PostalAreas
        for p in Products for t in Times for sc in ScenariosSet
        if (j, t) in open_jt
    )
    exp_transport_up = sum(
        prob_s * CostSupplierCandidate[(k, j)] * w_sol[(k, j, t, sc)]
        for k in Suppliers for j in open_wh
        for t in Times for sc in ScenariosSet
        if (j, t) in open_jt
    )

    log(f"\nCost breakdown:")
    log(f"  Setup:                     £{setup_val:>15,.2f}")
    log(f"  Operating:                 £{operating_val:>15,.2f}")
    log(f"  E[Transport (WH→Cust)]:   £{exp_transport_down:>15,.2f}")
    log(f"  E[Transport (Sup→WH)]:    £{exp_transport_up:>15,.2f}")
    total = setup_val + operating_val + exp_transport_down + exp_transport_up
    log(f"  Total:                     £{total:>15,.2f}")

    log(f"\nWarehouse build schedule:")
    log(f"  {'ID':<6} {'Postal Dist':<14} {'Period':>6} {'Capacity':>10} {'Setup £':>12}")
    for j, t in sorted(warehouses, key=lambda wh: wh[1]):
        pd_name = Candidates_df.loc[j, "Postal District"]
        log(f"  {j:<6} {pd_name:<14} {t:>6} {u[j]:>10,} {f[j]:>12,}")
    log(f"  Total: {len(warehouses)} warehouses")

    log(f"\nCumulative open warehouses by period:")
    wh_sorted = sorted(warehouses, key=lambda wh: wh[1])
    for t in Times:
        open_in_t = [(j, bt) for (j, bt) in wh_sorted if bt <= t]
        names = [Candidates_df.loc[j, "Postal District"] for (j, _) in open_in_t]
        log(f"  Period {t:>2}: {len(open_in_t)} warehouses — {', '.join(names)}")

log(f"\nDone. Output saved to part3_output.txt")

sys.stdout = _tee._stdout
_tee.close()
