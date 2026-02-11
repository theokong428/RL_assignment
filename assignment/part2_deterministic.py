# -*- coding: utf-8 -*-
"""
part2_deterministic.py — Part 2: Multi-Period MECWLP (Deterministic)
多时段确定性MECWLP模型

包含原始版 (z=binary) 和优化版 Part 2b (z=continuous)。
Includes original (z=binary) and optimised Part 2b (z=continuous).

独立运行: python part2_deterministic.py
"""

import xpress as xp
import time
import sys
from datetime import datetime

xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

from shared_data import load_all

TIME_LIMIT = -600  # 10 min


# =============================================================================
# Logging helper
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

_tee = Tee("part2_output.txt", sys.stdout)
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
SuppliersByProduct = D["SuppliersByProduct"]
AggDemandPeriods   = D["AggDemandPeriods"]
CostSupplierCandidate = D["CostSupplierCandidate"]
CostCandidateCustomer = D["CostCandidateCustomer"]
f, g, u, s = D["f"], D["g"], D["u"], D["s"]
Candidates_df = D["Candidates_df"]


# =============================================================================
# Helper: build, solve, report
# =============================================================================
def build_and_solve(label, z_binary=True):
    """
    构建并求解多时段MECWLP。
    z_binary=True  → 原始版 (z as binary)
    z_binary=False → 优化版 (z as continuous, integrality implied by z=Σy)
    """
    log(f"\n{'='*60}")
    log(f"{label}")
    log(f"{'='*60}")

    prob = xp.problem(label)
    xp.setOutputEnabled(True)

    # --- Variables ---
    y = {(j, t): xp.var(vartype=xp.binary, name=f"y_{j}_{t}")
         for j in Candidates for t in Times}

    if z_binary:
        z = {(j, t): xp.var(vartype=xp.binary, name=f"z_{j}_{t}")
             for j in Candidates for t in Times}
    else:
        z = {(j, t): xp.var(lb=0, ub=1, name=f"z_{j}_{t}")
             for j in Candidates for t in Times}

    x = {(j, area, p, t): xp.var(lb=0, ub=1, name=f"x_{j}_{area}_{p}_{t}")
         for j in Candidates for area in PostalAreas
         for p in Products for t in Times}

    w = {(k, j, t): xp.var(lb=0, name=f"w_{k}_{j}_{t}")
         for k in Suppliers for j in Candidates for t in Times}

    prob.addVariable(*y.values(), *z.values(), *x.values(), *w.values())

    nvar = len(y) + len(z) + len(x) + len(w)
    log(f"Variables: y={len(y)}, z={len(z)}, x={len(x)}, w={len(w)}, total={nvar}")

    # --- Objective ---
    prob.setObjective(
        xp.Sum(f[j] * y[j, t] for j in Candidates for t in Times)
        + xp.Sum(g[j] * z[j, t] for j in Candidates for t in Times)
        + xp.Sum(
            CostCandidateCustomer[(j, area)]
            * AggDemandPeriods.get((area, p, t), 0)
            * x[j, area, p, t]
            for j in Candidates for area in PostalAreas
            for p in Products for t in Times
        )
        + xp.Sum(
            CostSupplierCandidate[(k, j)] * w[k, j, t]
            for k in Suppliers for j in Candidates for t in Times
        ),
        sense=xp.minimize,
    )

    # --- Constraints ---
    ncon = 0

    # (C1) Cumulative open: z_{j,t} = Σ_{τ≤t} y_{j,τ}
    for j in Candidates:
        for t in Times:
            prob.addConstraint(z[j, t] == xp.Sum(y[j, tau] for tau in range(1, t + 1)))
            ncon += 1

    # (C2) Build at most once: Σ_t y_{j,t} ≤ 1
    for j in Candidates:
        prob.addConstraint(xp.Sum(y[j, t] for t in Times) <= 1)
        ncon += 1

    # (C3) Demand satisfaction: Σ_j x_{j,i,p,t} = 1
    for area in PostalAreas:
        for p in Products:
            for t in Times:
                prob.addConstraint(xp.Sum(x[j, area, p, t] for j in Candidates) == 1)
                ncon += 1

    # (C5) Warehouse capacity
    for j in Candidates:
        for t in Times:
            prob.addConstraint(
                xp.Sum(
                    AggDemandPeriods.get((area, p, t), 0) * x[j, area, p, t]
                    for area in PostalAreas for p in Products
                ) <= u[j] * z[j, t]
            )
            ncon += 1

    # (C6) Flow balance per product
    for j in Candidates:
        for p in Products:
            for t in Times:
                prob.addConstraint(
                    xp.Sum(w[k, j, t] for k in SuppliersByProduct[p])
                    == xp.Sum(
                        AggDemandPeriods.get((area, p, t), 0) * x[j, area, p, t]
                        for area in PostalAreas
                    )
                )
                ncon += 1

    # (C7) Supplier capacity
    for k in Suppliers:
        for t in Times:
            prob.addConstraint(xp.Sum(w[k, j, t] for j in Candidates) <= s[k])
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

    obj_val = None
    if sol in (xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
        obj_val = prob.attributes.objval
        best_bound = prob.attributes.bestbound
        mip_gap = abs(obj_val - best_bound) / (1e-10 + abs(obj_val))
        log(f"Objective: £{obj_val:,.2f}")
        log(f"Best bound: £{best_bound:,.2f}")
        log(f"MIP gap: {mip_gap * 100:.2f}%")

        # Extract open warehouses
        warehouses = []
        open_wh = set()
        open_jt = set()
        setup_val = 0.0
        operating_val = 0.0
        for j in Candidates:
            for t in Times:
                if prob.getSolution(y[j, t]) > 0.5:
                    warehouses.append((j, t))
                    setup_val += f[j]
                if prob.getSolution(z[j, t]) > 0.5:
                    open_wh.add(j)
                    open_jt.add((j, t))
                    operating_val += g[j]

        # Transport costs (only open warehouses)
        transport_down = sum(
            CostCandidateCustomer[(j, area)]
            * AggDemandPeriods.get((area, p, t), 0)
            * prob.getSolution(x[j, area, p, t])
            for j in open_wh for area in PostalAreas
            for p in Products for t in Times
            if (j, t) in open_jt
        )
        transport_up = sum(
            CostSupplierCandidate[(k, j)] * prob.getSolution(w[k, j, t])
            for k in Suppliers for j in open_wh for t in Times
            if (j, t) in open_jt
        )

        log(f"\nCost breakdown:")
        log(f"  Setup:              £{setup_val:>15,.2f}")
        log(f"  Operating:          £{operating_val:>15,.2f}")
        log(f"  Transport (WH→Cust):£{transport_down:>15,.2f}")
        log(f"  Transport (Sup→WH): £{transport_up:>15,.2f}")
        total = setup_val + operating_val + transport_down + transport_up
        log(f"  Total:              £{total:>15,.2f}")

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

    return obj_val, solve_time


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    log(f"Part 2: Multi-Period MECWLP — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    obj2, time2 = build_and_solve("Part 2: Original (z=binary)", z_binary=True)
    obj2b, time2b = build_and_solve("Part 2b: Optimised (z=continuous)", z_binary=False)

    log(f"\n{'='*60}")
    log("Comparison: Part 2 vs Part 2b")
    log(f"{'='*60}")
    if obj2 is not None and obj2b is not None:
        log(f"  Original:  £{obj2:,.2f}  ({time2:.1f}s)")
        log(f"  Optimised: £{obj2b:,.2f}  ({time2b:.1f}s)")
        diff = obj2b - obj2
        log(f"  Difference: £{diff:,.2f} ({diff / obj2 * 100:+.2f}%)")
    else:
        log("  Cannot compare: at least one model has no feasible solution")

    log(f"\nDone. Output saved to part2_output.txt")

    sys.stdout = _tee._stdout
    _tee.close()
