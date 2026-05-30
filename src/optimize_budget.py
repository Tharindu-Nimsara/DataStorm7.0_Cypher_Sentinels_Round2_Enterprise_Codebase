"""Phase 5 — LKR 5,000,000 trade-spend optimiser for Western-province outlets.

Why this exists
---------------
New deliverable #2 + Business Viability (25%). Having estimated each outlet's *latent
potential*, the business question becomes: given a fixed promotional budget, where do we spend
to unlock the most *incremental* volume? Spending on an outlet already near its ceiling is
wasted; spending on a high-gap, responsive outlet pays off. This module turns the potential
estimates into a concrete, defensible allocation.

The model
---------
* **Opportunity gap.** ``gap_i = max(potential_i − recent_actual_i, 0)`` — litres of unmet
  demand we could realistically capture. ``recent_actual`` is the outlet's historical mean
  monthly volume (``vol_mean``); the gap is what stands between today and its estimated ceiling.

* **Response function (diminishing returns).** ``incremental_i(s) = gap_i · (1 − e^{−k_i·s})``.
  As spend ``s`` rises, incremental volume approaches the gap asymptotically — the first
  rupees on an outlet do the most work, matching how trade promotions actually behave. The
  responsiveness ``k_i`` is keyed to the outlet's *constraint type*:
    - **supply-constrained** outlets (hitting their ceiling) respond to spend that adds
      physical capacity / visibility (coolers, merchandising) → higher ``k`` (fast payoff once
      the bottleneck is relieved);
    - **demand-led** outlets respond to spend that pulls shoppers (discounts) → moderate ``k``.

* **Optimisation.** maximise ``Σ incremental_i(s_i)`` s.t. ``Σ s_i ≤ 5,000,000``, ``0 ≤ s_i ≤
  cap_i``. The objective is concave (sum of concave terms), so we solve it exactly with
  **cvxpy**; a greedy marginal-return allocator is kept as a transparent fallback /
  cross-check and is the one we'd explain on stage.

* **Spend-type tag.** Each funded outlet is tagged discount / cooler / merchandising according
  to its constraint, so the allocation is actionable, not just a number.

Outputs ``reports/cypher_sentinels_budget_allocations.csv`` with columns
``Outlet_ID, Trade_Spend_Allocation_LKR`` (Western only). Idempotent.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
REPORTS = ROOT / "reports"
LOG_DIR = ROOT / "logs"
REPORTS.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TEAMNAME = "cypher_sentinels"
OUTPUT_CSV = REPORTS / f"{TEAMNAME}_budget_allocations.csv"

TOTAL_BUDGET_LKR = 5_000_000.0
PER_OUTLET_CAP_LKR = 50_000.0     # no single outlet may absorb >1% of the budget — keeps the
                                  # plan spread across many kades, not dumped on a handful
RANDOM_STATE = 42

# Constraint severity threshold (matches the XAI / gold 'supply-constrained' definition).
SUPPLY_CONSTRAINT_THRESHOLD = 0.12

# Responsiveness k (per LKR) by constraint type. Calibrated so that the *cap* spend captures a
# sensible fraction of an outlet's gap: with k≈9e-5, LKR 50k relieves ~99% of a supply
# bottleneck (1−e^{−4.5}); demand-led k≈4.5e-5 captures ~89% at the cap (1−e^{−2.25}),
# reflecting that a discount nudges demand but can't fully close the gap. Documented, and the
# allocation's *ranking* is insensitive to the exact value (it's monotonic in gap·k).
K_SUPPLY = 9.0e-5
K_DEMAND = 4.5e-5


def setup_logging() -> logging.Logger:
    log = logging.getLogger("optimize_budget")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / "optimize_budget.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def build_western_frame(log: logging.Logger) -> pd.DataFrame:
    """Western outlets with opportunity gap, responsiveness k, and spend-type tag."""
    diag = pd.read_parquet(GOLD / "_predictions_diagnostic.parquet")
    gold = pd.read_parquet(GOLD / "outlet_features.parquet")

    w = diag[diag["Province"] == "Western"].copy()
    w = w.merge(gold[["Outlet_ID", "vol_mean", "Cooler_Count"]], on="Outlet_ID",
                how="left", suffixes=("", "_g"))
    if "primary_distributor" in gold.columns:
        w = w.merge(gold[["Outlet_ID", "primary_distributor"]], on="Outlet_ID", how="left")

    w["recent_actual"] = w["vol_mean"]
    w["gap"] = (w["potential_final"] - w["recent_actual"]).clip(lower=0)

    supply = w["censoring_score"] > SUPPLY_CONSTRAINT_THRESHOLD
    w["constraint_type"] = np.where(supply, "supply-constrained", "demand-led")
    w["k"] = np.where(supply, K_SUPPLY, K_DEMAND)
    # spend type: supply outlets with <=1 cooler need a cooler; other supply outlets need
    # merchandising (shelf/visibility); demand-led outlets respond to discounts.
    w["spend_type"] = np.where(
        supply & (w["Cooler_Count"] <= 1), "cooler",
        np.where(supply, "merchandising", "discount"))

    log.info("Western outlets: %d  (supply-constrained=%d, demand-led=%d).",
             len(w), int(supply.sum()), int((~supply).sum()))
    log.info("Total opportunity gap = %.0f L/month  (mean %.1f, max %.0f).",
             w["gap"].sum(), w["gap"].mean(), w["gap"].max())
    return w[["Outlet_ID", "primary_distributor", "Outlet_Type", "Outlet_Size",
              "recent_actual", "potential_final", "gap", "constraint_type", "k", "spend_type"]]


def greedy_allocate(gap: np.ndarray, k: np.ndarray, budget: float, cap: float,
                    step: float, log: logging.Logger) -> np.ndarray:
    """Transparent marginal-return allocator: repeatedly fund the outlet with the highest
    incremental litres-per-rupee at its current spend, in fixed steps, until the budget runs
    out. This is the version we'd explain on stage — sort by marginal gain, fund the best.

    Marginal gain of one more rupee at spend s is d/ds [gap(1-e^{-k s})] = gap·k·e^{-k s},
    which is decreasing in s — so a greedy fill on a concave objective is near-optimal.
    """
    n = len(gap)
    spend = np.zeros(n)
    n_steps = int(budget // step)
    # marginal value at current spend for every outlet; we pop the max each step. Using a
    # vectorised argmax per step is O(steps·n); with ~9k outlets and ~1000 steps that's fine.
    for _ in range(n_steps):
        marginal = gap * k * np.exp(-k * spend)
        marginal[spend >= cap - 1e-9] = -np.inf      # respect the per-outlet cap
        j = int(np.argmax(marginal))
        if marginal[j] <= 0:
            break
        spend[j] = min(spend[j] + step, cap)
    log.info("  Greedy: funded %d outlets, spent %.0f of %.0f LKR.",
             int((spend > 0).sum()), spend.sum(), budget)
    return spend


def cvxpy_allocate(gap: np.ndarray, k: np.ndarray, budget: float, cap: float,
                   log: logging.Logger) -> np.ndarray | None:
    """Exact concave-program solve: maximise Σ gap_i(1−e^{−k_i s_i}) s.t. Σs≤budget, 0≤s≤cap.

    Returns None if cvxpy is unavailable so the caller uses the greedy result."""
    try:
        import cvxpy as cp
    except Exception as e:  # pragma: no cover
        log.warning("  cvxpy unavailable (%s) — using greedy allocation only.", e)
        return None

    n = len(gap)
    s = cp.Variable(n, nonneg=True)
    # gap·(1 − exp(−k s)) is concave in s; maximise its sum. The exp term needs a solver that
    # handles the exponential cone — ECOS would, but isn't bundled with modern cvxpy, so we
    # use CLARABEL (cvxpy's default conic solver) and fall back to SCS, then greedy.
    objective = cp.Maximize(cp.sum(cp.multiply(gap, 1 - cp.exp(-cp.multiply(k, s)))))
    constraints = [cp.sum(s) <= budget, s <= cap]
    prob = cp.Problem(objective, constraints)

    available = cp.installed_solvers()
    for solver in ("CLARABEL", "SCS"):
        if solver not in available:
            continue
        try:
            prob.solve(solver=solver)
        except Exception as e:
            log.warning("  cvxpy %s failed (%s) — trying next solver.", solver, e)
            continue
        if s.value is not None:
            spend = np.clip(s.value, 0, cap)
            log.info("  cvxpy (%s): status=%s, funded %d outlets, spent %.0f LKR.",
                     solver, prob.status, int((spend > 1.0).sum()), spend.sum())
            return spend
    log.warning("  no conic solver succeeded — using greedy allocation.")
    return None


def incremental_volume(gap: np.ndarray, k: np.ndarray, spend: np.ndarray) -> np.ndarray:
    return gap * (1 - np.exp(-k * spend))


def main() -> None:
    log = setup_logging()
    log.info("Phase 5 — 5M LKR Western budget optimiser start.")

    w = build_western_frame(log)
    gap = w["gap"].to_numpy(dtype=float)
    k = w["k"].to_numpy(dtype=float)

    # Exact concave solve, with the greedy allocator as a transparent cross-check.
    log.info("Solving allocation (cvxpy exact + greedy cross-check)...")
    spend_cvx = cvxpy_allocate(gap, k, TOTAL_BUDGET_LKR, PER_OUTLET_CAP_LKR, log)
    spend_greedy = greedy_allocate(gap, k, TOTAL_BUDGET_LKR, PER_OUTLET_CAP_LKR,
                                   step=1000.0, log=log)

    if spend_cvx is not None:
        inc_cvx = incremental_volume(gap, k, spend_cvx).sum()
        inc_greedy = incremental_volume(gap, k, spend_greedy).sum()
        log.info("Projected incremental volume — cvxpy=%.0f L  greedy=%.0f L  (greedy/cvxpy=%.3f).",
                 inc_cvx, inc_greedy, inc_greedy / max(inc_cvx, 1e-9))
        spend = spend_cvx
        method = "cvxpy (exact concave)"
    else:
        spend = spend_greedy
        method = "greedy (cvxpy unavailable)"

    w["Trade_Spend_Allocation_LKR"] = np.round(spend, 2)
    w["projected_incremental_L"] = incremental_volume(gap, k, spend)

    total_spent = w["Trade_Spend_Allocation_LKR"].sum()
    total_inc = w["projected_incremental_L"].sum()
    n_funded = int((w["Trade_Spend_Allocation_LKR"] > 1.0).sum())
    log.info("Chosen method: %s.", method)
    log.info("Allocated %.2f LKR (<= %.0f) across %d of %d Western outlets.",
             total_spent, TOTAL_BUDGET_LKR, n_funded, len(w))
    log.info("Projected incremental volume: %.0f L/month (%.1f%% of the %0.f L total gap).",
             total_inc, 100 * total_inc / max(gap.sum(), 1), gap.sum())

    # ── By-distributor + by-spend-type summary ──
    by_dist = (w[w["Trade_Spend_Allocation_LKR"] > 1.0]
               .groupby("primary_distributor")
               .agg(outlets_funded=("Outlet_ID", "count"),
                    spend_LKR=("Trade_Spend_Allocation_LKR", "sum"),
                    incremental_L=("projected_incremental_L", "sum")).round(0))
    log.info("By distributor:\n%s", by_dist.to_string())
    by_type = (w[w["Trade_Spend_Allocation_LKR"] > 1.0]
               .groupby("spend_type")
               .agg(outlets=("Outlet_ID", "count"),
                    spend_LKR=("Trade_Spend_Allocation_LKR", "sum"),
                    incremental_L=("projected_incremental_L", "sum")).round(0))
    log.info("By spend type:\n%s", by_type.to_string())

    # ── Submission CSV: exact two columns, Western only ──
    submission = w[["Outlet_ID", "Trade_Spend_Allocation_LKR"]].copy()
    submission.to_csv(OUTPUT_CSV, index=False)
    log.info("Wrote %s — %d rows, cols=%s", OUTPUT_CSV, len(submission),
             submission.columns.tolist())

    # Diagnostic with the reasoning columns for the report + app.
    diag_cols = ["Outlet_ID", "primary_distributor", "Outlet_Type", "Outlet_Size",
                 "recent_actual", "potential_final", "gap", "constraint_type",
                 "spend_type", "Trade_Spend_Allocation_LKR", "projected_incremental_L"]
    w[diag_cols].to_parquet(GOLD / "budget_allocation_detail.parquet", index=False)
    log.info("Wrote diagnostic: %s", GOLD / "budget_allocation_detail.parquet")


if __name__ == "__main__":
    main()
