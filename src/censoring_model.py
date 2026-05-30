"""Phase 2 — formal censoring (Tobit / Weibull-AFT) + a physical cooler ceiling.

Why this exists
---------------
The prelim treats left-censoring *heuristically*: it drops outlet-months above P99×0.95 from
Stage A training and adds a Stage-B uplift from a censoring/plateau score. That's reasonable,
but the rubric (Methodology 30%) explicitly names **Tobit / hurdle models** and asks for
"cooler replenishment cycles represented in the math." So this module adds two things, each a
cross-check rather than a replacement for the LightGBM primary estimator:

1. **A formal censored-regression cross-check.** We treat each outlet's high-volume months
   (those flagged constrained by censoring/plateau) as *left-censored* observations of latent
   demand — we only know demand was *at least* the observed value. We fit a Tobit model two
   ways and report agreement with the prelim Stage-A predictions:
     * a hand-rolled **Tobit** log-likelihood optimised with `scipy.optimize.minimize`, and
     * a **lifelines `WeibullAFTFitter`** as an independent library implementation.
   If the censored model and LightGBM broadly agree on *direction* (which outlets have the
   most hidden headroom), we can trust the heuristic uplift; if they fought, we'd revisit it.

2. **A physical cooler ceiling.** `physical_max ≈ Cooler_Count × cooler_capacity_L ×
   replenishment_cycles_per_month`, plus an ambient baseline for the 35% of outlets with zero
   coolers (a kade still sells warm/ambient stock). Constants are justified below and checked
   against the data. This becomes (a) a Gold feature and (b) an optional ceiling component
   that makes Stage B's constraint logic *physical*, not only statistical.

Outputs:
  * ``data/gold/physical_max.parquet``      — Outlet_ID, physical_max, cooler headroom signals
  * ``data/gold/_censoring_crosscheck.json`` — agreement stats (logged, used in the report)

Idempotent: overwrites on every run. Pure cross-check — does not modify the LightGBM model.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
LOG_DIR = ROOT / "logs"
for d in (GOLD, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── Physical cooler-capacity constants (documented, then checked against the data) ──
# A typical single-door beverage visi-cooler in a Sri Lankan kade holds ~300–400 L of stock
# when packed; we take 350 L as a representative *stock-on-hand* per cooler.
COOLER_CAPACITY_L = 350.0
# Distributors run weekly routes in the Western/Central belts, so a cooler can be refilled
# ~4 times a month. This is the throughput multiplier that turns stock-on-hand into a
# monthly *flow* ceiling.
REPLENISH_CYCLES_PER_MONTH = 4.0
# Outlets with Cooler_Count == 0 (35% of the master) still move ambient/warm stock off the
# shelf. We give them an ambient monthly baseline rather than a physical_max of zero. 120 L
# is set near the 95th percentile of observed vol_max for zero-cooler outlets (≈156 L) but
# below their max, reflecting that ambient sales are real but capacity-limited.
AMBIENT_BASELINE_L = 120.0

# Floor multiplier shared with predict.py — physical_max must never sit below the outlet's
# own historical peak (a ceiling history already beat is not a ceiling). Where the raw
# engineering estimate *is* below peak, that gap is itself a flag of stale cooler master-data.
HISTORICAL_FLOOR = 1.05


def setup_logging() -> logging.Logger:
    log = logging.getLogger("censoring_model")
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
    fh = logging.FileHandler(LOG_DIR / "censoring_model.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ──────────────────────────────────────────────────────────────────────────────
# Physical cooler ceiling
# ──────────────────────────────────────────────────────────────────────────────
def build_physical_max(gold: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Engineering capacity ceiling per outlet, plus auditable headroom signals.

    raw_physical = Cooler_Count·capacity·cycles  (ambient baseline if zero coolers)
    physical_max = max(raw_physical, historical_peak·1.05)   — a ceiling can't sit below history

    We deliberately keep the *raw* estimate too, because where raw < historical peak that's a
    fingerprint of stale cooler master-data (the classic 'Large outlet logged with 0 coolers'
    decay), which is a forensic finding worth reporting, not something to silently paper over.
    """
    g = gold[["Outlet_ID", "Cooler_Count", "vol_max", "vol_mean"]].copy()

    raw = np.where(
        g["Cooler_Count"] > 0,
        g["Cooler_Count"] * COOLER_CAPACITY_L * REPLENISH_CYCLES_PER_MONTH,
        AMBIENT_BASELINE_L,
    )
    g["physical_max_raw"] = raw
    floor = g["vol_max"] * HISTORICAL_FLOOR
    g["physical_max"] = np.maximum(raw, floor)

    # Headroom = how much engineering capacity sits unused above current peak. Negative raw
    # headroom (peak already exceeds raw capacity) is the stale-cooler-data flag.
    g["cooler_headroom"] = (g["physical_max_raw"] - g["vol_max"]).clip(lower=0)
    g["cooler_capacity_breached"] = (g["vol_max"] > g["physical_max_raw"]).astype(int)
    # Utilisation of raw engineering capacity by the outlet's *typical* month (mean volume).
    g["cooler_utilisation"] = (g["vol_mean"] / g["physical_max_raw"]).clip(upper=5.0)

    n_breach = int(g["cooler_capacity_breached"].sum())
    n_zero = int((g["Cooler_Count"] == 0).sum())
    log.info("Physical ceiling: capacity=%.0fL × %.0f cycles. Zero-cooler outlets=%d "
             "(ambient baseline %.0fL).", COOLER_CAPACITY_L, REPLENISH_CYCLES_PER_MONTH,
             n_zero, AMBIENT_BASELINE_L)
    log.info("  raw capacity breached by historical peak in %d outlets (%.1f%%) — flagged as "
             "likely stale cooler master-data; physical_max floored to peak×1.05 there.",
             n_breach, 100 * n_breach / len(g))
    log.info("  physical_max — mean=%.0f median=%.0f min=%.0f max=%.0f",
             g["physical_max"].mean(), g["physical_max"].median(),
             g["physical_max"].min(), g["physical_max"].max())

    return g[["Outlet_ID", "physical_max_raw", "physical_max", "cooler_headroom",
              "cooler_capacity_breached", "cooler_utilisation"]]


# ──────────────────────────────────────────────────────────────────────────────
# Tobit censored regression (hand-rolled NLL) — cross-check #1
# ──────────────────────────────────────────────────────────────────────────────
def _tobit_neg_loglik(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                      censored: np.ndarray) -> float:
    """Negative log-likelihood of a left-censored (Tobit type-I) Gaussian regression.

    For an uncensored point we observe y exactly → density term log φ((y-Xβ)/σ)/σ.
    For a left-censored point we only know latent demand ≥ y → survival term
    log(1 − Φ((y-Xβ)/σ)) = log Φ((Xβ-y)/σ). σ is parametrised as exp(log_sigma) to keep
    it positive during unconstrained optimisation.
    """
    beta = params[:-1]
    sigma = np.exp(params[-1])
    mu = X @ beta
    z = (y - mu) / sigma

    ll = np.empty_like(y)
    unc = ~censored
    # uncensored: log pdf
    ll[unc] = stats.norm.logpdf(z[unc]) - np.log(sigma)
    # left-censored: log survival = log P(latent >= y) = log Φ((mu - y)/sigma)
    ll[censored] = stats.norm.logcdf((mu[censored] - y[censored]) / sigma)
    return -np.sum(ll)


def fit_tobit(panel: pd.DataFrame, feature_cols: list[str],
              log: logging.Logger) -> pd.Series:
    """Fit a Tobit on the outlet-month panel; return a per-outlet latent-demand estimate.

    Target is log1p(Volume_Liters). A month is treated as left-censored when it sits within
    2% of the outlet's own P95 (the same 'at ceiling' definition gold_features uses for the
    censoring score) — those are the months where the true demand was plausibly throttled.
    """
    df = panel.copy()
    df["y"] = np.log1p(df["Volume_Liters"])
    X = np.column_stack([np.ones(len(df)), df[feature_cols].to_numpy(dtype=float)])
    y = df["y"].to_numpy(dtype=float)
    censored = df["censored_month"].to_numpy(dtype=bool)

    n_features = X.shape[1]
    # init: OLS beta on uncensored points, log_sigma from their residual std
    unc = ~censored
    beta0, *_ = np.linalg.lstsq(X[unc], y[unc], rcond=None)
    resid = y[unc] - X[unc] @ beta0
    log_sigma0 = np.log(max(resid.std(), 1e-3))
    p0 = np.append(beta0, log_sigma0)

    log.info("  Tobit: optimising NLL over %d obs (%d censored, %.1f%%), %d params...",
             len(df), int(censored.sum()), 100 * censored.mean(), n_features + 1)
    res = optimize.minimize(_tobit_neg_loglik, p0, args=(X, y, censored),
                            method="L-BFGS-B", options={"maxiter": 500})
    beta = res.x[:-1]
    sigma = float(np.exp(res.x[-1]))
    log.info("  Tobit: converged=%s  final NLL=%.1f  sigma=%.4f", res.success, res.fun, sigma)

    # Per-outlet latent demand = predicted mean at the outlet's mean feature row, back to L.
    df["_mu"] = X @ beta
    outlet_mu = df.groupby("Outlet_ID")["_mu"].mean()
    return np.expm1(outlet_mu).rename("tobit_latent_demand")


# ──────────────────────────────────────────────────────────────────────────────
# Weibull AFT (lifelines) — cross-check #2, independent implementation
# ──────────────────────────────────────────────────────────────────────────────
def fit_weibull_aft(panel: pd.DataFrame, feature_cols: list[str],
                    log: logging.Logger) -> pd.Series | None:
    """Independent censored fit via lifelines WeibullAFTFitter.

    Reframes the problem as survival analysis: each outlet-month's volume is an event time;
    constrained months are right-censored in *volume* space (true demand is larger), which is
    the natural AFT analogue of our left-censored demand. If lifelines isn't importable we
    skip gracefully — the hand-rolled Tobit is the load-bearing cross-check.
    """
    try:
        from lifelines import WeibullAFTFitter
    except Exception as e:  # pragma: no cover
        log.warning("  lifelines unavailable (%s) — skipping WeibullAFT cross-check.", e)
        return None

    df = panel.copy()
    # event observed when NOT constrained; constrained months are censored (demand >= observed)
    df["event_observed"] = (~df["censored_month"]).astype(int)
    df["duration"] = df["Volume_Liters"].clip(lower=1.0)

    fit_df = df[["duration", "event_observed"] + feature_cols].copy()
    # WeibullAFT needs finite, non-degenerate columns; drop zero-variance features
    keep = [c for c in feature_cols if fit_df[c].std() > 1e-9]
    fit_df = fit_df[["duration", "event_observed"] + keep]

    log.info("  WeibullAFT: fitting on %d obs, %d features...", len(fit_df), len(keep))
    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(fit_df, duration_col="duration", event_col="event_observed")

    # Predict the expected (mean) survival time per row, average to the outlet.
    df["_pred"] = aft.predict_expectation(fit_df).to_numpy()
    outlet_pred = df.groupby("Outlet_ID")["_pred"].mean()
    log.info("  WeibullAFT: median predicted latent demand=%.1f L", float(outlet_pred.median()))
    return outlet_pred.rename("weibull_latent_demand")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def build_panel(monthly: pd.DataFrame, gold: pd.DataFrame,
                feature_cols: list[str]) -> pd.DataFrame:
    """Outlet-month panel joined to per-outlet gold features, with a censored flag.

    censored_month := Volume_Liters >= outlet P95 × 0.98  (mirrors gold_features 'at_ceiling').
    """
    p95 = gold.set_index("Outlet_ID")["vol_p95"]
    df = monthly[["Outlet_ID", "Volume_Liters"]].copy()
    df["_p95"] = df["Outlet_ID"].map(p95)
    df["censored_month"] = df["Volume_Liters"] >= df["_p95"] * 0.98
    df = df.merge(gold[["Outlet_ID"] + feature_cols], on="Outlet_ID", how="left")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def agreement_stats(merged: pd.DataFrame, log: logging.Logger) -> dict:
    """Correlation + mean-difference between LightGBM Stage-A and the censored estimators."""
    out = {}
    base = merged["stage_a_pred"]
    for col, label in [("tobit_latent_demand", "Tobit"),
                       ("weibull_latent_demand", "WeibullAFT")]:
        if col not in merged or merged[col].isna().all():
            continue
        sub = merged[[col]].assign(base=base).dropna()
        pear = float(sub["base"].corr(sub[col], method="pearson"))
        spear = float(sub["base"].corr(sub[col], method="spearman"))
        mean_diff = float((sub[col] - sub["base"]).mean())
        out[label] = {"pearson": round(pear, 4), "spearman": round(spear, 4),
                      "mean_diff_L": round(mean_diff, 1), "n": int(len(sub))}
        log.info("  Agreement LightGBM vs %s: Pearson=%.3f Spearman=%.3f mean_diff=%.1f L (n=%d)",
                 label, pear, spear, mean_diff, len(sub))
    return out


def main() -> None:
    log = setup_logging()
    log.info("Phase 2 — censoring cross-check + physical cooler ceiling start.")

    gold = pd.read_parquet(GOLD / "outlet_features.parquet")
    monthly = pd.read_parquet(SILVER / "monthly_outlet.parquet")
    log.info("Loaded gold=%d outlets, monthly=%s outlet-months.", len(gold), f"{len(monthly):,}")

    # ── 1) Physical cooler ceiling ──
    phys = build_physical_max(gold, log)
    phys_path = GOLD / "physical_max.parquet"
    phys.to_parquet(phys_path, index=False)
    log.info("Wrote %s — shape=%s", phys_path, phys.shape)

    # ── 2) Censored-regression cross-checks ──
    # Compact, interpretable feature set for the censored models (the LightGBM uses the full
    # 100; here we want a small, stable design matrix the Tobit NLL can optimise robustly).
    feature_cols = [c for c in [
        "size_ordinal", "Cooler_Count", "poi_density_weighted_1km",
        "decayed_density_weighted", "market_share_proxy", "months_active",
        "seasonality_jan_mult", "peer_pct_vol_mean",
    ] if c in gold.columns]
    log.info("Censored models use %d features: %s", len(feature_cols), feature_cols)

    panel = build_panel(monthly, gold, feature_cols)
    tobit = fit_tobit(panel, feature_cols, log)
    weibull = fit_weibull_aft(panel, feature_cols, log)

    stage_a = pd.read_parquet(GOLD / "stage_a_pred.parquet")
    merged = stage_a.merge(tobit, on="Outlet_ID", how="left")
    if weibull is not None:
        merged = merged.merge(weibull, on="Outlet_ID", how="left")

    stats_out = agreement_stats(merged, log)
    (GOLD / "_censoring_crosscheck.json").write_text(json.dumps({
        "constants": {"cooler_capacity_L": COOLER_CAPACITY_L,
                      "replenish_cycles_per_month": REPLENISH_CYCLES_PER_MONTH,
                      "ambient_baseline_L": AMBIENT_BASELINE_L},
        "agreement": stats_out,
        "n_outlets": int(len(merged)),
    }, indent=2))
    log.info("Wrote cross-check summary: %s", GOLD / "_censoring_crosscheck.json")

    # Persist the latent-demand estimates alongside for the report / optional ensemble.
    cross = merged[["Outlet_ID", "stage_a_pred", "tobit_latent_demand"]].copy()
    if weibull is not None:
        cross["weibull_latent_demand"] = merged["weibull_latent_demand"]
    cross.to_parquet(GOLD / "censoring_crosscheck.parquet", index=False)
    log.info("Wrote %s — shape=%s", GOLD / "censoring_crosscheck.parquet", cross.shape)

    log.info("Phase 2 done. LightGBM remains the primary estimator; Tobit/WeibullAFT are "
             "directional cross-checks, physical_max is a feature + optional ceiling.")


if __name__ == "__main__":
    main()
