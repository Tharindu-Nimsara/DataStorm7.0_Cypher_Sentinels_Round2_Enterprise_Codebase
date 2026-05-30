"""Phase 1 — spatial upgrade: distance-decay POI gravity + competitive density.

Why this exists
---------------
The prelim spatial features were flat ring counts ("how many schools within 1 km").
That throws away *how close* each POI is: a bus stand 80 m from a kade pulls far more
footfall than one 1.9 km away, yet both counted equally in the 2 km ring. The final-round
rubric explicitly wants non-linear gravity / distance-decay signals plus a competitive-
saturation measure, so this module replaces the flat counts with two things:

1. **Per-type distance-decayed POI density.** For every POI type we apply a decay weight
   that falls off with distance, using a *per-type bandwidth* (a bus stop's pull decays
   fast; a hospital reaches further). See ``DECAY_BANDWIDTHS`` for the justification.

2. **Competitive catchment density.** Using a BallTree (haversine) over the real outlet
   coordinates, we count and distance-weight competing outlets within ~500 m, then form a
   ``market_share_proxy`` saturation index = own_pull / (own_pull + Σ competitor_pull).

A note on the data we actually have
-----------------------------------
Our Overpass cache stores *cumulative ring counts* per POI type (500 m / 1 km / 2 km), not
individual POI coordinates — re-scraping ~20k outlets for point geometry at 1 req/s would
cost 5+ hours and the cache is only ~82% complete. So the POI decay here is **ring-based
gravity**: we difference the cumulative rings into shell counts (0–500, 500–1000,
1000–2000 m) and evaluate the decay kernel at each shell's representative midpoint. This is
an honest, defensible approximation of a continuous gravity model and is documented as such
in the report. The *competitor* signal, by contrast, uses a genuine BallTree over exact
outlet coordinates — no approximation there.

Outputs ``data/gold/poi_decay_features.parquet`` (one row per master outlet), which
``gold_features.py`` merges in. Idempotent: overwrites on every run.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import dq_checks as dq

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
POI_CACHE = ROOT / "data" / "external" / "poi_cache"
REJECTED = ROOT / "data" / "rejected_records"
LOG_DIR = ROOT / "logs"
for d in (GOLD, REJECTED, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# POI types must match poi_scrape.py / gold_features.py.
POI_TYPES = ["school", "university", "bus_station", "hospital",
             "place_of_worship", "marketplace", "restaurant", "tourism"]
RING_EDGES = [0, 500, 1000, 2000]              # metres; cache holds the cumulative edges
SHELLS = [(0, 500), (500, 1000), (1000, 2000)]  # shells we difference the rings into
SHELL_MIDPOINTS_M = {(0, 500): 250.0, (500, 1000): 750.0, (1000, 2000): 1500.0}

# ── Per-type decay bandwidths (λ, in metres) for an exponential kernel w=exp(-d/λ) ──
# Justification — these encode how far each POI type's "pull" on a beverage kade reaches:
#   * bus_station (λ=180): commuter footfall is hyper-local; a stand two streets away does
#     little. Short λ so the 250 m shell dominates.
#   * restaurant (λ=200): impulse / on-premise demand is also very local.
#   * marketplace (λ=300): markets draw shoppers from a slightly wider catchment.
#   * school (λ=250) & place_of_worship (λ=300): regular but localised gatherings.
#   * university (λ=450) & tourism (λ=500): destinations people travel across town for.
#   * hospital (λ=500): regional draw — staff, visitors, nearby pharmacies/eateries.
# Values are deliberately in the 150–500 m band tasks.md suggested; we run a λ-sensitivity
# check at the end to confirm the feature *ranking* is stable to ±50% changes in λ.
DECAY_BANDWIDTHS = {
    "bus_station": 180.0,
    "restaurant": 200.0,
    "school": 250.0,
    "marketplace": 300.0,
    "place_of_worship": 300.0,
    "university": 450.0,
    "hospital": 500.0,
    "tourism": 500.0,
}

# Gravity kernel exponent for the alternative w = 1/(d^β + ε) score (reported as a
# cross-check; the exponential kernel is primary because λ is interpretable in metres).
GRAVITY_BETA = 1.5
GRAVITY_EPS = 1.0

# A POI weighted-importance score (same spirit as the prelim POI_WEIGHTS) so the combined
# decayed density emphasises high-footfall anchors.
POI_IMPORTANCE = {
    "school": 2.0, "university": 1.0, "bus_station": 3.0, "hospital": 2.0,
    "place_of_worship": 1.0, "marketplace": 1.0, "restaurant": 1.0, "tourism": 1.0,
}

# Competitor catchment: other beverage outlets within this radius compete for the same
# trade. 500 m ≈ a 5-minute walk, the realistic substitution range for a kade purchase.
COMPETITOR_RADIUS_M = 500.0
COMPETITOR_LAMBDA_M = 250.0          # decay for competitor "pull"
EARTH_RADIUS_M = 6_371_000.0


def setup_logging() -> logging.Logger:
    log = logging.getLogger("poi_decay")
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
    fh = logging.FileHandler(LOG_DIR / "poi_decay.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def load_poi_cache(master_ids: set[str], log: logging.Logger) -> pd.DataFrame:
    """Read every cached Overpass JSON into a tidy frame of cumulative ring counts.

    One row per (outlet, poi_type) with columns r500/r1000/r2000. Outlets without a cache
    file simply don't appear here; the caller imputes them with a flag downstream.
    """
    cached = sorted(p for p in POI_CACHE.glob("*.json") if p.stem in master_ids)
    log.info("POI cache files for master outlets: %d", len(cached))

    rows = []
    bad = 0
    for path in cached:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            bad += 1
            continue
        counts = data.get("counts", {})
        rec = {"Outlet_ID": data.get("Outlet_ID", path.stem)}
        for poi in POI_TYPES:
            ring = counts.get(poi, {})
            rec[f"{poi}__r500"] = float(ring.get("500", 0) or 0)
            rec[f"{poi}__r1000"] = float(ring.get("1000", 0) or 0)
            rec[f"{poi}__r2000"] = float(ring.get("2000", 0) or 0)
        rows.append(rec)
    if bad:
        log.warning("  %d cache files were unreadable and skipped.", bad)
    return pd.DataFrame(rows)


def run_input_dq_checks(coords: pd.DataFrame, poi_raw: pd.DataFrame,
                        log: logging.Logger) -> pd.DataFrame:
    """Apply the reusable DQ checks to the *external* spatial inputs (rubric rule 3).

    Coordinates feed the BallTree, so out-of-bounds lat/lon would silently poison the
    competitor neighbour search. We quarantine such rows (never drop) and proceed with the
    clean set. POI ring counts must be non-negative and monotonically non-decreasing across
    radii (a 1 km ring can't hold fewer POIs than the 500 m ring it contains); violations
    are quarantined too.
    """
    # 1) Geo bounds on the coordinates that drive the BallTree.
    coords_ok, coords_bad = dq.check_geo_bounds(coords, "Latitude", "Longitude")
    if len(coords_bad):
        log.warning("  DQ: %d coordinate rows outside Sri Lanka bounds → quarantined.",
                    len(coords_bad))
        coords_bad.to_csv(REJECTED / "poi_decay_coords_rejected.csv", index=False)
    else:
        # idempotent: an empty quarantine file still overwrites any stale one
        coords.iloc[0:0].assign(failure_reason=pd.Series(dtype=str)).to_csv(
            REJECTED / "poi_decay_coords_rejected.csv", index=False)

    # 2) POI ring sanity: non-negative and r500 <= r1000 <= r2000 per type.
    bad_mask = pd.Series(False, index=poi_raw.index)
    reasons = pd.Series("", index=poi_raw.index)
    for poi in POI_TYPES:
        c5, c10, c20 = f"{poi}__r500", f"{poi}__r1000", f"{poi}__r2000"
        neg = (poi_raw[c5] < 0) | (poi_raw[c10] < 0) | (poi_raw[c20] < 0)
        non_mono = (poi_raw[c10] < poi_raw[c5]) | (poi_raw[c20] < poi_raw[c10])
        hit = neg | non_mono
        reasons = reasons.mask(hit & (reasons == ""), f"{poi} rings negative or non-monotonic")
        bad_mask |= hit
    poi_bad = poi_raw.loc[bad_mask].copy()
    if len(poi_bad):
        poi_bad["failure_reason"] = reasons.loc[bad_mask]
        log.warning("  DQ: %d outlets had inconsistent POI rings → quarantined.", len(poi_bad))
        poi_bad.to_csv(REJECTED / "poi_decay_rings_rejected.csv", index=False)
    else:
        poi_raw.iloc[0:0].assign(failure_reason=pd.Series(dtype=str)).to_csv(
            REJECTED / "poi_decay_rings_rejected.csv", index=False)
        log.info("  DQ: all cached POI rings are non-negative and monotonic.")

    return coords_ok


def compute_decay_density(poi_raw: pd.DataFrame, log: logging.Logger,
                          bandwidths: dict[str, float] | None = None) -> pd.DataFrame:
    """Ring-based gravity: difference cumulative rings into shells, weight by the decay
    kernel evaluated at each shell midpoint, sum per type.

    For an exponential kernel ``w(d)=exp(-d/λ)`` the decayed density of type *t* is::

        decayed_t = Σ_shell  shell_count_t(shell) · exp(-midpoint(shell) / λ_t)

    We also compute a gravity-kernel variant ``1/(d^β+ε)`` as a reported cross-check, and a
    single combined score that folds in each type's footfall importance.
    """
    bandwidths = bandwidths or DECAY_BANDWIDTHS
    out = poi_raw[["Outlet_ID"]].copy()
    combined_exp = np.zeros(len(poi_raw), dtype=float)
    combined_grav = np.zeros(len(poi_raw), dtype=float)

    for poi in POI_TYPES:
        c5 = poi_raw[f"{poi}__r500"].to_numpy()
        c10 = poi_raw[f"{poi}__r1000"].to_numpy()
        c20 = poi_raw[f"{poi}__r2000"].to_numpy()
        # cumulative -> shell counts; clip at 0 so any tiny non-monotonic noise can't go negative
        shell_counts = {
            (0, 500): np.clip(c5, 0, None),
            (500, 1000): np.clip(c10 - c5, 0, None),
            (1000, 2000): np.clip(c20 - c10, 0, None),
        }
        lam = bandwidths[poi]
        exp_score = np.zeros(len(poi_raw), dtype=float)
        grav_score = np.zeros(len(poi_raw), dtype=float)
        for shell, n in shell_counts.items():
            d = SHELL_MIDPOINTS_M[shell]
            exp_score += n * np.exp(-d / lam)
            grav_score += n / (d ** GRAVITY_BETA + GRAVITY_EPS)
        out[f"decayed_density_{poi}"] = exp_score
        out[f"gravity_density_{poi}"] = grav_score
        combined_exp += exp_score * POI_IMPORTANCE[poi]
        combined_grav += grav_score * POI_IMPORTANCE[poi]

    out["decayed_density_weighted"] = combined_exp
    out["gravity_density_weighted"] = combined_grav
    log.info("Decay density built for %d cached outlets (%d POI types, exp+gravity kernels).",
             len(out), len(POI_TYPES))
    return out


def compute_competitor_density(coords: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """BallTree (haversine) over exact outlet coordinates → competitive catchment signal.

    For each outlet we look up *other* outlets within ``COMPETITOR_RADIUS_M`` and form:
      * ``n_competitors_500m`` — how many rival kades share the catchment,
      * ``competitor_decay_weight`` — Σ exp(-d/λ) over those rivals (closer rivals weigh more),
      * ``market_share_proxy``    — own_pull / (own_pull + competitor_pull), where own_pull=1;
        1.0 means a local monopoly, →0 means a saturated catchment.

    A BallTree gives O(n log n) radius queries instead of an O(n²) all-pairs loop — at ~19.5k
    outlets that's the difference between a second and minutes.
    """
    pts = coords.dropna(subset=["Latitude", "Longitude"]).copy()
    log.info("Building BallTree over %d outlet coordinates (haversine).", len(pts))

    lat_rad = np.radians(pts["Latitude"].to_numpy())
    lon_rad = np.radians(pts["Longitude"].to_numpy())
    X = np.column_stack([lat_rad, lon_rad])
    tree = BallTree(X, metric="haversine")

    radius_rad = COMPETITOR_RADIUS_M / EARTH_RADIUS_M
    lambda_rad = COMPETITOR_LAMBDA_M / EARTH_RADIUS_M

    # query_radius with distances; results include the point itself (distance 0).
    idxs, dists = tree.query_radius(X, r=radius_rad, return_distance=True, sort_results=True)

    n_comp = np.empty(len(pts), dtype=int)
    comp_weight = np.empty(len(pts), dtype=float)
    for i, (neigh, dd) in enumerate(zip(idxs, dists)):
        # drop self (the zero-distance entry for this row)
        mask = neigh != i
        d_others = dd[mask]
        n_comp[i] = d_others.size
        # exp(-d/λ) in radian space == exp(-d_metres/λ_metres); weight closer rivals more
        comp_weight[i] = float(np.exp(-d_others / lambda_rad).sum()) if d_others.size else 0.0

    own_pull = 1.0
    market_share = own_pull / (own_pull + comp_weight)

    res = pd.DataFrame({
        "Outlet_ID": pts["Outlet_ID"].to_numpy(),
        "n_competitors_500m": n_comp,
        "competitor_decay_weight": comp_weight,
        "market_share_proxy": market_share,
    })
    log.info("Competitor density — mean rivals=%.2f  median=%.0f  max=%d  | "
             "market_share mean=%.3f (1.0=monopoly)",
             res["n_competitors_500m"].mean(), res["n_competitors_500m"].median(),
             int(res["n_competitors_500m"].max()), res["market_share_proxy"].mean())
    return res


def impute_decay_features(features: pd.DataFrame, decay_cols: list[str],
                          log: logging.Logger) -> pd.DataFrame:
    """Cluster-median imputation for outlets with no POI cache, keeping the flag.

    Mirrors the prelim ``impute_spatial`` pattern: fill from the (Outlet_Type, Province)
    median of *observed* outlets, fall back to the global median, and preserve
    ``poi_decay_imputed`` as a model feature so the booster can learn that imputed rows are
    less certain. (Reuses the existing ``poi_imputed`` flag concept but namespaced to this
    module's outputs so the two stay independently auditable.)
    """
    needs = features["poi_decay_imputed"] == 1
    log.info("Imputing decay features for %d outlets without a POI cache.", int(needs.sum()))

    observed = features.loc[features["poi_decay_imputed"] == 0]
    cluster_med = observed.groupby(["Outlet_Type", "Province"])[decay_cols].median()
    global_med = observed[decay_cols].median()

    for idx in features.index[needs]:
        key = (features.at[idx, "Outlet_Type"], features.at[idx, "Province"])
        vals = cluster_med.loc[key] if key in cluster_med.index else global_med
        for c in decay_cols:
            v = vals[c]
            features.at[idx, c] = float(v) if pd.notna(v) else 0.0
    features[decay_cols] = features[decay_cols].fillna(0.0)
    return features


def lambda_sensitivity_check(poi_raw: pd.DataFrame, base: pd.DataFrame,
                             log: logging.Logger) -> None:
    """Vary every λ by ±50% and confirm the combined-density *ranking* is stable.

    A decay model is only trustworthy if small, defensible changes to the bandwidths don't
    reshuffle which outlets look spatially strong. We recompute the weighted density at
    0.5×λ and 1.5×λ and report Spearman rank correlation against the base — high correlation
    means our exact λ choices aren't load-bearing, which is what we want to claim in the
    report.
    """
    base_rank = base.set_index("Outlet_ID")["decayed_density_weighted"]
    for scale in (0.5, 1.5):
        scaled_bw = {k: v * scale for k, v in DECAY_BANDWIDTHS.items()}
        alt = compute_decay_density(poi_raw, log=logging.getLogger("_quiet"),
                                    bandwidths=scaled_bw)
        alt_rank = alt.set_index("Outlet_ID")["decayed_density_weighted"]
        joined = pd.concat([base_rank, alt_rank], axis=1, keys=["base", "alt"]).dropna()
        rho = joined["base"].corr(joined["alt"], method="spearman")
        log.info("  λ×%.1f sensitivity: Spearman rank corr vs base = %.4f", scale, rho)


def main() -> None:
    log = setup_logging()
    log.info("Phase 1 — POI distance-decay + competitor density start.")

    master = pd.read_parquet(SILVER / "outlet_master_clean.parquet")
    coords = pd.read_parquet(SILVER / "outlet_coordinates_clean.parquet")
    log.info("Loaded master=%d outlets, coords=%d outlets.", len(master), len(coords))

    master_ids = set(master["Outlet_ID"])
    poi_raw = load_poi_cache(master_ids, log)

    # DQ on the external spatial inputs, then keep only clean coordinates for the BallTree.
    coords_clean = run_input_dq_checks(coords, poi_raw, log)

    # ── POI decay density (cached outlets only; impute the rest) ──
    decay = compute_decay_density(poi_raw, log)
    decay_cols = [c for c in decay.columns if c != "Outlet_ID"]

    # ── λ-sensitivity stability check ──
    lambda_sensitivity_check(poi_raw, decay, log)

    # ── Competitor density via BallTree over clean coords ──
    competitor = compute_competitor_density(coords_clean, log)

    # ── Assemble one row per master outlet ──
    feats = master[["Outlet_ID", "Outlet_Type"]].copy()
    feats = feats.merge(decay, on="Outlet_ID", how="left")
    feats["poi_decay_imputed"] = feats["decayed_density_weighted"].isna().astype(int)

    # Province is needed for cluster-median imputation; derive it the same way gold does.
    monthly = pd.read_parquet(SILVER / "monthly_outlet.parquet")
    prov = _outlet_province(monthly)
    feats = feats.merge(prov, on="Outlet_ID", how="left")
    feats["Province"] = feats["Province"].fillna("Unknown")

    feats = impute_decay_features(feats, decay_cols, log)

    # Competitor features: outlets with no coords get 0 rivals / monopoly proxy + a flag.
    feats = feats.merge(competitor, on="Outlet_ID", how="left")
    feats["competitor_imputed"] = feats["n_competitors_500m"].isna().astype(int)
    feats["n_competitors_500m"] = feats["n_competitors_500m"].fillna(0).astype(int)
    feats["competitor_decay_weight"] = feats["competitor_decay_weight"].fillna(0.0)
    feats["market_share_proxy"] = feats["market_share_proxy"].fillna(1.0)

    out_cols = (["Outlet_ID"] + decay_cols +
                ["poi_decay_imputed", "n_competitors_500m", "competitor_decay_weight",
                 "market_share_proxy", "competitor_imputed"])
    out = feats[out_cols].copy()

    assert len(out) == len(master), f"row drift: {len(out)} != master {len(master)}"
    assert out["Outlet_ID"].is_unique, "duplicate Outlet_IDs in poi_decay output"
    assert out.isna().sum().sum() == 0, "nulls remain in poi_decay output"

    out_path = GOLD / "poi_decay_features.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Wrote %s — shape=%s", out_path, out.shape)
    log.info("Imputed POI-decay rows: %d (%.1f%%); competitor-imputed: %d (%.1f%%)",
             int(feats["poi_decay_imputed"].sum()), 100 * feats["poi_decay_imputed"].mean(),
             int(feats["competitor_imputed"].sum()), 100 * feats["competitor_imputed"].mean())


def _outlet_province(monthly: pd.DataFrame) -> pd.DataFrame:
    """Province from each outlet's most-frequent Distributor_ID prefix (matches gold_features)."""
    prefix_to_prov = {"DIST_W_": "Western", "DIST_C_": "Central",
                      "DIST_NW_": "North-Western", "DIST_S_": "Southern"}
    primary = (monthly.groupby("Outlet_ID")["Distributor_ID"]
               .agg(lambda s: s.mode().iloc[0] if len(s) else None))

    def to_prov(d):
        if pd.isna(d):
            return "Unknown"
        for pre, prov in prefix_to_prov.items():
            if d.startswith(pre):
                return prov
        return "Unknown"

    return primary.map(to_prov).rename("Province").reset_index()


if __name__ == "__main__":
    main()
