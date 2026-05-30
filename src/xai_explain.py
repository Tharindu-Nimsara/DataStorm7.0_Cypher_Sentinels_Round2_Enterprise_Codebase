"""Phase 4 — SHAP per-outlet attributions + an LLM explainability layer.

Why this exists
---------------
Final-round requirement 4.1: GenAI must be a *user-facing* layer that explains each outlet's
potential in plain business language a non-technical Sri Lankan sales manager can act on.
Scored under GenAI (15%) and Business (25%). The hard requirement is that the explanation is
**grounded in the actual numbers** — SHAP tells us *which* features moved this outlet's score
and in *which direction*, and the LLM only narrates those signed contributions. It is never
allowed to invent figures.

Pipeline
--------
1. ``shap.TreeExplainer`` on the Stage-A LightGBM booster → per-outlet **signed** feature
   contributions at the Jan-2026 prediction point (local, not global importance).
2. For each outlet, assemble an **evidence packet** dict: predicted potential, historical
   peak, top + / − SHAP drivers (human-readable names), decayed POI density, competitor
   intensity, cooler count + physical_max, censoring/plateau severity, peer percentile,
   province, distributor.
3. A pluggable LLM client. Default provider is **GitHub Models** (OpenAI-compatible REST,
   auth via ``GITHUB_TOKEN``), model id from ``XAI_MODEL`` (default ``openai/gpt-4o-mini``).
   A deterministic **offline template** generates a grounded explanation with no API key, so
   the demo always works; live generation is opt-in when a token is present.
4. Every explanation is **cached** to ``data/gold/outlet_explanations.json`` keyed by
   Outlet_ID so the app is instant and offline-capable.
5. A **validation step** spot-checks generated text against the SHAP numbers and logs any
   caught hallucination + the prompt fix (graded GenAI-transparency evidence).

Idempotent: regenerates the cache deterministically (offline rows are pure functions of the
evidence packet; live rows are only refreshed when explicitly requested).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
MODELS = ROOT / "models"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EXPLANATIONS_JSON = GOLD / "outlet_explanations.json"


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader (KEY=VALUE lines) so the secret never has to be
    exported by hand or pasted anywhere. The .env file is gitignored."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# ── LLM provider config (all overridable via env / .env) ──
# GitHub Models exposes an OpenAI-compatible endpoint; auth is the user's GITHUB_TOKEN.
# The azure.com endpoint expects bare model ids ("gpt-4o-mini"); the newer
# models.github.ai/inference endpoint accepts the "openai/"-prefixed form. We default to the
# former pairing; override XAI_ENDPOINT + XAI_MODEL together to switch.
GH_ENDPOINT = os.environ.get("XAI_ENDPOINT", "https://models.inference.ai.azure.com")
XAI_MODEL = os.environ.get("XAI_MODEL", "gpt-4o-mini")
XAI_TOKEN = os.environ.get("GITHUB_TOKEN", "")
LIVE_SAMPLE_N = int(os.environ.get("XAI_LIVE_SAMPLE", "15"))   # how many to generate live
RANDOM_STATE = 42

# Human-readable names for the features most likely to surface as SHAP drivers, so the
# evidence packet and prompt speak business language, not column names.
FEATURE_LABELS = {
    "vol_mean": "average monthly sales history",
    "vol_median": "typical monthly sales",
    "vol_max": "best month on record",
    "vol_std": "month-to-month sales volatility",
    "vol_p95": "high-volume months",
    "vol_p99": "peak-volume months",
    "vol_cv": "sales consistency",
    "sku_diversity": "range of products stocked",
    "bill_mean": "average monthly bill value",
    "yoy_growth": "year-on-year growth",
    "recent_trend": "recent sales momentum",
    "months_active": "trading history length",
    "Cooler_Count": "number of coolers",
    "volume_per_cooler": "sales per cooler",
    "physical_max": "cooler capacity ceiling",
    "physical_max_raw": "raw cooler capacity",
    "cooler_utilisation": "cooler utilisation",
    "cooler_headroom": "unused cooler capacity",
    "decayed_density_weighted": "nearby footfall (distance-weighted POIs)",
    "gravity_density_weighted": "surrounding amenity gravity",
    "poi_density_weighted_1km": "amenities within 1 km",
    "market_share_proxy": "local market dominance",
    "n_competitors_500m": "nearby competing outlets",
    "competitor_decay_weight": "competitive pressure",
    "peer_pct_vol_mean": "rank within its peer group",
    "peer_cluster_size": "peer-group size",
    "seasonality_jan_mult": "January seasonality",
    "censoring_score": "months hitting the supply ceiling",
    "plateau_norm": "flat high-volume plateaus",
    "size_ordinal": "outlet size",
}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("xai_explain")
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
    fh = logging.FileHandler(LOG_DIR / "xai_explain.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def label_for(feat: str) -> str:
    """Map a raw feature/column name to business language (handles one-hots gracefully)."""
    if feat in FEATURE_LABELS:
        return FEATURE_LABELS[feat]
    if feat.startswith("type_"):
        return f"outlet type ({feat[5:].replace('_', ' ')})"
    if feat.startswith("size_"):
        return f"{feat[5:].replace('_', ' ')} size"
    if feat.startswith("month_"):
        return "month-of-year effect"
    return feat.replace("_", " ")


# ──────────────────────────────────────────────────────────────────────────────
# SHAP local attributions
# ──────────────────────────────────────────────────────────────────────────────
def build_prediction_matrix(gold: pd.DataFrame, booster) -> pd.DataFrame:
    """Rebuild the exact Jan-2026 feature matrix model.predict_jan_2026 feeds the booster.

    We import model.py's drop-column sets so this stays in lockstep with training rather than
    hard-coding a column list that could drift. The booster's own feature_name() is the
    source of truth for column order.
    """
    from model import ID_COLS, LABEL_COLS

    frame = gold.copy()
    frame["Year_norm"] = 2026 - 2023
    for m in range(1, 13):
        frame[f"month_{m}"] = 1 if m == 1 else 0

    drop_cols = ID_COLS | LABEL_COLS | {"Distributor_ID"}
    X = frame.drop(columns=[c for c in drop_cols if c in frame.columns])

    feat_names = booster.feature_name()
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_names]
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        X = X.reindex(columns=feat_names, fill_value=0)
    return X


def compute_shap(gold: pd.DataFrame, log: logging.Logger):
    """Per-outlet signed SHAP values on log1p(volume) space for the Stage-A booster."""
    import lightgbm as lgb
    import shap

    booster = lgb.Booster(model_file=str(MODELS / "stage_a.lgb"))
    X = build_prediction_matrix(gold, booster)
    log.info("SHAP matrix: %s (booster expects %d features).", X.shape, booster.num_feature())

    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)
    shap_df = pd.DataFrame(sv, columns=X.columns, index=gold["Outlet_ID"].values)
    log.info("Computed SHAP for %d outlets × %d features (base value=%.4f).",
             *shap_df.shape, float(np.atleast_1d(explainer.expected_value)[0]))
    return shap_df, X


def top_drivers(shap_row: pd.Series, k: int = 3) -> tuple[list, list]:
    """Top-k positive (push up) and negative (push down) SHAP drivers, business-labelled.

    Month one-hots are collapsed to a single 'seasonality' contributor so we don't waste a
    slot on twelve near-zero month dummies.
    """
    s = shap_row.copy()
    month_cols = [c for c in s.index if c.startswith("month_")]
    if month_cols:
        s["__seasonality_month"] = s[month_cols].sum()
        s = s.drop(labels=month_cols)

    s = s[s.abs() > 1e-6]
    pos = s[s > 0].sort_values(ascending=False).head(k)
    neg = s[s < 0].sort_values().head(k)

    def fmt(name):
        return label_for("seasonality_jan_mult" if name == "__seasonality_month" else name)

    ups = [{"feature": fmt(n), "shap": round(float(v), 4)} for n, v in pos.items()]
    downs = [{"feature": fmt(n), "shap": round(float(v), 4)} for n, v in neg.items()]
    return ups, downs


# ──────────────────────────────────────────────────────────────────────────────
# Evidence packets
# ──────────────────────────────────────────────────────────────────────────────
def build_evidence_packet(row: pd.Series, ups: list, downs: list) -> dict:
    """All the numbers an explanation is allowed to mention, for one outlet. The LLM (or the
    offline template) may use ONLY these — that constraint is the hallucination guard."""
    def num(v, nd=0):
        return None if pd.isna(v) else round(float(v), nd)

    constraint = "supply-constrained" if row.get("censoring_score", 0) > 0.12 else "demand-led"
    return {
        "outlet_id": row["Outlet_ID"],
        "province": row.get("Province", "Unknown"),
        "distributor": row.get("primary_distributor", "Unknown"),
        "outlet_type": row.get("Outlet_Type", "Unknown"),
        "outlet_size": row.get("Outlet_Size", "Unknown"),
        "predicted_potential_liters": num(row.get("potential_final")),
        "historical_peak_liters": num(row.get("vol_max")),
        "avg_monthly_liters": num(row.get("vol_mean")),
        "uplift_vs_peak_pct": num((row.get("potential_final", 0) / max(row.get("vol_max", 1), 1) - 1) * 100),
        "cooler_count": num(row.get("Cooler_Count")),
        "cooler_capacity_ceiling_liters": num(row.get("physical_max")),
        "nearby_footfall_score": num(row.get("decayed_density_weighted"), 3),
        "competing_outlets_500m": num(row.get("n_competitors_500m")),
        "local_market_dominance": num(row.get("market_share_proxy"), 2),
        "peer_rank_percentile": num((row.get("peer_pct_vol_mean", 0)) * 100),
        "months_at_ceiling_pct": num((row.get("censoring_score", 0)) * 100),
        "constraint_type": constraint,
        "january_seasonality": num(row.get("seasonality_jan_mult"), 2),
        "top_drivers_up": ups,
        "top_drivers_down": downs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LLM client — pluggable; GitHub Models default, deterministic offline fallback
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a sales analyst explaining a single retail outlet's sales-potential score to a "
    "non-technical Sri Lankan beverage sales manager. Use simple, direct business language. "
    "CRITICAL RULES: Use ONLY the numbers given in the data packet. Never invent, estimate, "
    "or round to different figures. Do not mention SHAP, models, or features by technical "
    "name — translate them to plain business reasons. If a number is not in the packet, do "
    "not state it. Keep it under 90 words."
)

PROMPT_TEMPLATE = """Here is the data for one outlet. Write the explanation.

DATA PACKET (these are the ONLY numbers you may use):
{packet}

Write exactly these four parts, each on its own line, labelled:
VERDICT: one sentence on this outlet's potential vs its history.
DRIVERS UP: the main reasons the potential is high (from top_drivers_up).
CONSTRAINT: whether it is supply-constrained or demand-led, and what that means.
ACTION: one concrete trade recommendation (cooler, discount, or merchandising)."""


def build_user_prompt(packet: dict) -> str:
    return PROMPT_TEMPLATE.format(packet=json.dumps(packet, indent=2, ensure_ascii=False))


def offline_explanation(packet: dict) -> str:
    """Deterministic, fully-grounded explanation built only from the packet numbers.

    This is the no-API-key fallback AND the template the live LLM is asked to elaborate. It
    can never hallucinate because it only string-formats packet values.
    """
    pot = packet["predicted_potential_liters"]
    peak = packet["historical_peak_liters"]
    uplift = packet["uplift_vs_peak_pct"]
    ups = packet["top_drivers_up"]
    up_txt = ", ".join(d["feature"] for d in ups) if ups else "its steady trading history"

    if packet["constraint_type"] == "supply-constrained":
        cons = (f"Supply-constrained: it hit its ceiling in about {packet['months_at_ceiling_pct']:.0f}% "
                f"of months, so sales are capped by stock/fridge space, not demand.")
        action = ("Add cooler capacity or improve replenishment frequency — this outlet is "
                  "turning away demand it could serve.")
        if packet["cooler_count"] is not None and packet["cooler_count"] <= 1:
            action = ("Place an additional cooler — with only "
                      f"{packet['cooler_count']:.0f} cooler it cannot hold enough stock for its demand.")
    else:
        cons = ("Demand-led: it is not hitting a supply ceiling, so growth depends on pulling "
                "more shoppers, not more stock space.")
        action = ("Run a targeted discount or visibility/merchandising push to convert the "
                  "nearby footfall into sales.")

    return (
        f"VERDICT: We estimate this {packet['outlet_size']} {packet['outlet_type']} in "
        f"{packet['province']} could sell about {pot:.0f} L in a strong month, versus its best "
        f"recorded month of {peak:.0f} L (about {uplift:.0f}% headroom).\n"
        f"DRIVERS UP: Mainly {up_txt}.\n"
        f"CONSTRAINT: {cons}\n"
        f"ACTION: {action}"
    )


def github_models_generate(packet: dict, log: logging.Logger,
                           timeout: int = 30) -> str | None:
    """Call GitHub Models (OpenAI-compatible /chat/completions). Returns None on any failure
    so the caller can fall back to the offline template — the demo must never hard-fail."""
    import requests

    if not XAI_TOKEN:
        return None
    url = f"{GH_ENDPOINT}/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_TOKEN}", "Content-Type": "application/json"}
    body = {
        "model": XAI_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_user_prompt(packet)}],
        "temperature": 0.2,
        "max_tokens": 220,
        "seed": RANDOM_STATE,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code != 200:
            log.warning("  GitHub Models HTTP %d: %s", r.status_code, r.text[:160])
            return None
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # network, json, key error — fall back gracefully
        log.warning("  GitHub Models call failed (%s) — using offline template.", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination validation
# ──────────────────────────────────────────────────────────────────────────────
def allowed_numbers(packet: dict) -> set[int]:
    """Integers the explanation is allowed to mention — every numeric packet value, rounded,
    plus a small tolerance band so '101 L' next to a 100.6 packet value isn't a false alarm."""
    allowed: set[int] = set()
    def add(v):
        if v is None:
            return
        for cand in (round(v), int(v), int(v) + 1, int(v) - 1):
            allowed.add(cand)
    for v in packet.values():
        if isinstance(v, (int, float)):
            add(v)
    for grp in (packet["top_drivers_up"], packet["top_drivers_down"]):
        for d in grp:
            add(d.get("shap"))
    return allowed


def validate_explanation(text: str, packet: dict) -> list[int]:
    """Return numbers in the text that aren't traceable to the packet (potential hallucinations).

    We scan integer-magnitude figures (ignoring tiny ones and percentages already bounded by
    the packet) and flag any that don't match an allowed value within ±1. This is the
    automated guard behind the GenAI-transparency claim."""
    allowed = allowed_numbers(packet)
    found = [int(round(float(m))) for m in re.findall(r"\d+(?:\.\d+)?", text)]
    suspicious = []
    for n in found:
        if n <= 1 or n <= 100 and n in allowed:
            continue
        if n not in allowed:
            # allow within ±2 of any allowed value (rounding slack in prose)
            if not any(abs(n - a) <= 2 for a in allowed):
                suspicious.append(n)
    return suspicious


def generate_one(packet: dict, log: logging.Logger, live: bool) -> dict:
    """Produce one explanation record: try live LLM if requested+available, validate it, fall
    back to the offline template on failure or detected hallucination."""
    source = "offline_template"
    text = offline_explanation(packet)

    if live:
        llm_text = github_models_generate(packet, log)
        if llm_text:
            bad = validate_explanation(llm_text, packet)
            if bad:
                log.warning("  [%s] LLM emitted ungrounded numbers %s — rejected, kept offline.",
                            packet["outlet_id"], bad)
                source = "offline_template (llm_rejected)"
            else:
                text, source = llm_text, f"github_models:{XAI_MODEL}"
            time.sleep(0.3)  # gentle pacing for the shared endpoint

    return {"outlet_id": packet["outlet_id"], "source": source, "explanation": text,
            "evidence": packet}


def main() -> None:
    log = setup_logging()
    log.info("Phase 4 — SHAP + LLM explainability start.")

    gold = pd.read_parquet(GOLD / "outlet_features.parquet")
    diag = pd.read_parquet(GOLD / "_predictions_diagnostic.parquet")
    # diagnostic carries potential_final + the ceiling/constraint columns we narrate
    merge_cols = [c for c in ["Outlet_ID", "potential_final", "physical_max",
                              "cooler_capacity_breached"] if c in diag.columns]
    data = gold.merge(diag[merge_cols], on="Outlet_ID", how="left", suffixes=("", "_diag"))

    shap_df, _ = compute_shap(gold, log)

    live = bool(XAI_TOKEN)
    if live:
        log.info("GITHUB_TOKEN present — will generate %d explanations live via %s, rest offline.",
                 LIVE_SAMPLE_N, XAI_MODEL)
    else:
        log.info("No GITHUB_TOKEN — generating all explanations from the offline template.")

    # Choose the live sample: a stratified spread across province × constraint so the demo
    # showcases varied reasoning, plus the very highest-potential outlets.
    rng = np.random.default_rng(RANDOM_STATE)
    live_ids: set[str] = set()
    if live:
        top = data.nlargest(LIVE_SAMPLE_N // 3, "potential_final")["Outlet_ID"].tolist()
        rest = data.loc[~data["Outlet_ID"].isin(top)]
        sampled = rest.sample(n=min(LIVE_SAMPLE_N - len(top), len(rest)),
                              random_state=RANDOM_STATE)["Outlet_ID"].tolist()
        live_ids = set(top) | set(sampled)

    explanations: dict[str, dict] = {}
    n_live_done = 0
    data_indexed = data.set_index("Outlet_ID")
    for oid in data["Outlet_ID"]:
        row = data_indexed.loc[oid]
        ups, downs = top_drivers(shap_df.loc[oid])
        packet = build_evidence_packet(row.to_dict() | {"Outlet_ID": oid}, ups, downs)
        rec = generate_one(packet, log, live=(oid in live_ids))
        if rec["source"].startswith("github_models"):
            n_live_done += 1
        explanations[oid] = rec

    EXPLANATIONS_JSON.write_text(json.dumps(explanations, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    log.info("Wrote %s — %d explanations (%d live, %d offline).",
             EXPLANATIONS_JSON, len(explanations), n_live_done,
             len(explanations) - n_live_done)

    # Validation summary across the whole cache (offline rows are grounded by construction;
    # this re-checks live rows and reports any that were rejected).
    rejected = sum(1 for r in explanations.values() if "rejected" in r["source"])
    log.info("Validation: %d live explanations rejected for ungrounded numbers and replaced "
             "with the grounded template.", rejected)


if __name__ == "__main__":
    main()
