"""Generate report-grade charts to reports/figures/.

Produces:
    - rejected_records_bar.png       (Page 2)
    - sri_lanka_outlets.png          (Page 3)
    - methodology_flow.png           (Page 4)
    - predictions_histogram.png      (Page 5)
    - predictions_geo_heatmap.png    (Page 5)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
REJECTS = ROOT / "data" / "rejected_records"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# OCTAVE / John Keells Group house style — teal headings on clean white, to match the
# official problem-statement document the judges are reading.
NAVY = "#1F6F86"      # primary teal (section headers, bars) — kept var name for minimal diff
GREY = "#5B6770"      # slate grey body/axis
LIGHT = "#DCEAEF"     # pale teal fill
ACCENT = "#4FB0C6"    # bright teal accent
PINK = "#E6007E"      # OCTAVE magenta, used sparingly for emphasis

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GREY,
    "axes.labelcolor": GREY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.titlecolor": NAVY,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# 1. Rejected records bar
# ============================================================
def chart_rejected_records():
    """Bar chart of quarantined rows by failure reason category."""
    reason_counts: dict[str, int] = {}
    for f in REJECTS.glob("*_rejected.csv"):
        df = pd.read_csv(f)
        if "failure_reason" not in df.columns:
            continue
        # Bucket the long reasons into short labels
        for r, n in df["failure_reason"].value_counts().items():
            short = r.split(" or ")[0]
            if "duplicate" in short.lower():
                short = "Duplicate keys"
            elif "outside" in short.lower() and "lanka" in short.lower():
                short = "Coords outside Sri Lanka"
            elif "price_per_liter" in short.lower():
                short = "Price-per-liter outlier"
            elif "Volume_Liters" in short or "volume" in short.lower():
                short = "Volume ≤ 0 or non-numeric"
            elif "Total_Bill_Value" in short:
                short = "Bill ≤ 0 or non-numeric"
            elif "outlet_size" in short.lower():
                short = "Outlet_Size imputed/normalized"
            elif "outlet_type" in short.lower():
                short = "Outlet_Type typo/whitespace fixed"
            elif "cooler" in short.lower():
                short = "Cooler_Count fixed"
            elif "P99.9" in short or "p999" in short.lower():
                short = "Volume > P99.9 per SKU"
            elif "duplicate_key" in short.lower():
                short = "Duplicate keys"
            else:
                short = short[:40]
            reason_counts[short] = reason_counts.get(short, 0) + int(n)

    sr = pd.Series(reason_counts).sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sr.index, sr.values, color=NAVY, edgecolor="white")
    ax.set_xlabel("Records quarantined")
    ax.set_title("Quarantined records by failure reason (all 5 datasets)")
    ax.bar_label(bars, padding=3, fontsize=8, color=GREY,
                 labels=[f"{int(v):,}" for v in sr.values])
    fig.tight_layout()
    fig.savefig(FIG / "rejected_records_bar.png", dpi=150)
    plt.close(fig)
    print(f"  rejected_records_bar.png — {len(sr)} categories, {int(sr.sum()):,} total quarantined")


# ============================================================
# 2. Sri Lanka outlet map
# ============================================================
def chart_sri_lanka_map():
    """Scatter all outlets by lat/lon, colored by Outlet_Type. Approximates a map."""
    coords = pd.read_parquet(SILVER / "outlet_coordinates_clean.parquet")
    master = pd.read_parquet(SILVER / "outlet_master_clean.parquet")
    joined = coords.merge(master, on="Outlet_ID")
    joined = joined.dropna(subset=["Latitude", "Longitude"])
    # Restrict to Sri Lanka box for the chart
    joined = joined[(joined["Latitude"].between(5.9, 9.9)) &
                    (joined["Longitude"].between(79.5, 82.0))]

    types = sorted(joined["Outlet_Type"].unique())
    palette = [NAVY, GREY, "#3D5A80", "#98C1D9", "#E0FBFC", "#293241", "#5C7A99", "#8B9CB0"]

    fig, ax = plt.subplots(figsize=(7, 8))
    for i, t in enumerate(types):
        sub = joined[joined["Outlet_Type"] == t]
        ax.scatter(sub["Longitude"], sub["Latitude"], s=3, alpha=0.5,
                   color=palette[i % len(palette)], label=t)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"All {len(joined):,} outlets, colored by type")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=7, frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG / "sri_lanka_outlets.png", dpi=150)
    plt.close(fig)
    print(f"  sri_lanka_outlets.png — {len(joined):,} outlets plotted")


# ============================================================
# 3. Methodology flow diagram
# ============================================================
def chart_methodology_flow():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.2, 7)
    ax.axis("off")
    ax.set_title("Four-Stage Framework: Latent Outlet Potential",
                 fontsize=12, color=NAVY, weight="bold", y=1.02)

    boxes = [
        # (x, y, w, h, text, fill)
        (0.4, 5.0, 2.3, 1.0, "Silver\nmonthly_outlet\n447,562 rows", LIGHT),
        (0.4, 1.6, 2.3, 1.0, "Gold features\n20,000 × 105", LIGHT),

        (3.4, 5.4, 2.6, 1.0, "Stage A — LightGBM\n(uncensored months)", ACCENT),
        (3.4, 4.0, 2.6, 1.0, "Stage B — Constraint\ncensoring + plateau", ACCENT),
        (3.4, 2.6, 2.6, 1.0, "Stage C — Peer\n85th percentile", ACCENT),
        (3.4, 1.2, 2.6, 1.0, "Physical ceiling\nCoolers × cap × cycles", ACCENT),

        (6.9, 3.6, 3.0, 1.6, "raw = max(\n  peak × 1.05,\n  Stage A,\n  Peer 85th )", NAVY),

        (10.6, 3.6, 3.0, 1.6, "× seasonality (Jan)\n× constraint uplift\nclip to [floor,\n min(peer99×1.5,\n physical_max)]", NAVY),
    ]

    for x, y, w, h, text, fill in boxes:
        text_color = "white" if fill == NAVY else NAVY
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor=NAVY, facecolor=fill)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))

    # silver -> Stage A ; gold -> B, C, physical
    arrow(2.7, 5.5, 3.4, 5.9)
    arrow(2.7, 2.1, 3.4, 4.5)
    arrow(2.7, 2.1, 3.4, 3.1)
    arrow(2.7, 2.1, 3.4, 1.7)
    # A, B, C -> raw max ; physical -> final clip
    arrow(6.0, 5.9, 6.9, 4.6)
    arrow(6.0, 4.5, 6.9, 4.4)
    arrow(6.0, 3.1, 6.9, 4.2)
    arrow(6.0, 1.7, 10.6, 3.9)            # physical ceiling feeds the final clip
    arrow(9.9, 4.4, 10.6, 4.4)

    ax.text(7.0, 0.3,
            "Stage A predicts demand under censoring; the max() never undercuts history; "
            "the constraint uplift rewards capped outlets; the physical cooler ceiling caps the result.",
            ha="center", va="center", color=GREY, fontsize=8.5, style="italic")

    fig.tight_layout()
    fig.savefig(FIG / "methodology_flow.png", dpi=150)
    plt.close(fig)
    print("  methodology_flow.png")


# ============================================================
# 4. Predictions histogram
# ============================================================
def chart_predictions_histogram():
    sub = pd.read_csv(ROOT / "reports" / "cypher_sentinels_predictions.csv")
    vals = sub["Maximum_Monthly_Liters"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(vals, bins=80, color=NAVY, edgecolor="white")
    ax.axvline(np.median(vals), color="white", linestyle="--", linewidth=1.5,
               label=f"Median {np.median(vals):,.0f} L")
    ax.set_xlabel("Predicted Maximum_Monthly_Liters (Jan 2026)")
    ax.set_ylabel("Number of outlets")
    ax.set_title("Distribution of predicted potential across 20,000 outlets")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "predictions_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  predictions_histogram.png — median {np.median(vals):,.0f} L, max {vals.max():,.0f} L")


# ============================================================
# 5. Predictions geographic heatmap
# ============================================================
def chart_predictions_geo():
    sub = pd.read_csv(ROOT / "reports" / "cypher_sentinels_predictions.csv")
    coords = pd.read_parquet(SILVER / "outlet_coordinates_clean.parquet")
    df = sub.merge(coords, on="Outlet_ID")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[(df["Latitude"].between(5.9, 9.9)) & (df["Longitude"].between(79.5, 82.0))]

    # Log-scale color for visibility
    df["log_pot"] = np.log10(df["Maximum_Monthly_Liters"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["log_pot"],
                    s=4, alpha=0.6, cmap="viridis")
    cb = plt.colorbar(sc, ax=ax, label="log₁₀(Maximum_Monthly_Liters)", shrink=0.7)
    cb.outline.set_visible(False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Predicted potential by location ({len(df):,} outlets)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FIG / "predictions_geo_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  predictions_geo_heatmap.png — {len(df):,} outlets")


# ============================================================
# 6. Stage A feature importance
# ============================================================
def chart_feature_importance():
    fi = pd.read_csv(GOLD / "_stage_a_feature_importance.csv").head(15)
    fi = fi.sort_values("gain", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi["feature"], fi["gain"], color=NAVY, edgecolor="white")
    ax.set_xlabel("Gain")
    ax.set_title("Stage A LightGBM — top 15 features by gain")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  feature_importance.png — top feature: {fi['feature'].iloc[-1]}")


# ============================================================
# 7. Segment mean potential (finals)
# ============================================================
def chart_segment_means():
    """Mean predicted potential by outlet size — the 'who has the upside' story."""
    diag = pd.read_parquet(GOLD / "_predictions_diagnostic.parquet")
    sub = pd.read_csv(ROOT / "reports" / "cypher_sentinels_predictions.csv")
    d = diag.merge(sub, on="Outlet_ID")
    order = ["Small", "Medium", "Large", "Extra Large"]
    means = d.groupby("Outlet_Size")["Maximum_Monthly_Liters"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(means.index, means.values, color=[LIGHT, ACCENT, NAVY, "#14505f"],
                  edgecolor="white")
    ax.bar_label(bars, labels=[f"{v:,.0f} L" for v in means.values],
                 padding=3, color=GREY, fontsize=9)
    ax.set_ylabel("Mean predicted potential (L/month)")
    ax.set_title("Latent potential rises sharply with outlet size")
    fig.tight_layout()
    fig.savefig(FIG / "segment_means.png", dpi=150)
    plt.close(fig)
    print(f"  segment_means.png — XL {means['Extra Large']:,.0f} vs Small {means['Small']:,.0f} L")


# ============================================================
# 8. Western budget allocation: spend vs incremental volume
# ============================================================
def chart_budget_allocation():
    """By-distributor spend vs projected incremental volume + spend-type split."""
    bd = pd.read_parquet(GOLD / "budget_allocation_detail.parquet")
    funded = bd[bd["Trade_Spend_Allocation_LKR"] > 1.0]
    by_dist = funded.groupby("primary_distributor").agg(
        spend=("Trade_Spend_Allocation_LKR", "sum"),
        incr=("projected_incremental_L", "sum")).sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(by_dist))
    w = 0.38
    ax1.bar(x - w/2, by_dist["spend"] / 1e6, w, label="Spend (M LKR)", color=NAVY)
    ax1b = ax1.twinx()
    ax1b.bar(x + w/2, by_dist["incr"] / 1e3, w, label="Incremental (k L)", color=ACCENT)
    ax1.set_xticks(x)
    ax1.set_xticklabels(by_dist.index, fontsize=8)
    ax1.set_ylabel("Spend (M LKR)", color=NAVY)
    ax1b.set_ylabel("Incremental volume (k L/mo)", color=ACCENT)
    ax1b.spines["top"].set_visible(False)
    ax1.set_title("5M LKR allocation by distributor")

    by_type = funded.groupby("spend_type")["Trade_Spend_Allocation_LKR"].sum() / 1e6
    by_type = by_type.sort_values(ascending=True)
    bars = ax2.barh(by_type.index, by_type.values, color=[LIGHT, ACCENT, NAVY][:len(by_type)],
                    edgecolor="white")
    ax2.bar_label(bars, labels=[f"{v:.2f}M" for v in by_type.values], padding=3,
                  color=GREY, fontsize=9)
    ax2.set_xlabel("Spend (M LKR)")
    ax2.set_title("Spend by recommended type")
    fig.tight_layout()
    fig.savefig(FIG / "budget_allocation.png", dpi=150)
    plt.close(fig)
    print(f"  budget_allocation.png — {len(funded)} outlets funded")


# ============================================================
# 9. SHAP driver example (one worked outlet)
# ============================================================
def chart_shap_example():
    """Signed driver chart for one illustrative outlet, from the cached explanations."""
    import json
    ex_path = GOLD / "outlet_explanations.json"
    if not ex_path.exists():
        print("  (skip shap_example — no explanations cache)")
        return
    ex = json.loads(ex_path.read_text(encoding="utf-8"))
    diag = pd.read_parquet(GOLD / "_predictions_diagnostic.parquet")
    # pick a high-potential, supply-constrained outlet for an interesting story
    cand = diag[diag["censoring_score"] > 0.12].sort_values("potential_final", ascending=False)
    oid = next((o for o in cand["Outlet_ID"] if o in ex), None) or next(iter(ex))
    rec = ex[oid]
    ev = rec["evidence"]
    drivers = ([(d["feature"], d["shap"]) for d in ev.get("top_drivers_up", [])] +
               [(d["feature"], d["shap"]) for d in ev.get("top_drivers_down", [])])
    drivers = sorted(drivers, key=lambda t: t[1])
    labels = [d[0] for d in drivers]
    vals = [d[1] for d in drivers]
    colors = [PINK if v < 0 else NAVY for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(labels, vals, color=colors, edgecolor="white")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("SHAP contribution (log-volume space)")
    ax.set_title(f"Why outlet {oid} scores {ev['predicted_potential_liters']:.0f} L "
                 f"(peak {ev['historical_peak_liters']:.0f} L)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "shap_example.png", dpi=150)
    plt.close(fig)
    print(f"  shap_example.png — outlet {oid}")
    return oid


def main():
    print("Generating report charts...")
    chart_rejected_records()
    chart_sri_lanka_map()
    chart_methodology_flow()
    chart_predictions_histogram()
    chart_predictions_geo()
    chart_feature_importance()
    chart_segment_means()
    chart_budget_allocation()
    chart_shap_example()
    print("Done. Figures in:", FIG)


if __name__ == "__main__":
    main()
