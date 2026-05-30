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

NAVY = "#1B2A4E"
GREY = "#6C757D"
LIGHT = "#D6DBDF"
ACCENT = "#A4B7CB"

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
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Three-Stage Methodology: Latent Outlet Potential",
                 fontsize=12, color=NAVY, weight="bold")

    boxes = [
        # (x, y, w, h, text, fill)
        (0.5, 4, 2.2, 1.0, "Silver\nmonthly_outlet\n447,562 rows", LIGHT),
        (0.5, 1, 2.2, 1.0, "Gold features\n20,000 × 77", LIGHT),

        (3.5, 4, 2.5, 1.0, "Stage A\nLightGBM\n(uncensored months only)", ACCENT),
        (3.5, 2.5, 2.5, 1.0, "Stage B\nConstraint detection\ncensoring + plateau", ACCENT),
        (3.5, 1, 2.5, 1.0, "Stage C\nPeer 85th percentile", ACCENT),

        (7, 3, 3, 1.5, "max(\n  peak × 1.05,\n  stage_A,\n  peer_85th\n)", NAVY),

        (10.5, 3, 3, 1.5, "× seasonality_jan26\n× constraint_uplift\nclip [floor, ceiling]", NAVY),
    ]

    for x, y, w, h, text, fill in boxes:
        text_color = "white" if fill == NAVY else NAVY
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor=NAVY, facecolor=fill)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=8)

    # Arrows
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))

    arrow(2.7, 4.5, 3.5, 4.5)
    arrow(2.7, 1.5, 3.5, 3)
    arrow(2.7, 1.5, 3.5, 1.5)
    arrow(6, 4.5, 7, 3.75)
    arrow(6, 1.5, 7, 3.75)
    arrow(6, 3, 7, 3.75)
    arrow(10, 3.75, 10.5, 3.75)

    # Bottom caption
    ax.text(7, 0.2,
            "Stage A predicts under censoring; the max() never undercuts history; "
            "constraint uplift rewards capped outlets.",
            ha="center", va="center", color=GREY, fontsize=9, style="italic")

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


def main():
    print("Generating report charts...")
    chart_rejected_records()
    chart_sri_lanka_map()
    chart_methodology_flow()
    chart_predictions_histogram()
    chart_predictions_geo()
    chart_feature_importance()
    print("Done. Figures in:", FIG)


if __name__ == "__main__":
    main()
