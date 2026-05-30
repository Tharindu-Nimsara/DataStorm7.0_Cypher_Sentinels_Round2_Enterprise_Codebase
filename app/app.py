"""Outlet Intelligence — Streamlit app (Phase 6).

A locally-runnable decision tool for the trade-marketing team. It reads ONLY precomputed
artifacts (predictions, gold features, the cached SHAP+LLM explanations, the budget
allocation) — it never retrains a model or calls an API at request time, so it loads fast and
works offline. The LLM explanations were generated and cached in Phase 4; here we just display
them.

Run:  streamlit run app/app.py

Required features (final-round spec 6):
  1. Browse outlet-level predictions across all 20k outlets (searchable, sortable table).
  2. Filter by province and/or distributor.
  3. Drill into one outlet → potential, historical peak, SHAP driver chart, a map with the
     outlet, and the cached LLM explanation.
  4. Western-province budget-allocation view (recommended spend per outlet).
  5. A map of predicted potential across outlets (the hero visual).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
SILVER = ROOT / "data" / "silver"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Outlet Intelligence — Cypher Sentinels",
                   page_icon="🧊", layout="wide")

# ── Brand palette (dark navy + grey, matching the report) ──
NAVY = "#0B1F3A"
ACCENT = "#3DA5D9"
GREY = "#6B7280"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached — read precomputed artifacts once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading precomputed predictions…")
def load_data() -> pd.DataFrame:
    """One row per outlet: prediction + reasoning columns + coordinates, joined from the
    artifacts each pipeline stage already wrote. Pure read — nothing is recomputed."""
    preds = pd.read_csv(REPORTS / "cypher_sentinels_predictions.csv")
    diag = pd.read_parquet(GOLD / "_predictions_diagnostic.parquet")
    # The diagnostic is the authoritative source for prediction/constraint columns
    # (potential_final, censoring_score, physical_max, Province, …). From gold we only pull
    # the extra context columns NOT already in diag, to avoid _x/_y merge collisions.
    want = ["primary_distributor", "Cooler_Count", "decayed_density_weighted",
            "n_competitors_500m", "market_share_proxy"]
    gold = pd.read_parquet(GOLD / "outlet_features.parquet")
    gold_cols = ["Outlet_ID"] + [c for c in want
                                 if c in gold.columns and c not in diag.columns]
    gold = gold[gold_cols]
    coords = pd.read_parquet(SILVER / "outlet_coordinates_clean.parquet")

    df = (preds.merge(diag, on="Outlet_ID", how="left")
                .merge(gold, on="Outlet_ID", how="left")
                .merge(coords, on="Outlet_ID", how="left"))
    df["constraint_type"] = (df["censoring_score"] > 0.12).map(
        {True: "supply-constrained", False: "demand-led"})
    df["uplift_pct"] = (df["Maximum_Monthly_Liters"] / df["vol_max"].clip(lower=1) - 1) * 100
    return df


@st.cache_data
def load_explanations() -> dict:
    path = GOLD / "outlet_explanations.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_budget() -> pd.DataFrame:
    path = GOLD / "budget_allocation_detail.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # fall back to the bare submission CSV if the detail parquet isn't present
    csv = REPORTS / "cypher_sentinels_budget_allocations.csv"
    return pd.read_csv(csv) if csv.exists() else pd.DataFrame()


# valid Sri Lanka box — drop the 240 garbage-coord outlets from map layers only
def _has_valid_coords(df: pd.DataFrame) -> pd.Series:
    return df["Latitude"].between(5.9, 9.9) & df["Longitude"].between(79.5, 82.0)


# ──────────────────────────────────────────────────────────────────────────────
# Map helpers (pydeck)
# ──────────────────────────────────────────────────────────────────────────────
def potential_color(v: float, vmax: float) -> list[int]:
    """Blue (low) → amber (high) ramp on predicted potential."""
    t = min(max(v / vmax, 0), 1) if vmax else 0
    return [int(40 + 215 * t), int(120 + 40 * t), int(220 - 160 * t), 160]


def hero_map(df: pd.DataFrame):
    import pydeck as pdk
    m = df[_has_valid_coords(df)].copy()
    vmax = m["Maximum_Monthly_Liters"].quantile(0.97)
    m["color"] = m["Maximum_Monthly_Liters"].apply(lambda v: potential_color(v, vmax))
    m["radius"] = 60 + (m["Maximum_Monthly_Liters"].clip(upper=vmax) / vmax * 240)
    layer = pdk.Layer(
        "ScatterplotLayer", data=m[["Latitude", "Longitude", "color", "radius",
                                    "Outlet_ID", "Maximum_Monthly_Liters"]],
        get_position=["Longitude", "Latitude"], get_fill_color="color",
        get_radius="radius", pickable=True, opacity=0.6)
    view = pdk.ViewState(latitude=float(m["Latitude"].median()),
                         longitude=float(m["Longitude"].median()), zoom=8)
    return pdk.Deck(layers=[layer], initial_view_state=view,
                    map_style="road",
                    tooltip={"text": "{Outlet_ID}\n{Maximum_Monthly_Liters} L"})


def outlet_map(row: pd.Series):
    import pydeck as pdk
    if not (5.9 <= row.get("Latitude", 0) <= 9.9):
        return None
    pt = pd.DataFrame([{"Latitude": row["Latitude"], "Longitude": row["Longitude"]}])
    layer = pdk.Layer("ScatterplotLayer", data=pt,
                      get_position=["Longitude", "Latitude"],
                      get_fill_color=[61, 165, 217, 220], get_radius=180, pickable=False)
    view = pdk.ViewState(latitude=float(row["Latitude"]),
                         longitude=float(row["Longitude"]), zoom=13)
    return pdk.Deck(layers=[layer], initial_view_state=view, map_style="road")


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    explanations = load_explanations()
    budget = load_budget()

    st.title("🧊 Outlet Intelligence")
    st.caption("Latent monthly-volume potential for ~20,000 Sri Lankan outlets — "
               "Jan 2026. Team Cypher Sentinels · Data Storm v7.0.")

    # ── KPI strip ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outlets", f"{len(df):,}")
    c2.metric("Mean potential", f"{df['Maximum_Monthly_Liters'].mean():.0f} L")
    c3.metric("Total latent volume", f"{df['Maximum_Monthly_Liters'].sum()/1e6:.2f} M L")
    c4.metric("Supply-constrained", f"{(df['constraint_type']=='supply-constrained').mean()*100:.1f}%")

    # ── Sidebar filters ──
    st.sidebar.header("Filters")
    provinces = sorted(df["Province"].dropna().unique().tolist())
    sel_prov = st.sidebar.multiselect("Province", provinces, default=provinces)
    dists = sorted(df.loc[df["Province"].isin(sel_prov), "primary_distributor"]
                   .dropna().unique().tolist())
    sel_dist = st.sidebar.multiselect("Distributor", dists, default=dists)
    search = st.sidebar.text_input("Search Outlet_ID")

    fdf = df[df["Province"].isin(sel_prov) & df["primary_distributor"].isin(sel_dist)]
    if search:
        fdf = fdf[fdf["Outlet_ID"].str.contains(search, case=False, na=False)]
    st.sidebar.caption(f"{len(fdf):,} outlets match")

    tab_map, tab_browse, tab_outlet, tab_budget = st.tabs(
        ["🗺️ Potential map", "📋 Browse", "🔎 Outlet detail", "💰 Western budget"])

    # ── 5) Hero map ──
    with tab_map:
        st.subheader("Predicted potential across outlets")
        st.caption("Dot size & colour ∝ predicted January-2026 potential (liters). "
                   "Outliers capped at the 97th percentile for the colour ramp.")
        try:
            st.pydeck_chart(hero_map(fdf), use_container_width=True)
        except Exception as e:
            st.warning(f"Map unavailable ({e}); showing a sample table instead.")
            st.dataframe(fdf.head(500)[["Outlet_ID", "Province", "Maximum_Monthly_Liters"]])

    # ── 1+2) Browse + filters ──
    with tab_browse:
        st.subheader("Outlet predictions")
        cols = ["Outlet_ID", "Province", "primary_distributor", "Outlet_Type",
                "Outlet_Size", "vol_max", "Maximum_Monthly_Liters", "uplift_pct",
                "constraint_type"]
        show = (fdf[cols].rename(columns={
            "primary_distributor": "Distributor", "vol_max": "Hist. peak (L)",
            "Maximum_Monthly_Liters": "Potential (L)", "uplift_pct": "Uplift %",
            "constraint_type": "Constraint"}))
        st.dataframe(show.sort_values("Potential (L)", ascending=False),
                     use_container_width=True, height=560,
                     column_config={"Uplift %": st.column_config.NumberColumn(format="%.0f%%")})
        st.download_button("Download this view (CSV)", show.to_csv(index=False),
                           "outlet_view.csv", "text/csv")

    # ── 3) Outlet drill-down ──
    with tab_outlet:
        ids = fdf["Outlet_ID"].tolist()
        if not ids:
            st.info("No outlets match the current filters.")
        else:
            oid = st.selectbox("Choose an outlet", ids,
                               index=int(fdf["Maximum_Monthly_Liters"].argmax()))
            row = fdf[fdf["Outlet_ID"] == oid].iloc[0]
            left, right = st.columns([1, 1])

            with left:
                st.markdown(f"### {oid}")
                st.write(f"**{row['Outlet_Size']} {row['Outlet_Type']}** · "
                         f"{row['Province']} · {row['primary_distributor']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Potential", f"{row['Maximum_Monthly_Liters']:.0f} L")
                m2.metric("Historical peak", f"{row['vol_max']:.0f} L")
                m3.metric("Uplift", f"{row['uplift_pct']:.0f}%")
                st.write(f"**Constraint:** {row['constraint_type']}  ·  "
                         f"**Cooler ceiling:** {row.get('physical_max', float('nan')):.0f} L  ·  "
                         f"**Competitors ≤500m:** {int(row.get('n_competitors_500m', 0))}")

                rec = explanations.get(oid)
                if rec:
                    st.markdown("#### Why this score")
                    src = rec["source"]
                    badge = "🤖 LLM (gpt-4o-mini)" if src.startswith("github_models") else "📝 grounded template"
                    st.caption(f"Explanation source: {badge}")
                    st.info(rec["explanation"])

            with right:
                rec = explanations.get(oid)
                if rec:
                    st.markdown("#### Top drivers (SHAP)")
                    ev = rec["evidence"]
                    drivers = ([{"feature": d["feature"], "impact": d["shap"]}
                                for d in ev.get("top_drivers_up", [])] +
                               [{"feature": d["feature"], "impact": d["shap"]}
                                for d in ev.get("top_drivers_down", [])])
                    if drivers:
                        chart_df = pd.DataFrame(drivers).set_index("feature")
                        st.bar_chart(chart_df, horizontal=True, color=ACCENT)
                        st.caption("Positive = pushes potential up; negative = pulls it down.")
                m = outlet_map(row)
                if m is not None:
                    st.markdown("#### Location")
                    st.pydeck_chart(m, use_container_width=True)

    # ── 4) Western budget view ──
    with tab_budget:
        st.subheader("Western-province trade-spend allocation (LKR 5,000,000)")
        if budget.empty:
            st.info("Run `python src/optimize_budget.py` to generate the allocation.")
        else:
            funded = budget[budget["Trade_Spend_Allocation_LKR"] > 1.0]
            b1, b2, b3 = st.columns(3)
            b1.metric("Budget allocated",
                      f"{budget['Trade_Spend_Allocation_LKR'].sum()/1e6:.2f} M LKR")
            b2.metric("Outlets funded", f"{len(funded):,}")
            if "projected_incremental_L" in budget.columns:
                b3.metric("Projected incremental",
                          f"{budget['projected_incremental_L'].sum():,.0f} L/mo")
            if "spend_type" in funded.columns:
                st.markdown("**Spend by type**")
                by_type = funded.groupby("spend_type")["Trade_Spend_Allocation_LKR"].sum()
                st.bar_chart(by_type, color=ACCENT)
            disp_cols = [c for c in ["Outlet_ID", "primary_distributor", "constraint_type",
                                     "spend_type", "gap", "Trade_Spend_Allocation_LKR",
                                     "projected_incremental_L"] if c in funded.columns]
            st.dataframe(funded[disp_cols].sort_values("Trade_Spend_Allocation_LKR",
                         ascending=False), use_container_width=True, height=420)


if __name__ == "__main__":
    main()
