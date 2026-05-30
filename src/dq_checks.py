# Reusable DQ checks. Each returns (passed, rejected_with_failure_reason).
# Never silent-drop — callers persist rejected rows to data/rejected_records/.

import re
import numpy as np
import pandas as pd


def _attach_reason(df, reason):
    out = df.copy()
    out["failure_reason"] = reason
    return out


def check_duplicates(df, key_cols):
    dup = df.duplicated(subset=key_cols, keep="first")
    rejected = _attach_reason(df.loc[dup], f"duplicate on {key_cols}")
    return df.loc[~dup].copy(), rejected


def check_nulls(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"check_nulls: columns not in df: {missing}")

    null_mask = df[required_cols].isna().any(axis=1)
    if not null_mask.any():
        return df.copy(), df.iloc[0:0].assign(failure_reason=pd.Series(dtype=str))

    rejected = df.loc[null_mask].copy()
    rejected["failure_reason"] = "null in: " + df.loc[null_mask, required_cols].isna().apply(
        lambda r: ",".join(r.index[r].tolist()), axis=1
    )
    return df.loc[~null_mask].copy(), rejected


def check_referential_integrity(df, col, ref_values, ref_label="reference"):
    ref_set = set(pd.Series(list(ref_values)).dropna().tolist())
    valid = df[col].isin(ref_set)
    rejected = _attach_reason(df.loc[~valid], f"{col} not in {ref_label}")
    return df.loc[valid].copy(), rejected


def check_value_range(df, col, min_val=None, max_val=None):
    s = pd.to_numeric(df[col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if min_val is not None:
        mask &= s >= min_val
    if max_val is not None:
        mask &= s <= max_val
    mask &= s.notna()
    rejected = _attach_reason(df.loc[~mask], f"{col} outside [{min_val}, {max_val}] or non-numeric")
    return df.loc[mask].copy(), rejected


def check_format(df, col, allowed_values=None, regex=None, dtype=None):
    # pass exactly one of allowed_values / regex / dtype
    provided = [x for x in (allowed_values, regex, dtype) if x is not None]
    if len(provided) != 1:
        raise ValueError("check_format: pass exactly one of allowed_values / regex / dtype")

    if allowed_values is not None:
        allowed_set = set(allowed_values)
        mask = df[col].isin(allowed_set)
        reason = f"{col} not in {sorted(allowed_set) if len(allowed_set) <= 20 else 'allowed set'}"
    elif regex is not None:
        pattern = re.compile(regex)
        mask = df[col].astype(str).apply(lambda v: bool(pattern.fullmatch(v)))
        reason = f"{col} fails regex {regex!r}"
    else:
        if dtype in ("int", "int64", "integer"):
            coerced = pd.to_numeric(df[col], errors="coerce")
            mask = coerced.notna() & (coerced == coerced.astype("Int64").astype("float"))
        elif dtype in ("float", "float64", "numeric"):
            mask = pd.to_numeric(df[col], errors="coerce").notna()
        elif dtype in ("datetime", "date"):
            mask = pd.to_datetime(df[col], errors="coerce").notna()
        else:
            raise ValueError(f"check_format: unsupported dtype {dtype!r}")
        reason = f"{col} not coercible to {dtype}"

    rejected = _attach_reason(df.loc[~mask], reason)
    return df.loc[mask].copy(), rejected


def check_geo_bounds(df, lat_col="Latitude", lon_col="Longitude",
                     lat_min=5.9, lat_max=9.9, lon_min=79.5, lon_max=82.0):
    # Sri Lanka bounding box by default
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    in_box = lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
    rejected = _attach_reason(
        df.loc[~in_box],
        f"coords outside box lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] or null",
    )
    return df.loc[in_box].copy(), rejected


def check_price_consistency(df, vol_col="Volume_Liters", bill_col="Total_Bill_Value",
                            group_col="SKU_ID", lower_pct=0.01, upper_pct=0.99):
    # reject rows whose price-per-liter is outside [P1, P99] within their SKU group
    vol = pd.to_numeric(df[vol_col], errors="coerce")
    bill = pd.to_numeric(df[bill_col], errors="coerce")
    price = bill / vol.where(vol > 0)

    bounds = (
        price.groupby(df[group_col])
        .agg(p_lo=lambda s: s.quantile(lower_pct), p_hi=lambda s: s.quantile(upper_pct), n="count")
        .reset_index()
    )
    merged = df.assign(_price=price).merge(bounds, on=group_col, how="left")

    is_singleton = merged["n"].fillna(0) <= 1  # singleton groups have no distribution to check
    in_range = (merged["_price"] >= merged["p_lo"]) & (merged["_price"] <= merged["p_hi"])
    mask = (is_singleton | in_range) & merged["_price"].notna()

    rejected = _attach_reason(
        df.loc[~mask.values],
        f"price_per_liter outside [P{int(lower_pct*100)},P{int(upper_pct*100)}] within {group_col}",
    )
    return df.loc[mask.values].copy(), rejected
