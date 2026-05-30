# Stage A: LightGBM for expected demand.
# Trained only on outlet-months where vol < outlet_P99 * 0.95 (drops the
# censored top of each outlet's distribution). Target is log1p(volume).
# 5-fold GroupKFold by Outlet_ID, refit on all data with mean best-iter.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
MODELS = ROOT / "models"
LOG_DIR = ROOT / "logs"
GOLD.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CENSOR_THRESHOLD = 0.95          # exclude months at >=95% of outlet's P99
RANDOM_STATE = 42
N_FOLDS = 5

# Columns we never feed to the model: identifiers, labels, the target source.
ID_COLS = {"Outlet_ID", "primary_distributor"}
LABEL_COLS = {"Outlet_Size", "Outlet_Type", "Province",
              "seasonality_jan_label", "peer_cluster", "poi_density_tier"}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("model")
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
    fh = logging.FileHandler(LOG_DIR / "model.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def build_training_set(
    monthly: pd.DataFrame, gold: pd.DataFrame, log: logging.Logger
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    log.info("Joining monthly_outlet (%d rows) to gold features (%d outlets)...",
             len(monthly), len(gold))

    p99_lookup = gold.set_index("Outlet_ID")["vol_p99"]
    monthly = monthly.copy()
    monthly["outlet_p99"] = monthly["Outlet_ID"].map(p99_lookup)
    monthly["censored"] = monthly["Volume_Liters"] >= monthly["outlet_p99"] * CENSOR_THRESHOLD

    n_before = len(monthly)
    train = monthly[~monthly["censored"]].copy()
    log.info("Censoring filter (vol < outlet_P99 × %.2f): %d → %d rows (%.1f%% kept).",
             CENSOR_THRESHOLD, n_before, len(train), 100 * len(train) / max(n_before, 1))

    train = train.merge(gold, on="Outlet_ID", how="left")
    train["Year_norm"] = train["Year"] - 2023
    month_dummies = pd.get_dummies(train["Month"], prefix="month").astype(int)
    train = pd.concat([train, month_dummies], axis=1)

    train["target_log"] = np.log1p(train["Volume_Liters"])
    y = train["target_log"]
    groups = train["Outlet_ID"]

    drop_cols = {
        "Volume_Liters", "Total_Bill_Value", "target_log",
        "outlet_p99", "censored",
        "Distributor_ID",                    # high-cardinality, captured via Province
        "Year", "Month",                     # replaced by Year_norm + month dummies
        "n_unique_skus", "n_transactions",   # monthly-aggregate cols, not per-outlet
    } | ID_COLS | LABEL_COLS

    X = train.drop(columns=[c for c in drop_cols if c in train.columns])
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.warning("Dropping non-numeric columns from X: %s", non_numeric)
        X = X.drop(columns=non_numeric)
    log.info("Feature matrix: %s", X.shape)

    return X, y, groups


def train_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, log: logging.Logger) -> lgb.Booster:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    kf = GroupKFold(n_splits=N_FOLDS)
    fold_metrics = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        dtr = lgb.Dataset(X_tr, y_tr)
        dva = lgb.Dataset(X_va, y_va, reference=dtr)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)],
        )
        pred = model.predict(X_va, num_iteration=model.best_iteration)
        rmse = float(np.sqrt(((pred - y_va) ** 2).mean()))
        fold_metrics.append(rmse)
        best_iters.append(model.best_iteration)
        log.info("  fold %d: RMSE(log)=%.4f best_iter=%d", fold, rmse, model.best_iteration)

    mean_rmse = float(np.mean(fold_metrics))
    log.info("CV mean RMSE(log) = %.4f  |  mean best_iter = %d",
             mean_rmse, int(np.mean(best_iters)))

    final_iters = int(np.mean(best_iters))
    log.info("Refitting on full data for %d iterations...", final_iters)
    dall = lgb.Dataset(X, y)
    final = lgb.train(params, dall, num_boost_round=final_iters)

    (LOG_DIR / "model_cv_metrics.json").write_text(
        json.dumps({"fold_rmse_log": fold_metrics, "mean_rmse_log": mean_rmse,
                    "n_folds": N_FOLDS, "censor_threshold": CENSOR_THRESHOLD,
                    "n_features": X.shape[1], "n_train_rows": int(len(X))}, indent=2)
    )
    return final


def predict_jan_2026(model: lgb.Booster, gold: pd.DataFrame, X_template: pd.DataFrame,
                     log: logging.Logger) -> pd.DataFrame:
    # Gold carries Jan-2026 seasonality + holiday counts; here we just set
    # Year_norm and Jan month-of-year one-hot to match the training matrix.
    log.info("Building Jan-2026 prediction frame for %d outlets...", len(gold))

    pred_frame = gold.copy()
    pred_frame["Year_norm"] = 2026 - 2023
    for m in range(1, 13):
        pred_frame[f"month_{m}"] = 1 if m == 1 else 0

    drop_cols = ID_COLS | LABEL_COLS | {"Distributor_ID"}
    X_pred = pred_frame.drop(columns=[c for c in drop_cols if c in pred_frame.columns])

    missing = [c for c in X_template.columns if c not in X_pred.columns]
    extra = [c for c in X_pred.columns if c not in X_template.columns]
    if missing:
        log.warning("  adding %d missing columns to X_pred (zero-filled): %s",
                    len(missing), missing[:5])
        for c in missing:
            X_pred[c] = 0
    if extra:
        X_pred = X_pred.drop(columns=extra)
    X_pred = X_pred[X_template.columns]

    non_numeric = X_pred.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X_pred = X_pred.drop(columns=non_numeric)
        X_pred = X_pred.reindex(columns=[c for c in X_template.columns if c not in non_numeric])

    log_pred = model.predict(X_pred)
    raw_pred = np.expm1(log_pred)

    out = pd.DataFrame({"Outlet_ID": pred_frame["Outlet_ID"].values,
                        "stage_a_pred": raw_pred})
    log.info("Stage A pred — mean=%.1f median=%.1f min=%.1f max=%.1f",
             out["stage_a_pred"].mean(), out["stage_a_pred"].median(),
             out["stage_a_pred"].min(), out["stage_a_pred"].max())
    return out


def main() -> None:
    log = setup_logging()
    log.info("Stage A model start.")

    monthly = pd.read_parquet(SILVER / "monthly_outlet.parquet")
    gold = pd.read_parquet(GOLD / "outlet_features.parquet")
    log.info("Loaded silver monthly_outlet=%d rows, gold features=%d outlets.",
             len(monthly), len(gold))

    X, y, groups = build_training_set(monthly, gold, log)
    log.info("Training set ready. X.shape=%s y.shape=%s", X.shape, y.shape)

    booster = train_cv(X, y, groups, log)
    booster.save_model(str(MODELS / "stage_a.lgb"))
    log.info("Saved booster to %s", MODELS / "stage_a.lgb")

    fi = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(GOLD / "_stage_a_feature_importance.csv", index=False)
    log.info("Top 15 features by gain:\n%s", fi.head(15).to_string(index=False))

    pred = predict_jan_2026(booster, gold, X, log)
    pred.to_parquet(GOLD / "stage_a_pred.parquet", index=False)
    log.info("Wrote %s", GOLD / "stage_a_pred.parquet")


if __name__ == "__main__":
    main()
