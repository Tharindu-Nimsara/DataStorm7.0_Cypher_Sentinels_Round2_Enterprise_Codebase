# Read raw CSVs and log shape / schema / dtypes. No transforms.

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = ROOT / "data" / "bronze"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = {
    "transactions": "transactions_history_final.csv",
    "outlet_master": "outlet_master.csv",
    "outlet_coordinates": "outlet_coordinates.csv",
    "distributor_seasonality": "distributor_seasonality_details.csv",
    "holiday_list": "holiday_list.csv",
}

EXPECTED_COLUMNS = {
    "transactions": [
        "Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID",
        "Product_Name", "Volume_Liters", "Total_Bill_Value",
    ],
    "outlet_master": ["Outlet_ID", "Outlet_Size", "Cooler_Count", "Outlet_Type"],
    "outlet_coordinates": ["Outlet_ID", "Latitude", "Longitude"],
    "distributor_seasonality": ["Distributor_ID", "Year", "Month", "Seasonality_Index"],
    "holiday_list": ["Date", "Holiday_Name", "Holiday_Type"],
}


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("bronze_ingest")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Force UTF-8 stdout — Windows cp1252 default crashes on non-ASCII
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(LOG_DIR / "bronze_ingest.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def ingest() -> dict[str, pd.DataFrame]:
    log = _setup_logging()
    log.info("Bronze ingest start. BRONZE_DIR=%s", BRONZE_DIR)

    if not BRONZE_DIR.exists():
        log.error("Bronze dir does not exist: %s", BRONZE_DIR)
        sys.exit(2)

    missing = [
        name for name, fn in EXPECTED_FILES.items()
        if not (BRONZE_DIR / fn).exists()
    ]
    if missing:
        for name in missing:
            log.error("MISSING bronze file: %s (%s)", name, EXPECTED_FILES[name])
        log.error(
            "Place the Kaggle CSVs in data/bronze/ before re-running. "
            "Expected filenames: %s",
            list(EXPECTED_FILES.values()),
        )
        sys.exit(3)

    frames: dict[str, pd.DataFrame] = {}
    for name, fn in EXPECTED_FILES.items():
        path = BRONZE_DIR / fn
        log.info("Reading %s → %s", fn, name)
        df = pd.read_csv(path)
        frames[name] = df

        log.info("  shape=%s", df.shape)
        log.info("  columns=%s", df.columns.tolist())
        log.info("  dtypes=\n%s", df.dtypes.to_string())

        expected = EXPECTED_COLUMNS[name]
        unexpected = [c for c in df.columns if c not in expected]
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            log.warning("  MISSING expected columns: %s", missing_cols)
        if unexpected:
            log.warning("  UNEXPECTED columns: %s", unexpected)

        log.info("  head(3)=\n%s", df.head(3).to_string())

    log.info("--- ROW COUNT SUMMARY ---")
    for name, df in frames.items():
        log.info("  %-25s %10d rows", name, len(df))

    log.info("Bronze ingest complete.")
    return frames


if __name__ == "__main__":
    ingest()
