"""Parallel, multi-endpoint POI scraper for ~20k outlets.

Improvements over the single-endpoint version:
    * Multiple Overpass endpoints in round-robin assignment (each worker pinned
      to one endpoint so per-endpoint rate limits are respected).
    * 2km radius queried ONCE per outlet; 500m and 1km counts derived locally
      via haversine bucketing. Cuts subquery count ~3×.
    * Regex-union over OSM tags so one subquery covers many POI types. Smaller
      response, fewer timeouts.
    * Concurrent worker pool (one thread per endpoint).
    * Same JSON cache schema as before — gold_features.py needs no change.

Resumable: skips outlets whose `OUT_*.json` already exists.
"""
from __future__ import annotations

import json
import logging
import math
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
COORDS_CSV = ROOT / "data" / "bronze" / "outlet_coordinates.csv"
CACHE_DIR = ROOT / "data" / "external" / "poi_cache"
SCRAPE_LOG = ROOT / "data" / "external" / "scrape_log.csv"
LOG_DIR = ROOT / "logs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Multiple endpoints — workers round-robin so per-endpoint throttling kicks in
# independently. If one endpoint is slow, others keep going.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]

# We query a single 2km radius and bucket locally for 500m/1km/2km.
QUERY_RADIUS_M = 2000
RADII_BUCKETS = [500, 1000, 2000]

# Tag-based POI classification. Each POI "type" is a list of (key, value) pairs;
# if an OSM element matches ANY pair, it counts for that type.
POI_QUERIES = {
    "school": [("amenity", "school")],
    "university": [("amenity", "university"), ("amenity", "college")],
    "bus_station": [("amenity", "bus_station"), ("public_transport", "station"),
                    ("highway", "bus_stop")],
    "hospital": [("amenity", "hospital"), ("amenity", "clinic")],
    "place_of_worship": [("amenity", "place_of_worship")],
    "marketplace": [("amenity", "marketplace"), ("shop", "supermarket")],
    "restaurant": [("amenity", "restaurant"), ("amenity", "cafe"),
                   ("amenity", "fast_food")],
    "tourism": [("tourism", "hotel"), ("tourism", "attraction"),
                ("tourism", "museum"), ("tourism", "guest_house")],
}

# Build a flat set of (key, value) we care about — used to filter elements locally.
ALL_TAG_PAIRS: list[tuple[str, str]] = []
for pairs in POI_QUERIES.values():
    ALL_TAG_PAIRS.extend(pairs)

# Per-endpoint pacing — minimum interval between requests on the SAME endpoint.
# (Per-endpoint, so 4 endpoints = effective 4 req/sec total without hammering one.)
PER_ENDPOINT_RATE_LIMIT_SEC = 1.2

MAX_RETRIES = 5
BACKOFF_BASE_SEC = 4.0    # 4, 8, 16, 32, 64
REQUEST_TIMEOUT_SEC = 90
USER_AGENT = "DataStorm7-LatentPotential/1.0 (academic; +contact: tharinduonline1080@gmail.com)"

# Concurrency: one worker per endpoint = no contention on per-endpoint rate.
NUM_WORKERS = len(OVERPASS_ENDPOINTS)

_STOP = False
_LOG_LOCK = threading.Lock()
_SCRAPE_LOG_LOCK = threading.Lock()


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("poi_scrape")
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
    fh = logging.FileHandler(LOG_DIR / "poi_scrape.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def _build_compact_query(lat: float, lon: float) -> str:
    """Single Overpass query covering all POI types at the largest radius.

    Groups tags by key to minimize subqueries. Returns center coordinates for
    ways/relations so local distance bucketing works.
    """
    # Group tag-pair values per key. Some keys (amenity, tourism, public_transport,
    # highway, shop) appear in many POI types.
    by_key: dict[str, list[str]] = {}
    for k, v in ALL_TAG_PAIRS:
        by_key.setdefault(k, []).append(v)

    parts: list[str] = []
    for k, vals in by_key.items():
        vals_unique = sorted(set(vals))
        vre = "|".join(vals_unique)
        # Nodes only — 90%+ of amenity tagging is on nodes; ways/relations are rare
        # for these POI types and add 2× query size for marginal recall.
        parts.append(f'node["{k}"~"^({vre})$"](around:{QUERY_RADIUS_M},{lat},{lon});')

    body = "".join(parts)
    return f"[out:json][timeout:40];({body});out tags;"


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _count_by_type_radius(elements: list[dict], lat: float, lon: float) -> dict[str, dict[str, int]]:
    """Bucket elements by (poi_type, radius). Local haversine — no extra requests."""
    out: dict[str, dict[int, set]] = {
        poi: {r: set() for r in RADII_BUCKETS} for poi in POI_QUERIES
    }
    for el in elements:
        tags = el.get("tags", {}) or {}
        if "lat" in el and "lon" in el:
            elat, elon = el["lat"], el["lon"]
        elif "center" in el:
            elat, elon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        dist = _haversine_m(lat, lon, elat, elon)
        osm_id = (el.get("type"), el.get("id"))

        for poi_name, tag_pairs in POI_QUERIES.items():
            if not any(tags.get(k) == v for k, v in tag_pairs):
                continue
            for r in RADII_BUCKETS:
                if dist <= r:
                    out[poi_name][r].add(osm_id)

    # Cache JSON uses str keys for radii
    return {poi: {str(r): len(ids) for r, ids in by_r.items()} for poi, by_r in out.items()}


def _query_endpoint(endpoint: str, query: str, log: logging.Logger) -> dict | None:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}"
            if resp.status_code in (429, 502, 503, 504):
                wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1)) + random.uniform(0, 1)
                with _LOG_LOCK:
                    log.warning("  [%s] attempt %d %s; sleep %.1fs", endpoint.split("//")[1].split("/")[0], attempt, last_err, wait)
                time.sleep(wait)
                continue
            with _LOG_LOCK:
                log.warning("  [%s] attempt %d %s; retrying", endpoint, attempt, last_err)
        except requests.RequestException as e:
            last_err = type(e).__name__
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            with _LOG_LOCK:
                log.warning("  [%s] attempt %d %s; sleep %.1fs", endpoint, attempt, last_err, wait)
            time.sleep(wait)
    with _LOG_LOCK:
        log.error("  [%s] gave up after %d attempts: %s", endpoint, MAX_RETRIES, last_err)
    return None


def _append_scrape_log(outlet_id: str, status: str, n_elements: int | None) -> None:
    with _SCRAPE_LOG_LOCK:
        new = not SCRAPE_LOG.exists()
        with open(SCRAPE_LOG, "a", encoding="utf-8") as f:
            if new:
                f.write("timestamp_utc,Outlet_ID,status,n_elements\n")
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"{ts},{outlet_id},{status},{n_elements if n_elements is not None else ''}\n")


def _install_signal_handlers(log: logging.Logger) -> None:
    def _handle(signum, frame):
        global _STOP
        with _LOG_LOCK:
            log.warning("Signal %s received — finishing in-flight queries then stopping.", signum)
        _STOP = True

    signal.signal(signal.SIGINT, _handle)
    try:
        signal.signal(signal.SIGTERM, _handle)
    except (AttributeError, ValueError):
        pass


def _worker_loop(endpoint: str, queue: list, lock: threading.Lock,
                 progress: dict, total: int, log: logging.Logger) -> None:
    """Each worker pulls outlets off the shared queue and queries its pinned endpoint."""
    last_request_ts = 0.0
    while True:
        if _STOP:
            return
        with lock:
            if not queue:
                return
            row = queue.pop()
        outlet_id = row["Outlet_ID"]
        lat, lon = row["Latitude"], row["Longitude"]

        if pd.isna(lat) or pd.isna(lon):
            _append_scrape_log(outlet_id, "skipped_null_coords", None)
            with lock:
                progress["skip"] += 1
            continue

        # Per-endpoint pacing
        delta = time.time() - last_request_ts
        if delta < PER_ENDPOINT_RATE_LIMIT_SEC:
            time.sleep(PER_ENDPOINT_RATE_LIMIT_SEC - delta)

        query = _build_compact_query(lat, lon)
        t0 = time.time()
        data = _query_endpoint(endpoint, query, log)
        last_request_ts = time.time()

        if data is None:
            _append_scrape_log(outlet_id, "failed", None)
            with lock:
                progress["fail"] += 1
            continue

        elements = data.get("elements", [])
        counts = _count_by_type_radius(elements, lat, lon)
        payload = {
            "Outlet_ID": outlet_id,
            "Latitude": lat,
            "Longitude": lon,
            "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_elements": len(elements),
            "counts": counts,
            "endpoint": endpoint,
        }
        (CACHE_DIR / f"{outlet_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        _append_scrape_log(outlet_id, "ok", len(elements))

        with lock:
            progress["ok"] += 1
            done = progress["ok"] + progress["fail"] + progress["skip"]
            if done % 100 == 0:
                elapsed = time.time() - progress["start"]
                rate = done / max(elapsed, 1)
                eta_h = (total - done) / max(rate, 1e-6) / 3600
                with _LOG_LOCK:
                    log.info(
                        "  progress %d / %d  ok=%d fail=%d skip=%d  rate=%.2f/s  ETA=%.1fh",
                        done, total, progress["ok"], progress["fail"], progress["skip"],
                        rate, eta_h,
                    )


def scrape() -> None:
    log = _setup_logging()
    _install_signal_handlers(log)
    log.info("POI scraper v2 (multi-endpoint, parallel) start.")
    log.info("Endpoints: %d, workers: %d", len(OVERPASS_ENDPOINTS), NUM_WORKERS)

    if not COORDS_CSV.exists():
        log.error("Missing %s — run bronze_ingest first.", COORDS_CSV)
        sys.exit(2)

    coords = pd.read_csv(COORDS_CSV)
    log.info("Loaded %d coordinate rows.", len(coords))

    cached = {p.stem for p in CACHE_DIR.glob("*.json")}
    log.info("Resume mode: %d outlets already cached.", len(cached))

    todo = coords[~coords["Outlet_ID"].isin(cached)].reset_index(drop=True)
    log.info("To scrape: %d outlets.", len(todo))
    if len(todo) == 0:
        log.info("Nothing to do.")
        return

    queue = todo.to_dict("records")  # pop from the end → LIFO, doesn't matter
    queue_lock = threading.Lock()
    progress = {"ok": 0, "fail": 0, "skip": 0, "start": time.time()}

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = []
        for i in range(NUM_WORKERS):
            endpoint = OVERPASS_ENDPOINTS[i % len(OVERPASS_ENDPOINTS)]
            futures.append(ex.submit(_worker_loop, endpoint, queue, queue_lock,
                                     progress, len(todo), log))
        for f in futures:
            f.result()

    log.info("Scrape end. ok=%d fail=%d skip=%d. Cache total: %d",
             progress["ok"], progress["fail"], progress["skip"],
             len(list(CACHE_DIR.glob("*.json"))))


if __name__ == "__main__":
    scrape()
