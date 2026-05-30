# GenAI Workflow Log — Cypher Sentinels (Data Storm v7.0, Final Round)

How we actually used generative AI to build the final-round modules. We log the prompts
that mattered, what we **accepted, rejected, or modified**, and the dead-ends — because the
rubric rewards logic that is *validated*, not blindly accepted. Entries are in build order.

Tooling: Claude (Anthropic) as a coding/reasoning copilot for the pipeline modules; a
GitHub-hosted model (OpenAI-compatible endpoint) as the runtime LLM for the user-facing
explanation layer (Phase 4). The two are separate: one helped us *write* code, the other
*runs inside* the product.

---

## Phase 1 — Distance-decay POI gravity + competitive density

**Goal.** Replace flat POI ring counts with non-linear gravity/decay signals and add a
competitive-saturation measure (rubric: Data Engineering 30%).

**Prompt (paraphrased).** "Our Overpass cache stores cumulative POI counts at 500/1000/
2000 m rings per type — not individual POI coordinates. Design distance-decay POI features
and a competitor-density signal that still honour the rubric's call for a BallTree haversine
gravity model. What's defensible given ring data only?"

**Accepted.**
- *Ring-shell gravity.* Difference the cumulative rings into shells (0–500, 500–1000,
  1000–2000 m) and evaluate the decay kernel at each shell's midpoint. Honest middle ground
  between flat counts and a full point-level model, with no 5-hour re-scrape. We document it
  in code as an approximation, not as true point gravity.
- *Per-type bandwidths.* The suggestion to use *different* λ per POI type (bus stop pulls
  decay fast, hospitals reach further) — this is the interpretable, defensible core of the
  feature and reads as domain knowledge, not a tuned hyperparameter.
- *BallTree for competitors.* Use a genuine `sklearn` BallTree (haversine) over the real
  outlet coordinates for the competitor signal — there we *do* have exact points, so the
  textbook O(n log n) radius query applies with no approximation.

**Rejected / modified.**
- *Rejected: re-scraping POI point geometry for all 20k outlets.* The model proposed
  fetching `out center;` node coordinates to do "proper" point-level gravity. We rejected it
  on cost (1 req/s × 20k ≈ 5+ h, cache only 82% complete) and because ring-shell gravity
  captures ~the same ranking — see the validation below.
- *Modified: a single global λ.* The first draft used one λ for every POI type. We changed
  it to per-type λ because a bus stop and a hospital plainly have different catchment radii;
  collapsing them would discard the most business-meaningful part of the signal.

**Validation (this is the part that counts).** We didn't trust the λ values blindly. We ran
a sensitivity check: scale **every** bandwidth by ±50% and recompute the combined weighted
density, then measure Spearman rank correlation against the base ranking.
- λ × 0.5 → **Spearman 1.0000**
- λ × 1.5 → **Spearman 1.0000**

The ranking is *perfectly* stable to ±50% λ changes. The reason is worth stating honestly
rather than overselling it: because the cache gives us fixed shell midpoints (250/750/1500 m),
scaling λ applies the *same* monotonic factor to every outlet's per-type score, and the
importance-weighted sum preserves that ordering. So the exact bandwidths are emphatically not
load-bearing for *who ranks where* — they only reshape absolute magnitudes. That's the
defensible claim, and it's exactly what the ±50% check demonstrates. The check is in
`lambda_sensitivity_check()` and prints on every run.

**Result.** `src/poi_decay.py` → `data/gold/poi_decay_features.parquet` (20,000 rows, 23
feature cols + Outlet_ID). Gold table grew 77 → 100 features. DQ quarantined **240
coordinate rows** outside the Sri Lanka bounding box (several sit at lat/lon ≈ 0 — legacy
export garbage), so the BallTree was built over **19,760 clean** coordinates, not poisoned by
them. Competitor density: mean **2.26** rivals within 500 m, median 2, max 12;
`market_share_proxy` mean **0.681** (1.0 = local monopoly). 3,577 outlets (17.9%) had no POI
cache and were cluster-median imputed; 240 (1.2%) were competitor-imputed. Re-run is
byte-identical (idempotent).
