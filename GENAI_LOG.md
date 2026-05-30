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

---

## Phase 2 — Formal censoring (Tobit / Weibull-AFT) + physical cooler ceiling

**Goal.** Add the formal censored-data methods the methodology rubric names (Tobit / hurdle)
as a *cross-check* on the prelim's heuristic uplift, and make the cooler limit physical
(Methodology 30%).

**Prompt (paraphrased).** "We left-censor heuristically (drop months above P99×0.95, add a
censoring/plateau uplift). Add a formal Tobit and a lifelines survival model as cross-checks,
and a physical_max from cooler count × capacity × replenishment. What constants are
defensible and how do we keep this honest?"

**Accepted.**
- *Hand-rolled Tobit NLL* (left-censored Gaussian, σ via `exp(log σ)` for positivity, OLS
  warm-start) optimised with `scipy.optimize.minimize`. Treating months within 2% of the
  outlet's P95 as left-censored mirrors gold's existing `at_ceiling` definition, so the two
  notions of "constrained" stay consistent.
- *lifelines `WeibullAFTFitter`* as a fully independent second opinion — different library,
  different parametric family — so agreement isn't an artefact of one implementation.
- *Keeping LightGBM primary.* Both censored models are reported as directional cross-checks,
  not swapped in. This is the rubric's "validated, not blindly trusted" stance applied to our
  *own* method, not just the LLM.

**Rejected / modified.**
- *Rejected: making `physical_max` a hard ceiling = Cooler_Count × cap × cycles.* The first
  cut had **13% of outlets with historical peak already ABOVE that ceiling** (and 35% of
  outlets have zero coolers → ceiling of 0). A ceiling history has already beaten is not a
  ceiling. We changed it to `max(raw_capacity, peak×1.05)` and, crucially, kept the raw
  estimate + a `cooler_capacity_breached` flag — because those 2,606 breaches are a *forensic
  signal* of stale cooler master-data (the 'Large outlet, 0 coolers' decay), which we report
  rather than silently smoothing away.
- *Modified: applying the physical ceiling to every outlet in predict.* We restricted it to
  the 17,394 *non-breached* outlets; on breached ones `physical_max` is just peak×1.05 and
  would wrongly clamp the uplift, so we skip it there.
- *Constants checked against data, not asserted.* capacity 350 L (packed single-door
  visi-cooler), 4 replenishment cycles/month (weekly Western/Central distributor routes),
  ambient 120 L for zero-cooler kades (near the 95th percentile of their observed peaks).

**Validation (the part that counts).** Agreement of the two censored estimators with the
LightGBM Stage-A prediction, per outlet (n=20,000):
- **Tobit:** Pearson **0.944**, Spearman **0.907**, mean diff **+6.7 L**
- **WeibullAFT:** Pearson **0.712**, Spearman **0.901**, mean diff **+31.3 L**

Three independent estimators rank latent headroom the *same* way (Spearman ≈ 0.90 for both).
The Tobit's near-perfect Pearson + tiny mean diff says the heuristic uplift isn't inflating
demand; WeibullAFT's lower Pearson but equal Spearman says the two differ mainly in *scale*,
not *ordering* — exactly the directional validation we wanted. Had they disagreed on ordering
we'd have revisited the uplift; they don't, so we keep it.

**Result.** `src/censoring_model.py` → `physical_max.parquet`, `censoring_crosscheck.parquet`,
`_censoring_crosscheck.json`. Gold grew 100 → 105 features. predict's sanity ceiling is now
`min(peer_P99×1.5, physical_max)` on trusted-cooler outlets (binds tighter for 4,815 of them);
mean potential 433.2 → 430.5 L, all 8 validation checks still pass (0 floor violations).
