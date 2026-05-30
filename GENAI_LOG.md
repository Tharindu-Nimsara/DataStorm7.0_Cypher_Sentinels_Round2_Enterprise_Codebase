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

---

## Phase 3 — Retrain on the new feature set + regenerate predictions

**Goal.** Make the predictions reflect the Phase-1 spatial-decay and Phase-2 physical-ceiling
features before the optimizer, XAI, and app consume them.

**No new GenAI here — this phase is a disciplined re-run, logged for honesty.** We retrained
`model.py` unchanged (5-fold GroupKFold by Outlet_ID, target `log1p(Volume_Liters)`, seed 42)
on the expanded matrix and re-ran `predict.py` + `validate_predictions.py`.

**What we watched for, and what we found.**
- *Did the extra features hurt the demand model?* No. CV mean **RMSE(log) = 0.6098** (folds
  0.6087–0.6109), essentially the prelim's ~0.61. The spatial/physical features are stable
  additions, not noise that degrades Stage A.
- *Did the final distribution move?* Barely — mean held at **430.5 L**. This is itself a
  finding worth stating plainly rather than dressing up: the `max(floor, stage_a, peer_85)`
  formula is dominated by the historical floor and the peer-85th term, so Stage A rarely
  "wins" the max(). The new features mostly improve *explainability and the ceiling*, not the
  point estimate. We resisted the temptation to retune the formula just to make the number
  move — there's no leaderboard, and a stable, defensible framework beats a tweaked one.
- Top features by gain are unchanged (vol_mean, sku_diversity, vol_std…), confirming the
  demand signal still comes from transactional history, with spatial/physical features as
  context.

**Result.** Fresh `reports/cypher_sentinels_predictions.csv` (20,000 rows; mean 430.5,
median 214.8, range 77–2955 L). All 8 validation checks pass: 0 nulls, 0 non-positive, 0
floor violations, Large/XL 1318 ≫ Small 145, urban 807 > rural 419.

---

## Phase 4 — SHAP attributions + LLM explainability (the user-facing GenAI layer)

**Goal.** Final-round requirement 4.1: a GenAI layer that explains each outlet's score in plain
business language a Sri Lankan sales manager can act on — *grounded in the real numbers, not
trusted blindly*. Scored under GenAI (15%) and Business (25%). This is the phase where the
rubric's "rigorously validated rather than blindly accepting generated logic" is the whole
point, not a footnote.

**Provider.** GitHub Models, OpenAI-compatible REST, model `gpt-4o-mini`, auth via a
`GITHUB_TOKEN` in a gitignored `.env`. Pluggable (endpoint + model are env vars) with a
deterministic **offline template** fallback so the demo runs with no key.

**Setup friction we hit and fixed (logged honestly).**
1. *401 "models permission required."* The first fine-grained PAT lacked the **Models**
   account permission. Our client returned `None` and fell back to the template — the pipeline
   never hard-failed — and we added the permission.
2. *400 "Unknown model: openai/gpt-4o-mini."* The `models.inference.ai.azure.com` endpoint
   wants the **bare** id `gpt-4o-mini`; the `openai/`-prefix form is for the newer
   `models.github.ai/inference` endpoint. We probed all four combinations, confirmed three
   return 200, and set the default to the bare-id pairing. Both failures are exactly the kind
   of dead-end the rubric wants documented.

**Prompt design + hallucination guard (the core).**
- *System prompt* pins the role ("explain to a non-technical Sri Lankan beverage sales
  manager"), forbids technical jargon (no "SHAP"/"feature"), and states the hard rule: **use
  ONLY the numbers in the data packet; never invent or re-round figures.**
- *Evidence packet* per outlet is the single source of allowed numbers — predicted potential,
  historical peak, cooler ceiling, footfall, competitor count, peer rank, constraint type, and
  the top signed SHAP drivers translated to business phrases.
- *Automated validator* (`validate_explanation`) scans every figure in the generated text and
  rejects the explanation if any number isn't traceable to the packet (±2 rounding slack);
  rejects fall back to the grounded template.

**Validation — it caught real hallucinations.** Over a 15-outlet live sample, the guard
**rejected 3 (20%)** where `gpt-4o-mini` invented numbers not in the packet:
- `OUT_19648` — emitted `955` and `117` (neither in its packet) → rejected, replaced.
- `OUT_18885` — emitted `320` → rejected.
- `OUT_08815` — emitted `16` → rejected.
The other **12 were accepted** (e.g. `OUT_02141`: "predicted potential of 271 liters, higher
than its historical peak of 256 liters" — both figures verified against the packet). *Before*
adding the validator + the "ONLY these numbers" clause, an early prompt let the model pad
explanations with plausible-but-fabricated volumes; tightening the system prompt and adding
the programmatic check is what turned a nice-looking demo into a defensible one.

**Why this matters for the score.** We don't ask the judges to trust the LLM — we *show* a
20% fabrication rate caught and corrected automatically, with the grounded fallback meaning no
user ever sees an invented number. That's the validated-GenAI posture the rubric rewards.

**Result.** `src/xai_explain.py` → `data/gold/outlet_explanations.json`: 20,000 explanations
(12 live `gpt-4o-mini`, 19,988 grounded offline / live-rejected), each keyed by Outlet_ID with
its `source`, `explanation`, and full `evidence` packet. Caching makes the app instant and
offline-capable; live regeneration is opt-in when a token is present.

*Post-build audit (caught two of our own bugs, fixed both).* Auditing the cache, 227 offline
rows tripped our own validator. The cause wasn't the LLM — it was us: (1) raw column names
leaked into driver text ("peer p85 monthly", "poi restaurant 2000m") because we hadn't
labelled `peer_p85_monthly` / the `poi_<type>_<radius>m` columns; (2) the validator then
flagged the "85" inside "p85" as an ungrounded figure. We added the missing business labels +
a generic `poi_<type>_<radius>m` rule, and restricted the validator to standalone numeric
tokens (digits not adjacent to a letter). Re-audit: **0 / 20,000 ungrounded, 0 column leaks.**
Worth recording because it shows the validation net catching *our* mistakes, not just the
model's.

---

## Phase 5 — LKR 5,000,000 Western budget optimiser

**Goal.** Turn the potential estimates into a concrete trade-spend allocation across the 9,000
Western-province outlets that maximises *incremental* volume (Business Viability 25% +
deliverable #2).

**Prompt (paraphrased).** "We have potential per outlet. Formulate a 5M-LKR allocation that
maximises incremental volume with diminishing returns, where responsiveness depends on whether
an outlet is supply- or demand-constrained. cvxpy or greedy?"

**Accepted.**
- *Opportunity gap* `max(potential − recent_actual, 0)` with `recent_actual = vol_mean`.
  Western total gap ≈ **1.96M L/month**, mean 218 L.
- *Saturating response* `gap·(1−e^{−k·s})` — diminishing returns matching real promo
  behaviour, with `k` keyed to constraint type (supply-constrained relieved faster by
  cooler/merchandising spend; demand-led nudged more slowly by discounts).
- *Both an exact and an explainable solver.* cvxpy for the exact concave optimum **and** a
  greedy marginal-return allocator we can narrate on stage ("fund the best litres-per-rupee
  first"). Reporting both, with their agreement, is stronger than either alone.

**Rejected / modified.**
- *Rejected: a linear `min(α·s, gap)` response.* Simpler, but it has no diminishing returns,
  so the optimum degenerates to dumping the cap on the highest-gap outlets — unrealistic and
  not a good stage story. The concave form spreads spend sensibly.
- *Fixed: solver choice.* The first cut called `cvxpy` with `ECOS`, which **isn't bundled with
  modern cvxpy** — it fell back to greedy. We switched to the bundled **CLARABEL** conic
  solver (SCS fallback); the exact solve now runs (`status=optimal`).

**Validation.** The exact cvxpy solution and the transparent greedy allocator project
**154,524 L vs 154,475 L** incremental — agreement **1.000**. So the simple, stage-explainable
greedy is provably near-optimal here; we don't have to choose between rigour and
explainability.

**Result.** `reports/cypher_sentinels_budget_allocations.csv` (9,000 Western rows; exact cols
`Outlet_ID, Trade_Spend_Allocation_LKR`; sum 4,999,999.72 ≤ 5M; 0 negatives/nulls). 805
outlets funded (per-outlet cap 50k keeps spend spread; max single 17,530). Projected
**+154,524 L/month** (7.9% of the gap), balanced across DIST_W_01/02/03. Spend mix: 663
discount, 115 merchandising, 27 cooler.

---

## Phase 6 — Streamlit Outlet Intelligence app

**Goal.** A functional, locally-runnable app for business users (deliverable #4 + Business
Viability 25%): browse 20k predictions, filter, drill into one outlet with its SHAP chart +
map + LLM explanation, and view the Western budget.

**GenAI's role here is *display*, not generation.** The app reads the **precomputed** cache
from Phase 4 — it never calls the LLM at request time. That's a deliberate design choice we'd
defend: live API calls in a demo are slow, can fail on stage, and cost money per view; serving
the validated cache makes the demo instant, offline-capable, and reproducible. Each
explanation is badged in the UI by its `source` (🤖 LLM vs 📝 grounded template) so the viewer
knows which is which — transparency, not disguise.

**Validation.** Launched headless (`streamlit run`, health endpoint `ok`, page renders, zero
runtime errors in the log). Testing surfaced a **real merge-collision bug**: the diagnostic
parquet and the gold table both carry `censoring_score` / `physical_max`, so a naïve join
produced `_x`/`_y` columns and the constraint logic `KeyError`'d. Fixed `load_data` to pull
from gold only the columns *not already in* diag. Found before any judge would have — exactly
why we run the app, not just lint it.

**Result.** `app/app.py`, five tabs (potential map · browse · outlet detail · Western budget),
all reading precomputed artifacts; 20,000 rows load, 19,760 with map-valid coordinates (the
240 out-of-bounds rows are excluded from map layers only, not from the data).

---

## Phase 7 — Validation, idempotency, packaging, finals stats

**Goal.** Make the whole thing defensible and submittable: extend validation to the new
outputs, prove idempotency, dump the finals numbers the PDFs cite, and package the two
submission artifacts.

**No GenAI generation here — engineering discipline, logged for completeness.**

- **Extended validation.** `validate_predictions.py` now also checks the budget CSV (exact
  columns, Western-only, sum ≤ 5M, no negatives/nulls) and the explanations JSON (one entry
  per outlet, each non-empty with an evidence packet). All 8 + 5 + 3 checks pass; missing
  artifacts warn-and-skip so it runs at any stage.
- **Idempotency, proven not asserted.** Snapshotted content signatures of the seven
  deterministic outputs (poi_decay, gold, physical_max, predictions, budget, both
  rejected-records files), re-ran the whole deterministic chain, and re-diffed: **all seven
  byte-identical.** Crucially the rejected-records files don't duplicate on re-run — the
  property the rubric calls out explicitly.
- **Finals-stats dump.** `finals_stats.py` → `reports/finals_stats.{json,md}`: one traceable
  source for every figure the paper/deck quote (segment means, uplift mean, the 95%/5% max()
  split, urban/rural, the 2.05× upside gap, Tobit/Weibull agreement, the 5M allocation +
  efficiency). The finals segment means (XL 2156 / Large 1045 / Medium 351 / Small 145 L)
  landed within 1 L of the prelim baselines — independent evidence the framework is stable.
- **Submission packaging.** `make_repro_zip.py` builds the slot-3 "Reproducible Codebase" zip
  (code + bronze + 16k POI cache + outputs, 305 MB → 141 MB) with a pre-write assertion and a
  post-write leak check that confirmed **no `.env` / `CLAUDE.md` / `tasks.md` / scaffolding**
  made it in. The web app is split into its own repo with bundled artifacts so it runs
  standalone — matching the form's separate "web app repository" slot.

**Honest note on the two `cvxpy`/solver and GitHub-Models dead-ends** (logged in Phases 5 and
4): both were found by *running* the code, not by reading it — as were the merge-collision bug
(Phase 6) and the 227 validator false-positives (Phase 4). The pattern across the final round
was: build, run, let the failure surface, fix, re-verify. That loop — not a clean first
draft — is what the evidence in this log actually documents.
