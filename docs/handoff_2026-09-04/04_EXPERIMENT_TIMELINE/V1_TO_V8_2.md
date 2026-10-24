# Experiment Timeline: V1 through V8.2

## V1 - Ordinal GBDT baseline

**Status: CONFIRMED RESULT**

### Thesis

Treat `life_satisfaction` as an ordinal latent score. Train three strong but different tree regressors, blend their continuous predictions, then learn three thresholds for QWK.

### Core

- LightGBM regressor
- XGBoost regressor
- CatBoost regressor
- 5-fold `StratifiedGroupKFold`, group=`id`
- fold-local preprocessing and semantic feature engineering
- native NaN where supported
- no SMOTE/random oversampling
- no survey-weight sample weighting
- cross-fitted threshold evaluation

### Result

- Cross-fit OOF QWK: **0.479662**
- Public LB: **0.48003**

### Main lesson

CV transferred almost one-to-one to Public LB. The validation framework was credible. Error analysis indicated the biggest weighted disagreement was adjacent **class 1 <-> 2**, not rare class 4.

---

## V2 - Representation-diversity ordinal ensemble

**Status: CONFIRMED RESULT / EMPIRICAL CHAMPION**

### Thesis

Do not brute-force hyperparameters. Add representation and objective diversity while preserving V1's validated ordinal framework.

### Major changes

- repeated **2 x 5 grouped CV**, seeds 42 and 2026;
- nested fold-safe feature selection;
- raw survey-code representation;
- dictionary-valid semantic copies;
- nonstandard-code indicators;
- domain burden/missingness features and compact interaction bank;
- native categorical tree views;
- monotonic semantic candidate;
- multiclass expected-value and cumulative-ordinal sidecars;
- greedy sidecar selection based on OOF gain + residual diversity;
- permanent fix to **23,666 unique-ID** submission output.

### Result

- Cross-fit OOF QWK: **0.481367**
- Public LB: **0.48193**

### Why V2 matters beyond the score

Later “cleaner” rebuilds often underperformed. The empirical interpretation became that V2's redundancy was partly useful alternate representation, not merely feature bloat.

**Permanent decision:** keep the exact V2 executable pipeline frozen as the anchor.

---

## V3 - Structural/ontology representation

**Status: NEGATIVE RESULT**

### Hypotheses tested

- observed-code ontology;
- questionnaire skip/applicability states;
- fold-safe ordinal target encodings for high-signal coded categories;
- CatBoost dual numeric/categorical representation;
- hard numeric feature budget around 175;
- single grouped 5-fold evaluation with predeclared breakthrough gates.

### Outcome

The recovered run explicitly states that V3 **did not clear the predeclared breakthrough gate** and did not beat V2 internally. Any generated V3 submission was exploratory rather than a confident leaderboard attack.

### Lesson

The apparent survey structure was not hiding a large easy signal. More elaborate structural encoding did not solve the 0.48 ceiling.

---

## V4 - Breakthrough discovery system

**Status: NEGATIVE DISCOVERY ROUND**

### Purpose

Stop iterating tiny tree variants. Probe genuinely different inductive biases and kill weak ideas early.

### Candidate families

1. Full raw-categorical CatBoost.
2. Missing/special-response fingerprinting.
3. Retrieval / nearest-neighbor respondent similarity.
4. Deep tabular multitask embedding MLP.

### Experimental safeguards

- multiple untouched probe folds;
- inner calibration split;
- no one-fold promotion;
- no tree scaling;
- neural/retrieval scaling only when required;
- no SMOTE;
- `id` and `household_id` excluded from arbitrary numeric features;
- stop before full CV when there is no qualitative jump.

### Outcome

Retrospective evidence says questionnaire/ontology, fingerprinting, simple retrieval, and basic deep routes did **not** produce the required qualitative jump. Full raw-categorical CatBoost later remained only a weak/unstable sidecar.

---

## V5A - Boundary-1 breakthrough specialists

**Status: NEGATIVE RESULT, WITH IMPORTANT SCOPE CAVEAT**

### Thesis

Directly attack the dominant class 1 vs >1 / 1-vs-2 bottleneck rather than another generic 4-class model.

### Models

- all-raw-categorical CatBoost ordinal anchor;
- all-cat CatBoost boundary specialist;
- TabM boundary specialist with piecewise-linear numerical embeddings;
- RealMLP-TD boundary specialist.

### Protocol

Three outer probe folds before any full CV. Submission only if honest unique-ID cross-fit QWK could reach the aggressive leaderboard target.

### Outcome

No dramatic boundary signal was established. The important caveat is:

> V5 falsified **TabM/RealMLP as B1 binary specialists in this setup**, not TabM/RealMLP as full-target tabular models.

That distinction later kept them on the research backup list.

---

## V5B - Forensic/internal-signal attack

**Status: NEGATIVE RESULT / FROZEN**

### Thesis

If ordinary modeling is capped, audit legal structure *inside* the competition files: submission mapping, quasi-keys, repeated household/source patterns, and high-purity internal lookups.

### Safeguards

- no external NHIS/source data;
- lookup only from competition train;
- lookup must survive grouped OOF;
- compact LightGBM/XGBoost + all-cat CatBoost;
- no blind dictionary truncation of raw categorical survey codes.

### Outcome

The forensic lookup path did not produce reliable enough OOF gain and carried substantial leakage/overfit risk. It is marked **dead/frozen**.

---

## V6A - Error-centric ordinal learning

**Status: NEGATIVE / DIAGNOSTICALLY VALUABLE**

### Thesis

The remaining legal opportunity is not more features, but handling ambiguous adjacent labels and hard rows.

### Components

- compact V2-lite anchor;
- OOF soft-target distillation;
- OOF difficulty weighting;
- CORN-style conditional ordinal model;
- error-cohort EDA for `true1/pred2` and `true2/pred1`;
- multi-probe gates;
- fully cross-fitted blending and thresholds.

### Key design constraint

A roughly 140-feature cap prevented the 400+ engineered-feature expansion that had not paid off.

### Outcome

The new error-centric techniques did not create a meaningful new ranking signal. CORN and weighting/distillation were later frozen.

---

## V6B - Recovery/all-categorical portfolio

**Status: CONFIRMED NEGATIVE RESULT**

### Recovered final diagnostics

| Metric | V6 |
|---|---:|
| Core cross-fit QWK | **0.478899** |
| Final cross-fit QWK | **0.478832** |
| Unique-ID diagnostic | **0.480162** |
| `AUC(y>1)` | **0.779611** |
| `AUC(y>2)` | **0.872803** |
| `AUC(y>3)` | **0.808347** |
| `AUC(1 vs 2)` | **0.765352** |
| Global/calibration recipe QWK | **0.481761** |

Sidecar selection frequency:

- cumulative ordinal: 3/5 folds;
- all-categorical CatBoost: 1/5;
- XGB categorical: 1/5.

### Main lessons

1. Ranking did not collapse. Boundary-1 AUC stayed close to V2's roughly 0.780 region.
2. Score geometry and calibration were a bigger failure mode.
3. The roughly 0.00293 global-vs-honest-QWK optimism proved direct discrete-QWK blend selection can overfit.
4. “Cleaning” V2 did not reproduce V2. Freeze the exact V2 pipeline.

---

## V7 - Foundation breakthrough architecture

**Status: CONFIRMED DESIGN / FINAL PERFORMANCE NOT RECOVERED**

### Strategic shift

Once project notes recorded that public pretrained tabular foundation models were permitted, V7 became a breakthrough experiment rather than another tree iteration.

### Primary design

- foundation stage first;
- TabPFN-3 classifier + regressor;
- EXAONE Tabular classifier + regressor when runtime is available;
- grouped foundation OOF;
- run V2 champion-equivalent anchor only after a foundation candidate passes a promotion gate;
- positive Ridge on continuous OOF scores;
- thresholds learned separately and cross-fitted;
- exact checkpoint/runtime provenance;
- 23,666 unique-ID output.

### Frozen notebook-level promotion references

- foundation QWK gate: **0.4840**
- `AUC(y>1)` gate: **0.7850**
- `AUC(1 vs 2)` gate: **0.7700**

### Research interpretation

The project wanted a **real boundary move**, not another +0.0002 QWK fluctuation. Prior research notes treated B1 AUC around 0.80+ as genuine new signal and 0.83+ as very interesting.

### Open item

A final verified V7 foundation score is not available in the recovered archive and is therefore not invented here.

---

## V8 - Conditional foundation mixture

**Status: CONFIRMED DESIGN / IMPLEMENTATION RECOVERED**

### Why V8 exists

V1-V7 repeatedly showed that the tail boundaries are easier than the first boundary. V8 therefore refuses to relearn what V2 already does well.

### Architecture

1. **Part A:** exact V2 frozen anchor.
2. **Part B:** cross-fitted calibration of anchor into `P(y>2)` and `P(y>3)`.
3. **Part C:** successive-halving foundation specialists trained only on classes 1 and 2.
4. **Part D:** full CV only for promoted specialists.
5. **Part E:** positive Ridge combining V2 and promoted conditional scores, then cross-fitted thresholds.

Conditional reconstruction:

`P(y>1) = P(y>2) + (1-P(y>2)) * P(y=2 | y<=2)`

Ordinal score:

`s = 1 + P(y>1) + P(y>2) + P(y>3)`

### Specialist candidates

- EXAONE Tabular
- TabICLv2
- repaired TabPFN-3

### V8 predeclared gates

- probe folds: `[0, 2, 3]`
- fold-0 survive delta AUC: **0.0010**
- promote mean delta AUC: **0.0030**
- promote minimum wins: **2**
- promote mean conditional QWK gain: **0.0015**
- serious B1 AUC: **0.7860**
- submit cross-fit gate: **0.4875**

---

## V8.1 - Foundation package and provenance layer

**Status: CONFIRMED IMPLEMENTATION PHASE**

V8.1 deliberately runs after the frozen V2 pipeline so installing foundation runtimes cannot change the tree anchor that produced the baseline.

Recorded runtime/checkpoint settings include:

- TabPFN requested version: `8.5.0`
- TabPFN repo: `Prior-Labs/tabpfn_3`
- TabPFN revision: `e923ba9be85784206c9e2f43b0035c84d5fd5747`
- TabPFN classifier file: `tabpfn-v3-classifier-v3_default.ckpt`
- TabICL requested version: `2.1.1`
- TabICL classifier: `tabicl-classifier-v2-20260212.ckpt`
- TabICL regressor: `tabicl-regressor-v2-20260212.ckpt`
- EXAONE repo: `LG-AI-Research/EXAONE-Tabular`
- EXAONE checkpoint revision: `96946d008279216721e951dd3ee2f55a38877864`
- EXAONE classifier: `exaone-tabular-classifier-v1_default.safetensors`
- EXAONE regressor: `exaone-tabular-regressor-v1_default.safetensors`
- probe ensemble count: 4
- full ensemble count: 8
- EXAONE max VRAM budget: 13 GiB
- TabICL batch size: 2

The implementation also records model-file SHA256 when downloads succeed.

### Reproducibility weakness discovered for V8.3

The EXAONE *checkpoint* is pinned, but V8.1 installs the EXAONE Python runtime from the GitHub repository without a commit pin. That means runtime API behavior can drift even when weights do not. V8.3 should fix this by resolving and pinning a known-working runtime commit.

---

## V8.2 - Fold-safe foundation input views

**Status: CONFIRMED IMPLEMENTATION PHASE**

### Key rules

- no target encoding;
- no external semantic reconstruction;
- special-response codes on true continuous-looking fields become `NaN + flag`;
- missing indicators are selected inside each outer training fold;
- foundation numeric view is `float32`;
- TabICL additionally receives a mixed nominal/numeric representation inferred fold-safely.

### EXAONE specialist call in the recovered implementation

EXAONE is used on `cuda:0` and constructed with a public pinned checkpoint revision, ensemble count, seed, and VRAM budget. It is fit only on rows whose target is class 1 or 2, with the binary target `class 2 = 1`.

### Current blocker

The next requested project step is **V8.3 EXAONE runtime debugging**. No final EXAONE quality conclusion should be drawn until the runtime path itself has been made reproducible.
