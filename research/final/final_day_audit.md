# DATAQUEST 5.0 final-day audit

Audit date: 2026-09-10. The raw competition files, every saved notebook cell and output, material result artifacts, both handoff archives, and the complete diagnostic package were inspected. Primary artifacts take precedence over notebook prose and handoff summaries. Claims below are tagged implicitly by section: **raw/train-supported** means recomputed from competition train data or row-aligned OOF; **historical** means recovered from executed project artifacts; **documented public** means recorded locally but not independently queried from Kaggle; **literature inference** means supported by cited external sources but not measured here.

## Decision in one paragraph

Recovery remains the credible champion: historical OOF QWK `0.481867` and documented public `0.48334`. Its largest remaining loss is individual ranking between classes 1 and 2, which contributes **63.32%** of squared-label error; false-tail errors contribute another **25.45%**. The raw train set supports burden, health, mental-health, economic, functional, missing-state, and survey-context signal, but it does not support a separate hand-built “resilience” correction after conditioning on Recovery. Missing-cell reconstruction, historical stacking, neighborhoods, retrieval, and the tested foundation models are negative. The only material formulation left open at acceptable final-day risk is a **full-target, fold-trained TabM probability model** over preserved response states, paired with a fresh CatBoost regression anchor and a small ordinal CDF-loss auxiliary. This is a complementarity test, not evidence of a breakthrough. An honest expected gain is approximately `0.000–0.003` QWK; crossing `0.50682` is not supported by current evidence.

## Repository and evidence reconstruction

Eighteen notebooks exist: V1, V2, V3, V4, V5, V5.1, V6, V6.1, V7, V8, V8.1, V8.2, V8.3, V9, V11, V12, V14, and V15. Every saved code cell has an execution count and there are no Jupyter `error` output objects. Some cells intentionally caught runtime failures or skipped later stages, so “executed notebook” does not mean that every defined candidate produced predictions.

The `.fr-PIIJV3/results-v1_v12.zip` archive indexes/duplicates 286 result-tree entries and adds no independent experiment. The September 8 handoff contains 39 documentation entries; the September 4 handoff contains 20. Neither contains raw prediction arrays or notebooks. The September 8 handoff is accurate through V11 but stale for V12: it says ModernNCA was blocked before training, while later V12 primary metrics prove a successful runtime gate and a decisive two-fold quality failure.

### Compact experiment ledger

| Version | Hypothesis / representation / model | Validation and strongest signal | Public LB | Correlation / failure | Verdict and open question |
|---|---|---|---:|---|---|
| V1 | Raw survey codes, sentinel flags, 11 fold-safe repairs and semantic counts; LGB/XGB/Cat regressors | 5-fold grouped/stratified; global `0.480481`, fold-pure cuts `0.479662` | `0.48003` | Components clustered at `0.4773–0.4780`; a caught LightGBM CUDA failure preceded fallback | Genuine baseline; raw-state GBDT signal is real |
| V2 | Raw, semantic, nonstandard-code, monotonic, multiclass and cumulative-ordinal GBDT views | 2×5 grouped; core CF `0.479735`, final CF `0.481367`, global `0.481587`; cumulative sidecar gain `0.001087`, XGB-cat gain `0.000545` | `0.48193` | Sidecars correlate `0.9927–0.9975`; selection later shown optimistic | Genuine small improvement; still strongest core recipe |
| V3 | Questionnaire ontology, applicability paths, ordinal target encoding, structural dual views | Probe ordinal-TE gain `0.000640`; final CF `0.479948`, unique-ID `0.480846` | unknown | Ontology/path bundles neutral or negative; skip-rule artifact empty | Mostly falsified additive structure engineering |
| V4 | All-categorical CatBoost, deep embedding/multitask, retrieval, fingerprint | 3 folds; all-cat blend `+0.002925`, deep `+0.000935`, retrieval `-0.000177`, fingerprint `-0.001059` | none | Fold gate required `+0.015` or absolute `.52`; no full CV | No robust breakthrough; learned representation family not universally falsified |
| V5 | **Boundary-1 only** CatBoost, TabM-PLE, RealMLP and cross-family specialists | 3 folds; TabM mean blend `+0.000586` with 1/3 wins; RealMLP `-0.000455`; specialist AUCs `0.7757–0.7801` | none | Fold signs reverse; residual correlations roughly `.40–.50` because the injected target differs | Boundary specialist fails; full-target TabM/RealMLP remains untested |
| V5.1 | Household/quasi-key lookup and repeated-respondent aggregation | 5 folds; base `0.478556`, lookup `0.478569`, gain `0.000014` | none | OOF household coverage `5.2%`; exact-hash test coverage zero | Scientifically negligible; submission mapping audit is valuable |
| V6 | Error difficulty, distillation and CORN | Anchor `0.479158`; CORN `+0.000106`, distilled `-0.001327`, difficulty `-0.001559` | none | Predictable difficulty AUC `0.623`, but correction does not help | Tested error weighting/ordinal correction falsified |
| V6.1 | Recovery/all-cat portfolio and calibration | Core CF `0.478899`, final CF `0.478832`, global `0.481761` | unknown | Global optimism gap `0.002930`; fold blends unstable | Calibration/selection artifact, no honest gain |
| V7 | Full-target TabPFN classifier/regressor | Probe blend gains `+0.000259/+0.000804`; final anchor-only `0.481448` vs anchor `0.481716` | none | Residual corr about `.992`; boundary AUC declines; full path empty | Reject TabPFN rerun |
| V8 | Conditional foundation specialist with frozen tails | Only TabPFN fold 0: QWK `-0.003216`, AUC12 `-0.002992` | none | EXAONE config failure; TabICL OOM; full artifacts empty | Scientific and operational failure |
| V8.1 | EXAONE environment repair | No candidate prediction | none | Virtual environment preflight failure; outputs equal anchor | Engineering failure only |
| V8.2 | Pinned EXAONE install repair | No candidate prediction | none | Install/preflight failure; outputs equal anchor | Engineering failure only |
| V8.3 | EXAONE T4 attention repair | Tiny `128×101` smoke passes in 2.42 s | none | Real fold plan needs 33.41 GB vs 13.96 GB; `.npz` is JSON failure record | Runtime hypothesis tested, model quality untested and impractical |
| V9 | Full-target TabICLv2 regression | Folds 0/2: QWK `-0.009541`, AUC12 `-0.008910` | none | Residual corr `.988417`; full run skipped | Decisively negative transfer |
| V10 | xRFM mixed representation, 12 fixed configs | No valid outer prediction | none | Config 10 Cholesky failed after ~9.5 h; earlier API/kernel/641 GiB AGOP blockers | Engineering suspended; quality unknown, final-day risk prohibitive |
| Recovery | Convex blend of V2/V5.1/V6.1/V7 predictions | OOF `0.481867`; historical paired gain about `0.000939`, 4/5 positive, bootstrap P(gain>0) ≈`.913` | `0.48334` | Test-score corr with V2 ≈`.99932`; 558/23,666 labels changed | Current champion; small diversity/calibration gain |
| V11 | Historical prediction-space cumulative-logit stack over 43 streams | `0.478662`, delta `-0.003205`; all folds negative; 95% bootstrap `[-.004889,-.001462]` | none | Residual corr `.998834`; three duplicated fallback streams | Historical stacking falsified |
| V12 | ModernNCA learned geometry, 1,124/1,125 dimensions | Folds 0/2: mean QWK `-0.010239`, AUC12 `-0.004337` | none | Runtime succeeded; residual corr `.992318` | Scientifically negative; corrects stale handoff |
| V14 | Pretrained Xiaomi TabLDM synthetic-SCM regressor | Folds 0/2: mean QWK `-0.009724`, AUC12 `-0.007480` | none | 14.25 GiB on T4; residual corr `.991629` | Scientifically negative and resource-tight |
| V15 | Fold-safe probabilistic reconstruction of four survey fields, 54 posterior features | Reconstruction works, but standalone `0.475999` (`-0.005868`); best blend `-0.000640`; gated blend `-0.000848` | none | All standalone folds negative; residual corr `.997682` | Missing-cell reconstruction is not the target bottleneck |

The only documented public values recoverable from local project evidence are V1 `0.48003`, V2 `0.48193`, and Recovery `0.48334`. There is no Kaggle export in the repository, so these values remain documented rather than independently verified.

### Executed versus merely defined

- V4, V5, and V6 executed probes and deliberately skipped full candidate CV/test after their gates failed.
- V7 defined a full TabPFN branch, but `v7_full_fold_metrics.csv` is empty.
- V8 obtained only one TabPFN fold. V8.1/V8.2 never produced candidate predictions. V8.3 passed a tiny smoke test but failed planning before a real fold.
- V9 executed the runtime gate and two screen folds, then skipped five-fold/test.
- V10 has no trustworthy outer prediction or primary live result tree.
- V11 executed the historical stack but correctly withheld a submission.
- V12 and V14 executed runtime gates and two quality folds, then stopped after strong negative results.
- V15 executed all five candidate folds and blend evaluation; only the Recovery control was exported.

## Raw competition-data audit

The four raw files were re-read from `data/` and hashed. Train is `95,609 × 179`; test is `23,902 × 178`; the sample is `23,902 × 2`. There are 176 predictors after excluding `id`, `household_id`, and target. Raw SHA-256 values are preserved in `final_evidence/raw_audit.json`.

### Identity, duplicates, and the evaluator contract

- Train has 94,663 unique respondent IDs; test has 23,666. Exactly 946 train IDs and 236 test IDs occur twice. Every duplicate-ID group is identical in predictors and, for train, target. There are no conflicting duplicates.
- Train and test IDs do not overlap. No distinct IDs share an identical full predictor row, and no train row has an identical predictor hash in test.
- Train has 59,016 households; 11,620 household IDs occur in both train and test. Prior honest household lookup produced essentially zero gain, so household identity is excluded as a predictive feature. It remains an audit field.
- Sample-submission IDs are a sequential placeholder: its 23,902 ID sequence does not match raw test row IDs and has only 4,518 incidental set overlaps. Historical V2’s accepted contract uses one row per unique `test_final.id`, in first-occurrence order. The final notebook must assert exact duplicate equality, collapse by ID, and emit 23,666 rows under this proven competition contract while using the sample only to validate column names and target dtype.

The unique-ID class distribution is 42,552 class 1 (`44.951%`), 47,713 class 2 (`50.403%`), 3,511 class 3 (`3.709%`), and 887 class 4 (`0.937%`). The rare tail makes threshold selection and direct batch-kappa optimization high variance.

### Feature semantics and missing states

The table is a redesigned NHIS-like coded survey for years 2021–2024. The official 2021 NHIS adult codebook uses the same four-level life-satisfaction wording and documents distinct refused, not-ascertained, and don’t-know states. This is a high-confidence provenance inference, not license to acquire external respondent data.

Most predictors are low-cardinality coded states. Codes such as `7`, `8`, `9`, `90`, `97`, `98`, `99`, `130`, `140`, and `990` have feature-specific meanings; some are legitimate values in one field and special states in another. A global sentinel-to-NaN rule is invalid. `survey_weight` is continuous/high-cardinality (85,872 observed values), while pseudo-PSU and pseudo-stratum are design categories. The dictionary omits some observed states, so observed train-fold vocabularies are authoritative.

Mean per-feature missingness over unique respondents is `14.405%`. Per-row missing counts have quantiles 4, 10, 18, 24, 33, 45, and 60 at 0%, 1%, 25%, 50%, 75%, 99%, and 100%. Co-missingness is structured: month/year fields for flu vaccination correlate about `.85`; employment-hour/full-time routes about `.83`; missed-work/sick-leave and several insurance routes about `.82`. Class-conditional missingness changes substantially for employment, marital, walking, insurance, and anxiety routes. Missingness is therefore predictive questionnaire state, not a generic imputation defect.

### Redundancy and feature engineering

Several mappings are nearly deterministic on observed pairs: survey year→data year and interview month→quarter are exact; food-security 4-category→3-category and cancer-count→cancer-ever are exact; household children, detailed marital status, smoking status, and e-cigarette status recover supplied recodes at `99.67–99.92%`. These relations offer roughly 6.9k missing-value fill opportunities each, but V15 demonstrates that successful reconstruction does not improve target QWK. Earlier semantic counts, ontologies, routing paths, and deterministic repairs were mostly alternative views of existing information. Their value is inductive bias, not new information.

### Train/test split and survey-design structure

A fresh three-fold LightGBM adversarial probe on a fixed 24k-row sample yields AUC `0.501515` for raw predictors, `0.501329` for design variables, and `0.501431` for ID alone. There is no measurable train/test covariate shift in this test. ID’s Spearman correlation with source row order is `0.00137`, and no legitimate ID-order signal is evident. Survey year target means span about `1.591–1.615` and tail rates `4.34–5.24%`; month and quarter effects are small and already present as raw states. Survey weight has an extreme univariate train/test total-variation score because it is near-continuous, not because an adversarial model can exploit a split shift. It remains numeric, with no population-quota calibration.

The strongest single-variable conditional-entropy signals are general health, depression/anxiety frequency, income-to-poverty, food security, functional limits, work-limited status, memory/errand/social difficulty, income category, and marital status. These align with the same-survey literature. They still leave heavy overlap between classes.

## Train-only error forensics

The strongest available row-aligned historical OOF is Recovery, QWK `0.481867`, continuous-score RMSE `0.515267`, and squared-label error sum 42,126 after its saved fold-pure decisions.

| Transition family | Rows | Squared-label loss | Fraction of observed weighted loss |
|---|---:|---:|---:|
| `1 → 2` | 10,745 | 10,745 | 25.51% |
| `2 → 1` | 15,929 | 15,929 | 37.81% |
| false tail: true 1/2 → pred 3/4 | 6,703 | 10,720 | 25.45% |
| missed tail: true 3/4 → pred 1/2 | 1,709 | 3,947 | 9.37% |
| `3 ↔ 4` | 785 | 785 | 1.86% |

Thus 1↔2 contributes **63.32%**, all non-tail↔tail transitions **34.82%**, and 3↔4 only **1.86%**. Threshold-free AUC is `0.765796` for 1 versus 2, `0.872638` for non-tail versus tail, and `0.538005` for 3 versus 4. Class 1/2 overlap is the dominant correctable-looking loss; class 3/4 separation is close to random but affects few rows.

False-tail errors are concentrated among respondents with high depression frequency, poor general health, and food insecurity, but their subgroup mean residuals are close to zero. Recovery is not applying a simple global over-severity offset; it lacks reliable person-level discrimination inside burden strata. This makes a hand correction or class-weight increase unsafe.

### Protective / “resilience” hypothesis

Within folds, respondents were matched on four coarse burden variables before comparing marital status, home ownership, investment income, recent work, and paid sick leave. Raw target associations survive in sensible directions: for example, the matched marital contrast changes target mean by roughly `-0.158` and tail rate by `-0.0107`; home ownership changes target mean by roughly `-0.112`. However Recovery residual contrasts are small and sign-unstable: marital mean residual difference is about `+0.0032` with fold SD `0.0147`; home ownership about `-0.0063` with SD `0.0104`; investment, work, and sick leave are similarly negligible or inconsistent. These proxies are already available to the trees. Train OOF therefore rejects a dedicated, hand-scored resilience specialist. A general full-target learner may still discover interactions without hard-coding them.

## Diagnostic-package firewall

The mandated files were inspected first in the required order, all remaining files were then inspected, and every manifest SHA-256 matched. The package’s evidence tiers were internally coherent. However, it contains 3,603 binary-labeled Indonesian headline-veracity examples with string IDs, text titles, macro-F1 histories, and public scores around `.93–.96`. Competition data contain 94,663/23,666 unique numeric respondent IDs, 176 coded survey predictors, a four-level ordinal target, and QWK. ID intersection is zero and the schemas, task, metric, row counts, and historical model names are incompatible.

**Firewall result:** the package belongs to a different competition. No row-level, aggregate, “verified,” high-confidence, unresolved, or leaderboard-constrained finding is admissible for DATAQUEST 5.0. Zero hypotheses were promoted from it. All diagnoses and architecture decisions in this report come from raw competition train, historical competition OOF, or external methodological literature. This conclusion is stronger than trying to weight its tiers: even its strongest tier identifies facts about the wrong task.

## Leakage and validation audit

Historical improvements are small enough that validation design can dominate them.

- V6.1’s `0.00293` global-versus-cross-fit gap and V2’s sidecar selection show why thresholds and blend weights optimized on the same OOF labels cannot be called unbiased.
- Outer validation rows must remain outside feature typing, category maps, numeric scaling/binning, early stopping, threshold selection, blend selection, and model selection.
- Every scored outer fold needs a calibration subset drawn only from that fold’s outer-training rows. Fold-local cuts and blend weights then apply unchanged to outer validation.
- Duplicate IDs are exact and must be collapsed before folds; row weighting by duplicate count would otherwise overweight 946 respondents.
- Household IDs cross train/test and can encode survey sampling, but honest V5.1 evidence gives no useful lookup gain. IDs and household IDs are excluded from model inputs.
- Survey design variables may be legitimate features, but no sampling weight, design-cell prior, or class quota may determine final class counts.
- The final notebook will compare candidate and blend with a CatBoost anchor retrained from the same raw folds. Historical Recovery is context only and is never loaded.
- Five fixed outer folds, fold-local calibration, paired bootstrap, boundary AUCs, QWK transition loss, fold signs, and residual correlation are frozen before notebook implementation.

## Friend-pipeline comparison

Repository-wide and archive-wide searches found no friend notebook, OOF/test predictions, manifest, or score artifact. The reported public `0.48002` and claims that it collapsed special codes or dropped `survey_weight`, pseudo-stratum, and pseudo-PSU are unverified. Correlation and residual complementarity cannot be computed. Existing train evidence argues against silently adopting that preprocessing because observed special states carry class-conditional signal. The only useful future test would require row-aligned, provenance-audited friend OOF and test scores; it is outside this final independent notebook.

## Ranked remaining bottlenecks

| Rank | Bottleneck | Evidence for / against | Prior tests | Open? | Maximum plausible QWK upside | T4×2 risk |
|---:|---|---|---|---|---:|---|
| 1 | Latent 1↔2 ranking and absent subjective context | 63.32% of weighted loss; AUC `.7658`; same-survey psychometrics indicate real single-item noise. Against: many model families converge to similar residuals. | V5 narrow specialists failed; broad GBDTs strong | Full-target supervised representation remains open | about `.003–.008` in an unusually good result; expected lower | Medium |
| 2 | False-tail conditional discrimination | 25.45% of loss; tail AUC `.8726`. Against: high-burden subgroup residual means are near zero and class 4 is tiny. | V5/V6 conditional corrections weak | Only jointly with a full-target model | about `.001–.004` | Medium |
| 3 | Calibration and validation variance | Historical global optimism reaches `.00293`; Recovery gain CI includes zero. Against: V1/V2 public transfer is broadly aligned. | Many threshold/blend searches | Methodologically open, not a signal source | prevents losses; likely `<.002` gain | Low |
| 4 | Survey-state interaction representation | Distinct codes and routing are informative; full-target TabM/RealMLP untested. Against: V4 embeddings weak and trees already see the states. | V4 deep probe; V5 boundary-only neural | Yes, narrowly | about `.001–.005` | Medium |
| 5 | Protective/resilience moderation | Literature and raw matched target differences support association. Against: Recovery residual effects are tiny/sign-unstable. | GBDTs already include proxies | Dedicated head: no; implicit interaction: yes | dedicated gain near zero | Low |
| 6 | Missing-cell noise / routing reconstruction | Strong structured missingness and reconstructability. Against: V15 is decisively negative for target QWK. | V3/V15 | No for reconstruction-first | negligible | Medium |
| 7 | Exotic model capacity / geometry | Generic benchmarks suggest alternatives. Against: TabPFN, TabICL, ModernNCA, TabLDM, retrieval, and stacker are negative. | V4, V7–V14 | Routine rerun: no | low | High |
| 8 | Irreducible omitted variables and response noise | Emotional support, personality, adaptation, and exact question context are incompletely observed; single-item validity is imperfect. Against: “irreducible” cannot be measured exactly without repeats. | All | Not fixable from supplied inputs | caps all model upside | None |

### Improvement scale

Using Recovery’s OOF expected-disagreement denominator only as a local approximation, the weighted-error reduction required from public `0.48334` is:

| Target QWK | Delta | Approx. weighted-error reduction | Approx. relative reduction |
|---:|---:|---:|---:|
| `0.48850` | `0.00516` | 420 | 1.00% |
| `0.50682` | `0.02348` | 1,909 | 4.55% |
| `0.56232` | `0.07898` | 6,421 | 15.29% |
| `0.61541` | `0.13207` | 10,738 | 25.56% |

QWK’s expected-disagreement term changes with predicted marginals, so these are strategic scale estimates, not a conversion formula. Rank 4 requires a credible but nontrivial correction. `0.50682` requires a regime change larger than every verified gain since V1. Scores `.56232` and `.61541` would require removing error at a scale unsupported by the supplied variables and all train OOF evidence.

## Answers to the required scientific questions

1. **What fraction is 1↔2?** `63.32%` of Recovery’s observed squared-label loss. False/missed tail transitions together are `34.82%`; 3↔4 is `1.86%`.
2. **Are false tails systematic?** They cluster in high-burden phenotypes, but within those groups residual means are near zero. The model has uncertain individual ranking, not a correctable global severity bias.
3. **What distinguishes similar-burden respondents?** Marital/partner status, housing, investment income, employment, and leave benefits associate with the raw target after coarse burden matching. The supplied proxies explain little residual error after Recovery.
4. **Does resilience survive?** Raw association does; an incremental Recovery correction does not survive fold checks. A dedicated handcrafted score is rejected.
5. **Why did deep/foundation models fail?** V8.x includes genuine runtime compromises, but V9, V12, and V14 ran valid screens and lost boundary AUC and QWK. V4’s deep branch was diverse but weak. Lack of task adaptation explains some failures; convergence of residuals also indicates limited additional signal.
6. **Hierarchical or full target?** The evidence favors full-target probabilities. Boundary specialists and CORN have already failed; hard routing would amplify rare-tail variance.
7. **Why ordinal loss?** Order-aware cost matches long-distance penalties, but there is no train proof that a pure ordinal/QWK objective wins. Use four-class CE as the stable objective and a fixed small CDF auxiliary, then calibrate continuous expected score on fold-local labels.
8. **Unused code/design/skip signal?** Raw special states, design variables, and routing remain legitimate inputs, but previous GBDTs already use most of them. Preserve them; do not explode features or reconstruct missing cells.
9. **Can embeddings learn what trees miss?** It remains possible only because full-target TabM/RealMLP was never tested. V4/V5 make the prior modest, so the candidate must prove complementarity under paired OOF.
10. **Best expected-value candidate?** Full-target TabM with native raw-state categories, robust numeric PLE, CE plus small ordinal-CDF auxiliary, blended only if it clears strict gates against a from-scratch CatBoost anchor.

Supporting machine-readable evidence is in `research/final_evidence/`. It is research support only and is not an input to the final notebook.
