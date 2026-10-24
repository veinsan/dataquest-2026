# DATAQUEST 5.0 final architecture decision

**Research cutoff:** 2026-09-10  
**Task:** four-level ordinal prediction, `life_satisfaction ∈ {1,2,3,4}`  
**Primary metric:** Quadratic Weighted Kappa (QWK)  
**Implementation status:** design only. A later implementation must build a clean notebook from this specification; no existing draft should be treated as approved.

## 1. Final recommendation

Build a **from-scratch raw-state ordinal ensemble with one full-target TabM branch**:

1. a compact, reliable GBDT anchor made from LightGBM, XGBoost, and CatBoost regressors;
2. one cumulative LightGBM ordinal sidecar producing a coherent expected score from the three events `Y>1`, `Y>2`, and `Y>3`;
3. one full-target TabM probability model trained with four-class cross-entropy plus a fixed small cumulative-distribution penalty;
4. fold-local, nested selection of the cumulative-sidecar weight, TabM weight, and three ordinal cuts;
5. automatic rejection of TabM unless it improves untouched outer folds consistently, with the GBDT ordinal anchor retained as the control.

This is the best expected-value final experiment, not a predicted breakthrough. The remaining error is dominated by weak person-level ranking between classes 1 and 2, while the tested hand-built, reconstruction, neighborhood, stacking, and foundation-model corrections have failed. Full-target TabM is one of the few material formulations that is still open: V5 tested TabM only as a binary boundary-1 specialist, not as a joint four-class model.

The evidence does **not** support expecting QWK above `0.50682`. A favorable outcome may reach the high `.48s`; a stable gain of `.001–.003` over a matched anchor is more plausible than a gain of `.02+`. The current Recovery submission remains the operational benchmark and should not be displaced if the new branch fails the frozen gates.

## 2. Evidence rules and firewall

The conclusions use three distinct evidence classes:

- **Primary competition evidence:** raw released train/test files and recomputed train-only or row-aligned OOF statistics.
- **Historical competition evidence:** executed notebooks and their lower-level CSV/JSON/NPZ/log artifacts. These establish what was tried, but historical threshold or blend selection can be optimistic.
- **External evidence:** original papers, official repositories/model cards, and authoritative survey documentation. This informs priors and mechanisms; it does not supply respondents, labels, coefficients, priors, or attainable QWK.

### Diagnostic-package disposition

The required package under `research/ground_truth_final_package/` was read in the mandated order, then every remaining file was inspected. All manifest hashes matched and its internal evidence tiers were understandable. It is nevertheless a package for a different task: 3,603 Indonesian headline-veracity examples, string identifiers, binary labels, and macro-F1 histories. DATAQUEST 5.0 contains numeric respondent IDs, 176 coded survey predictors, four ordinal labels, and QWK. The two ID spaces have zero overlap and no shared schema, target, model output, or error geometry.

**Firewall decision: zero package-derived hypotheses are admitted.** No row, label, tier, disagreement, public-score constraint, class count, model output, text phenotype, or aggregate from that package may influence the eventual pipeline. The architecture, gates, subgroup definitions, and expected performance below come only from released competition train data, competition OOF, and external literature. The final implementation must not read or mention the package.

## 3. Exhaustive repository reconstruction

Eighteen experiment notebooks were inspected cell by cell: V1, V2, V3, V4, V5, V5.1, V6, V6.1, V7, V8, V8.1, V8.2, V8.3, V9, V11, V12, V14, and V15. Every saved code cell has an execution count and no notebook contains a Jupyter `error` output object. Several experiments caught failures in ordinary output or deliberately skipped later stages; an “executed notebook” therefore does not imply that every defined branch produced predictions.

All material result CSV/JSON/NPZ/log/manifest files were inspected, including shapes, folds, OOF/test columns, thresholds, confusion matrices, correlations, bootstrap outputs, feature records, model metadata, and branch gates. Binary model artifacts were checked only for metadata/configuration. The two handoff archives were used as indices and reconciled with primary files. The September 8 handoff is stale for V12: primary artifacts prove that ModernNCA passed its runtime gate and then failed both quality folds. The large `.fr-PIIJV3` archive duplicates/indexes the result tree and adds no independent experiment.

### Historical experiment ledger

| Version | Tested hypothesis and representation | Honest result or strongest screen | Failure mode / interpretation | Final verdict |
|---|---|---|---|---|
| **V1** | Raw codes, sentinel flags, 11 fold-safe repairs, semantic counts; LGB/XGB/Cat regression | 5-fold grouped/stratified; global OOF `.480481`, fold-pure cuts `.479662`; documented public `.48003` | Component QWKs cluster around `.4773–.4780`; caught LightGBM CUDA failure followed by fallback | Proven baseline and credible CV→public transfer |
| **V2** | Raw, semantic, nonstandard-code, categorical, monotonic, multiclass, and cumulative-ordinal GBDT views | 2×5 grouped; core CF `.479735`; final CF `.481367`, global `.481587`; documented public `.48193` | Cumulative sidecar gain `.001087`; XGB-cat gain `.000545`; correlations `.9927–.9975`; later evidence shows selection optimism risk | Genuine small improvement; strongest reproducible model family |
| **V3** | Questionnaire ontology, routing/applicability paths, ordinal target encoding, dual structural models | Ordinal-TE probe `+.000640`; final CF `.479948`, unique-ID `.480846` | Ontology/path bundles neutral or negative; skip-rule artifact effectively empty | Additive structure engineering mostly falsified |
| **V4** | All-categorical CatBoost, deep embedding/multitask model, retrieval, fingerprint | Three probes: all-cat blend `+.002925`; deep `+.000935`; retrieval `-.000177`; fingerprint `-.001059` | None met preregistered breakthrough gate; no five-fold candidate | Weak diversity exists, no robust breakthrough |
| **V5** | Binary `1` versus `>1` CatBoost, TabM-PLE, RealMLP, cross-family specialists | Three folds; TabM blend `+.000586` with 1/3 wins; RealMLP `-.000455`; specialist AUCs `.7757–.7801` | Fold signs reverse; full run correctly skipped | Boundary-specialist formulation rejected; full-target TabM/RealMLP remains open |
| **V5.1** | Household/quasi-key lookup and repeated-respondent aggregation | Base `.478556`; lookup `.478569`; gain `.000014` | OOF household coverage only `5.2%`; exact-hash test coverage zero | Scientifically negligible; useful submission-contract audit |
| **V6** | Error difficulty, distillation, weighting, CORN | Anchor `.479158`; CORN `+.000106`; distillation `-.001327`; difficulty weighting `-.001559` | Error difficulty AUC `.623` did not translate into QWK | Tested weighting and CORN correction rejected |
| **V6.1** | Core/all-cat/recovery/calibration portfolio | Core CF `.478899`; final CF `.478832`; global `.481761` | Global optimism gap `.002930`; fold choices unstable | Calibration/selection artifact, no honest gain |
| **V7** | Full-target TabPFN classifier and regressor | Probe blend gains `+.000259/+.000804`; anchor-only final `.481448` vs anchor `.481716` | Residual correlation ≈`.992`; boundary AUC declined; full candidate file empty | Reject routine TabPFN rerun |
| **V8** | Conditional foundation specialist with frozen tails | Only TabPFN fold 0 produced quality metrics: QWK `-.003216`, AUC12 `-.002992` | EXAONE revision failure; TabICL memory failure; full tables empty | Scientific and operational failure |
| **V8.1** | EXAONE environment repair | No candidate prediction | Virtual-environment preflight failed; saved outputs equal anchor | Engineering failure only |
| **V8.2** | Pinned EXAONE install repair | No candidate prediction | Install/preflight failed; saved outputs equal anchor | Engineering failure only |
| **V8.3** | EXAONE T4 attention repair | Tiny `128×101` smoke test passed in 2.42 s | Real fold estimated 33.41 GB against 13.96 GB; `.npz`-named artifact is a JSON failure record | Quality remains untested, but final-day formulation is impractical |
| **V9** | Full-target TabICLv2 regression | Folds 0/2: mean QWK `-.009541`, AUC12 `-.008910` | Residual correlation `.988417`; full stage skipped | Decisive negative transfer |
| **V10** | xRFM mixed representation with 12 fixed configurations | No valid outer prediction | Cholesky failure after ~9.5 h; earlier API/kernel, 641.55 GiB AGOP, and Jacobian-axis failures | Scientific quality unknown; operational risk unacceptable |
| **Recovery** | Convex blend of V2, V5.1, V6.1, and V7 prediction streams | OOF `.481867`; paired historical gain about `.000939`, 4/5 positive, bootstrap P(gain>0) ≈`.913`; documented public `.48334` | Test score correlation with V2 ≈`.99932`; only 558/23,666 labels changed | Current champion; small diversity/calibration gain rather than new signal regime |
| **V11** | Cumulative-logit stacker over 43 historical streams | `.478662`, delta `-.003205`; all five folds negative; bootstrap 95% `[-.004889,-.001462]` | Residual correlation `.998834`; duplicated fallback streams present | Historical prediction stacking falsified |
| **V12** | ModernNCA learned geometry | Folds 0/2: mean QWK `-.010239`, AUC12 `-.004337` | Runtime did succeed; residual correlation `.992318` | Scientifically negative; handoff blocker claim obsolete |
| **V14** | Pretrained Xiaomi TabLDM synthetic-SCM regression | Folds 0/2: mean QWK `-.009724`, AUC12 `-.007480` | 14.25 GiB on T4; residual correlation `.991629` | Scientifically negative and resource-tight |
| **V15** | Fold-safe probabilistic reconstruction of four survey fields, 54 posterior features | Reconstruction succeeds, but target QWK `.475999` (`-.005868`); best Recovery blend `-.000640`; gated blend `-.000848` | All standalone folds negative; residual correlation `.997682` | Missing-cell reconstruction is not the target bottleneck |

The only locally documented public scores are V1 `.48003`, V2 `.48193`, and Recovery `.48334`. No raw leaderboard export exists, so they remain documented rather than independently re-queried.

### Defined versus actually run

- V4–V6 ran probes and correctly skipped full candidate CV after failed gates.
- V7 defined a full TabPFN path, but its full-fold table is empty.
- V8 obtained one TabPFN fold; V8.1/V8.2 obtained no candidate prediction; V8.3 passed only a tiny smoke test.
- V9 ran two screen folds and stopped.
- V10 never produced a trustworthy outer prediction.
- V11 ran its meta-model but no submission after the negative result.
- V12 and V14 passed runtime checks, ran two quality folds, and stopped after strong losses.
- V15 ran five candidate folds and blend tests; only its Recovery control was exported.

### Friend-pipeline evidence

No friend notebook, aligned OOF/test prediction, manifest, result directory, handoff member, or raw record of the reported `.48002` exists. Its claimed special-code collapsing and feature drops cannot be verified, and residual complementarity cannot be computed. The raw train audit argues against copying that preprocessing: distinct response states have different target distributions. The friend pipeline is excluded until provenance-audited, row-aligned OOF and test scores exist.

## 4. Raw dataset audit

### Shape, identity, and duplicates

- Raw train: `95,609 × 179`; raw test: `23,902 × 178`; sample: `23,902 × 2`.
- Predictors after removing target, `id`, and `household_id`: **176**.
- Unique train IDs: **94,663**. Unique test IDs: **23,666**.
- Exactly 946 train IDs and 236 test IDs occur twice. Each duplicate pair is value-identical; train targets are consistent.
- Train/test ID overlap is zero. Distinct IDs do not share identical full predictor rows, and no full predictor hash crosses train/test.
- Train contains 59,016 households; 11,620 household IDs cross train/test. Within-train household label concordance is `46.12%` versus `45.76%` under the class prior, only `+0.36` percentage points. Prior honest household lookup was negligible.
- ID versus source-row-order Spearman correlation is `.00137`; no legitimate ordering signal was found.

Raw release hashes:

| File | SHA-256 |
|---|---|
| `train_final.csv` | `0ccc7f7aa3d9e7520d1edd0cae30c1ac119fcb4c508fdd7c9ebc871ceaee2056` |
| `test_final.csv` | `c7027da38aba915e6a9952d21c6cd960b22c172e0ccf3f9f570d63d07e008740` |
| `data_dictionary.csv` | `4ee1269e4e10fd62225916bb8eff1cd51930a211cb8488d6c97a5b4817e4f631` |
| `sample_submission.csv` | `3b27035c9910f24c7a0cf29f85c1382be423f58cda0524bbd7b0a24ab9e65abf` |

The unique-train class distribution is:

| Class | Rows | Share |
|---:|---:|---:|
| 1 | 42,552 | 44.951% |
| 2 | 47,713 | 50.403% |
| 3 | 3,511 | 3.709% |
| 4 | 887 | 0.937% |

Class 4 has fewer than 900 unique respondents, so tail class weights, isotonic calibrators, and direct mini-batch kappa objectives have high variance.

### Submission contract

The supplied sample has 23,902 sequential placeholder IDs and does not match `test_final.id`; only 4,518 values overlap incidentally. Historical submission artifacts reveal two conventions: V1 wrote 23,902 sample rows, while V2 and every later Recovery-family submission write **23,666 unique `test_final.id` values in first-occurrence order**. The documented best public score belongs to the latter lineage. The later implementation must therefore:

1. assert duplicate test rows are identical;
2. predict once per unique test respondent;
3. emit 23,666 unique test IDs in first-occurrence order;
4. use the sample only to validate the two output column names and integer target type;
5. record the sample/competition-contract mismatch explicitly in its run audit.

### Feature typing and response states

The data are a high-confidence match to the redesigned National Health Interview Survey Sample Adult family for 2021–2024. The target coding exactly matches the official four-category life-satisfaction item. This is provenance/semantic evidence, not permission to retrieve public respondent microdata. No predictor name contains another life-satisfaction outcome or obvious companion target.

Of 176 predictors, 168 have at most 32 observed states. The eight higher-cardinality fields are `survey_weight` (85,872), `income_to_poverty_ratio` (1,084), `weight_pounds` (203), `pseudo_psu` (111), `days_missed_work` (95), `hours_worked_per_week` (94), `age` (70), and `pseudo_stratum` (52). Survey weight is continuous; pseudo-PSU and pseudo-stratum are design categories.

The local dictionary has 187 entries, including eight variables absent from the CSVs, and does not fully document observed states such as `7/8/9/90/97/99/140/990/999`. Those numbers are feature-specific: some are refused/don’t-know/not-ascertained states, while others are legitimate detailed categories. Blanket sentinel-to-NaN conversion is invalid.

The response states also carry train signal. For example, `depression_frequency=8`, `=9`, and actual NaN have materially different target means/tail rates; `food_security_4cat=8` also differs from actual NaN. The architecture must preserve the exact feature/value identity of every low-cardinality state.

### Missingness and routing

Mean cell missingness is `14.405%`. Per-row missing-count quantiles at minimum/1%/25%/50%/75%/99%/maximum are `4/10/18/24/33/45/60`. Mean missing predictor counts rise by target: `24.60`, `25.68`, `28.92`, and `29.79` for classes 1–4.

Co-missingness shows survey routing rather than random deletion:

- flu-vaccine month/year correlation ≈`.848`;
- hours worked/full-time ≈`.829`;
- missed-work/paid-sick-leave ≈`.818`;
- multiple plan-1 insurance fields ≈`.813–.818`.

Class-conditional missingness spreads reach about `.26` for some functional/employment fields and `.24–.25` across insurance routes. Missingness is useful state information, but V15 proves that reconstructing the missing values is not itself useful for target QWK. Preserve state; do not build a reconstruction stage.

### Redundancy

Observed mappings are nearly deterministic:

| Source → destination | Support | Accuracy |
|---|---:|---:|
| survey year → data year | 80,169 | 100% |
| interview month → quarter | 80,112 | 100% |
| food-security 4-cat → 3-cat | 79,993 | 100% |
| cancer-count → cancer-ever | 80,110 | 100% |
| children in family → household | 79,928 | 99.864% |
| detailed → coarse marital state | 79,995 | 99.671% |
| smoking status → ever smoked | 80,225 | 99.849% |
| e-cigarette status → ever used | 79,879 | 99.922% |

These mappings create alternative views, not information. V3/V15 show that extra repairs and reconstruction do not improve the target reliably. The final design retains supplied raw fields but avoids another feature-expansion layer.

### Time and train/test shift

Unique-row target means by survey year are `1.5914`, `1.6100`, `1.6154`, and `1.6066` for 2021–2024; tail rates are `5.24%`, `4.49%`, `4.50%`, and `4.34%`. Quarter means span `1.6006–1.6144`; month means span `1.5954–1.6233`. These modest context effects justify retaining year/month/quarter, not constructing a temporal prior.

A fresh bounded adversarial validation used 12k unique train plus 12k unique test rows, three folds, seed 20260910, and a 150-tree LightGBM classifier:

| View | Train-vs-test AUC |
|---|---:|
| Raw predictors excluding IDs | `.501515` |
| Survey-design variables | `.501329` |
| ID only | `.501431` |

There is no measurable broad split shift in this screen. `survey_weight` has a misleadingly high univariate total-variation value because it is nearly unique; the multivariate adversarial test does not exploit it. Keep it numeric, but never use it as a sample weight or population-class quota.

### Strong train signal and its limit

The lowest single-feature conditional entropies belong to general health, depression/anxiety frequency, income-to-poverty, food security, work limitation, memory/errand/social difficulty, income category, and marital state. These agree with survey literature. They still leave heavy overlap, particularly within classes 1 and 2.

## 5. Train-only OOF error forensics

The row-aligned Recovery OOF contains one prediction per unique respondent. It has QWK `.481867`, expected-score RMSE `.515267`, accuracy `.621066`, and squared-label error sum 42,126. Its weights/cuts were historically selected, so the absolute QWK is diagnostic rather than a fresh unbiased estimate; the rowwise transition geometry remains useful.

### Weighted transition decomposition

| Transition family | Rows | Squared-label loss | Fraction of weighted loss |
|---|---:|---:|---:|
| `1 → 2` | 10,745 | 10,745 | 25.51% |
| `2 → 1` | 15,929 | 15,929 | 37.81% |
| False tail: true 1/2 → predicted 3/4 | 6,703 | 10,720 | 25.45% |
| Missed tail: true 3/4 → predicted 1/2 | 1,709 | 3,947 | 9.37% |
| `3 ↔ 4` | 785 | 785 | 1.86% |

Direct 1↔2 errors contribute **63.32%** of weighted loss. Non-tail↔tail transitions contribute **34.82%**. Direct 3↔4 contributes only **1.86%**.

Threshold-free AUCs reinforce the diagnosis:

- class 1 versus 2: `.765796`;
- non-tail versus tail: `.872638`;
- class 3 versus 4: `.538005`.

The main potentially learnable bottleneck is subtle 1↔2 ordering. Tail membership is substantially better ranked, although false-tail discretization remains expensive. Class 3 versus 4 is near chance but too sparse to justify a dedicated final-day model.

### Margin and high-burden geometry

The 10% of respondents closest to a Recovery cut have `49.87%` classification error and weighted error `.582` per row; the farthest 10% have `15.78%` error and weighted error `.203`. This supports continuous score calibration and uncertainty audits, not row-specific overrides.

A target-free high-burden phenotype was defined for analysis as at least three of: `general_health_status ∈ {4,5}`, `depression_frequency ∈ {1,2}`, `anxiety_frequency ∈ {1,2}`, `food_security_4cat ∈ {3,4}`, `worry_about_medical_costs = 1`, `work_limited_by_health = 1`, and `difficulty_social_activities = 1`. It contains 10,210 respondents (`10.79%`) but accounts for:

- 4,636/6,703 false-tail rows (`69.16%`);
- 7,990/10,720 false-tail weighted loss (`74.53%`).

Its actual tail rate is `24.49%`, Recovery error rate `62.96%`, and weighted error `.969` per row. Yet mean `target − score` residual is only `+.0029`. Recovery is not applying a simple global over-severity offset; it lacks reliable person-level ranking within a highly overlapping phenotype. Hand-lowering this subgroup would exchange false tails for missed tails and is not justified.

### Protective/resilience check

Fold-repeated exact-burden-stratum comparisons held general health, depression frequency, food security, and medical-cost worry fixed, then compared candidate protective factors. Marriage and home ownership retain real raw-target associations:

- married versus not: target-mean difference ≈`-.158`, tail difference ≈`-1.07` percentage points;
- home ownership: target-mean difference ≈`-.112`, tail difference ≈`-.69` percentage points.

Investment income, recent work, and paid leave have smaller associations. However Recovery residual contrasts are tiny and sign-unstable across folds: approximately `+.0032` for marriage and `-.0063` for home ownership, with fold ranges crossing zero. The trees already consume these fields.

**Conclusion:** protective context is epidemiologically plausible and visible in the target, but a missing broad “resilience representation” is not supported by OOF residuals. Do not create a resilience score, specialist gate, class correction, or subgroup threshold. Let the full-target model learn interactions softly and require outer-fold evidence.

## 6. Literature synthesis

### Survey provenance and target behavior

- The [official 2021 NHIS Sample Adult codebook](https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/NHIS/2021/Adult-codebook.pdf) documents the matching four-category life-satisfaction item and distinct refused/not-ascertained/don’t-know states. This supports feature-specific state preservation.
- [CDC/NCHS life-satisfaction decision documentation](https://omb.report/icr/202303-0920-002/doc/129730201.pdf) and [Hoyer, Ochiai & Blakey (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548803/) describe question-order/context effects: asking after health questions can shift respondents toward health-oriented judgments. This supports retaining survey context and expecting irreducible response noise.
- [Miller et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/37565865/) analyzed 2021 NHIS and identified psychological distress, unmarried status, poor general health, weak support, and food insecurity as major correlates. It used a binary outcome and does not provide a four-class QWK benchmark.
- [Barger, Donoho & Wayment (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6036920/) found social relationships add explanatory value beyond health and socioeconomic status. This motivates soft interaction capacity, not a hand-built proxy.
- [Cheung & Lucas (2014)](https://pubmed.ncbi.nlm.nih.gov/24890827/) found substantial but imperfect validity for single-item life-satisfaction measurement. Missing personality, emotional-support detail, adaptation, and question context constrain predictability.

No same-input, same-split, four-class QWK study was found. Literature cannot supply a numerical ceiling or a defensible claim that `.50682` is reachable.

### Method/model comparison

| Method | Primary source | Relevance | Internal evidence against it | Verdict |
|---|---|---|---|---|
| Raw-state CatBoost/LGB/XGB | [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), [LightGBM](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [XGBoost](https://www.kdd.org/kdd2016/subtopic/view/xgboost-a-scalable-tree-boosting-system/670/) | Strong, reliable mixed-table anchor; categorical states and nonlinear splits fit this dataset | V1/V2 show diminishing returns, but they remain the only repeatedly positive family | **Required anchor** |
| Cumulative/all-threshold ordinal heads | [Frank & Hall (2001)](https://doi.org/10.1007/3-540-44795-4_13), [Pedregosa et al. (2017)](https://www.jmlr.org/beta/papers/v18/15-495.html) | Three ordered binary events provide a coherent expected score and cheap objective diversity | V6 CORN neutral; V2 cumulative sidecar only modestly positive | **Keep one compact sidecar** |
| CORAL/CORN neural heads | [CORAL](https://github.com/Raschka-research-group/coral-pytorch), [CORN](https://doi.org/10.1007/s10044-023-01181-9) | Enforce or construct rank-consistent ordinal probabilities | V6 direction essentially neutral; rare later boundaries have few examples | Do not make the primary architecture |
| Squared cumulative-distribution loss | [Hou, Yu & Samaras (2016)](https://arxiv.org/abs/1611.05916) | Penalizes probability mass by ordinal distance at negligible extra cost | It is not QWK and can over-smooth class 4 | **Use only as fixed 0.10 auxiliary beside CE** |
| Direct differentiable kappa | [de la Torre, Puig & Valls (2018)](https://doi.org/10.1016/j.patrec.2017.05.018) | Metric-aligned in principle | Mini-batch marginal denominator is unstable under severe class imbalance; prior calibration optimism is material | Reject as primary loss |
| Full-target TabM | [Gorishniy, Kotelnikov & Babenko, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html), [official repository](https://github.com/yandex-research/tabm) | Parameter-efficient MLP ensemble; one-hot categorical states plus PLE numerics; fits 95k-row scale without row attention | V5 boundary-only TabM was weak/unstable; V4 generic deep branch small | **Best remaining supervised candidate** |
| Full-target RealMLP | [Holzmüller, Grinsztajn & Steinwart, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html) | Strong defaults on 1k–500k-row tables and often complementary to GBDT | V5 boundary RealMLP was negative and weaker than TabM; larger dependency surface | Scientifically open, second choice |
| Mitra-v2 fine-tuning | [technical report](https://arxiv.org/abs/2609.04540), [official model card](https://huggingface.co/autogluon/mitra-classifier-2) | Synthetic-only, Apache-2.0, ~77M-parameter row-column model; most different prior; observed classifier revision `edada0d20759c58ada8c8605c25f22f6e98ea5f0` | Six-day-old release, no independent replication, eight-bag 50-step recipe, unproven T4 reliability; other foundation transfers failed | Bounded future screen, not final default |
| TabPFN/TabICL | [TabPFN](https://www.nature.com/articles/s41586-024-08328-6), [TabICLv2](https://arxiv.org/abs/2602.11139) | Strong generic foundation-model benchmarks | V7/V9 reduce QWK and boundary AUC on this exact task | Reject routine rerun |
| ModernNCA / retrieval / TabLDM | Their official method families plus local artifacts | Alternative learned geometry | V4/V12/V14 consistently negative with high residual correlation | Reject |
| Masked reconstruction / VIME | [VIME](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf) | Could learn questionnaire co-state | V15 reconstructs fields successfully but lowers QWK | Reject reconstruction-first design |
| Historical prediction stacking | Generic stacking literature | Could combine residual diversity | V11 is decisively negative; streams are extremely correlated | Reject |
| Hard burden/resilience mixture | Survey association literature | Targets concentrated false-tail subgroup | OOF subgroup mean residual is calibrated; prior specialists fail | Reject hard routing and manual correction |

The [TabArena 2025 benchmark](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html) supports a portfolio prior: GBDTs remain strong, supervised deep models become useful with adequate data/budget, and generic foundation-model rank does not guarantee transfer. It does not predict this competition’s QWK.

## 7. True remaining bottlenecks

| Rank | Bottleneck | Evidence for | Evidence against | Still open? | Plausible upside | T4×2 risk |
|---:|---|---|---|---|---:|---|
| 1 | Latent 1↔2 ranking | 63.32% of weighted loss; AUC `.7658`; single-item/context noise literature | Many model families converge to correlated residuals | Full-target supervised state interaction is narrowly open | Expected `0–.003`; exceptional `.005–.008` | Medium |
| 2 | High-burden false-tail discrimination | 10.8% subgroup contains 74.5% of false-tail weighted loss | Mean residual nearly zero; hard correction trades error direction | Joint probabilistic model only | `0–.002` incremental | Medium |
| 3 | Calibration/selection variance | Historical global optimism up to `.00293`; Recovery gain CI crosses zero | V1/V2 public transfer broadly follows CV | Yes as evaluation discipline, not new information | Avoids losses; unlikely >`.002` | Low |
| 4 | Survey-state interaction representation | Distinct codes/routing are informative; full-target TabM untested | Trees already see most states; V4/V5 deep probes weak | Yes, modest prior | `0–.004` | Medium |
| 5 | Protective/resilience context | Literature and matched raw-target contrasts | Residual contrasts tiny/sign-unstable | Dedicated component: no | Near zero as separate head | Low |
| 6 | Missing-value reconstruction | Strong routing and redundant items | V15 decisive target failure | No | Negligible | Medium |
| 7 | Exotic capacity/geometry | Generic benchmark possibility | TabPFN, TabICL, NCA, TabLDM, retrieval, stacker all negative | Routine rerun: no | Low expected | High |
| 8 | Omitted subjective covariates/label noise | Emotional support, personality, adaptation, and exact context incompletely observed | Exact irreducible fraction cannot be measured | Not recoverable from supplied inputs | Defines ceiling | None |

### Required error reductions

Using Recovery OOF’s expected-disagreement denominator as a local approximation only:

| Target QWK | Delta from `.48334` | Approx. weighted-error reduction | Approx. relative reduction |
|---:|---:|---:|---:|
| `.48850` | `.00516` | 420 | 1.00% |
| `.50682` | `.02348` | 1,909 | 4.55% |
| `.56232` | `.07898` | 6,421 | 15.29% |
| `.61541` | `.13207` | 10,738 | 25.56% |

QWK’s chance-agreement denominator changes with predicted marginals, so these are scale estimates rather than a scoring conversion. Rank 4 is difficult but plausible under a favorable correction. `.50682` requires a regime change larger than every verified improvement since V1. `.56232` and `.61541` are unsupported by the available signal and experiments.

## 8. Candidate comparison

Four genuinely different choices were compared. Scores run from 1 (poor) to 5 (strong); expected upside is a prior, not a result.

| Candidate | Expected QWK upside | Orthogonality | Train support | Literature support | Leakage safety | Compute feasibility | Final-day reliability | Total / 35 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A. Raw-state GBDT ordinal anchor + full-target TabM** | 3 | 3 | 4 | 5 | 5 | 4 | 4 | **28** | **Choose** |
| B. Same anchor + full-target RealMLP | 3 | 3 | 3 | 5 | 5 | 4 | 3 | 26 | Second-ranked future experiment; do not substitute in this run |
| C. Mitra-v2 four-class fine-tune | 4 | 5 | 1 | 3 | 4 | 1 | 1 | 19 | Too risky for final default |
| D. Burden-gated hierarchical specialists | 2 | 3 | 1 | 3 | 5 | 4 | 3 | 21 | Reject: OOF does not support the gate |

Candidate A wins because it preserves the proven GBDT regime, tests the one supervised formulation left open by V5, avoids pretrained-data provenance ambiguity, and already has a known T4-compatible official implementation. It does not claim that TabM alone will beat trees. Its purpose is to earn a small blend weight through untouched OOF or be discarded.

## 9. Exact recommended architecture

### 9.1 Data unit and excluded fields

- Collapse exact duplicate respondent IDs after asserting equality of every predictor and train target.
- Work with 94,663 unique train respondents and 23,666 unique test respondents.
- Exclude `id` and `household_id` from every model input.
- Do not use duplicate count, household lookup, survey weight as a sample weight, external respondents, pseudo labels, or test-derived class proportions.
- Keep `survey_weight` as an ordinary numeric predictor. Keep pseudo-PSU and pseudo-stratum as ordinary categorical predictors.

### 9.2 Common raw-state representation

Use all 176 raw predictors. Do not perform feature selection, target encoding, imputation-model training, ontology expansion, or target-driven interaction generation.

**Categorical/state view**

- Treat every low-cardinality survey item as a feature-specific categorical token.
- Give actual NaN its own token.
- Preserve every observed special code as its exact token; never merge `7`, `8`, `9`, `90`, `97`, `98`, `99`, and related states globally.
- Treat `pseudo_psu` and `pseudo_stratum` as categories.
- Category vocabularies are learned on each fit partition only; outer-validation/test unknowns map to a reserved unknown token.

**Numeric view**

Use numeric copies for `survey_weight`, `income_to_poverty_ratio`, `age`, `height_inches`, `weight_pounds`, `hours_worked_per_week`, `days_missed_work`, `emergency_room_visits`, `years_in_current_home`, family/household adult and child counts, and `cancer_types_count` when present.

- Fit median and IQR on the fit partition only; clip standardized values to `[-12,12]` for TabM.
- For continuous fields with documented special states, retain a categorical state token and mask only the numeric copy. Fixed state sets: weight `{996,997,999}`, height `{96,97,98,99}`, age `{97,99}`, hours worked `{990}`, and missed-work days `{997,998,999}`.
- Add one total raw-missing-count numeric feature. Do not add hundreds of pairwise flags or missing-pattern IDs.

### 9.3 GBDT core anchor

Train three regressors on the same fold partitions and average their continuous scores equally:

| Model | Frozen starting configuration | Representation |
|---|---|---|
| LightGBM regressor | RMSE; up to 4,500 trees; lr `.025`; 63 leaves; min child 80; feature fraction `.80`; bagging fraction `.85`; L1 `.05`; L2 `.50`; early stop 180 | Nonnegative fold-fit category indices with explicit NaN/unknown states; raw numeric view |
| XGBoost regressor | squared error; histogram tree method with native categorical handling enabled; up to 4,000 trees; lr `.025`; depth 7; min child weight 8; subsample `.85`; colsample `.80`; L1 `.05`; L2 `1.0`; early stop 180 | Same fold-fit categorical dtypes/state indices and numeric view |
| CatBoost regressor | RMSE; up to 4,000 trees; lr `.035`; depth 8; L2 5; early stop 180; no file writes | Native string categories and numeric view |

These are fixed starting budgets, not a search space. If a library/version requires a syntactic parameter adjustment, preserve the stated learning problem and record the change. Do not run HPO.

The **core score** is the equal mean of the three predictions. Equal weighting is fixed before validation to avoid spending calibration labels on highly correlated core models.

### 9.4 Cumulative ordinal sidecar

Train three LightGBM binary classifiers on `Y>1`, `Y>2`, and `Y>3` using the same state-index/numeric view:

- binary log loss;
- up to 3,000 trees, lr `.025`, 31 leaves, min child 100, feature fraction `.80`, bagging fraction `.85`, early stop 150;
- no oversampling, focal loss, or inverse-frequency weighting;
- project predicted cumulative probabilities to a nonincreasing sequence if numerical crossing occurs.

The sidecar expected score is:

`1 + P(Y>1) + P(Y>2) + P(Y>3)`.

Within each outer fold’s calibration subset, select sidecar weight `a` from `{0, .15, .30, .45}` in:

`anchor_score = (1-a) × core_score + a × cumulative_score`.

Tie-break toward smaller `a`. Learn the three cuts jointly on the same calibration subset. This sidecar is retained because V2 supplied the only repeated positive ordinal-objective evidence; it is not treated as a breakthrough by itself.

### 9.5 Full-target TabM branch

Use official `tabm==0.0.3` and `rtdl_num_embeddings==0.0.12`, trained from random initialization. No pretrained weights are involved.

| Item | Frozen value |
|---|---|
| Output | four logits per parameter-efficient ensemble member |
| Ensemble members `k` | 16 |
| Backbone | 2 MLP blocks, width 256, package default activation/dropout unless API requires explicit values |
| Numeric encoding | fold-fit piecewise-linear embeddings, 32 bins, embedding width 12 |
| Categorical encoding | native TabM one-hot state input; no claim of learned entity embeddings |
| Batch size | 1,024; reduce once to 768 only on actual CUDA OOM |
| Optimizer | AdamW, lr `.002`, weight decay `.0003` |
| Epoch budget | 70; patience 8 |
| Precision | CUDA mixed precision, finite-loss checks each batch |
| Primary loss | mean four-class cross-entropy over all 16 members |
| Ordinal auxiliary | `0.10 ×` mean squared difference between predicted and one-hot cumulative distributions over the first three cuts |
| Class weighting | none |
| Early-stop metric | calibration CE plus the same fixed cumulative penalty, never calibration QWK per epoch |

Average softmax probabilities across the 16 members. The TabM continuous score is the probability expectation over class values 1–4.

The cumulative penalty is deliberately small. Cross-entropy supplies stable probabilities and all class information; the auxiliary discourages long-distance mass. Direct kappa, focal weighting, hard tail routing, and custom ranking pairs add instability without train support.

### 9.6 Final candidate blend

On the fold-local calibration subset, select TabM weight `b` from `{0, .10, .20, .30, .40, .50, .65, .80, 1.0}`:

`candidate_score = (1-b) × anchor_score + b × tabm_expected_score`.

For every `b`, fit three ordered cuts with a minimum separation of `.025`. Select the highest calibration QWK and tie-break toward smaller `b`. Record every tried weight and cut set. Apply the chosen weight/cuts unchanged to that outer validation fold.

No second grid, class-weight sweep, loss-weight sweep, seed sweep, subgroup correction, or global OOF threshold rescue is allowed.

## 10. Frozen validation contract

### 10.1 Splits

1. Deduplicate by respondent ID after equality assertions.
2. Create five deterministic outer `StratifiedKFold` splits with shuffle and seed `20260910`.
3. Stratify on target × survey year. If any combined stratum has fewer than five rows, back off only that split definition to target alone and record it.
4. Inside each outer-training partition, reserve 12.5% with one deterministic stratified shuffle split, seed `20261000 + fold`, for preprocessing-independent early stopping, component weighting, and cuts.
5. Fit category maps, numeric scales/bins, and every model only on the remaining fit partition.
6. Never use the outer validation labels for model, feature, epoch, weight, cut, or branch selection.

Household grouping is not required because household ID is excluded, household concordance is only marginally above chance, and train/test households overlap. Respondent duplicate isolation is mandatory.

### 10.2 Stage-S stop rule

TabM is the only expensive novel branch. Predeclare outer folds **0 and 2** as Stage S. The GBDT anchor may continue for all five folds regardless; train TabM on folds 1, 3, and 4 only if all Stage-S conditions hold:

- mean candidate-blend QWK delta versus anchor ≥ `0.0000`;
- neither fold delta < `-0.0025`;
- mean 1↔2 AUC delta ≥ `-0.0020`;
- predictions are finite, probability rows sum to one within `1e-5`, and residual correlation is below `0.9995` or the candidate changes at least 0.5% of fold labels.

This screen is permissive enough not to require a lucky early gain but stops another clearly negative foundation/deep transfer. If it fails, do not try RealMLP or Mitra in the same run.

### 10.3 Required outer metrics

For anchor, TabM, and candidate blend, save per fold and pooled fold-pure:

- QWK and accuracy;
- expected-score RMSE;
- AUC for class 1 versus 2, non-tail versus tail, and class 3 versus 4;
- 4×4 confusion matrix;
- weighted transitions: `1→2`, `2→1`, false tail, missed tail, `3↔4`, distance-2, and distance-3;
- score correlation and residual correlation versus anchor;
- prediction changes by transition;
- performance by survey year, raw-missing-count quintile, threshold-margin decile, and the fixed high-burden audit group;
- test score/probability quantiles by fold for shift monitoring, without using them to tune labels or quotas.

### 10.4 Paired uncertainty

After all five candidate folds, run 2,000 respondent-level paired bootstrap replicates with seed `20264000` on:

`QWK(candidate fold-pure labels) − QWK(anchor fold-pure labels)`.

Report mean delta, median delta, `P(delta>0)`, and the 2.5/97.5% interval. Bootstrap the fixed OOF decisions; do not re-optimize weights or cuts inside bootstrap replicates.

### 10.5 Final promotion gates

Promote TabM into the deployment ensemble only if **all** conditions hold:

1. pooled fold-pure candidate QWK gain over anchor ≥ `.0015`;
2. at least 4 of 5 fold deltas are strictly positive;
3. worst fold delta ≥ `-.0015`;
4. paired bootstrap `P(delta>0) ≥ .90`;
5. pooled 1↔2 AUC delta ≥ `-.0010`;
6. false-tail weighted loss ≤ `1.03 ×` anchor false-tail weighted loss;
7. no year subgroup loses more than `.004` QWK when it contains all four classes;
8. all integrity, finite-value, and probability checks pass.

If any gate fails, the scientific conclusion is “full-target TabM did not earn inclusion.” Retain the GBDT anchor artifacts as a control and keep the existing Recovery submission rather than spending the last submission on the failed candidate.

## 11. Fold calibration and test aggregation

Do not optimize one global OOF threshold and report it as unbiased. Preserve each fold’s independently learned score scale with **cut normalization**:

1. Within fold `f`, calibration produces ordered cuts `c1,c2,c3` for its selected score.
2. Map that fold’s score piecewise linearly so `c1→1.5`, `c2→2.5`, and `c3→3.5`; extend below/above using the slope of the nearest interval.
3. The outer-fold hard label is therefore identical to applying its original cuts, while normalized scores share a common scale.
4. Apply the same fold-specific component weights, cuts, and normalization to that fold’s test predictions.
5. Average normalized test scores across the five fold models.
6. Apply fixed cuts `[1.5,2.5,3.5]` once to the averaged normalized score.

This uses only fold-training calibration labels, avoids a median-threshold scale assumption, and requires no global OOF refit. If TabM is rejected, perform the same aggregation with the GBDT anchor scores/cuts.

The deployment ensemble remains an average of fold models trained on roughly 70% of unique train rows after outer validation and inner calibration. Do not retrain on all rows unless a separate calibration method is specified; otherwise retraining changes the score scale without independent cuts.

## 12. Resource and reliability contract

### T4×2 schedule

- GPU 0: TabM for the current fold.
- GPU 1: CatBoost, then XGBoost for the same fold.
- LightGBM: CPU concurrently if the installed GPU backend is unreliable; otherwise GPU 1 when free.
- Never shard one model across both GPUs. Parallelize independent models/folds only.
- Release model objects and CUDA cache after every fold; keep only predictions and small manifests.

### Single-GPU fallback

Run CatBoost → XGBoost → TabM sequentially on GPU 0, with LightGBM on CPU. The learning problem, seeds, partitions, and model budgets remain unchanged. A no-GPU session should stop immediately with a clear accelerator message rather than silently attempting a multi-hour CPU TabM run.

### Memory/runtime expectations

- Expected T4×2 wall time: roughly 3–5 hours if Stage S passes; shorter if TabM stops early.
- Expected one-T4 wall time: roughly 6–9 hours.
- TabM should remain comfortably below 16 GB with batch 1,024. On verified CUDA OOM, clear cache and retry once at batch 768; any further OOM is a hard failure.
- Pin TabM/rtdl numeric-embedding versions exactly. Record Python, CUDA, PyTorch, CatBoost, LightGBM, XGBoost, pandas, NumPy, SciPy, and scikit-learn versions.
- No external model weights or data downloads are needed. Internet is required only if the two small pinned packages are absent.

## 13. Later implementation outputs

The separate implementation task must save, from the clean raw-data run:

- unique-respondent OOF probabilities/scores/classes for every component and selected blend;
- outer-fold assignments and inner fit/calibration counts;
- per-fold and pooled metric table;
- the full calibration grid with component weights, cuts, and selected rows;
- confusion/weighted-transition table;
- subgroup and margin audit table;
- paired-bootstrap distribution summary;
- test continuous scores and TabM class probabilities per fold and averaged;
- raw input hashes;
- feature typing, category cardinalities, numeric columns, special-state sets, seeds, best iterations/epochs, library versions, device use, and runtimes;
- one concise run summary with Stage-S and final gate decisions;
- a submission only under the unique-ID contract, with 23,666 rows, first-occurrence test-ID order, integer labels in `{1,2,3,4}`, finite predictions, and exact expected columns.

The implementation must read only the four competition CSVs plus files it creates in its own working directory. It must not load prior OOF/test predictions, prior folds, prior thresholds, prior submissions, handoff files, or any file from the diagnostic package.

## 14. Rejected directions

The following are closed for the last submission unless materially new train evidence appears:

- row/household/quasi-key lookup;
- class quotas or survey-weighted prior matching;
- global special-code collapse;
- missing-cell reconstruction, VIME, or imputation pretraining;
- ontology/path/large aggregate feature expansion;
- hard burden or resilience routing;
- boundary-1-only specialists;
- CORN as the main model;
- error-difficulty weighting and distillation;
- historical prediction stacking;
- TabPFN or TabICL reruns;
- ModernNCA, Xiaomi TabLDM, retrieval/fingerprint reruns;
- EXAONE and xRFM final-day engineering;
- direct mini-batch QWK optimization;
- Mitra-v2 without an isolated future runtime/quality screen.

These rejections are formulation-specific. They do not claim that every future version of each model family is universally poor; they state that another final-day attempt has lower expected value than the chosen experiment.

## 15. Answers to the ten deciding questions

1. **What fraction of weighted error is 1↔2?** `63.32%`; false/missed tail contributes `34.82%`; 3↔4 contributes `1.86%`.
2. **Are false tails systematic?** They concentrate in high-burden rows, but mean residual is calibrated. The problem is individual uncertainty, not a safe global correction.
3. **What distinguishes similar-burden respondents?** Marital/partner status, home ownership, investment income, employment, and benefits associate with raw satisfaction, but most incremental residual signal is already captured.
4. **Does resilience survive OOF?** The raw association survives; a separate Recovery correction does not. Reject a hand-built resilience component.
5. **Why did previous deep/foundation models fail?** Some EXAONE/xRFM runs were engineering failures, but TabICL, ModernNCA, TabLDM, and deep/retrieval screens also lost valid QWK/AUC. Lack of task adaptation explains only part; limited extra signal and inductive-bias mismatch are the stronger conclusion.
6. **Hierarchical or full-target?** Full-target probabilities are better supported. Prior specialists/CORN failed, and hard routing amplifies rare-tail variance.
7. **Why use an ordinal loss?** A small cumulative auxiliary respects distance and costs almost nothing. There is no support for replacing stable CE/regression with direct QWK; fold-local cuts remain necessary.
8. **Is unused signal present in special codes/design/skip patterns?** Those states are useful and must be preserved, but prior trees already exploit most of them. The open question is whether TabM learns joint state interactions, not whether missingness exists.
9. **Can a supervised model learn what trees miss?** Possibly, because full-target TabM was never tested. V4/V5 make the prior modest, so outer-fold complementarity must be demonstrated.
10. **Which candidate has the best expected value?** A proven raw-state GBDT ordinal anchor plus one full-target TabM branch with strict nested admission gates.

## 16. Expected outcome and decision after the run

Interpret the future result conservatively:

- **Stage S fails:** stop TabM; no architecture rescue. Keep Recovery.
- **Full candidate gain `<.0015` or unstable:** reject TabM even if one global threshold looks better. Keep Recovery.
- **Gain `.0015–.003`, stable and gate-compliant:** a credible small advance; inspect public submission as one final independent test.
- **Gain `.003–.006`, stable:** strong result with a realistic chance to approach/cross rank 4.
- **Gain `>.01`:** treat as suspicious until every fold, preprocessing boundary, duplicate, target proxy, and calibration path is re-audited.
- **Claimed path to `>.50682`:** unsupported before the run and still requires a very large, stable OOF gain. No current paper, benchmark, subgroup, or historical experiment justifies expecting it.

The final research conclusion is therefore narrow: **preserve the survey states, keep the proven tree regime, and spend the remaining experiment budget on one honest full-target TabM complementarity test.** If it cannot beat the matched anchor without calibration leakage, the evidence favors accepting the current ceiling rather than inventing another high-variance correction.
