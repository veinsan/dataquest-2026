# DATAQUEST 5.0 final architecture decision

**Research cutoff:** 2026-09-10  
**Correct diagnostic source:** `research/oqdq5_output/`  
**Task:** four-level ordinal prediction, `life_satisfaction ∈ {1,2,3,4}`  
**Primary metric:** Quadratic Weighted Kappa (QWK)  
**Status:** frozen research and implementation specification. No model or submission was produced in this phase.

## 1. Executive decision

Build a **from-scratch V2-family ordinal GBDT control plus one full-target TabM ordinal-regression branch**:

1. an equal-weight LightGBM, XGBoost, and CatBoost regression core;
2. one cumulative LightGBM sidecar for `Y>1`, `Y>2`, and `Y>3`;
3. one randomly initialized TabM model with exact survey-state tokens, piecewise-linear numeric embeddings, and a single scalar output trained by mean squared error independently over its parameter-efficient ensemble members;
4. fold-local selection of the cumulative-sidecar weight, TabM weight, and three ordered QWK cuts on an inner calibration split;
5. untouched outer-fold promotion gates that automatically remove TabM if it does not add stable ranking signal.

This replaces the earlier cross-entropy-first TabM draft. The corrected diagnostic bundle is itself a LightGBM probability reconstruction followed by exact class-count constraints. Its expected score is extremely close to the historical GBDT family, its train-only analogue adds essentially no Recovery residual signal, and forcing known train class counts lowers QWK by about `.013`. Independently, the executed V2 multiclass expected-value branch underperformed the regression and cumulative branches. The final neural experiment must therefore produce an unconstrained scalar ordinal score; it must not imitate the reconstruction's probability-plus-quota mechanism.

The remaining open hypothesis is narrow. V4 already tested a full-target embedding MLP with regression and cumulative heads, obtaining only a `+.000935` mean blend gain with one win in three probes. V5 tested TabM only for binary class `1` versus `>1`, obtaining `+.000586` with one win in three. The new information is TabM's parameter-efficient internal ensemble and numeric feature embeddings applied to the whole ordinal target. The target formulation alone is not new.

A stable gain of `.001–.003` over the matched GBDT control is plausible; `.005–.008` would be exceptional. The evidence does not support expecting `0.50682`, and it provides no defensible path to `0.56232` or `0.61541`. The existing Recovery submission at public `.48334` remains the operational fallback.

## 2. Evidence hierarchy and diagnostic firewall

The architecture follows this evidence order:

1. **Released competition train data and leakage-safe OOF evidence** decide what may enter the pipeline.
2. **Executed historical notebooks and row-aligned artifacts** establish what was tested, what actually ran, and which gains were ranking gains versus calibration or selection noise.
3. **`research/oqdq5_output/`** is a diagnostic oracle of limited reliability. Official aggregate source/count evidence is strong; its per-row test labels are a model-derived candidate and are not ground truth.
4. **Primary external sources** supply survey semantics, model mechanisms, and broad priors. They supply no respondents, labels, coefficients, target priors, or expected leaderboard score.

The diagnostic directory is analysis-only. Nothing in it may be used as a feature, label, pseudo-label, sample weight, threshold, cut initializer, class quota, calibration target, blend weight, model-selection score, test constraint, row rule, ID rule, or submission edit. The official aggregate class totals are also forbidden from the eventual pipeline. A diagnostic observation is admitted to the design only when released-train OOF independently supports it.

### 2.1 Correct-package provenance and schema

The directory contains exactly nine artifacts. It has no manifest, generator, model configuration, fold record, seed, package lock, calibration report, or direct source-row join.

| Artifact | Rows / structure | Meaning established by audit | Reliability |
|---|---:|---|---|
| `README.txt` | 491 bytes | Declares NHIS Sample Adult 2021–2024, four-class life satisfaction, aggregate totals, and candidate-only row labels | Useful provenance claim; explicitly disclaims exact row labels |
| `reconstruction_report.json` | JSON | Names `LSATIS4_A` and 2021 `LSATIS4R_A`; calls the method “LightGBM probability ranking + exact official class-count constraints” | Clear method summary, incomplete reproduction record |
| `class_count_constraints.csv` | 4 rows | Year totals, full totals, released-train counts, arithmetic test remainder, and assigned candidate counts | Aggregate totals independently corroborated; forbidden for modeling |
| `ground_truth_evidence.csv` | 23,666 rows | `id`, four probabilities, raw argmax, constrained label, assigned-label confidence, raw probability margin, quota-adjustment flag | Source-model output, not factual confidence |
| `ground_truth_unique.csv` | 23,666 rows | One constrained candidate label per unique test ID plus diagnostics | Model-derived candidate labels |
| `ground_truth.csv` | 23,902 rows | Candidate labels expanded to raw test rows, including exact duplicate IDs | Correct raw-row alignment, candidate labels only |
| `duplicate_test_ids.csv` | 472 rows | The 236 duplicated test IDs with all released predictors | Matches released exact duplicates |
| `duplicate_train_targets.csv` | 1,892 rows | The 946 duplicated train IDs and their consistent targets | Matches released train duplicates |
| `submission_ground_truth_candidate.csv` | 23,902 rows | Candidate labels mapped to sample-template IDs | Invalid for this competition contract: only 4,518 incidental ID joins and 19,384 blank labels |

The audited SHA-256 hashes are:

| Artifact | SHA-256 |
|---|---|
| `README.txt` | `c83fdcfdcf9ba0754420bbfaea1260efa60dc3d84827e72fa1ff8e309ba382b7` |
| `reconstruction_report.json` | `7d01f067c0146aed07e734356d09346ab9684c6f030ac258dcc811274088fc1a` |
| `class_count_constraints.csv` | `c894e331fb7926750068dcba5ba36e740156108bfc6a6b7c1a7f0d9edc9dfbd6` |
| `ground_truth_evidence.csv` | `d2f5a95b4f1a08400cd7da1291d34242d08e548fa752d6d1dda14f3e25f9a221` |
| `ground_truth_unique.csv` | `3ffbe4d9401d1901358bc50342c008531042ddec441f9bb1bd66d505fdad01cb` |
| `ground_truth.csv` | `f864c12effd707388ff809b7e3682d79ae0673f0ee4e4271659f3fafc967821f` |
| `duplicate_test_ids.csv` | `45c8f39f184146aae6ab9818718714ec96559ac420abd9e7ed4ed13522c97ce6` |
| `duplicate_train_targets.csv` | `c8fdfad8ce9c3e206e0a53fe370ef1977b460bcd961f6b13f95b58fe5de56b38` |
| `submission_ground_truth_candidate.csv` | `741b1c1bd8f826bcb675f5c58ba1cc55391c73fb15a0738c7990883560152f31` |

The annual counts in the bundle match the official 2021, 2022, 2023, and 2024 Sample Adult codebooks: class totals over the four years are `53,190 / 59,641 / 4,389 / 1,109`. The released unique-train counts are `42,552 / 47,713 / 3,511 / 887`, so the arithmetic remainder is `10,638 / 11,928 / 878 / 222`. This validates the source family and aggregate arithmetic, not any row assignment. Primary references: [2021 codebook](https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/NHIS/2021/Adult-codebook.pdf), [2022 codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/adult-codebook.pdf), [2023 codebook](https://ftp.cdc.gov/pub/health_Statistics/NCHs/Dataset_Documentation/NHIS/2023/Adult-codebook.pdf), and [2024 codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2024/Adult-codebook.pdf).

### 2.2 Alignment and meaning checks

- Unique train and test IDs are disjoint; their union is exactly the integers `1…118,329` with no gaps. This supports a randomized split from the declared full record universe.
- `ground_truth.csv` has exactly the raw test length and exact raw test ID order. Duplicate IDs receive consistent candidate labels.
- The unique/evidence files contain exactly the unique test-ID set but use their own order; they must be joined by `id`, never aligned positionally.
- The four probability columns sum to one with maximum numerical error below `6e-16`; `raw_argmax` is exact.
- `margin` is the difference between the largest and second-largest source-model probabilities.
- `confidence` is the probability assigned to the **constrained** label. On quota-adjusted rows it is often not the maximum probability and must not be interpreted as probability that the candidate label is true.
- No direct CDC microdata key, row hash, or cryptographic match proves any individual label.
- ID is randomized: adjacent train IDs have essentially zero target correlation, adjacent-label agreement equals the prior expectation, and ID-only train/test adversarial AUC is `.5014`. The contiguous full ID universe is not a feature opportunity.

### 2.3 Admissibility decision

Three levels must remain separate:

- **Verified:** source family, target code family, full aggregate counts, released-row duplicate structure, and arithmetic count remainder.
- **Observed source-model behavior:** probabilities, margins, raw argmax, and which rows the constraint optimizer moved.
- **Unverified:** every constrained per-row candidate label and every error claim that uses it as truth.

The package is useful for finding model-family disagreement and for falsifying unsafe ideas. It cannot establish hidden-test accuracy, QWK, class transitions, or subgroup residual direction. Its strongest architectural contribution is negative: it shows why another LightGBM probability model, exact quota, or test-side tail correction is not a new signal source.

## 3. Repository and historical experiment reconstruction

Eighteen experiment notebooks were inspected cell by cell: V1, V2, V3, V4, V5, V5.1, V6, V6.1, V7, V8, V8.1, V8.2, V8.3, V9, V11, V12, V14, and V15. Saved outputs, caught exceptions, conditional skips, result tables, NPZ/JSON metadata, OOF/test streams, thresholds, confusion matrices, correlations, and resource records were reconciled. A populated execution count does not imply that a conditional candidate branch ran.

### 3.1 V1 through latest experiment ledger

| Version | Tested hypothesis | Honest result / strongest valid screen | Interpretation and disposition |
|---|---|---|---|
| **V1** | Raw/sentinel/semantic ordinal LGB, XGB, Cat regressors | Fold-pure `.479662`; public `.48003` | Proven baseline and credible OOF→LB transfer |
| **V2** | Representation-diverse GBDT plus multiclass and cumulative objectives | Core `.479735`; final `.481367`; public `.48193` | Cumulative LGB added `.001087`; XGB-categorical added `.000545`; strongest reproducible family |
| **V3** | Questionnaire ontology, routes, aggregates, ordinal target encoding | Final `.479948`; unique-ID `.480846`; best small probe `+.000640` | Large additive semantic engineering did not survive |
| **V4** | All-categorical CatBoost, retrieval, fingerprint, and full-target embedding MLP with regression+cumulative heads | All-cat blend `+.002925` with 3/3 wins; deep blend `+.000935` with 1/3; retrieval `-.000177`; fingerprint `-.001059` | No full five-fold promotion. Generic full-target embedding/multitask learning is already weakly tested |
| **V5** | Binary `1` versus `>1` CatBoost, TabM-PLE, RealMLP, and specialist ensemble | TabM blend `+.000586`, 1/3 wins; RealMLP `-.000455`; specialist AUC `.7757–.7801` | Boundary specialist rejected. Full-target TabM scalar regression remains untested |
| **V5.1** | Household/quasi-key lookup | `+.000014`; OOF household coverage `5.2%`; zero exact-hash test coverage | Closed |
| **V6** | Difficulty weighting, distillation, CORN | CORN `+.000106`; distillation `-.001327`; weighting `-.001559` | Closed as correction mechanisms |
| **V6.1** | Core/all-cat/sidecar portfolio and calibration | Core `.478899`; final `.478832`; global `.481761`; all-cat standalone `.478443`, selected in only 1/5 folds | Global optimism `.002930`; V4 all-cat signal did not become a stable full result |
| **V7** | Full-target TabPFN classifier/regressor | Small probes, but final anchor `.481448` versus `.481716`; candidate full table empty | No gain; routine rerun closed |
| **V8** | Conditional foundation specialist | Only one TabPFN fold yielded quality: QWK `-.003216`, AUC12 `-.002992` | EXAONE revision and TabICL memory failures; no viable branch |
| **V8.1–V8.3** | EXAONE environment and T4 repair | Tiny `128×101` smoke test only; full fold estimated 33.41 GB against 13.96 GB | Operationally closed for T4×2 |
| **V9** | Full-target TabICLv2 regression | Folds 0/2 mean QWK `-.009541`, AUC12 `-.008910` | Decisive negative transfer |
| **V10/xRFM** | Mixed representation with 12 configurations | No valid outer prediction after API/kernel, 641.55 GiB AGOP, Cholesky, and Jacobian failures | Quality unknown; final-day operational risk unacceptable |
| **Recovery** | Convex blend of V2/V5.1/V6.1/V7 streams | OOF `.481867`; about `+.000939`, 4/5 positive, bootstrap `P(>0)≈.913`; public `.48334` | Current champion; 558/23,666 test labels changed versus V2, score correlation ≈`.99932` |
| **V11** | Stacker over 43 historical streams | `.478662`, `-.003205`; all five folds negative | Correlated historical prediction stacking closed |
| **V12** | ModernNCA learned neighborhood | Folds 0/2 mean `-.010239`; AUC12 `-.004337` | Scientifically negative |
| **V14** | Pretrained Xiaomi TabLDM regression | Folds 0/2 mean `-.009724`; AUC12 `-.007480`; 14.25 GiB | Scientifically negative and resource-tight |
| **V15** | Fold-safe reconstruction of four survey fields and 54 posterior features | Standalone `.475999` (`-.005868`); best Recovery blend `-.000640`; every fold negative | Missing-cell reconstruction is not the target bottleneck |

No V13 notebook/result exists. V10 produced no reliable quality result. The only documented public scores with local artifacts are V1 `.48003`, V2 `.48193`, and Recovery `.48334`; no leaderboard export is present.

### 3.2 Ranking gain versus calibration or noise

- The V1→V2 OOF and public improvements have the same direction and similar magnitude. V2's cumulative sidecar also improves threshold-free boundary AUC slightly. This is the clearest genuine ranking improvement.
- V6.1 reports global `.481761` but fold-pure `.478832`. That `.002930` gap is calibration/model-selection optimism, not a new representation.
- Recovery is a small but credible operational gain: 4/5 historical folds improve and public rises by `.00141`, yet its test score is almost identical to V2. It is diversity at the margin, not a regime change.
- V4's all-categorical CatBoost had three positive probes, but V6.1 later obtained a weaker standalone full branch and selected it once. It remains a possible tiny sidecar, not the best final experiment.
- V4's deep MLP and V5's TabM specialist establish a low prior for a neural breakthrough. They do not test TabM scalar full-target regression.
- Valid full-target foundation/geometry screens—TabICLv2, ModernNCA, and TabLDM—lowered both QWK and 1↔2 AUC. Their failure was scientific, not only operational.

### 3.3 What was actually executed

- V4–V6 ran multi-fold probes and correctly skipped conditional full candidates.
- V7's full foundation table is empty.
- V8 obtained one TabPFN quality fold; V8.1/V8.2 obtained no candidate prediction; V8.3 obtained a smoke test only.
- V9, V12, and V14 ran their declared two-fold quality screens and stopped after losses.
- V10 never produced an auditable outer prediction.
- V11 ran its stacker but no candidate submission.
- V15 ran all five candidate folds and blend tests; only the Recovery control was exported.

### 3.4 Friend/reference solution

No friend notebook, aligned OOF/test prediction, manifest, result directory, handoff member, or raw record of the reported public `.48002` is present. Claims about its special-code collapse and feature drops cannot be verified. Distinct response states have different released-train target distributions, so that preprocessing should not be copied. The friend pipeline contributes no admissible model input or architecture decision.

## 4. Released dataset diagnosis

### 4.1 Unit, target, and duplicates

- Raw train: `95,609 × 179`; raw test: `23,902 × 178`; sample template: `23,902 × 2`.
- After excluding target, `id`, and `household_id`, there are 176 predictors.
- Unique train respondents: **94,663**; unique test respondents: **23,666**.
- Exactly 946 train IDs and 236 test IDs occur twice. Each pair is value-identical; train targets agree.
- Train/test ID overlap is zero. No distinct respondent shares a complete predictor hash, within or across splits.
- Class counts after deduplication are `42,552 / 47,713 / 3,511 / 887`, or `44.951% / 50.403% / 3.709% / .937%`.

Class 4 has fewer than 900 unique respondents. Direct class weighting, tail specialists, flexible calibration per tail, and mini-batch kappa losses have high variance.

Raw release hashes:

| File | SHA-256 |
|---|---|
| `train_final.csv` | `0ccc7f7aa3d9e7520d1edd0cae30c1ac119fcb4c508fdd7c9ebc871ceaee2056` |
| `test_final.csv` | `c7027da38aba915e6a9952d21c6cd960b22c172e0ccf3f9f570d63d07e008740` |
| `data_dictionary.csv` | `4ee1269e4e10fd62225916bb8eff1cd51930a211cb8488d6c97a5b4817e4f631` |
| `sample_submission.csv` | `3b27035c9910f24c7a0cf29f85c1382be423f58cda0524bbd7b0a24ab9e65abf` |

### 4.2 Submission unit

The sample contains sequential placeholder IDs and does not align to `test_final.id`; only 4,518 values overlap incidentally. The best-scoring V2/Recovery lineage writes one row per unique actual test ID. A later implementation must assert exact duplicate equality, predict once per unique respondent, preserve first-occurrence unique test-ID order, and output exactly 23,666 rows. The sample is used only for column names and target dtype.

### 4.3 Feature semantics and state preservation

The predictors strongly match the 2021–2024 NHIS Sample Adult survey. Of 176 predictors, 168 have at most 32 observed states. Higher-cardinality fields are `survey_weight`, `income_to_poverty_ratio`, `weight_pounds`, `pseudo_psu`, `days_missed_work`, `hours_worked_per_week`, `age`, and `pseudo_stratum`.

The local dictionary is incomplete and sometimes semantically wrong. It omits observed states such as `7/8/9/90/97/98/99/140/990/999`. More materially, it describes `difficulty_social_activities` as `1=yes, 2=no`, while the official 2021 source coding is `1=no difficulty`, `2=some difficulty`, `3=a lot of difficulty`, and `4=cannot do at all`; `7/9/90` are separate response states. This caused an earlier burden audit and V5's simple “count code 1” functional feature to point in the wrong direction. The corrected architecture preserves raw feature/value identity and does not create that count.

Special values are feature-specific. `depression_frequency=8`, `=9`, and actual NaN have different target distributions; the same is true for other response fields. Global sentinel-to-NaN conversion is invalid.

### 4.4 Missingness, routing, and redundancy

Mean cell missingness is `14.405%`. Row missing-count quantiles at minimum/1%/25%/50%/75%/99%/maximum are `4/10/18/24/33/45/60`. Mean missing counts rise from `24.60` in class 1 to `29.79` in class 4.

Co-missingness reflects questionnaire routes: flu-vaccine month/year correlation is about `.848`, hours/full-time `.829`, missed-work/paid-sick-leave `.818`, and several plan-1 insurance pairs `.813–.818`. Missing state is predictive. V15 shows that reconstructing missing cells is not predictive enough to help the target.

Several supplied fields are deterministic or near-deterministic views of one another: year/recode, month/quarter, food-security 4/3 category, cancer count/ever, children family/household, detailed/coarse marital state, and smoking/ecigarette status/ever. They may offer convenient split geometry but not new information. V3 and V15 reject another large repair or expansion layer.

### 4.5 Train/test shift and latent structure

Target means by survey year are `1.5914`, `1.6100`, `1.6154`, and `1.6066`; tail rates are `5.24%`, `4.49%`, `4.50%`, and `4.34%`. Keep year/month/quarter as context features; do not construct year priors.

A fresh bounded adversarial validation—12k unique train plus 12k unique test rows, three folds, fixed 150-tree LightGBM—gave AUC `.501515` on raw predictors, `.501329` on survey-design fields, and `.501431` on ID alone. There is no broad shift to correct. `survey_weight` remains an ordinary predictor and must never become a sample weight or population quota.

Household IDs overlap across train/test, but within-train household label concordance is only `46.12%` versus a `45.76%` prior. Historical lookup adds `+.000014`. There is no useful respondent graph or stable latent household embedding.

The strongest single-variable signals are general health, depression/anxiety frequency, income-to-poverty, food security, work limitation, functional difficulty, income category, and marital state. They are broad latent health/mental-health/SES axes, but classes 1 and 2 overlap heavily within them.

## 5. Legitimate train-OOF error geometry

The row-aligned unique-respondent Recovery OOF is the authoritative forensic reference. Its absolute score reflects historical model/threshold selection, but every error below uses released labels.

Recovery OOF has QWK `.481867`, accuracy `.621066`, expected-score RMSE `.515267`, and hard-label squared-error sum `42,126`. Fold QWK ranges from `.468064` to `.496010`, which is large relative to all verified gains and justifies paired fold and bootstrap gates.

### 5.1 Confusion matrix and weighted transitions

| True \ predicted | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|
| **1** | 31,036 | 10,745 | 717 | 54 |
| **2** | 15,929 | 25,852 | 5,454 | 478 |
| **3** | 176 | 1,198 | 1,646 | 491 |
| **4** | 141 | 194 | 294 | 258 |

| Transition family | Rows | Squared-label loss | Share of weighted loss |
|---|---:|---:|---:|
| `1 → 2` | 10,745 | 10,745 | 25.51% |
| `2 → 1` | 15,929 | 15,929 | 37.81% |
| False tail: true 1/2 → predicted 3/4 | 6,703 | 10,720 | 25.45% |
| Missed tail: true 3/4 → predicted 1/2 | 1,709 | 3,947 | 9.37% |
| `3 ↔ 4` | 785 | 785 | 1.86% |

Direct 1↔2 mistakes contribute **63.32%** of weighted loss. Cross-tail mistakes contribute **34.82%**. Direct 3↔4 contributes only **1.86%**.

Threshold-free AUCs are `.765796` for class 1 versus 2, `.872638` for non-tail versus tail, and `.538005` for class 3 versus 4. The largest learnable opportunity is fine ordering inside the 1/2 majority. Tail membership is better ranked, while class 3 versus 4 is sparse and close to chance.

### 5.2 QWK marginal geometry and the false-tail trap

Recovery predicts class counts `47,282 / 37,989 / 8,111 / 1,281`, compared with actual `42,552 / 47,713 / 3,511 / 887`. It intentionally predicts 2.31 times as many class-3 rows and 1.44 times as many class-4 rows. This is not sufficient evidence of bad calibration: QWK's expected-disagreement denominator changes with predicted marginals.

Assigning the Recovery score to the **known train class counts by rank** raises accuracy from `.621066` to `.653930` and reduces mean squared hard-label error from about `.4450` to `.3920`, but QWK falls from `.481867` to `.468942`. Applying the same exact-count rank rule to the V2 multiclass expected score gives `.467826`. Therefore:

- class-count matching is not QWK calibration;
- reducing apparent false tails can lower QWK;
- no official or reconstructed test quota belongs in inference;
- the final model must improve ordering and let train-only QWK cuts choose marginals.

### 5.3 Boundary confidence and ambiguity

The 10% of rows nearest a Recovery cut have `49.87%` classification error and weighted error `.582` per row. The farthest 10% have `15.78%` error and weighted error `.203`. The errors are concentrated in overlap regions, not isolated IDs. A continuous score plus honest cut calibration is appropriate; row-specific patches are not.

### 5.4 Corrected high-burden interaction

For diagnosis only, define burden count from seven released, target-free states:

- `general_health_status ∈ {4,5}`;
- `depression_frequency ∈ {1,2}`;
- `anxiety_frequency ∈ {1,2}`;
- `food_security_4cat ∈ {3,4}`;
- `worry_about_medical_costs = 1`;
- `work_limited_by_health = 1`;
- `difficulty_social_activities ∈ {2,3,4}` under the official code semantics.

The corrected high-burden group, burden count at least three, contains 11,766 train respondents (`12.43%`). Its actual tail rate is `22.52%`, while Recovery predicts tail for `62.80%`. Among its actual non-tail rows, `55.63%` are false tails. It contains `75.65%` of all false-tail rows and `79.74%` of false-tail weighted loss, yet only `25.49%` of total weighted loss.

The mean `target − Recovery score` residual in the group is `+.00042`: the score is calibrated on average. The model cannot distinguish which high-burden respondents report satisfaction, rather than applying a uniformly excessive severity offset.

Conditional false-tail rates among actual non-tail rows rise sharply with burden count: `.049%`, `1.39%`, `11.99%`, `37.37%`, `70.04%`, and `90.47%` for counts 0 through 5. This is an interaction/heterogeneity problem, not a safe gate.

### 5.5 Protective context and missing signal

Within repeated exact burden strata, marriage and home ownership retain raw target associations: target-mean differences are about `-.158` and `-.112`, with tail-rate differences around `-1.07` and `-.69` percentage points. Investment income, work, and paid leave are smaller protective correlates.

Recovery residual contrasts are tiny and sign-unstable: about `+.0032` for marriage and `-.0063` for home ownership, with fold ranges crossing zero. Existing trees already consume these fields. The train data supports soft interaction learning, not a resilience score, manual subgroup correction, special threshold, or mixture gate.

## 6. Correct diagnostic-package forensics

### 6.1 How the candidate labels were made

The source LightGBM raw argmax counts are `9,985 / 13,482 / 137 / 62`. The constraint stage changes them to `10,638 / 11,928 / 878 / 222`.

Exactly 1,590 of 23,666 rows (`6.72%`) are quota-adjusted. The transitions are:

| Raw argmax → constrained label | Rows |
|---|---:|
| `1→1` | 9,984 |
| `1→4` | 1 |
| `2→1` | 654 |
| `2→2` | 11,928 |
| `2→3` | 776 |
| `2→4` | 124 |
| `3→3` | 102 |
| `3→4` | 35 |
| `4→4` | 62 |

Of the 1,100 constrained tail labels, 936 (`85.09%`) were moved away from the source argmax. Mean assigned-label confidence is `.673` on untouched rows and `.363` on adjusted rows. The tail is consequently the least trustworthy part of the candidate, and the single raw `1→4` move illustrates that quota optimization can create a long transition unsupported by argmax.

The constrained label has Spearman correlation `.8819` with source expected score. It is a rank-and-quota product, not a direct microdata label.

### 6.2 It does not rank historical submissions reliably

Diagnostic QWK against the constrained candidate is:

| Historical labels | Diagnostic QWK | Predicted counts 1/2/3/4 |
|---|---:|---:|
| V1 | `.854509` | 11,538 / 9,893 / 1,910 / 325 |
| V2 | `.855884` | 11,276 / 10,006 / 2,086 / 298 |
| V3 | `.858636` | 11,010 / 10,187 / 2,126 / 343 |
| V5.1 | `.861582` | 10,860 / 10,431 / 1,952 / 423 |
| V6 | `.853765` | 11,452 / 9,886 / 1,995 / 333 |
| V6.1 | `.864786` | 11,103 / 10,244 / 1,944 / 375 |
| V7 | `.849924` | 11,374 / 9,836 / 2,045 / 411 |
| V8 | `.860053` | 10,714 / 10,484 / 2,123 / 345 |
| V8.1 | `.849681` | 11,518 / 9,842 / 2,059 / 247 |
| V8.2 | `.849722` | 11,343 / 9,846 / 2,129 / 348 |
| V8.3 | `.854749` | 11,328 / 9,982 / 2,033 / 323 |
| Recovery | `.850627` | 11,641 / 9,651 / 2,039 / 335 |

This ordering contradicts legitimate evidence. V6.1 is best under the diagnostic candidate despite honest fold-pure `.478832`; Recovery is worse than V2 by `.00526` despite improving public score from `.48193` to `.48334`. V1→V2 changes 878 labels, and the candidate splits them almost evenly, 433 versus 439, with six matched by neither prediction. V2→Recovery changes 558; the candidate favors V2 on 345 and Recovery on 209, with four matched by neither. On the 459 changes not touched by the quota optimizer, it favors V2 314 to 143; on the 99 adjusted changes, it favors Recovery 66 to 31. The diagnostic score mainly measures agreement with the source classifier and its quota stage.

No final architecture, weight, cut, or submission decision may use this table.

### 6.3 Source-model similarity and residual test

The source expected score correlates `.97857` Pearson and `.99016` Spearman with Recovery test score. It is close to a model family already explored.

The nearest leakage-safe train analogue supplies three independent checks:

1. V2's LightGBM multiclass expected-value branch has cross-fitted QWK `.477327`, below CatBoost regression `.479761`, the cumulative branch `.478959`, and final V2 `.481367`.
2. The difference between V2 multiclass expected score and Recovery score has Pearson correlation `.0014` and Spearman `.0051` with the true Recovery residual; a one-variable OLS correction coefficient is only `.0209`.
3. Within Recovery-predicted tails, the classifier-minus-Recovery delta has true-tail AUC `.4918`, below chance. It does not re-rank the false-tail problem.

The candidate therefore provides no independently supported classifier residual. This is why the final TabM branch uses scalar regression rather than four-class CE, CDF loss, or candidate-inspired class calibration.

### 6.4 Diagnostic disagreements and their limitation

Treating the candidate label as a temporary comparison target, Recovery's squared disagreement is 3,046:

| Candidate→Recovery transition family | Rows | Squared disagreement | Share |
|---|---:|---:|---:|
| `1→2` | 184 | 184 | 6.04% |
| `2→1` | 1,186 | 1,186 | 38.94% |
| Candidate non-tail → Recovery tail | 1,310 | 1,313 | 43.11% |
| Candidate tail → Recovery non-tail | 36 | 74 | 2.43% |
| `3↔4` | 289 | 289 | 9.49% |

The corrected high-burden test group contains 3,041 rows (`12.85%`) and captures `70.38%` of candidate-defined false-tail rows and `70.45%` of their squared disagreement. Its candidate tail rate is `33.38%`, Recovery tail rate `63.04%`, and mean `candidate label − Recovery score` residual `+.2620`.

That `+.2620` residual conflicts with the released-train OOF residual `+.00042`. It is reconstruction bias, largely produced by a source classifier that predicts few raw tails and then fills a fixed quota. It must not become a subgroup correction.

One qualitative pattern does replicate without using candidate labels: conditional false-tail rates by burden score are similar in train OOF and diagnostic comparison through the common range—approximately `.05%/.07%`, `1.39%/1.43%`, `11.99%/10.62%`, and `37.37%/35.70%` for burden 0–3. This confirms that historical models react strongly to accumulated adverse states. Train OOF simultaneously proves that the group mean score is calibrated, so the admissible response is a soft full-target interaction model with an outer-fold gate.

### 6.5 Final diagnostic verdict

The bundle helps locate disagreements, audit response-code semantics, and close unsafe count/probability ideas. It cannot tell which historical model is right per test row. The following are explicitly rejected:

- fitting or selecting against candidate QWK;
- quota matching or using the official arithmetic remainder at inference;
- lowering Recovery tails in the diagnostic high-burden group;
- treating source `confidence` as correctness probability;
- using candidate-defined hard/easy rows, IDs, transitions, or class totals;
- using the source probability vector as a feature or teacher.

## 7. Literature and model-family evidence

### 7.1 Survey and target evidence

- The official NHIS codebooks document distinct valid, refused, don't-know, and not-ascertained states, supporting feature-specific state preservation.
- [CDC/NCHS life-satisfaction question documentation](https://omb.report/icr/202303-0920-002/doc/129730201.pdf) and [Hoyer, Ochiai & Blakey (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548803/) describe question-order/context effects. A single response after health questions contains real subjective and contextual noise.
- [Miller et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/37565865/) identify psychological distress, unmarried status, poor general health, weak support, and food insecurity as major 2021 NHIS correlates. It is a binary analysis and supplies no four-class QWK benchmark.
- [Barger, Donoho & Wayment (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6036920/) find incremental social-relationship signal beyond health and socioeconomic status. This supports interaction capacity, not a hand-built resilience rule.
- [Cheung & Lucas (2014)](https://pubmed.ncbi.nlm.nih.gov/24890827/) report useful but imperfect validity for single-item life satisfaction. Personality, adaptation, emotional support, and response context are incompletely represented here.

No same-input, same-split four-class study provides a numerical ceiling.

### 7.2 Method evidence

| Family | External evidence | Exact-task evidence | Decision |
|---|---|---|---|
| GBDT regression | CatBoost, LightGBM, and XGBoost remain strong mixed-table baselines | V1/V2 and public scores are the only repeated positive lineage | Required control |
| Cumulative ordinal classifiers | All-threshold decomposition yields a coherent expected score | V2 sidecar `+.001087`; V6 CORN neutral | Keep one compact LGB sidecar |
| TabM | [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) and [official repository](https://github.com/yandex-research/tabm) support parameter-efficient ensembles, one-hot categorical input, and piecewise/periodic numeric embeddings; the repository requires optimizing the mean member loss rather than the loss of the mean | V5 proves T4 runtime and weak boundary signal; full-target scalar regression is open | Chosen novel branch |
| Numeric feature embeddings | [Official implementation/paper](https://github.com/yandex-research/rtdl-num-embeddings) shows gains from vector-valued numeric encoding before an MLP | This survey has continuous plus ordered coded fields; V5 used PLE successfully at runtime | Use fold-fit PLE |
| RealMLP | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html) gives strong defaults on medium tabular data | V5 boundary RealMLP was negative and below TabM | Second choice, not same-run fallback |
| Multiclass CE / expected value | Stable general classification objective | V2 multiclass expected branch is weak; correct diagnostic source adds no residual | Reject as final TabM objective |
| CDF/EMD ordinal loss | [Hou, Yu & Samaras (2016)](https://arxiv.org/abs/1611.05916) penalizes distant probability mass | No exact-task gain; CE/CDF would add an arbitrary objective after the source-classifier audit | Reject in this run |
| Direct differentiable kappa | Metric alignment in principle | Rare classes make the batch marginal denominator unstable; historical calibration optimism is already material | Reject |
| Mitra-v2 | [Technical report](https://arxiv.org/abs/2609.04540) and [official model card](https://huggingface.co/autogluon/mitra-classifier-2) describe synthetic-only pretraining and a 50-step, eight-fold bagged fine-tune | Six-day-old report, no independent exact-task evidence, CUDA/dependency burden, and prior foundation transfers negative | High-upside future screen; too risky for final default |
| TabPFN/TabICL | Strong generic foundation benchmarks | V7/V9 lower exact-task QWK and AUC | Closed |
| ModernNCA/retrieval/TabLDM | Learned neighborhood or generative representations | V4/V12/V14 negative with high residual correlation | Closed |
| Reconstruction/self-supervision | VIME-style methods can learn co-state structure | V15 reconstructs survey values but harms target QWK | Closed |

[TabArena 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html) supports the portfolio prior: GBDTs remain strong, deep models can catch up with adequate budgets and ensembles, foundation models are strongest on smaller tables, and validation overfitting can over-select deep models. [TabReD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html) finds GBDT and simple embedding MLPs strongest on richer real-world tables. Neither predicts competition QWK; both favor a guarded GBDT-plus-simple-deep experiment over another exotic transfer.

## 8. Bottleneck diagnosis and hypothesis closure

### 8.1 True remaining bottlenecks

| Rank | Bottleneck | Evidence | What remains learnable | Plausible incremental upside |
|---:|---|---|---|---:|
| 1 | Fine 1↔2 ranking | 63.32% of weighted error; AUC `.7658`; model residuals highly correlated | Smooth interactions among exact survey states and ordered values | `0–.003`; exceptional `.005–.008` |
| 2 | High-burden heterogeneity | 12.43% of train contains 79.74% of false-tail loss; mean score residual is zero | Which adverse-state combinations coexist with retained satisfaction | `0–.002` as part of full-target model |
| 3 | Calibration/selection variance | Fold QWK `.468–.496`; V6.1 optimism `.00293` | Better validation discipline, not more information | Prevents loss |
| 4 | Survey-state representation | Exact special/missing states matter; trees may miss smooth distributed interactions | One-hot state tokens plus ordered PLE in TabM | `0–.004` |
| 5 | Rare-tail ordering | tail AUC `.8726`, 3↔4 `.5380`, only 887 class-4 rows | Limited; specialist variance is high | Small |
| 6 | Omitted subjective factors / response noise | Survey literature and persistent cross-model residual agreement | Not recoverable from released predictors | Defines ceiling |

### 8.2 Closed hypotheses

The following are closed for the last submission unless new released-train OOF evidence appears:

- ID, row order, household/quasi-key, duplicate count, or respondent retrieval;
- class-count matching, survey-weighted prior matching, or diagnostic quota calibration;
- global special-code collapse or generic imputation;
- ontology/path/large aggregate feature expansion;
- masked reconstruction, denoising, VIME, or test-distribution pretraining;
- hard burden/resilience routing or manual tail correction;
- boundary-1-only specialists and a dedicated 3↔4 model;
- CORN as the primary model, difficulty weighting, distillation, and focal/class weighting;
- historical prediction stacking;
- another TabPFN, TabICL, ModernNCA, retrieval, fingerprint, or TabLDM run;
- EXAONE/xRFM final-day engineering;
- four-class LightGBM probability reconstruction or probability-teacher distillation;
- direct mini-batch QWK optimization;
- threshold, blend, class, subgroup, or row decisions from `oqdq5_output`.

### 8.3 Surviving hypotheses

Only two model hypotheses remain credible:

1. **Full-target TabM scalar regression** may capture distributed state interactions with a different inductive bias while retaining the metric-aligned continuous target.
2. **Full-target RealMLP regression** remains scientifically open, but its V5 boundary result and dependency/runtime surface give it lower final-day expected value.

Mitra-v2 has greater theoretical upside and orthogonality but insufficient exact-task and operational evidence for the only remaining run. It should not silently replace TabM if TabM fails.

### 8.4 Magnitude required

Using Recovery OOF's expected-disagreement denominator only as a local approximation:

| Target QWK | Delta from `.48334` | Approx. weighted-error reduction | Relative reduction |
|---:|---:|---:|---:|
| `.48850` | `.00516` | 420 | 1.00% |
| `.50682` | `.02348` | 1,909 | 4.55% |
| `.56232` | `.07898` | 6,421 | 15.29% |
| `.61541` | `.13207` | 10,738 | 25.56% |

QWK's denominator moves with predicted marginals, so these are scale estimates. Crossing `.50682` requires a regime change much larger than every verified gain since V1.

## 9. Candidate comparison and final selection

Scores are priors from 1 (poor) to 5 (strong), not measured QWK.

| Candidate | Upside | Orthogonality | Exact-task support | Leakage safety | T4×2 feasibility | Final-day reliability | Total / 30 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **GBDT control + full-target TabM regression** | 3 | 3 | 4 | 5 | 5 | 4 | **24** | **Choose** |
| GBDT + full-target RealMLP regression | 3 | 3 | 3 | 5 | 4 | 3 | 21 | Second choice for a later isolated run |
| GBDT + all-categorical CatBoost | 2 | 2 | 4 | 5 | 5 | 5 | 23 | Lower novelty; V6.1 weakens V4's probe gain |
| Mitra-v2 fine-tune | 5 | 5 | 1 | 4 | 2 | 1 | 18 | Too new and operationally fragile |
| Burden-gated ordinal specialists | 2 | 3 | 1 | 5 | 4 | 3 | 18 | Train OOF rejects the gate |

All-categorical CatBoost has the highest reliability but repeats a tested representation. TabM has the best balance of remaining novelty, train support, random-initialization rule clarity, and known T4 runtime. Its branch weight can become zero, so the experiment cannot force a weak neural model into the final score.

## 10. Frozen architecture blueprint

### 10.1 Data unit and exclusions

1. Load only the four released competition CSVs.
2. Assert train duplicate IDs have identical predictors and target; assert test duplicate IDs have identical predictors.
3. Keep the first row per ID, yielding 94,663 train and 23,666 test respondents.
4. Exclude `id` and `household_id` from every model.
5. Do not use duplicate count, target encoding, pseudo-labels, source probabilities, external respondents, or test-derived statistics other than applying fit transformations.
6. Keep `survey_weight` as a numeric predictor; keep `pseudo_psu` and `pseudo_stratum` as categorical state predictors.

### 10.2 Minimal tree representation

Use all 176 raw predictors plus one `raw_missing_count` feature.

For LightGBM and XGBoost regression:

- preserve released numeric codes exactly;
- preserve actual NaN for native tree handling;
- add binary special-state flags only for `weight_pounds ∈ {996,997,999}`, `height_inches ∈ {96,97,98,99}`, `age ∈ {97,99}`, `hours_worked_per_week = 990`, `days_missed_work ∈ {997,998,999}`, and `income_to_poverty_ratio = 110`;
- do not repair redundant fields, standardize, one-hot encode, or collapse response states.

For CatBoost:

- treat every outer-fit feature with at most 120 non-null states as a string categorical feature, except `survey_weight` and `income_to_poverty_ratio`;
- map actual NaN to `__NA__` and keep every numeric response code as a distinct string;
- keep higher-cardinality physical/continuous fields numeric, masking no raw values;
- add the same special-state flags and missing count.

These are complementary views of the same released information. The tree core does not use the diagnostic package or hand-built burden features.

### 10.3 GBDT regression core

Train on the fold fit partition and early-stop on its inner calibration partition:

| Model | Frozen configuration |
|---|---|
| LightGBM regressor | `objective=regression_l2`, up to 5,500 trees, lr `.025`, 31 leaves, min child 40, feature fraction `.85`, bagging fraction `.85`, bagging frequency 1, L1 `.05`, L2 `4.0`, early stop 220; GPU `max_bin=63`, CPU fallback allowed |
| XGBoost regressor | squared error, histogram method, up to 5,500 trees, lr `.025`, depth 7, min child weight 8, subsample `.85`, colsample `.85`, L1 `.05`, L2 `4.0`, early stop 220 |
| CatBoost regressor | RMSE, up to 5,500 trees, lr `.025`, depth 8, L2 5, early stop 220, native categories, `allow_writing_files=False` |

Use seed `20262000 + 100×fold + {11,23,37}` for LGB/XGB/Cat. If a GPU library fails for a known backend reason, use its CPU mode with the same learning problem and record the fallback. Do not alter hyperparameters after seeing outer results.

The core score is the fixed equal mean of the three continuous predictions. No core-weight search is allowed.

### 10.4 Cumulative ordinal sidecar

Train three LightGBM binary classifiers on `Y>1`, `Y>2`, and `Y>3` with the tree representation:

- binary log loss;
- up to 3,500 trees, lr `.025`, 31 leaves, min child 80, feature/bagging fractions `.85`, L2 `4.0`, early stop 180;
- no class weighting, focal loss, resampling, or quotas;
- enforce `P(Y>1) ≥ P(Y>2) ≥ P(Y>3)` by successive minimum projection.

The sidecar score is:

`cumulative_score = 1 + P(Y>1) + P(Y>2) + P(Y>3)`.

On the inner calibration split, test `a ∈ {0, .15, .30, .45}` in:

`anchor_score = (1-a) × core_score + a × cumulative_score`.

For each `a`, optimize cuts under the procedure in §11.2. Choose highest calibration QWK, breaking ties toward smaller `a`.

### 10.5 Survey-state TabM representation

Use `tabm==0.0.3` and `rtdl_num_embeddings==0.0.12`, initialized from scratch.

**Categorical tokens**

- Include every raw predictor with at most 120 observed states in the outer fit partition, except `record_type`, `adult_sample_flag`, `imputation_number`, and `household_respondent_flag`, and except `survey_weight`/`income_to_poverty_ratio`.
- Each feature has its own vocabulary. `feature=value`, actual `__NA__`, and every special code are distinct states.
- Learn vocabularies on the fit partition only; map unseen calibration/outer/test values to index 0.
- Add a separate state token for each high-cardinality numeric field: `__NA__`, each documented special code, and `__OBSERVED__`. This preserves the state of `survey_weight`, `income_to_poverty_ratio`, and `weight_pounds` even when their raw values are not one-hot encoded. Drop a derived token if it is constant in the fit partition.
- TabM uses its native one-hot categorical input. Do not add learned entity embeddings: V4 already tested a learned categorical-embedding MLP, and low-cardinality one-hot is the official TabM path.

**Numeric and ordered dual view**

Include numeric copies of:

- continuous/physical: `survey_weight`, `income_to_poverty_ratio`, `age`, `height_inches`, `weight_pounds`, `hours_worked_per_week`, and `days_missed_work`;
- counts: `children_in_family`, `children_in_household`, `adults_in_family`, `adults_in_household`, `cancer_types_count`, `emergency_room_visits`, and `years_in_current_home`;
- ordered survey fields: `general_health_status`, `depression_frequency`, `anxiety_frequency`, `food_security_3cat`, `food_security_4cat`, `education_level`, `worry_about_medical_costs`, and all raw `difficulty_*` fields.

Keep `income_poverty_category` and `time_since_last_doctor_visit` categorical only. Their observed domains (`1…14` plus `140`, and `0…9`) are richer than the local dictionary and are not safely ordered without a complete field-level code map. The continuous `income_to_poverty_ratio` already supplies the ordered income signal.

The valid numeric domains are frozen:

| Fields | Values retained in numeric copy |
|---|---|
| `age` | `18…85`; mask `97/99` |
| `height_inches` | every finite value below 96; mask `96/97/98/99` |
| `weight_pounds` | every finite value below 996; mask `996/997/999` |
| `hours_worked_per_week` | every finite value except `990` |
| `days_missed_work` | every finite value except `997/998/999` |
| `income_to_poverty_ratio` | every finite value except `110`; `11.0` remains valid |
| `survey_weight` | every finite value |
| children counts | integers `0…3` |
| adult counts | integers `1…3` |
| `cancer_types_count` | integers `0…6` |
| `emergency_room_visits` | integers `0…4` |
| `years_in_current_home` | integers `1…5` |
| `general_health_status`, `depression_frequency`, `anxiety_frequency` | `1…5` |
| `food_security_3cat` | `1…3` |
| `food_security_4cat` | `1…4` |
| `education_level` | `1…10`; mask `97/99/990` |
| `worry_about_medical_costs` | `1…3` |
| every `difficulty_*` field | `1…4` |

Mask every value outside its frozen domain in the numeric copy while preserving its raw categorical or high-card state token. Add a missing/invalid indicator for every numeric copy with both states present.

Fit median and IQR on the fit partition, fill the numeric copy with the fit median, divide by fit IQR, and clip to `[-12,12]`. Add standardized `raw_missing_count`. Compute 32 fold-fit quantile bins and use `PiecewiseLinearEmbeddings` version B with embedding width 12. Every statistic and bin boundary is fit-only.

This dual view is the survey-native representation: categorical tokens preserve response meaning; numeric PLE exposes order and smooth nonlinear effects; TabM mixes them jointly. It introduces no hand-selected respondent rule.

### 10.6 TabM scalar ordinal regressor

| Item | Frozen value |
|---|---|
| Output | `d_out=1` scalar for each internal member |
| Ensemble members `k` | 16 |
| Backbone | TabM, 2 blocks, width 256 |
| Batch size | 1,024; one OOM retry at 768 |
| Optimizer | AdamW, lr `.002`, weight decay `.0003` |
| Epoch budget | 70 |
| Early stopping | RMSE of the averaged 16-member inner-calibration score, patience 8, minimum improvement `1e-5` |
| Loss | mean squared error over all rows and all 16 member outputs |
| Precision | CUDA mixed precision with finite-loss and finite-gradient checks |
| Class/sample weighting | none |

Optimize the mean of the 16 **individual member losses** against the raw labels `1…4`, not the loss of their averaged prediction. Do not standardize the target. At inference, average the 16 scalar outputs. Do not round or force the score into `[1,4]` before blending.

No CE head, CDF penalty, cumulative neural head, rank-pair loss, direct QWK loss, reconstruction pretraining, test adaptation, feature masking pretext task, or second neural seed is allowed. This isolates the TabM architecture/representation hypothesis and avoids an arbitrary loss sweep.

### 10.7 Final candidate score

On the same fold-local calibration subset, test:

`b ∈ {0, .10, .20, .30, .40, .50, .65, .80, 1.0}`

in:

`candidate_score = (1-b) × anchor_score + b × tabm_score`.

For every `b`, optimize three cuts under §11.2. Choose the highest calibration QWK and break ties toward smaller `b`. Record the full grid. Apply the chosen `a`, `b`, and cuts unchanged to the untouched outer fold and that fold's test prediction.

For standalone diagnostic QWK of the core, cumulative sidecar, and TabM, fit a separate cut vector on the same inner calibration rows and apply it once to the outer fold. Those component cuts are diagnostic only and never replace the selected anchor/candidate cuts.

There is no second weight grid, HPO, seed sweep, subgroup blend, class correction, or global OOF rescue.

## 11. Frozen validation and calibration protocol

### 11.1 Outer and inner splits

1. Deduplicate by respondent ID after equality assertions.
2. Build five outer `StratifiedKFold` splits with shuffle and seed `20260910`.
3. Stratify on target × survey year; if any combined stratum has fewer than five rows, use target alone and record the fallback.
4. Within each outer-training partition, reserve 12.5% using one stratified shuffle split with seed `20261000 + fold`, using the same target×year rule.
5. Fit preprocessing, maps, bins, models, early stopping, component weights, and cuts using only outer-fit/inner-calibration rows.
6. Never use outer-validation labels for epochs, transformations, features, weights, cuts, loss selection, or retries.

Household grouping is not used: household is excluded, household signal is negligible, and household overlap across train/test is part of the competition distribution. Duplicate respondent isolation remains mandatory.

### 11.2 Ordered-cut optimization

For a continuous score `s`, predict `1 + digitize(s,[t1,t2,t3])`.

- Require `t1 < t2 < t3` and adjacent separation at least `.025`.
- Use deterministic Nelder–Mead optimization with maximum 550 iterations from exactly three fixed starts: `[1.45,1.95,2.40]`, `[1.55,2.03,2.52]`, and `[1.65,2.15,2.65]`.
- Select the best calibration QWK; ties choose the lexicographically smaller cut vector.
- Do not initialize from test values, diagnostic counts, or required class proportions.

Outer-fold labels produced by these inner-calibration cuts are the only honest labels used for promotion. A globally optimized OOF cut may be reported as optimistic context but cannot replace fold-pure QWK.

### 11.3 Stage-S falsification gate

Predeclare outer folds **0 and 2** for Stage S. Train all GBDT control components on all five folds if needed, but train TabM on folds 1, 3, and 4 only when the first two candidate folds satisfy all conditions:

- mean candidate-minus-anchor QWK ≥ `0.0000`;
- neither QWK delta < `-.0025`;
- mean 1↔2 AUC delta ≥ `-.0020`;
- mean expected-score RMSE increase ≤ `.0030`;
- all values are finite and the model changes at least `.5%` of anchor labels or score correlation is below `.9995`;
- no preprocessing, CUDA, loss, mapping, or row-integrity failure occurred.

This is a falsification screen, not promotion. If it fails, stop TabM and do not substitute RealMLP, Mitra, CE, another seed, or a diagnostic correction in the same run.

### 11.4 Required diagnostics

For core, cumulative sidecar, anchor, TabM, and candidate, record per fold and pooled fold-pure:

- QWK and accuracy;
- expected-score RMSE and mean residual;
- AUC for class 1 versus 2, non-tail versus tail, and class 3 versus 4;
- 4×4 confusion matrix;
- hard-label counts;
- weighted transitions `1→2`, `2→1`, false tail, missed tail, `3↔4`, distance 2, and distance 3;
- score Pearson/Spearman and residual correlation versus anchor;
- changed-label transition matrix and whether each change corrects or breaks the anchor;
- QWK/error by year, raw-missing quintile, cut-margin decile, and the fixed corrected high-burden audit group;
- chosen epochs/iterations, `a`, `b`, cuts, runtime, memory peak, and device;
- test score quantiles and inter-fold correlation for shift monitoring only.

No test diagnostic label or class count may appear in the implementation metrics.

### 11.5 Paired uncertainty

After five folds, run 2,000 respondent-level paired bootstrap replicates with seed `20264000` on fixed fold-pure labels:

`QWK(candidate) − QWK(anchor)`.

Report mean, median, 2.5/97.5 percentiles, and `P(delta>0)`. Do not re-optimize weights or cuts inside bootstrap replicates.

### 11.6 Full promotion and submission conditions

TabM enters the deployment score only if all conditions hold:

1. pooled fold-pure candidate gain over the matched anchor is at least `.0015`;
2. at least four of five fold deltas are strictly positive;
3. the worst fold delta is at least `-.0015`;
4. paired bootstrap `P(delta>0) ≥ .90`;
5. pooled 1↔2 AUC delta is at least `-.0010`;
6. false-tail weighted loss is at most `1.03 ×` the anchor value;
7. no survey year with all four classes loses more than `.004` QWK;
8. the anchor itself clears sanity floors of `.476` fold-pure QWK, `.760` 1↔2 AUC, and `.868` tail AUC; otherwise investigate a broken control rather than promoting a relative gain;
9. every integrity, finite-value, fold-isolation, and reproducibility check passes.

If any condition fails, TabM is rejected. Keep the existing Recovery submission; do not spend the last submission on the anchor-only rerun unless it independently and materially beats a from-scratch matched Recovery control.

Even if all gates pass, submit only when the static firewall audit confirms that the notebook reads the released competition files and its own temporary outputs only, and contains no diagnostic labels, IDs, counts, paths, thresholds, or derived rules.

## 12. Fold-to-test score aggregation

Each fold has its own selected `a`, `b`, and cuts. Normalize its continuous score before averaging:

1. map `t1→1.5`, `t2→2.5`, and `t3→3.5` piecewise linearly;
2. use slope `1/(t2-t1)` below `t2`, and `1/(t3-t2)` above `t2`, extending the nearest segment outside the cuts;
3. apply the same mapping to that fold's outer score and test score;
4. average the five normalized test scores;
5. apply fixed cuts `[1.5,2.5,3.5]` once.

This preserves each fold's hard outer labels while putting fold test scores on a shared scale. It uses no test label, count, or distribution target. Do not average fold hard labels, match class totals, or apply a diagnostic quota.

If TabM is rejected after full validation, the deployment candidate does not fall back automatically to a new anchor submission. Recovery remains the default because it has the strongest actual public evidence.

## 13. T4×2 resource and reproducibility contract

### 13.1 Schedule

- GPU 0: TabM for the current fold.
- GPU 1: CatBoost, followed by XGBoost for that fold.
- LightGBM: CPU concurrently when its GPU backend is unavailable; otherwise GPU 1 when free.
- Do not shard a single model across the two T4s.
- Release model objects and CUDA cache after every fold; retain predictions and manifests only.

Expected wall time is roughly 3–5 hours on T4×2 when Stage S passes and shorter when it fails. On one T4, run CatBoost, XGBoost, and TabM sequentially with LightGBM on CPU; expect roughly 6–9 hours. A no-GPU run stops before TabM rather than silently using CPU.

TabM `k=16`, width 256, two blocks, and batch 1,024 already fit T4-class hardware in V5. On a verified CUDA OOM, clear cache and retry once at batch 768. A second OOM is a hard failure; do not shrink the model and reinterpret it as the same candidate.

### 13.2 Reproducibility

- Pin `tabm==0.0.3` and `rtdl_num_embeddings==0.0.12`.
- Record Python, CUDA, driver, PyTorch, LightGBM, XGBoost, CatBoost, pandas, NumPy, SciPy, and scikit-learn versions.
- Set Python, NumPy, and PyTorch CPU/CUDA seeds; record any nondeterministic CUDA operation warning.
- Save raw file hashes, fold/inner indices, feature lists, vocab cardinalities, numeric valid masks, medians/IQRs, PLE bins, best iterations/epochs, calibration grids, and cut maps.
- Use one declared seed per fold; do not select among seeds.
- Record caught backend fallbacks and ensure they do not change rows, targets, or preprocessing.

No pretrained weights, external data, historical predictions, or research artifact is needed. Internet is required only if the two pinned small packages are absent.

## 14. Expected failure modes and fixed fallback behavior

| Failure | Evidence/trigger | Required action |
|---|---|---|
| TabM repeats tree score | correlation ≥`.9995`, negligible label changes, no paired gain | Reject; keep Recovery |
| TabM worsens 1↔2 ranking | Stage-S/full AUC gate fails | Reject; do not rescue with thresholds |
| TabM compresses or exaggerates tails | false-tail or missed-tail loss rises; unstable cuts | Reject under gates |
| Calibration subset overfits | positive calibration but negative/variable outer folds | Reject; outer results control |
| One lucky fold drives gain | fewer than 4/5 positive or bootstrap fails | Reject |
| GBDT control is unexpectedly weak | sanity floors fail | Audit implementation; no submission |
| Package/install/CUDA failure | import, finite-loss, or second OOM failure | Stop novel branch; no substitute model |
| Test distribution looks different | score quantiles/inter-fold correlation unusual | Report only; do not tune counts/cuts |
| Apparent gain exceeds `.01` | far beyond historical evidence | Treat as contamination/validation alarm and re-audit before any submission |

## 15. Later implementation deliverables

A separate implementation task can translate this document directly. Its clean run must retain:

- unique-respondent OOF scalar scores and hard labels for every component;
- outer and inner split indices/counts;
- complete calibration grids and chosen `a`, `b`, cuts;
- fold and pooled metric, confusion, transition, margin, year, missingness, and corrected burden tables;
- paired bootstrap summary;
- fold test scores and normalized aggregate score;
- raw hashes, feature typing, state vocabularies, numeric valid ranges, special masks, PLE bins, seeds, versions, devices, best iterations/epochs, memory, and runtime;
- one Stage-S/final-gate decision record;
- a submission only if §11.6 passes, with 23,666 unique IDs in first-occurrence test order and integer labels in `{1,2,3,4}`.

The notebook must be created only when explicitly requested. It must train from zero and read only the released competition CSVs plus files it creates in its own working directory. Historical OOF/test scores, historical folds/cuts, handoffs, and every file under `research/oqdq5_output/` are prohibited model inputs.

## 16. Final assessment

The correct diagnostic package does not reveal a legal hidden correction. Its aggregate provenance is credible, but its row labels reproduce a LightGBM probability ranking under a quota. It overstates agreement with that model family, reverses the legitimate V2-versus-Recovery ordering, and assigns most tail candidates against raw argmax. The train-only replay independently rejects quota matching and the multiclass expected-score residual.

The legitimate evidence says:

- most recoverable loss is still 1↔2 ordering;
- false tails are concentrated in adverse-state combinations, but their group mean score is already calibrated;
- special codes, missingness, and survey order must be represented, not manually repaired;
- reconstruction, lookup, hard specialists, stacking, neighborhood learning, and tested foundation transfers are closed;
- a simple interaction-aware supervised model is the only credible remaining representation test.

The final run is therefore **GBDT ordinal control + full-target TabM scalar regression**, with the neural branch admitted only by untouched OOF. Passing the frozen gates could justify a last independent leaderboard test and may plausibly move into the upper `.48s`. Crossing `0.50682` would require an improvement an order of magnitude larger than the expected TabM contribution; `0.56232` and `0.61541` are not supported by the released-data evidence. If TabM does not pass, the correct action is to keep Recovery rather than manufacture confidence from the reconstructed test candidate.
