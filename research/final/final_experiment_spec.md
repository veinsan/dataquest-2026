# Final experiment specification

Frozen: 2026-09-10, before implementation of `notebooks/v_final.ipynb`. This specification uses only raw competition train evidence, historical competition OOF for diagnosis, and public methodological literature. The final notebook does not load this file or any historical artifact.

## Objective and decision boundary

The practical objective is a submission that improves the documented Recovery public QWK `0.48334` without test-side calibration. Train-only error analysis identifies weak 1↔2 ranking and high-burden false-tail uncertainty as the remaining modelable losses. A hand-built resilience score, missing-cell reconstruction, more historical stacking, and routine foundation-model reruns are excluded.

The final experiment asks one question:

> Does a full-target probability model with a supervised TabM representation learn stable, complementary respondent-level ordering that a fresh raw-state CatBoost regressor misses?

This was not answered by V5, which trained TabM only as a binary `1` versus `>1` specialist on three folds. The test is deliberately narrow: one candidate model, one anchor, one predeclared blend grid, and no rescue HPO.

## Candidate architectures

### Candidate A — full-target TabM probability ensemble plus CatBoost anchor

- **Learning problem:** four-class classification on labels 1–4. Each TabM ensemble member outputs four logits. Its continuous ordinal score is the mean class expectation across members.
- **Representation:** exact response values become feature-specific categorical states; actual missing is its own state. Continuous/count features receive fold-fit median/IQR scaling, clipping, missing indicators, and piecewise-linear numeric embeddings. Known special codes in continuous fields receive explicit state tokens and are masked only in that numeric copy. IDs and household IDs are excluded. One global raw missing-count feature is allowed; no ontology, lookup, target encoding, reconstruction, or feature explosion.
- **Model:** official `tabm==0.0.3`, `k=16`, two MLP blocks, width 256, numeric PLE from `rtdl_num_embeddings==0.0.12`. Random initialization; no pretrained weights.
- **Loss:** unweighted four-class cross-entropy plus fixed `0.10 ×` squared cumulative-distribution loss. No inverse-frequency weighting or direct batch-QWK loss.
- **Anchor:** CatBoost regression trained from scratch on the same fit/calibration partitions, with low-cardinality states as strings and continuous fields numeric.
- **Tail handling:** learned jointly through class probabilities. No hard routing, oversampling, quota, or special tail threshold.
- **Complementarity:** TabM learns a shared nonlinear state representation and probability distribution; CatBoost supplies the strongest proven piecewise-tree bias. Blend alpha and cuts are chosen only on fold-local calibration data.
- **Footprint:** roughly one T4 for TabM and one for CatBoost; run concurrently on T4×2 and sequentially on one GPU. Expected 2–4 hours depending on early stopping and Kaggle load, below 16 GB per device.
- **Falsification:** reject if paired outer-OOF gain is `<0.0015`, fewer than four folds improve, any fold loses more than `0.0015`, bootstrap `P(ΔQWK>0)<0.90`, 1↔2 AUC falls by more than `.001`, or false-tail weighted loss rises materially.
- **Why still open:** V5 used only a boundary-1 binary target. V4’s custom deep probe was smaller/different and weak, while no historical run combined full-target TabM probabilities, stable CE, ordinal auxiliary, and honest five-fold comparison.

### Candidate B — full-target RealMLP-TD plus CatBoost anchor

- **Learning problem and representation:** four-class probabilities from raw categorical states and robust fold-fit numeric preprocessing.
- **Model/objective:** official RealMLP tuned defaults with cross-entropy; expected class score; optional fixed CDF auxiliary only if supported without replacing the official trainer.
- **Complementarity:** tuned normalization, embeddings, smooth activations, and ensembling differ from CatBoost and TabM’s parameter-sharing bias.
- **Footprint:** one T4, moderate memory, but `pytabkit` adds a large dependency surface and its trainer makes custom ordinal loss/audit hooks less reliable.
- **Falsification:** the same paired gates as Candidate A.
- **Why still open:** V5 tested RealMLP only on boundary 1. It was weaker than TabM there (`-0.000455` mean blend delta), so full-target use remains scientifically open but has a poorer local prior.

### Candidate C — bounded Mitra-v2 task fine-tuning

- **Learning problem:** native four-class classification with expected-score inference; one strictly time-boxed fold-pair screen before any five-fold work.
- **Representation/model:** official `autogluon/mitra-classifier-2`, synthetic-only pretrained 2D row-column transformer, exact public revision and Apache-2.0 provenance recorded.
- **Objective:** official 50-step fine-tuning recipe, without importing external respondent data; no eight-bag replication unless the screen passes.
- **Complementarity:** synthetic pretraining provides the most different prior among candidates.
- **Footprint:** high. The September 2026 stack is six days old, pulls AutoGluon/TabArena dependencies and a ~303 MB checkpoint, and its reported recipe bags eight fine-tunes. T4 behavior is not locally established.
- **Falsification:** immediate rejection on dependency/runtime failure, >14.5 GB steady memory, >90 minutes per screen fold, or mean two-fold QWK delta below zero.
- **Why still open:** it is a new synthetic-only model rather than the TabPFN/TabICL/TabLDM variants already tested. Its novelty is also the main operational risk.

## Candidate comparison and selection

Scores are 1 (poor) to 5 (strong), fixed before the notebook. “Upside” is expected value, not a claimed measured gain.

| Candidate | QWK upside | Orthogonality | Train support | Literature support | Leakage safety | Compute feasibility | Final-day reliability | Total / 35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Full-target TabM + CatBoost | 3 | 3 | 4 | 5 | 5 | 4 | 4 | **28** |
| B. Full-target RealMLP + CatBoost | 3 | 3 | 3 | 5 | 5 | 4 | 3 | **26** |
| C. Mitra-v2 fine-tune | 4 | 5 | 1 | 3 | 4 | 1 | 1 | **19** |

**Chosen: Candidate A.** TabM has the best combination of an actually open learning formulation, a slight positive local prior in V5, an official small package, no pretrained-data provenance question, and a direct implementation path already proven to run on Kaggle T4×2. RealMLP is close scientifically but adds machinery without a better local signal. Mitra-v2 is reserved for a future bounded screen because final-day runtime and release maturity dominate its speculative upside.

## Frozen data and feature contract

1. Discover a directory containing exactly the four raw files: `train_final.csv`, `test_final.csv`, `data_dictionary.csv`, and `sample_submission.csv`. Search `/kaggle/input` first, then `./data` and `.`. Reject ambiguity unless candidates contain identical file hashes.
2. Assert target labels exactly `{1,2,3,4}` and required columns `id`, `household_id`, and `life_satisfaction`.
3. For each duplicate ID, assert all raw predictors and train targets are identical. Collapse to first occurrence before fitting or scoring.
4. Exclude `id` and `household_id` from all model inputs. Do not use respondent multiplicity as a feature or sample weight.
5. Preserve low-cardinality values as exact string tokens with a separate actual-NaN token. Treat `pseudo_psu` and `pseudo_stratum` as categories.
6. Numeric base fields are the intersection of the data with: `survey_weight`, `income_to_poverty_ratio`, `age`, `height_inches`, `weight_pounds`, `hours_worked_per_week`, `days_missed_work`, `emergency_room_visits`, `years_in_current_home`, `children_in_family`, `children_in_household`, `adults_in_family`, `adults_in_household`, and `cancer_types_count`. Add raw missing count.
7. Continuous special-code masks are fixed from train/dictionary semantics: weight `{996,997,999}`, height `{96,97,98,99}`, age `{97,99}`, hours `{990}`, and missed-work days `{997,998,999}`. The original state is retained categorically; only the numeric copy is masked.
8. Category vocabularies, medians, IQRs, numeric PLE bins, and all learned transformations are fit on the fold’s fit partition only. Outer validation and test unknowns map to index zero.

## Frozen validation contract

### Outer and inner splits

- Work at the 94,663 unique-respondent unit.
- Five deterministic `StratifiedKFold` outer folds, shuffled with seed `20260910`. Stratification key is target × observed survey year, with rare combined strata backed off to target alone if any count is below five.
- Inside each outer-training split, one deterministic `StratifiedShuffleSplit` reserves 10% for calibration/early stopping. The split seed is `20261000 + fold` and uses the same backed-off strata.
- The outer validation fold is never used for preprocessing, early stopping, epoch choice, threshold choice, alpha choice, feature choice, or model selection.

### Model budgets

| Component | Frozen budget |
|---|---|
| CatBoost anchor | RMSE; 3,500 iterations; learning rate `.035`; depth 8; L2 5; early stopping 180; fold seed `20262000+fold` |
| TabM | `k=16`; 2 blocks × 256; batch 1,024; max 70 epochs; AdamW lr `.002`, weight decay `.0003`; patience 8 on calibration CE+CDF loss; fold seed `20263000+fold` |
| Numeric embeddings | 32 fold-fit quantile/PWL bins where supported; embedding width 12 |
| Blend grid | alpha `{0,.10,.20,.30,.40,.50,.60,.75,1.0}` on `alpha×TabM + (1-alpha)×CatBoost` expected scores |
| Threshold search | three deterministic Nelder–Mead starts, sorted cuts, minimum separation `.025`; calibration labels only |

For every fold, tune separate anchor, TabM, and blend thresholds on calibration predictions. Choose blend alpha by calibration QWK, breaking ties toward smaller alpha. Apply the chosen alpha and cuts unchanged to outer validation. Global OOF threshold re-optimization is prohibited for reported scores.

### Metrics and uncertainty

Save per-fold and pooled fold-pure:

- QWK, accuracy and expected-score RMSE;
- AUC for 1 versus 2, non-tail versus tail, and 3 versus 4;
- full confusion matrix;
- squared-error transitions: 1→2, 2→1, false tail, missed tail, and 3↔4;
- candidate/anchor score correlation and residual correlation;
- high-burden subgroup QWK/error using a fixed target-free definition: at least three severe states among poor general health, frequent depression, frequent anxiety, low food security, high cost worry, health-limited work, and social-activity difficulty;
- 2,000 paired respondent-level bootstrap replicates of `QWK(blend)-QWK(anchor)` with seed `20264000`.

The historical Recovery score is never a baseline input. The paired baseline is the notebook’s fresh CatBoost OOF under the exact same splits.

### Promotion and final-submission policy

Promote the blend only when all conditions hold:

1. pooled fold-pure QWK gain over anchor ≥ `0.0015`;
2. at least 4 of 5 fold deltas are strictly positive;
3. worst fold delta ≥ `-0.0015`;
4. bootstrap `P(ΔQWK>0) ≥ 0.90`;
5. 1↔2 AUC delta ≥ `-0.0010`;
6. false-tail weighted loss does not exceed anchor by more than 3%.

If the gate fails, emit the from-scratch anchor submission and identify the candidate as rejected in `summary.json`. Do not tune a second alpha grid, class weights, alternative losses, extra seeds, or global cuts to rescue it.

For test inference, average each branch’s five continuous fold predictions. If promoted, use the median fold-local selected alpha and median fold-local cuts; otherwise use the median anchor cuts. This is a deployment aggregation and is not reported as an unbiased OOF score.

The proven historical competition contract is one row per unique `test_final.id` in first-occurrence order (23,666 rows), despite the raw-row placeholder sample. The notebook validates exact output column names and integer label dtype against the sample, documents the row-count mismatch in its audit, and asserts unique IDs/order, finite predictions, and legal labels.

## Resource and failure plan

- With two GPUs, run CatBoost on CUDA device 1 while TabM uses device 0 for each fold. With one GPU, run them sequentially on device 0. No model spans devices.
- Require at least one CUDA GPU for the full experiment and raise a direct accelerator error otherwise.
- Use mixed precision for TabM, batch 1,024, inference batch 4,096, no data-loader workers, and immediate model/cache cleanup after each fold.
- Install only `tabm==0.0.3` and `rtdl_num_embeddings==0.0.12` when absent/mismatched. Record Python, platform, package versions, CUDA devices, file hashes, feature lists, seeds, best epochs, thresholds, and alphas.
- If the second GPU fails, catch that branch’s device error and rerun the fold sequentially on GPU 0. Model/data/schema errors are fatal and exposed with fold/component context.
- No local full training is part of notebook construction. Static compilation and small synthetic unit checks cover thresholding, ordinal loss, duplicate validation, and transition decomposition.

## Required notebook outputs

Under `/kaggle/working/oqdq5_final/`:

- `oof_predictions.csv`
- `fold_metrics.csv`
- `threshold_calibration_audit.csv`
- `confusion_and_transition_audit.csv`
- `subgroup_metrics.csv`
- `test_scores.csv`
- `feature_model_manifest.json`
- `submission.csv`
- `summary.json`

Only `submission.csv` is intended for upload. All artifacts are created from raw inputs during that notebook run.
