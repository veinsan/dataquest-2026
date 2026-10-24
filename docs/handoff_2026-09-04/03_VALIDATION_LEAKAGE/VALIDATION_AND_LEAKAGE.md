# Validation, Leakage, and Calibration Lessons

## 1. The non-negotiable validation unit is `id`

Every reported validation prediction must be out-of-fold by respondent `id`.

Why:

- train has 95,609 rows but only 94,663 unique IDs;
- duplicate IDs have consistent targets and identical payloads;
- row-random splitting can put an identical respondent on both sides of a fold.

Primary split family used by the project:

- `StratifiedGroupKFold`
- group = `id`
- V1: 5 grouped folds
- V2 and later anchors: repeated 2 x 5 grouped folds with seeds 42 and 2026 when compute budget allowed.

## 2. Fold-safe means more than grouped splitting

The following must be fit/selected only on the training portion of an outer fold:

- redundancy repair mappings;
- frequency/count encodings if they depend on observed fold distributions;
- target encoding;
- categorical-signal ranking;
- feature selection;
- lookup/quasi-key experts;
- missing-flag selection;
- nominal-column inference if supervised;
- blend weights;
- threshold calibration for an honest held-out fold.

V1 explicitly rebuilt preprocessing and feature engineering from scratch inside each fold. This protected both duplicate leakage and unsupervised mapping/frequency leakage.

## 3. Global threshold QWK is diagnostic, not the honest score

QWK needs three thresholds to map a continuous prediction into 1-4. Optimizing those thresholds on all OOF labels and then scoring the same labels is mildly optimistic.

Preferred estimate:

For fold `f`:

1. learn thresholds on OOF predictions from all other folds;
2. apply them to fold `f`;
3. concatenate all held-out predictions;
4. report the final cross-fitted QWK.

This rule became especially important after V6.

## 4. V6 proved direct discrete-QWK blending can overfit

Recovered V6 values:

- global/calibration recipe QWK: **0.481761**
- honest final cross-fit QWK: **0.478832**
- optimism gap: about **0.00293**

The latent ranking signal had not collapsed: `AUC(y>1)=0.779611`, very close to V2's roughly 0.780 region. The larger failure was the geometry/calibration of the combined score.

Permanent lesson:

> Do not select ensemble weights because a tiny change improves discrete QWK by +0.0001 or +0.0002 on the same calibration labels.

V7/V8 therefore separate:

- **smooth blending**: positive Ridge / non-negative continuous combination;
- **metric discretization**: QWK thresholds learned afterward and evaluated cross-fitted.

## 5. Submission-row mismatch is a data-contract bug, not a modeling opportunity

`sample_submission.csv` mirrors 23,902 physical test rows, while the validated evaluator contract is one prediction per **23,666 unique IDs**.

Never “fix” this using manual label mapping. The correct operation is deterministic aggregation of duplicate rows sharing the same test `id`, then one prediction per ID.

## 6. Household overlap is a leakage trap

11,619 household IDs occur in both train and test. This can tempt a lookup strategy, but household membership is not the same as a respondent label.

V5's forensic rule was correct in spirit:

- a lookup signal is only legal/useful if constructed from competition train data;
- it must survive honest grouped OOF;
- coverage and purity must be measured without validation labels;
- no external NHIS/source reconstruction is allowed.

The forensic lookup path failed to show enough robust gain and is now frozen.

## 7. Model-research leakage boundary

Public pretrained tabular foundation weights were treated as allowable only after the project recorded organizer permission. Even then:

Allowed project behavior:

- public/open model checkpoint;
- synthetic-pretrained prior;
- local inference/training on competition features;
- exact repository/checkpoint revision logging.

Rejected behavior:

- task-relevant external labeled data;
- reconstructing respondent identity/source records;
- cloud/API inference with undisclosed state;
- manual test mapping;
- leaderboard-driven test edits.

## 8. Validation hierarchy used for promotion

A candidate should be judged in this order:

1. Runtime correctness and fold integrity.
2. Boundary ranking, especially `AUC(1 vs 2)` and `AUC(y>1)`.
3. Cross-fitted QWK, not global-threshold QWK.
4. Residual diversity versus the V2 anchor.
5. OOF/test score-distribution alignment.
6. Only then Public LB as an external transfer check.

## 9. What “honest” means for V8

V8 keeps V2 as a frozen anchor and gives foundation models only the 1/2 specialist task. All supervised feature/view decisions are made inside the outer training side.

Probe-fold QWK with fold-local optimized thresholds is explicitly marked **diagnostic only**. Promotion should rely on preregistered fold-level AUC/QWK criteria across multiple folds, followed by full cross-fitted evaluation.
