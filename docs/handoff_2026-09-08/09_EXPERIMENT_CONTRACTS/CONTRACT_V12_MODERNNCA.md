# V12 — ModernNCA Learned-Geometry Full-Target Ordinal Regression

**This is the active frozen scientific design. Current V12 has no scientific result yet.**

## Primary hypothesis

A supervised learned respondent embedding can create a target-relevant neighborhood geometry that improves first-boundary ranking beyond recovery/V2. V4's fixed retrieval failed, but may have failed because raw similarity was wrong rather than because local structure is absent.

## Null hypothesis

Even after supervised geometry learning, neighborhood information cannot improve 1↔2/y>1 ranking; any diversity comes with weaker accuracy.

## Entity/data contract

- one row per unique respondent `id`;
- exact duplicates collapsed;
- `id` and `household_id` excluded from model matrix;
- frozen primary V2 5-fold assignment;
- Stage-S folds 0 and 2;
- no lookup, external respondent data, target encoding, or historical prediction features.

## Target

Continuous full target `life_satisfaction in {1,2,3,4}` under regression/MSE. Do not convert V12 to B1-only, multiclass, CORN, direct-QWK, or pairwise ranking.

## Representation

- target-independent typing from dictionary + audited schema;
- nominal survey codes: preserve observed categories; outer-train one-hot / TALENT `tabr_ohe` equivalent;
- known ordered survey scales: numeric ordinal;
- genuine continuous fields: established sentinels -> NaN + flag, train-side median/imputation/normalization as applicable;
- compact missing indicators + `missing_count`;
- no V2 semantic-copy explosion, household features, prediction-stack features, ontology/path expansion.

## Architecture

Official ModernNCA regression via TALENT. Core mechanism:
1. numeric embeddings/encoder create learned representation;
2. during training, stochastic candidate subset controlled by `sample_rate`;
3. Euclidean distances to candidates;
4. softmax over negative distances;
5. weighted candidate labels produce regression prediction.

TALENT method semantics audited in latest debugging:
- `cat_policy='tabr_ohe'`
- `num_policy='none'`
- AdamW
- regression label mean/std processing
- full train support used at validation/prediction
- output de-normalized for regression.

## Hyperparameter contract

Exactly **8 fixed candidates** drawn from the official ModernNCA/TALENT HPO space with RNG seed `20260908`. The same candidate manifest must be used in every fold. HPO candidate comparison uses training seed 42 only and selects **inner RMSE**, not QWK. After selection, refit selected config with seeds **42 and 2026**, average continuous predictions.

Statistical hyperparameters such as `sample_rate` are not memory-only controls and must not be changed adaptively after observing runtime/quality.

## Stage R

Purpose: runtime correctness only, no quality conclusion.

Must exercise realistic:
- full outer-training row count for fold 0;
- full transformed feature dimensionality;
- actual ModernNCA training step/epochs;
- stochastic candidate sampling;
- full train-support validation prediction;
- finite outputs;
- telemetry for dimensions, memory, runtime, package/source provenance.

A tiny kernel smoke test alone is insufficient.

## Stage S

Run folds 0 and 2, one isolated process per T4.

Compare ModernNCA to:
- **frozen recovery champion** — primary control;
- V2 anchor — secondary diagnostic.

Required diagnostics:
- AUC 1↔2
- AUC y>1, y>2, y>3
- RMSE, MAE, Spearman
- prediction/residual correlation vs recovery
- score disagreement quantiles
- geometry diagnostics: effective neighbors, neighbor target distribution/mass, distances, especially class 1/2

Geometry diagnostics are explanatory only, never model-selection features.

Cross-screen QWK calibration:
- fold 0 thresholds learned from fold 2;
- fold 2 thresholds learned from fold 0.

## Stage-S gates

**Gate A — ranking:** mean delta AUC1v2 >= +0.005 OR mean delta AUC y>1 >= +0.005, and no critical fold delta < -0.005.

**Gate B — QWK:** mean cross-screen delta QWK >= +0.002 vs recovery.

**Gate C — useful novelty:** delta QWK >= -0.0015, residual corr < 0.95, and critical AUC deltas not below -0.001.

Promote if at least one gate passes. Automatic kill if both folds show nonpositive critical AUC deltas and negative QWK. No extra HPO rescue.

## Full CV

Only after Stage-S promotion:
- run five frozen folds;
- selected config refit with seeds 42/2026 per fold;
- exactly one OOF score per respondent;
- cross-fitted fold thresholds;
- 2,000 respondent paired bootstrap vs recovery.

## Optional frozen blend after standalone proof

Only after full-CV proof, test:
`(1-alpha)*recovery + alpha*ModernNCA`, alpha in `{0,0.25,0.50,0.75,1}`.

Alpha selected by RMSE on other folds, not QWK. No dense grid and no V11-style meta model.

## Scientific promotion

Delta QWK >= +0.002 vs recovery and either >=4/5 nonnegative folds or bootstrap P(delta>0)>=0.95, with no critical-boundary regression.

## Submission gate

Candidate can be submitted only if:
- delta QWK >= +0.0015;
- >=4/5 nonnegative folds;
- minimum fold delta >= -0.0015;
- bootstrap P(delta>0)>=0.90;
- AUC1v2 non-regression;
- sane finite prediction distribution;
- no dependence on global calibration.

Top-3 challenge evidence: delta QWK >= +0.0025 OR delta AUC1v2 >= +0.005 with stable folds.  
Breakthrough scale: delta QWK >= +0.005 or delta AUC1v2 >= +0.010.

## Resource strategy

Kaggle T4x2:
- Stage-S fold 0 -> GPU 0
- Stage-S fold 2 -> GPU 1
- one model/process per GPU; no model parallelism
- full CV queued across two GPUs only after promotion.

No CPU approximation or reduced-support shortcut if it changes the frozen scientific question.

## Current implementation blocker

Latest run failed before training because TALENT was installed from floating `@main` but a stale `ModernNCA.py` SHA was hard-coded. Fix source pinning/provenance; do not weaken scientific gates or call V12 falsified.
