# Modeling and Feature Findings

## What consistently works

- continuous regression/expected ordinal score rather than unordered four-class prediction;
- grouped respondent validation;
- raw survey codes retained rather than aggressively sanitized;
- compact missingness/sentinel indicators;
- GBDT family (LightGBM/XGBoost/CatBoost) as strong core;
- V2 representation diversity: raw + semantic + nonstandard-code representation;
- small stable ordinal/categorical sidecars;
- smooth averaging before cross-fitted threshold calibration.

## What looks saturated

### Global tree representation
Multiple strong tree variants converge to nearly the same ranking. More ordinary tree HPO or interaction expansion is low expected value as a breakthrough strategy.

### Hand-engineered survey structure
V3 ontology/skip/path engineering failed to beat V2.

### Prediction-space information
Recovery averaging helped, but V11 learned stacking could not improve first-boundary AUC at all. The prediction manifold is now considered saturated for sophisticated stacking.

### Calibration-only changes
V6.1 and V11 prove calibration can move QWK materially without improving ranking. Global calibration is diagnostic only.

## Diversity lesson

Diversity is only useful if the diverse model is sufficiently accurate. V4/V5 repeatedly produced lower residual correlations with retrieval/neural specialists but little/no QWK gain. The project should seek **independent accurate ranking signal**, not low correlation as an objective by itself.

## Why full-target regression remains the preferred target formulation

Across V1/V2 and later controls, continuous full-target regression/latent ordinal score has been the most stable formulation. Binary B1 specialists, CORN, multiclass sidecars, and foundation specialists have not produced a stronger regime.
