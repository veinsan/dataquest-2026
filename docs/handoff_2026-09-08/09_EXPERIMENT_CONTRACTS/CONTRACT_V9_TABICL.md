# V9 TabICLv2 Contract and Final Status

## Hypothesis
Full-target TabICLv2 regression could add independent ranking signal where conditional foundation specialization failed.

## Design
- one unique respondent per id
- exclude id/household_id
- continuous full target 1–4
- full 199-feature audited representation
- TabICL 2.2.0 official regressor checkpoint
- Stage R runtime proof
- Stage S folds 0 and 2
- same-run V2 control
- cross-screen QWK thresholds
- full CV/test only if promoted

## Preregistered Stage-S gates
- ranking: mean delta AUC1v2 >= +0.005 OR mean delta AUC y>1 >= +0.005; no catastrophic delta below -0.005
- QWK: mean delta QWK >= +0.002
- diversity rescue: QWK >= -0.001 and residual corr < 0.98

## Actual result
- mean delta AUC1v2 = **-0.00890979**
- mean delta AUC y>1 = **-0.00845092**
- mean cross-screen delta QWK = **-0.00954102**
- mean residual corr = **0.98841706**
- promoted = false

Post-hoc audit bootstrap on two screen folds also strongly negative; no full CV/test/submission was produced.

**Status: scientifically falsified.**
