# V11 Historical Prediction Portfolio Ordinal Stacker — Frozen Contract and Outcome

## Hypothesis
Historical OOF prediction disagreements contain reproducible error information, especially around 1↔2, that a regularized cumulative-ordinal meta-model can exploit beyond the recovery convex blend.

## Data/eligibility
- CPU only
- no base-model retraining
- completed OOF/test streams only
- V9/V10 excluded
- 43 explicit eligible streams -> 40 unique after duplicate removal
- six provenance-defined model families
- frozen recovery score included as feature/control

## Meta model
Three regularized cumulative logistic heads for P(y>1), P(y>2), P(y>3), monotonic projected; continuous expected ordinal score. `C in {0.01,0.1,1,10}` selected by inner non-QWK criterion.

## Frozen gates
Falsify if delta QWK < +0.0005 and delta AUC1v2 < +0.001, or insufficient fold stability. Promotion required delta QWK >= +0.0015, >=4/5 positive folds, min fold >= -0.0015, bootstrap P>=0.90, and AUC1v2 non-regression. Top-3 challenge required delta QWK >= +0.0025 or delta AUC1v2 >= +0.004.

## Actual result
- recovery QWK: **0.4818672191**
- V11 crossfit QWK: **0.4786620928**
- delta: **-0.0032051263**
- positive folds: **0/5**
- AUC1v2: 0.76577484 vs recovery 0.76579642
- y>1 AUC: 0.77995660 vs 0.77998006
- prediction corr: 0.997158
- residual corr: 0.9988345
- 4,063 label changes (4.292%)
- 2,000-bootstrap P(delta>0)=0; 95% [-0.004889,-0.001462]
- global-threshold diagnostic still below recovery by ~-0.000308
- selected C=0.01 on every outer fold

No test prediction/submission file was created because promotion failed.

**Status: runtime success, scientific falsification, DO NOT SUBMIT.**
