# Recovery Ensemble — Current Champion

## Why it exists

After V10 consumed large GPU compute without producing a valid candidate, the project audited completed historical OOF/test artifacts and asked whether a valid submission could be recovered without new base-model training.

## Components and final weights

- V2 final continuous score: `0.5333147962860486`
- V5.1 final: `0.2160045364556684`
- V6.1 final: `0.20393903979487613`
- V7 stack: `0.0467416274634069`

Final thresholds:
`[1.5694444444444446, 2.011875, 2.5100000000000002]`

## Validation evidence before submission

Recovery analysis reported:
- frozen recovery OOF QWK used later by V11: **0.4818672191**
- comparison against the recovery-era V2 reference showed about **+0.000939 QWK**
- positive folds: **4/5**
- respondent bootstrap `P(delta>0) ~0.913`
- approximate 95% interval: `[-0.00036, +0.00221]`

The evidence was positive but not decisive. Submission was justified because a complete matching OOF/test prediction set existed and compute cost was negligible.

## Leaderboard confirmation

Public LB became **0.48334**, improving V2 by **+0.00141** and reaching rank 4.

This LB outcome materially strengthened the conclusion that small paired OOF improvements from near-peer averaging can transfer.

## Prediction behavior

Recovery remained extremely close to V2:
- continuous prediction correlation with V2 ~**0.99932**
- only **558 / 23,666 test IDs (2.36%)** changed discrete label
- largest change: **391 rows from 2 -> 1**
- other reported changes: 3->2 (56), 2->3 (46), 3->4 (38), 1->2 (26), 4->3 (1)

V2 thresholds were roughly `[1.5585, 2.0085, 2.5390]`; recovery moved to `[1.5694, 2.0119, 2.5100]`, especially raising the 1/2 cutpoint.

## Canonical interpretation

Recovery did **not** reveal a new raw-data signal. It most plausibly succeeded through:
1. variance cancellation across near-peer historical predictors;
2. more stable first-boundary decisions/calibration;
3. changing a small set of borderline respondents rather than rewriting the ranking globally.

V11 later showed that learning a more complex stack over the same prediction manifold did not uncover further ranking information.
