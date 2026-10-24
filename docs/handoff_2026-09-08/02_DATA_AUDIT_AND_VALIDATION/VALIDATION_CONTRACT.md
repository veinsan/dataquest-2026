# Validation Contract and Reliability

## Canonical entity

All reported OOF predictions must be out-of-fold by **unique respondent `id`**. Duplicate physical rows must never straddle train/validation.

## Frozen fold lineage

V2 established the anchor split family with repeated `2 x 5 StratifiedGroupKFold(groups=id)` using seeds 42 and 2026. Later resource-intensive experiments commonly use the **primary V2 5-fold unique-ID assignment** as a frozen outer split so candidates can be compared pairwise.

V9/V10/V12 Stage-S screening uses folds **0 and 2**.

## Fold-safe operations

The following must be trained/selected only on the training side of the relevant outer/inner split:
- imputation/scaling;
- categorical vocabularies;
- supervised nominal/feature inference;
- target encoding;
- feature selection;
- lookup/quasi-key rules;
- missing-flag selection;
- HPO/model selection;
- ensemble weights;
- threshold calibration for an honest held-out fold.

## Honest QWK calibration

Global thresholds optimized on all OOF labels and then scored on the same OOF are diagnostic only. Honest estimate:
1. for held-out fold `k`, fit three thresholds using predictions+labels from other folds only;
2. apply to fold `k`;
3. concatenate held-out classes;
4. compute QWK.

V6.1 and V11 proved that calibration can generate apparent gains on the same scale as meaningful leaderboard movement, so this rule is permanent.

## Validation-to-LB reliability

Two strongest positive transfers:
- V1 -> V2: cross-fit improvement ~+0.001705; LB improvement +0.00190.
- V2 -> recovery: validation recovery improvement ~+0.00094 over the recovery baseline; LB improvement +0.00141 (0.48193 -> 0.48334).

Thus tiny OOF gains can transfer, but run variance around ~1e-3 means they require paired/stable evidence.

## V2 rerun variance warning

Nominally exact V2-derived reruns produced materially different anchor QWK values:
- original V2 final: 0.481367
- V7 anchor: 0.481716
- V8 anchor: 0.480827
- V8.1 anchor: 0.481110
- V8.2 anchor: 0.481212
- V8.3 anchor: 0.481017

Observed range ~0.000890 for final anchor and even larger core variation. Gains below ~0.001 are not trustworthy without paired evidence, fold stability, and/or LB confirmation.

## Bootstrap lessons

Original V2 vs V1 respondent bootstrap was directionally positive but not definitive:
- approximate 95% interval for V2 score: 0.4754–0.4865
- paired V2–V1 delta median ~+0.00153
- 95% interval roughly [-0.00034, +0.00348]
- positive in ~94% resamples

Recovery pre-submit evidence:
- delta around +0.000939
- 4/5 positive folds
- `P(delta>0) ~0.913`
- 95% interval about [-0.00036, +0.00221]

LB 0.48334 independently confirmed the direction.

V11 negative bootstrap was decisive:
- 2,000 reps
- `P(delta>0)=0`
- 95% interval [-0.004889, -0.001462]
