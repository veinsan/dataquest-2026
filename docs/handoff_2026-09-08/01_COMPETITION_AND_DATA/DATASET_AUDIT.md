# Dataset Audit — Canonical Findings

## Target distribution

| Class | Count | Share |
|---:|---:|---:|
| 1 | 42,968 | 44.9414% |
| 2 | 48,189 | 50.4022% |
| 3 | 3,559 | 3.7225% |
| 4 | 893 | 0.9340% |

Classes 1+2 = **95.34%**. This concentration is the central reason the first ordinal boundary dominates QWK behavior.

## Duplicate respondents

- 95,609 physical train rows vs 94,663 unique train IDs.
- 23,902 physical test rows vs 23,666 unique test IDs.
- 946 train duplicate-row excess; 236 test duplicate-row excess.
- Approximately 1,892 train rows and 472 test rows participate in duplicate-ID groups.
- Audited duplicate groups are exact feature duplicates and train duplicates have consistent targets.
- No train/test respondent-ID overlap.

Implication: group/deduplicate by `id`; random row CV leaks identical respondents.

## Train-test exact payload overlap

After identifiers are removed, audit found **no exact train-test feature-payload duplicate**. This killed the simplest direct row-lookup hypothesis.

## Household structure

- unique train households: 59,016
- unique test households: 19,745
- overlapping train/test household IDs: 11,619
- roughly 53.5% of test rows belong to a household that also occurs in train

But household label signal is weak:
- among repeated train households, unanimous target only ~40.62%
- pairwise same-target probability ~46.12%
- baseline same-class probability ~45.76%

V5.1 lookup OOF coverage ~5.20%, test coverage ~7.48%, and covered OOF MAE ~0.523. Household transfer is therefore not a robust predictive mechanism.

## Missingness

- train cell missing rate ~14.21%
- test cell missing rate ~14.30%

Mean missing-feature count increases with satisfaction class:
- class 1: ~24.59
- class 2: ~25.69
- class 3: ~28.92
- class 4: ~29.79

Missingness is real signal, but broad missingness fingerprinting did not become a breakthrough. Keep compact explicit indicators and native missingness rather than huge missing-interaction banks.

## Train-test shift

Adversarial validation with identifiers removed yielded AUC **~0.506**, consistent with very small marginal distribution shift. This supports using honest CV as the primary model-selection signal.

## Survey-code semantics

Most `float64` fields are survey-coded categorical/ordinal states, not generic continuous numbers. The data dictionary is useful for semantics but **not a hard valid-code contract**. Audited rates of observed codes outside simplified dictionary listings were very high for several variables, e.g.:
- child_government_insurance ~92%
- other_government_insurance ~92%
- state_health_plan ~91%
- military_health_insurance ~86%
- income_poverty_category ~79%
- legal_marital_status ~77%
- education_level ~61%

Therefore mapping all “unexpected” codes to missing destroys information. V2's successful approach kept raw code + semantic-valid copy + nonstandard indicator.

## Continuous sentinel examples

Established response sentinels used in later preprocessing audits include:
- `weight_pounds`: 996, 997, 999
- `height_inches`: 96, 97, 98, 99
- `age`: 97, 99
- `hours_worked_per_week`: 990
- `days_missed_work`: 997, 998, 999
- `income_to_poverty_ratio`: 110

For genuine continuous fields, later neural/foundation views convert these to NaN + explicit special-response flag.

## Dictionary leakage audit

Eight dictionary fields were absent from actual train/test, including `salary_satisfaction`. No direct hidden satisfaction target leakage was found through those fields.

## Strongest recurring raw signals

Across feature importance/probe analyses, the most stable useful predictors include:
- general health
- depression frequency
- anxiety frequency
- social activity difficulty
- marital status
- medical-cost worry
- functional difficulty
- income/poverty measures
- missingness / mental-burden patterns

General health and depression show strong monotonic relation with the target, but global models already exploit most obvious marginal structure.
