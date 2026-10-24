# Dataset Audit

This audit summarizes the competition CSVs in the supplied `oq-dq-5.zip`. Raw rows are not reproduced here.

## 1. Core shapes

| File | Shape / rows |
|---|---:|
| Train | **95,609 x 179** |
| Test | **23,902 x 178** |
| Data dictionary | **187 x 3** |
| Sample submission | **23,902 x 2** |
| Unique train `id` | **94,663** |
| Unique test `id` | **23,666** |

There are 176 practical predictor columns after excluding `id`, `household_id`, and the target from the train schema.

## 2. Target distribution

`life_satisfaction` is strongly concentrated in classes 1 and 2:

| Class | Count | Share |
|---:|---:|---:|
| 1 | **42,968** | **44.9414%** |
| 2 | **48,189** | **50.4022%** |
| 3 | **3,559** | **3.7225%** |
| 4 | **893** | **0.9340%** |

Implications:

- Most rows lie on the exact boundary that repeatedly dominates model errors: **1 vs 2**.
- Classes 3 and 4 are rare, but V1-V7 diagnostics showed the higher ordinal boundaries are easier to rank than boundary 1.
- Naive class balancing is not automatically appropriate for QWK. The project repeatedly avoided SMOTE/random oversampling and instead used continuous ordinal modeling and boundary diagnostics.

## 3. Duplicate respondent structure

The difference between total rows and unique IDs is real:

- Train has **946 extra rows beyond unique IDs**. Approximately 1,892 rows are involved in duplicated-ID groups in the audited structure.
- Test has **236 extra rows beyond unique IDs**. Approximately 472 rows are involved in duplicated-ID groups.
- Duplicate train IDs have **consistent targets**.
- Duplicate payload rows are identical in the audited train/test duplicate groups.
- There is **no train/test overlap in respondent `id`**.

This is why `groups=id` is mandatory for validation. A random row split would allow identical respondent payloads to appear on both sides of a fold.

## 4. Household structure

- Unique train `household_id`: **59,016**
- Unique test `household_id`: **19,745**
- Household IDs appearing in both train and test: **11,619**

This overlap is informative for audit purposes but does **not** automatically justify a target lookup or direct household memorization strategy. V5's forensic path explicitly tested quasi-key/lookup ideas only under grouped OOF and failed to establish a reliable enough gain.

Use household structure as a diagnostic unless a future supervised household feature survives an honest, fold-safe protocol.

## 5. Missingness

The data are survey-like and contain substantial but closely aligned missingness:

- Train missing cells: **2,431,886**, mean cell missing rate **14.2099%**.
- Test missing cells: **608,441**, mean cell missing rate **14.3009%**.

The near match reduces concern about a broad train/test missingness shift, but missingness is still potentially semantic because questionnaire applicability and special-response codes can create structured patterns.

The project lesson is not “impute everything.” Trees keep native missingness where supported. Foundation models use explicitly engineered, fold-safe missing/special-response handling.

## 6. Cardinality profile

The dataset is overwhelmingly low-cardinality survey-coded:

- **162** predictor columns have <=10 unique observed values.
- **168** have <=50.
- **173** have <=120.

High-cardinality fields include:

- `id`: 94,663 unique in train
- `survey_weight`: 85,872
- `household_id`: 59,016
- `income_to_poverty_ratio`: 1,084
- `weight_pounds`: 203
- `pseudo_psu`: 111

This profile motivated several research directions:

- native categorical CatBoost/tree views;
- dual raw + semantic survey-code representations in V2;
- TabM/RealMLP research because the effective categorical space is compact;
- foundation models with explicit categorical/numeric handling.

## 7. Survey special-response codes

The project identified continuous-looking fields where high numeric codes are response sentinels rather than quantitative magnitudes. Examples carried into V7/V8 include:

- `weight_pounds`: 996, 997, 999
- `height_inches`: 96, 97, 98, 99
- `age`: 97, 99
- `hours_worked_per_week`: 990
- `days_missed_work`: 997, 998, 999
- `income_to_poverty_ratio`: 110

V8.2's foundation view converts these supported special responses to `NaN` plus an explicit flag rather than feeding them as literal magnitudes.

## 8. Data-dictionary lesson

The simplified dictionary is useful but should not be used to blindly invalidate every observed code. V2's successful representation kept:

- the raw code;
- a dictionary-valid semantic copy when useful;
- a nonstandard-code indicator.

Later attempts to over-clean or compress these alternate representations often reduced performance. This is part of why the exact V2 implementation is frozen as the empirical anchor.

## 9. Audit conclusion

The data do not look like a generic clean continuous table. They are a duplicated, survey-coded, sparse/semantic-missingness dataset with a very concentrated 1/2 target boundary. The most defensible modeling posture is therefore:

`grouped respondent validation + raw/semantic representation diversity + native missingness + ordinal continuous score + carefully cross-fitted calibration`.
