# Decision Register

| ID | Decision | Evidence / reason | Status |
|---|---|---|---|
| D001 | Validate by respondent `id` | Duplicate respondent rows exist; random rows leak | **Permanent** |
| D002 | Output one row per unique test `id` | 23,902 physical rows vs 23,666 evaluator IDs | **Permanent** |
| D003 | Keep continuous ordinal score + thresholds | QWK is ordinal and V1/V2 transferred well | **Permanent** |
| D004 | Freeze exact V2 executable anchor | 0.481367 CF / 0.48193 LB; cleaner rebuilds dropped | **Permanent** |
| D005 | Treat class 1/2 as primary bottleneck | Repeated error/AUC diagnostics V1-V6 | **Permanent diagnostic priority** |
| D006 | Do not directly micro-optimize blend weights on QWK | V6 global 0.481761 vs honest 0.478832 | **Permanent** |
| D007 | Forensic lookup/quasi-key path is dead | No robust grouped-OOF gain; leakage temptation | **Frozen** |
| D008 | Cumulative ordinal may remain optional sidecar | Selected 3/5 V6 folds | **Optional** |
| D009 | All-cat CatBoost / XGB categorical are not default sidecars | Only 1/5 V6 selection each | **Drop default** |
| D010 | Foundation models are the main breakthrough family after permission | Need new inductive bias; project rule notes permit them | **Active** |
| D011 | Use smooth positive Ridge/non-negative continuous blend | Avoid discrete-QWK calibration overfit | **Active** |
| D012 | V8 uses conditional 1/2 specialist + frozen V2 tails | Focus model capacity on actual hard boundary | **Active** |
| D013 | Foundation feature decisions must be fold-safe | Avoid hidden target leakage | **Permanent** |
| D014 | V8.3 must debug EXAONE runtime before architecture changes | Runtime failure is confounded with model quality | **NEXT** |
| D015 | Pin EXAONE runtime commit as well as checkpoint | V8.1 package install can drift | **V8.3 requirement** |
| D016 | If EXAONE fails reproducibly, disable it without loosening gates | Preserve experimental integrity | **Fail-closed** |
