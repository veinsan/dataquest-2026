# Leakage, Evaluator, and Provenance Rules

## Leakage traps already audited

### Duplicate respondents
Random row splits are invalid because duplicate IDs are exact copies. Always group/collapse by `id`.

### Household overlap
Household overlap is not a license to transfer labels. Household lookup was tested fold-safely and failed.

### Dictionary over-cleaning
Observed “nonstandard” survey codes are not leakage; deleting them is destructive preprocessing.

### Test/source reconstruction
Never map rows back to an external survey/source database to recover labels or latent identifiers.

### Leaderboard tuning
Do not turn public LB feedback into test pseudo-labels, manual edits, or per-row corrections.

### Calibration leakage
Thresholds/weights learned from the same OOF labels they score can create optimistic QWK. V6.1 is the key warning.

## Foundation/pretrained models

Project-recorded organizer guidance allowed public/open pretrained tabular models when reproducible and without relevant external task labels. For each foundation experiment, provenance must include package/repo revision, checkpoint revision/file/hash, seed, device, fold, feature list, and preprocessing.

## Evaluator row contract

Current canonical submission contract is one row per **23,666 unique test respondent IDs**. Historical V5.1 sample-row mapping experiments are documented as a dead/incorrect branch for current use.
