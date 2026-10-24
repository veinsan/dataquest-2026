# Objective Quest DATAQUEST 5.0 - Project Handoff

**Snapshot:** 2026-09-04, Asia/Jakarta  
**Scope:** complete documentation handoff from V1 through V8.2, with V8.3 as the next engineering step.  
**Artifact policy:** documentation only. No notebook, competition CSV, OOF/test prediction file, submission, model checkpoint, Kaggle log dump, or raw result artifact is included.

## Executive state

The project has one empirically proven modeling anchor and one unresolved next bottleneck:

1. **Exact V2 remains the strongest verified in-project model.** It achieved **cross-fitted OOF QWK 0.481367** and **Public LB 0.48193**, with very close CV-to-LB transfer.
2. V3 through V6 repeatedly failed to create a qualitative jump beyond the roughly **0.48 QWK ceiling**. This was useful negative evidence, not wasted work.
3. The dominant modeling bottleneck is the adjacent **class 1 <-> 2 boundary**. V6 still had `AUC(y>1)=0.779611` and `AUC(1 vs 2)=0.765352`, while the higher ordinal boundaries were substantially easier.
4. V6 also exposed a calibration failure mode: a calibration/global recipe at **0.481761 QWK** fell to **0.478832 honest cross-fit QWK**. Tiny direct-QWK blend gains are not trustworthy.
5. V7 moved the project to modern tabular foundation models while freezing V2 as the anchor. The recorded architecture prioritized TabPFN-3 and EXAONE, with strict grouped OOF and reproducibility logging.
6. V8 reframed the problem more sharply: do **not** ask a new model to relearn all four classes. Freeze the V2 ordinal tails and use foundation models as **1-vs-2 conditional specialists**.
7. V8.1 and V8.2 implemented the foundation runtime/reproducibility layer and fold-safe foundation views. The next step is **V8.3 EXAONE runtime debugging**, not another model redesign.

## Current recommendation

Do **not** change the V8 statistical design before the EXAONE runtime path is isolated and proven. V8.3 should be a runtime-first diagnostic pass with the following invariant:

> Make EXAONE install, import, checkpoint load, tiny fit, and fold-0 specialist prediction deterministic first. Only after the runtime works should its model quality be compared with the predeclared V8 gates.

If EXAONE cannot be made reproducible, fail closed: disable EXAONE and continue the same V8 conditional experiment with TabICLv2 and/or TabPFN-3. Do not compensate for a runtime failure by loosening validation gates or changing the leaderboard strategy.

## Folder map

| Folder | Purpose |
|---|---|
| `00_README` | Entry point, status, machine-readable handoff manifest |
| `01_COMPETITION_DATA` | Competition task and immutable modeling/submission contract |
| `02_DATASET_AUDIT` | Dataset shape, duplicates, missingness, IDs, target distribution, cardinality |
| `03_VALIDATION_LEAKAGE` | Grouped validation, calibration, leakage taxonomy, submission-contract lessons |
| `04_EXPERIMENT_TIMELINE` | Full V1 through V8.2 experiment history and status matrix |
| `05_LEADERBOARD` | Verified own scores and historical leaderboard reference targets |
| `06_MODEL_RESEARCH` | Model-family research and current priority order |
| `07_FAILED_IDEAS` | Dead/frozen ideas, why they failed, and what evidence could revive them |
| `08_NEXT_STEP_V8_3` | Current V8 state, EXAONE runtime debugging plan, pass/fail criteria |
| `09_RESEARCH_LOG` | Research chronology, decision register, source provenance, unresolved items |

## Status vocabulary used in this handoff

- **CONFIRMED RESULT**: numerical result recovered from completed experiment/handoff evidence.
- **CONFIRMED DESIGN**: implementation or experimental contract is recovered, but a final model metric is not available in the current archive.
- **NEGATIVE RESULT**: the experiment was run or evaluated enough to reject/promote down the hypothesis.
- **RESEARCH CONCLUSION**: conclusion from the project's prior model/literature audit, not a fresh September 4 web revalidation.
- **OPEN**: information or runtime outcome not recovered and must not be invented.

## Golden rules carried forward

1. Group validation by respondent `id`.
2. Preprocessing, supervised encodings, feature selection, lookup construction, and missingness selection must be learned on the training side of each fold only.
3. Keep **exact V2 executable behavior** as the anchor. Do not replace it with a cleaner approximation.
4. Separate smooth continuous blending from final QWK thresholding.
5. Treat the `1 <-> 2` boundary as the main promotion diagnostic.
6. The final evaluator contract is **23,666 unique test IDs**, despite 23,902 physical rows in `test_final.csv` and `sample_submission.csv`.
7. No external respondent/source reconstruction, manual test mapping, hardcoded test labels, pseudo-labeling from leaderboard feedback, or hidden label probing.
8. Reproducibility for foundation models is part of the model: package/runtime revision, checkpoint revision, SHA256, seeds, device, fold split, feature list, and preprocessing configuration must be logged.
