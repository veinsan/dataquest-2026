# Objective Quest DATAQUEST 5.0 — Complete Project Handoff

**Snapshot:** 2026-09-08 (Asia/Jakarta)  
**Coverage:** project history from the first dataset audit / V1 through the latest V12 implementation-debugging state.  
**Source priority:** actual experiment artifacts and executed outputs > leaderboard outcomes > current notebook implementation > frozen design documents > older inference/assumptions.

This ZIP is intentionally documentation-only. It contains no notebook, raw competition dataset, raw OOF/test prediction array, submission CSV, model checkpoint, or heavy result directory.

## Read these first

1. `CURRENT_ACTIONABLE_HANDOFF.md` — exact current state and immediate next action.
2. `../03_EXPERIMENT_HISTORY/EXPERIMENT_HISTORY_V1_V12.md` — chronological scientific/engineering history.
3. `../04_SUBMISSIONS_AND_LEADERBOARD/RECOVERY_ENSEMBLE_CHAMPION.md` — current champion and why it worked.
4. `../09_EXPERIMENT_CONTRACTS/CONTRACT_V12_MODERNNCA.md` — active frozen experiment contract.
5. `../07_FAILURES_AND_DEBUGGING/RUNTIME_DEBUGGING_HISTORY.md` — expensive failures and lessons that must not be repeated.

## Canonical one-paragraph state

The task is ordinal tabular prediction of `life_satisfaction` classes 1–4 under Quadratic Weighted Kappa (QWK), evaluated per unique respondent `id`. The strongest raw-model pipeline remains V2, but the current submission champion is a CPU recovery ensemble built from completed historical V2/V5.1/V6.1/V7 predictions; it achieved **Public LB 0.48334 and rank 4**. V11 attempted learned prediction-space ordinal stacking and was cleanly falsified: cross-fitted QWK **0.478662**, delta **-0.003205** vs frozen recovery OOF, **0/5** positive folds, bootstrap `P(delta>0)=0`. V12 was therefore selected to seek genuinely independent ranking signal through **ModernNCA learned respondent geometry**. The latest V12 Kaggle run stopped **before training** because the notebook installed floating TALENT `@main` but compared `ModernNCA` source bytes against an older hard-coded SHA; actual installed SHA was `02fce6...`, expected was `8c0e34...`. This is an implementation/provenance blocker, **not a scientific result**. Immediate work is to finish reproducible TALENT/ModernNCA pinning and downstream static/runtime correctness while preserving the frozen V12 scientific design, then run Stage R -> Stage S before any full CV.

## Status vocabulary

- **LB-CONFIRMED POSITIVE:** improvement has actual leaderboard confirmation.
- **SCIENTIFIC POSITIVE:** honest validation passed a preregistered promotion criterion.
- **SCIENTIFIC NEGATIVE / FALSIFIED:** valid experiment generated evidence against the tested hypothesis.
- **RUNTIME / ENGINEERING FAILURE:** model quality was not validly evaluated.
- **SUSPENDED:** scientifically open but not currently worth executing under resource/engineering constraints.
- **OPEN:** hypothesis not yet validly tested.
