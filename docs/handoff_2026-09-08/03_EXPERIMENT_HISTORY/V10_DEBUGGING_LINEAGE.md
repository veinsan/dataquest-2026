# V10 xRFM Debugging Lineage — Do Not Confuse Engineering with Science

V10 is the most expensive engineering lesson in the project. It never produced a scientifically valid outer Stage-S candidate, so xRFM quality is **unknown**.

## Frozen V10 scientific design

- xRFM full-target continuous regression
- unique respondent entity
- canonical mixed-type representation
- fixed 12-config nested RMSE search
- Stage R runtime proof
- Stage S folds 0 and 2
- one process per T4
- no silent candidate dropping
- cross-screen QWK calibration
- promote only via preregistered ranking/QWK/diversity gates

## Failure sequence

1. **Stage-S worker return code 2 / config API mismatch.** The preregistered Lpq configuration stored `p_interp` but xRFM 0.4.5 expected effective `norm_p`. Minimal translation was required; search values themselves were not changed.

2. **Kermac hardware incompatibility.** `lpq_kermac` required NVIDIA compute capability >= 8.0; Kaggle T4 is 7.5. Ten Lpq candidates were moved to xRFM's T4-compatible legacy/PyTorch Lpq path as a hardware compatibility correction rather than dropped.

3. **Stage-R memory failure not solved by `M_batch_size`.** Audit showed `M_batch_size` changes AGOP sample coverage, so it is statistical, not pure memory. It was frozen rather than adaptively reduced.

4. **Kernel solve memory workaround.** A strict in-place Cholesky solver was introduced to avoid large LU workspace. This kept the same KRR equation when positive-definite, but later became the source of the final numerical failure.

5. **641.55 GiB AGOP failure.** Official legacy Lpq `torch.func.jacrev` path materialized an enormous `centers × 8000 AGOP samples × features` cdist-backward intermediate. A prior `jacrev(chunk_size=1)` fix was proven to chunk the wrong axis. The corrected implementation chunked the **sample axis** while computing mathematically equivalent gradients/AGOP.

6. **Long (~9.5h) run reached almost-complete Stage S.** Stage R full-fit succeeded. Both Stage-S folds completed **11/12 fixed HPO configs**. Config 10 failed because strict Cholesky found the regularized kernel matrix numerically non-positive-definite (fold 2 leading minor order ~10015). Because all 12 configs were mandatory, no outer model selection/refit/prediction was validly produced.

7. **No V10 artifact recovery.** The uploaded archive named `results-v10-error.zip` was SHA-distinct but its contents were actually the V9 artifact tree (`oqdq5_v9/...`). Therefore no V10 Stage-S prediction artifact could be salvaged.

8. **Stop decision.** With large compute spent and no valid outer V10 predictions, the project suspended/stopped xRFM for current conditions. A proposed later solver fix (Cholesky where valid, same-system `torch.linalg.solve` fallback) was not established as a completed scientific run.

## Scientific interpretation

- Do **not** write “xRFM failed predictive quality.”
- Correct status: **runtime/engineering unresolved -> suspended; quality unknown**.
- Resuming xRFM only makes sense if compute/resource availability improves and there is a strong reason after V12, with all runtime fixes consolidated and preflight representative of every mandatory config.
