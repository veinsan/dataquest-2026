# Current Actionable Handoff — Read First

## 1. Where are we now?

The project is at **V12 implementation/debugging, before any ModernNCA scientific evaluation**. V12 is the active experiment because V11 falsified learned prediction-space stacking and the remaining near-term need is an **independent ranking representation**, especially on class 1 vs 2.

## 2. Current champion

**Recovery ensemble — Public LB 0.48334, current rank 4.**  
Scores currently above it in project context: **0.48853, 0.56232, 0.61541**.

The recovery submission was built CPU-only from completed historical artifacts:
- V2 final score: weight `0.5333147962860486`
- V5.1 final score: weight `0.2160045364556684`
- V6.1 final score: weight `0.20393903979487613`
- V7 stack: weight `0.0467416274634069`
- final thresholds: `[1.5694444444444446, 2.011875, 2.5100000000000002]`

Verified recovery submission SHA256 (file itself excluded from this handoff):  
`f8d5e0a8ce37eef1f71eba42dab3aa6945d234c1764c7394f78aa9729bbc3f75`

## 3. Active experiment

**V12 — ModernNCA Learned-Geometry Full-Target Ordinal Regression.**

Scientific question: can a supervised learned similarity space expose local/conditional respondent structure that global tree/foundation/prediction-space models did not exploit, improving the critical 1↔2 ranking?

## 4. What just happened?

The first V12 Kaggle run stopped before any training. Environment evidence:
- 2 x Tesla T4, ~14.56 GiB each, compute capability 7.5
- torch `2.10.0+cu128`
- numpy `2.0.2`
- scikit-learn `1.6.1`
- optuna `3.6.1`
- TALENT installation emitted dependency warnings that `qhoptim>=1.1.0` was initially absent; the setup cell later attempted to install it.

The hard failure occurred in the source-integrity cell:

`RuntimeError: Official ModernNCA source drift: expected 8c0e34271c2cbac32d5723e1d1ec5857eba63e91f5b3f3bd4ee2f23959aece6e, got 02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f.`

## 5. Exact current blocker

The notebook installed **`git+https://github.com/LAMDA-Tabular/TALENT.git@main`**, which is floating, then required an older exact file-byte SHA. The installed package still matched the ModernNCA scientific mechanism according to subsequent source-level audit, so the byte-SHA guard was brittle and stale.

This is an **implementation/provenance bug**, not evidence that ModernNCA is weak.

A later static debugging pass established that current official TALENT still uses:
- `ModernNCA` learned embedding + stochastic candidate sampling,
- Euclidean `torch.cdist`, softmax over negative distance, weighted candidate labels,
- `ModernNCAMethod` with `cat_policy='tabr_ohe'`, `num_policy='none'`, AdamW,
- full train-support prediction,
- regression de-normalization from `mean_std` label processing.

## 6. What should be done next?

**Do not redesign V12. Finish implementation/runtime correctness first.**

Immediate tasks:
1. Replace floating-source + stale-file-hash coupling with a reproducible exact TALENT Git commit pin.
2. Install all declared TALENT dependencies, including `qhoptim`, before importing TALENT.
3. Verify package commit provenance (`direct_url.json` or equivalent), API signatures, official ModernNCA default/HPO config semantics, and the exact eight frozen V12 configs.
4. Static-audit worker/process code, preprocessing, Stage R, Stage S, full-CV/test-resume path, artifact routing, and failure propagation.
5. Stage R must exercise realistic ModernNCA train + full-support validation prediction on fold 0.
6. Only if Stage R succeeds, run Stage S folds 0 and 2, one worker per T4.
7. Full five-fold CV/test inference is forbidden unless Stage-S promotion gate passes.

There is an **unexecuted local static draft** named `Objective_Quest_DATAQUEST_5_V12_ModernNCA_FIXED.ipynb` in the current workspace. It is not included here and must **not** be treated as runtime-validated or as resolution of the blocker. The next agent should review it only as a draft implementation aid.

## 7. Evidence/files the next agent should request/use

If available in the new chat, the most useful source artifacts are:
- historical `results.zip` (V1–V8.3 completed outputs),
- `results-v9.zip`,
- `results-v11.zip`,
- latest V12 source notebook,
- latest V12 `dataquest(3).log` / `download(3).txt`,
- competition CSVs only when actual data re-audit is needed,
- recovery submission only to reproduce/verify its champion predictions, never to retrain base models.

## 8. Frozen methodology that should not change casually

- evaluator entity = unique respondent `id`;
- no row-random CV across duplicate IDs;
- V12 target = continuous full-target regression `y in {1,2,3,4}`;
- exclude `id` and `household_id` as predictive model inputs;
- use target-independent semantic typing;
- nominal observed survey codes are categories, not invalid/missing just because dictionary lists differ;
- known ordered survey scales remain numeric ordinal;
- established continuous sentinel codes become NaN + explicit flag;
- full outer split is frozen to canonical V2 unique-ID fold assignment;
- Stage-S folds = 0 and 2;
- eight fixed official-space ModernNCA configs, HPO seed `20260908`;
- HPO selection metric = inner RMSE, not QWK;
- selected config refit with seeds 42 and 2026 and averaged;
- cross-fitted QWK calibration only;
- recovery `0.48334` is the frozen primary control;
- no global threshold rescue, no dense blend-weight micro-tuning;
- no fallback submission on failed experiment.

## 9. Do not repeat

- prediction-space stacking after V11;
- calibration-only rescue after V6.1/V11;
- generic V2 micro-HPO as a breakthrough strategy;
- household/quasi-key lookup;
- naive retrieval on fixed geometry;
- generic TabPFN/TabICL cycling;
- V10/xRFM runtime patching without clear resource availability and a strong reason to resume;
- treating runtime failures as scientific failures;
- treating small globally calibrated QWK gains as honest improvements.

## 10. How V12 success/failure is decided

### Stage-S promotion: at least one gate

**Gate A ranking:** mean delta AUC 1↔2 >= +0.005 OR mean delta AUC y>1 >= +0.005, with no critical fold delta < -0.005.  
**Gate B ordinal:** mean delta QWK >= +0.002 vs frozen recovery.  
**Gate C novelty:** delta QWK >= -0.0015, residual corr vs recovery < 0.95, and critical AUC does not regress below -0.001.

If both screened folds have nonpositive critical AUC deltas and negative QWK, kill V12. Do not rescue with extra HPO.

### Full-CV scientific improvement

Delta QWK >= +0.002 vs recovery and either >=4/5 nonnegative folds or bootstrap `P(delta>0)>=0.95`, with no critical-boundary regression.

### Submission gate

Delta QWK >= +0.0015; >=4/5 nonnegative folds; minimum fold delta >= -0.0015; bootstrap `P(delta>0)>=0.90`; 1↔2 AUC non-regression; sane predictions; no dependence on global calibration.

### Strong Top-3 challenge evidence

Delta QWK >= +0.0025 OR delta AUC 1↔2 >= +0.005 with stable folds.
