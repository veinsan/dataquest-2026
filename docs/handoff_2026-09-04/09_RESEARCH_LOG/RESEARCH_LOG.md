# Research Log

## 2026-08-29 - V1 baseline established

- Built ordinal continuous-score GBDT baseline.
- Grouped by respondent `id`.
- Public LB 0.48003 closely matched cross-fit 0.479662.
- Validation framework gained credibility.
- 1/2 adjacent errors identified as dominant.

## 2026-08-29 - V2 representation diversity

- Added repeated 2x5 grouped CV.
- Added raw + semantic + nonstandard-code views, native categorical branches, objective sidecars, nested feature selection.
- Fixed evaluator output to 23,666 unique IDs.
- Achieved 0.481367 cross-fit and 0.48193 Public LB.
- Became empirical champion.

## 2026-08-30 - V3 structural experiment

- Tested ontology, questionnaire paths, skip/applicability structure, ordinal TE, dual categorical views.
- Did not clear the preregistered breakthrough gate.
- Learned that structural survey engineering was not hiding a large easy signal.

## 2026-08-30 - V4 breakthrough discovery

- Explicit model-family discovery rather than booster tuning.
- Probed raw-cat CatBoost, fingerprints, retrieval, deep multitask MLP.
- No candidate showed the qualitative multi-fold jump needed for promotion.

## 2026-08-30 - V5 boundary and forensic branches

- Boundary branch: TabM/RealMLP/all-cat specialists for class 1 bottleneck.
- Forensic branch: internal quasi-key / household / mapping audit.
- No strong legal OOF breakthrough.
- Forensic lookup marked dead; TabM/RealMLP only falsified in tested B1 specialist form.

## 2026-08-31 - V6 error-centric and recovery

- Tested soft-target distillation, difficulty weighting, CORN, compact/recovery portfolios.
- Honest final CF 0.478832.
- Boundary AUC remained near old ceiling.
- Calibration/global recipe looked 0.00293 better than honest cross-fit, exposing direct-QWK blend overfit.
- Exact V2 elevated from “baseline philosophy” to **frozen executable anchor**.

## 2026-09-01 - V7 research pivot

- Project literature/model audit concluded another ordinary tree ensemble was low value.
- Modern foundation models became the main breakthrough hypothesis once organizer permission was recorded.
- TabPFN-3 primary, EXAONE high priority; TabICLv2/TabFM additional candidates.
- Smooth positive Ridge/non-negative continuous blending adopted instead of discrete-QWK weight search.

## 2026-09-01 to 2026-09-02 - V7 foundation implementation

- Implemented foundation-first probe with TabPFN/EXAONE and exact V2 fallback/anchor.
- Logged checkpoints, seeds, devices, revisions.
- Final verified foundation metric is not present in the recovered handoff and remains open.

## 2026-09-02 - V8 conditional redesign

- Reframed foundation model as **class1/2 specialist**, not full ordinal replacement.
- Froze V2 tails.
- Added successive-halving probe folds 0,2,3.
- Added TabICLv2 and repaired TabPFN rescue path around EXAONE.
- Established predeclared delta-AUC/QWK promotion gates.

## 2026-09-02 - V8.1 / V8.2

- V8.1: foundation package/checkpoint manifest after frozen V2.
- V8.2: fold-safe numeric/mixed foundation views, special-response NaN+flag, fold-selected missing flags.
- EXAONE runtime path remains the next explicit engineering problem.

## 2026-09-04 - Handoff snapshot

- Consolidated V1-V8.2, dataset audit, leaderboard references, model research, failed ideas, leakage/validation lessons.
- Defined V8.3 as runtime-debugging-only.
- No notebook or raw result included in this handoff.
