# Failed, Frozen, and Deprioritized Ideas

The purpose of this document is to prevent future sessions from unknowingly repeating experiments that already produced useful negative evidence.

## 1. “More engineered features”

**Status: DEAD AS A DEFAULT STRATEGY**

Evidence:

- V1-V6 repeatedly concentrated most useful gain in a compact subset;
- 400+ feature expansion did not produce a breakthrough;
- V6 intentionally capped the compact view around 140 features.

Revive only if a new feature block has a clear causal/semantic hypothesis and wins across honest multi-fold probes.

## 2. V3 structural ontology / questionnaire path engineering

**Status: FROZEN**

Included observed-code ontology, skip/applicability states, and supervised ordinal encodings. V3 did not beat V2 or clear its breakthrough gate.

Do not rebuild the same representation under a new name.

## 3. Target-encoding explosion

**Status: DEAD**

High-signal categorical TE was carefully made fold-safe, but larger encoding banks did not create a qualitative jump and increase overfit/calibration risk.

V8.2 explicitly uses **no target encoding** for foundation views.

## 4. Missingness / response fingerprinting

**Status: NEGATIVE IN TESTED FORM**

V4 tested row-level missing/special-response fingerprinting as a distinct representation. It did not produce the required multi-fold jump.

Missing indicators remain useful as small fold-selected features in V8.2. The dead idea is the fingerprinting representation as the main breakthrough model.

## 5. Simple retrieval / nearest-neighbor prediction

**Status: NEGATIVE**

V4's local respondent similarity route did not materially break the ceiling. Do not revisit simple retrieval without a substantially new representation or metric argument.

## 6. Generic deep embedding MLP

**Status: DEPRIORITIZED**

V4 basic deep tabular representation did not jump. This does **not** invalidate modern TabM/RealMLP/foundation architectures, which are materially different.

## 7. V5 TabM B1 specialist

**Status: NEGATIVE FOR THIS TASK FORMULATION ONLY**

The boundary specialist did not deliver dramatic new signal.

Do not incorrectly conclude “TabM failed globally.” A full-target TabM remains an unexhausted backup hypothesis.

## 8. V5 RealMLP B1 specialist

**Status: NEGATIVE FOR THIS TASK FORMULATION ONLY**

Same caveat as TabM: the negative evidence applies to the tested boundary specialist, not every RealMLP formulation.

## 9. Forensic quasi-key / household lookup

**Status: DEAD/FROZEN**

It was legal to audit internal competition structure, but a lookup expert had to survive honest grouped OOF. It did not show robust enough gain, and household overlap creates strong leakage/overfit temptation.

Do not use external respondent reconstruction or manually map households/test rows.

## 10. Soft-target distillation

**Status: FROZEN**

V6 used OOF soft targets to reduce ambiguity noise. No meaningful breakthrough emerged.

## 11. Difficulty weighting

**Status: FROZEN**

V6 derived sample difficulty from inner OOF errors rather than class balancing. It was methodologically defensible but not strong enough empirically.

## 12. CORN-style conditional ordinal model

**Status: FROZEN**

Retrospective gain was only around the tiny-noise regime. Do not spend another full session tuning it unless a new interaction with a genuinely stronger representation is proposed.

## 13. All-categorical CatBoost as a generic sidecar

**Status: DROP**

V6 selection frequency only 1/5 folds. It may remain useful as a specific diagnostic, but it no longer deserves a default place in every ensemble.

## 14. XGBoost categorical as a generic sidecar

**Status: DROP**

Also selected only 1/5 folds in V6.

## 15. Direct QWK blend-weight micro-tuning

**Status: DEAD**

This is one of the strongest lessons in the project.

V6 calibration/global QWK: **0.481761**  
V6 honest final cross-fit: **0.478832**

Tiny discrete-QWK improvements are too discontinuous and too easy to overfit. Use smooth blending first and cross-fit the thresholds afterward.

## 16. “Clean up V2”

**Status: DO NOT DO**

V3 structural, V5 forensic/compact, and V6 compact/recovery did not reproduce the empirical V2 advantage. V2's alternate representations appear to contain useful redundancy.

The anchor instruction is literal: **copy/freeze the exact V2 executable pipeline**.

## 17. Blindly chasing the historical 0.61541 target with tree tuning

**Status: REJECTED STRATEGY**

Project sensitivity analysis suggests the missing boundary signal is on the order of +0.11 AUC, not a typical hyperparameter adjustment. The correct search is for a genuinely different model prior or conditional decomposition, which is why V7/V8 use foundation models.
