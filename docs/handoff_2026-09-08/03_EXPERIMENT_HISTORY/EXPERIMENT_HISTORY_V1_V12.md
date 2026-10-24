# Experiment History V1–V12 — Scientific and Causal Timeline

## V1 — Ordinal GBDT baseline
**Status:** LB-CONFIRMED POSITIVE baseline
**Hypothesis:** LGB/XGB/Cat regression ensemble + ordinal thresholds can model survey target
**Validation/contract:** 5-fold StratifiedGroupKFold by id; cross-fitted thresholds
**Actual evidence:** crossfit QWK 0.479662; global 0.480481
**Public LB:** 0.48003
**Why this branch ended/continued:** Established reliable validation and ordinal formulation

## V2 — Representation-diverse ordinal GBDT champion
**Status:** LB-CONFIRMED POSITIVE; raw-model anchor
**Hypothesis:** raw+semantic survey-code views, categorical/ordinal sidecars and repeated grouped CV improve signal
**Validation/contract:** 2x5 SGKF id; nested FS; cross-fit thresholds
**Actual evidence:** final crossfit 0.481367; global 0.481587; unique-ID evaluator-aligned diagnostic later ~0.48062
**Public LB:** 0.48193
**Why this branch ended/continued:** Improved V1 and transferred to LB

## V3 — Structural ontology / questionnaire paths
**Status:** SCIENTIFIC NEGATIVE
**Hypothesis:** survey structure, applicability/skip states and ordinal TE reveal hidden signal
**Validation/contract:** single 5-fold grouped; fold-safe structural features
**Actual evidence:** crossfit 0.479948; unique-ID diagnostic 0.480846
**Public LB:** unverified
**Why this branch ended/continued:** Did not beat V2 / breakthrough gate

## V4 — Breakthrough discovery probes
**Status:** SCIENTIFIC NEGATIVE for tested forms
**Hypothesis:** all-cat trees, retrieval, fingerprinting or deep multitask representation yield large orthogonal gain
**Validation/contract:** probe folds 0,2,3 with preregistered +0.015 breakthrough gate
**Actual evidence:** all-cat +0.002925 mean probe; deep +0.000935; retrieval -0.000177; fingerprint -0.001059
**Public LB:** none
**Why this branch ended/continued:** No candidate cleared gate; diversity often came with weakness

## V5 — Boundary-1 specialists
**Status:** SCIENTIFIC NEGATIVE for B1-specialist formulation
**Hypothesis:** specialize directly on class1-vs2 using CatBoost/TabM/RealMLP
**Validation/contract:** probe folds 0,2,3; B1 AUC + blend QWK
**Actual evidence:** TabM mean B1 AUC ~0.77910, gain +0.000586; cross-family B1 AUC ~0.78007; RealMLP ~0.77571
**Public LB:** none
**Why this branch ended/continued:** No stable first-boundary breakthrough

## V5.1 — Forensic household/quasi-key lookup
**Status:** SCIENTIFIC NEGATIVE / CLOSED
**Hypothesis:** internal repeated structure can support legal target lookup
**Validation/contract:** fold-safe grouped lookup coverage/purity
**Actual evidence:** base 0.478556; forensic 0.478569; gain +0.0000135; lookup rejected
**Public LB:** unverified
**Why this branch ended/continued:** No robust gain; household association weak; mapping branch also exposed submission-ID confusion

## V6 — Error-centric ordinal
**Status:** SCIENTIFIC NEGATIVE
**Hypothesis:** distillation, difficulty weighting, CORN improve ambiguous rows
**Validation/contract:** grouped probes; promote only meaningful gain
**Actual evidence:** best CORN gain +0.000106; final 0.479158; adjacent-error separability AUC 0.6234
**Public LB:** unverified
**Why this branch ended/continued:** Difficulty is predictable but tested correction mechanisms did not help

## V6.1 — Recovery/all-cat compact portfolio
**Status:** SCIENTIFIC NEGATIVE; critical calibration lesson
**Hypothesis:** compact robust branches + stable sidecars recover V2 with less noise
**Validation/contract:** 2x5 grouped; cross-fit blend/thresholds
**Actual evidence:** honest final 0.478832; global recipe 0.481761; optimism ~+0.00293
**Public LB:** unverified
**Why this branch ended/continued:** Exposed direct-QWK calibration/selection optimism

## V7 — TabPFN-3 foundation breakthrough
**Status:** SCIENTIFIC NEGATIVE for TabPFN-3 formulation
**Hypothesis:** pretrained tabular prior provides new representation
**Validation/contract:** V2 frozen anchor; probe folds 0,2,3; full-target classifier/regressor branches
**Actual evidence:** TabPFN standalone ~0.4758/0.4760; critical AUC worse; v7 stack crossfit 0.481448 vs rerun anchor 0.481716
**Public LB:** no worth-submit result
**Why this branch ended/continued:** Foundation model ranking weaker; no branch promoted

## V8 — Conditional 1/2 foundation specialists
**Status:** MIXED: TabPFN negative; other families runtime-inconclusive
**Hypothesis:** freeze V2 tails and use foundation capacity only on hardest 1/2 boundary
**Validation/contract:** successive-halving probes; conditional AUC/QWK gates
**Actual evidence:** TabPFN conditional delta AUC1v2 -0.002992; delta y>1 -0.002188; diagnostic QWK gain -0.003216; TabICL/EXAONE runtime failures
**Public LB:** none
**Why this branch ended/continued:** Conditional decomposition did not rescue TabPFN; moved to runtime-specific EXAONE work

## V8.1 — EXAONE runtime isolation attempt 1
**Status:** RUNTIME FAILURE, quality unknown
**Hypothesis:** EXAONE may provide new B1 signal once runtime is reproducible
**Validation/contract:** preflight before probes
**Actual evidence:** V2 anchor 0.481110; EXAONE runtime failed creating venv
**Public LB:** none
**Why this branch ended/continued:** Engineering failure only

## V8.2 — EXAONE runtime isolation attempt 2 + fold-safe views
**Status:** RUNTIME FAILURE, quality unknown
**Hypothesis:** target-isolated install and corrected foundation views make EXAONE runnable
**Validation/contract:** preflight before quality
**Actual evidence:** V2 anchor 0.481212; EXAONE preflight subprocess failed
**Public LB:** none
**Why this branch ended/continued:** Engineering failure only

## V8.3 — EXAONE T4 runtime recovery
**Status:** RESOURCE/ENGINEERING INCONCLUSIVE; EXAONE quality unknown
**Hypothesis:** pinned package/checkpoint + T4 attention patch enables actual probe
**Validation/contract:** runtime proof then fold probe
**Actual evidence:** preflight/checkpoint SHA passed; actual fold inference infeasible from ~33.4 GiB reserved VRAM estimate on ~13 GiB T4 budget
**Public LB:** none
**Why this branch ended/continued:** Runtime progressed but no valid quality prediction

## V9 — TabICLv2 full-target in-context regression
**Status:** SCIENTIFICALLY FALSIFIED at Stage S
**Hypothesis:** full-target learned in-context prior may beat tree ranking where conditional specialist failed
**Validation/contract:** Stage R; Stage S folds 0,2; cross-screen thresholds; frozen V2 control
**Actual evidence:** mean dAUC1v2 -0.008910; mean dAUC y>1 -0.008451; mean dQWK -0.009541; residual corr 0.9884
**Public LB:** none
**Why this branch ended/continued:** All preregistered ranking/QWK/diversity gates failed strongly

## V10 — xRFM local feature-learning full-target regression
**Status:** SUSPENDED/STOPPED; scientific quality unknown
**Hypothesis:** supervised local feature geometry/adaptive partitions reveal conditional signal missed by global models
**Validation/contract:** fixed 12-config nested RMSE; Stage R; Stage S folds 0,2; no silent candidate drop
**Actual evidence:** no valid outer Stage-S predictions; latest long run completed Stage R and 11/12 inner configs on both screen folds but config 10 failed strict Cholesky
**Public LB:** none
**Why this branch ended/continued:** Repeated runtime/backend/memory/numerical failures consumed large compute; no valid candidate submission

## Recovery — CPU recovery ensemble V2+V5.1+V6.1+V7
**Status:** LB-CONFIRMED CHAMPION
**Hypothesis:** near-peer historical predictions can cancel variance and stabilize boundary decisions
**Validation/contract:** nested RMSE-based combination + cross-fit thresholds; paired bootstrap
**Actual evidence:** OOF ~0.481867; ~+0.000939 vs recovery baseline; 4/5 positive; P>0 ~0.913
**Public LB:** 0.48334 (rank 4)
**Why this branch ended/continued:** Small targeted boundary changes transferred positively to leaderboard

## V11 — Historical Prediction Portfolio Ordinal Stacker
**Status:** SCIENTIFICALLY FALSIFIED / DO NOT SUBMIT
**Hypothesis:** model disagreement contains learnable error structure beyond convex recovery averaging
**Validation/contract:** CPU-only nested cumulative logistic heads; same frozen 5 folds; cross-fit thresholds; 2000 bootstrap
**Actual evidence:** QWK 0.478662; delta -0.003205; 0/5 positive; AUC1v2 delta -0.0000216; P>0=0
**Public LB:** not submitted
**Why this branch ended/continued:** No independent ranking signal; mostly same prediction manifold, worse calibration/QWK

## V12 — ModernNCA Learned-Geometry Full-Target Regression
**Status:** ACTIVE IMPLEMENTATION DEBUGGING; no scientific result
**Hypothesis:** supervised neighborhood embedding exposes local respondent geometry and improves 1↔2 ranking
**Validation/contract:** frozen design: Stage R -> Stage S folds 0,2 -> full CV only if promoted
**Actual evidence:** latest run stopped before training: TALENT source SHA integrity mismatch
**Public LB:** none
**Why this branch ended/continued:** Current failure is dependency/provenance implementation issue, not model-quality evidence

# Causal decision chain

1. **V1 -> V2:** close CV-to-LB transfer validated the grouped ordinal framework; focus shifted to representation diversity rather than changing the metric formulation.
2. **V2 -> V3:** V2 gains were small, so the next hypothesis was hidden survey/questionnaire structure. V3 failed, lowering the posterior on hand-engineered structural ontology.
3. **V3 -> V4:** instead of more feature expansion, V4 explicitly searched for different inductive biases. It found a recurring trade-off: accurate models were highly correlated; diverse models were weaker.
4. **V4 -> V5:** because 1↔2 dominated error, V5 concentrated model capacity on that boundary. TabM/RealMLP diversity was real but ranking improvement was not.
5. **V5 -> V5.1/V6:** internal lookup and error-centric correction hypotheses were tested. Household/quasi-key signal collapsed; distillation/difficulty/CORN did not unlock the boundary.
6. **V6 -> V6.1:** calibration optimism became obvious. The project permanently separated continuous modeling from QWK thresholding and stopped trusting global discrete micro-gains.
7. **V6.1 -> V7/V8:** ordinary tree/deep tuning looked saturated. Organizer permission for open pretrained tabular models made foundation models the next major prior. TabPFN underperformed; conditional 1/2 specialization also failed.
8. **V8.x -> V9:** EXAONE remained quality-unknown because of runtime. TabICL had also been runtime-limited in V8, so V9 tested a clean full-target TabICLv2 design. It produced strong negative quality evidence and was killed.
9. **V9 -> V10:** two generic foundation routes were now low-prior. The remaining open idea was learned local feature relevance/geometry, so xRFM was selected. Runtime engineering repeatedly prevented valid Stage-S outer predictions; xRFM quality remains unknown.
10. **V10 crisis -> Recovery:** with no valid V10 candidate and exhausted/expensive GPU resources, completed historical OOF/test artifacts were reused. Simple convex recovery averaging produced LB 0.48334 and became champion.
11. **Recovery -> V11:** LB confirmation suggested prediction disagreement might contain more exploitable error structure. V11 tested this rigorously; it failed on all folds and boundary AUC, closing learned prediction-space stacking.
12. **V11 -> V12:** after prediction-space saturation, the next experiment had to change respondent representation. ModernNCA was chosen because fixed retrieval had shown diversity but poor geometry; ModernNCA directly learns supervised neighborhood geometry and is more diagnostic/operationally cleaner than TabR or resuming xRFM.
13. **Current V12:** no model-quality evidence exists yet. The only latest evidence is a pre-training source/provenance failure caused by floating TALENT main + stale byte hash. Fix implementation, then evaluate according to frozen gates.
