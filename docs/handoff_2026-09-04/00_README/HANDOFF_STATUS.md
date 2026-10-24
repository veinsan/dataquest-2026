# Handoff Status Snapshot

## What is known with high confidence

- Competition target: `life_satisfaction`, ordinal classes 1-4, evaluated with Quadratic Weighted Kappa (QWK).
- Train: **95,609 rows x 179 columns**.
- Test: **23,902 rows x 178 columns**.
- Unique train respondent IDs: **94,663**.
- Unique test respondent IDs: **23,666**.
- V1 Public LB: **0.48003**; V1 cross-fit OOF: **0.479662**.
- V2 Public LB: **0.48193**; V2 cross-fit OOF: **0.481367**.
- V6 honest final cross-fit QWK: **0.478832**, with dominant class 1/2 difficulty.
- V8 architecture is a frozen-V2 plus conditional-foundation-specialist design.
- V8.1 installs/logs foundation runtimes after V2 is frozen.
- V8.2 creates fold-safe foundation input views without target encoding.
- Next engineering task: isolate and repair **EXAONE runtime execution** in V8.3.

## What is intentionally not claimed

- No verified V7 final OOF or Public LB score was recovered from the available archive.
- No verified V8 foundation probe result was recovered from the available archive.
- `0.48853`, `0.56232`, and `0.61541` are historical leaderboard/reference targets used by the project, **not this team's model scores**.
- This handoff does not assert that the current top leaderboard score on 2026-09-04 is still any of those values.
- This handoff does not reconstruct or include any ground-truth labels for the competition test set.

## Best next action

Run a minimal EXAONE smoke path before spending another full repeated tree/foundation CV cycle. The smoke path must establish, in order:

`install -> import -> API signature -> checkpoint download/hash -> from_pretrained -> tiny fit -> predict_proba -> full-feature small fit -> fold-0 specialist`.

Only after all those pass should V8's predeclared `delta AUC`, conditional QWK, and promotion gates be evaluated.
