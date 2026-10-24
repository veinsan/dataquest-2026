# V8 Current State Before V8.3

## Statistical architecture is already defined

V8 is not blocked on “what model should we try?” It is blocked on proving that the EXAONE implementation path is stable enough to test the existing hypothesis.

### Frozen components

- Exact V2 anchor.
- Grouped respondent validation.
- V2 tail calibration to `P(y>2)` and `P(y>3)`.
- 1/2 conditional specialist formulation.
- Successive-halving probe folds `[0,2,3]`.
- Positive Ridge / smooth blend concept.
- Cross-fitted QWK thresholding.
- Unique-ID submission contract.
- Foundation checkpoint revisions.

### V8 conditional score

Given a foundation specialist probability:

`q = P(y=2 | y<=2)`

and V2-calibrated tails:

`p2 = P(y>2)`  
`p3 = P(y>3)`

construct:

`p1 = p2 + (1-p2)*q`

then:

`s = 1 + p1 + p2 + p3`

This leaves the higher ordinal structure anchored in V2 and assigns the new model only the boundary it is supposed to improve.

## Current candidates

- EXAONE Tabular
- TabICLv2
- TabPFN-3 rescue

## Current EXAONE implementation shape

The recovered V8 code attempts:

1. install the EXAONE runtime from its public GitHub repository;
2. import `EXAONETabularClassifier` / `EXAONETabularRegressor`;
3. download explicit classifier/regressor safetensors at a pinned HF revision;
4. instantiate with device, ensemble count, seed, `max_vram_bytes`, and checkpoint revision;
5. fit a binary class-1/class-2 training subset;
6. obtain `predict_proba` and use the class-2 probability.

## Why V8.3 must be runtime-first

Several distinct failures can look identical if the full notebook is run end-to-end:

- Git package dependency resolution;
- Python import/API drift;
- unsupported constructor argument;
- checkpoint/runtime incompatibility;
- HF download/cache issue;
- CUDA allocator or OOM;
- dtype/NaN/shape incompatibility;
- failure during internal feature selection;
- fit succeeds but probability/class semantics differ from assumptions.

A full V2 + all-foundation run obscures which layer failed. V8.3 should isolate them one by one.
