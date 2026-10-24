# V8.3 Exit Criteria

## PASS: EXAONE is ready for model evaluation

All must be true:

- [ ] EXAONE runtime package is pinned to a specific reproducible source/version/commit.
- [ ] Classifier and regressor import cleanly.
- [ ] Runtime API signatures match the adapter used by V8.
- [ ] Pinned HF checkpoint downloads successfully.
- [ ] Checkpoint SHA256 is recorded.
- [ ] `from_pretrained` succeeds on `cuda:0` with smoke ensemble count 1.
- [ ] Tiny competition-derived class1/2 fit succeeds.
- [ ] `predict_proba` is finite and class order is verified.
- [ ] Full-feature small-row fit succeeds.
- [ ] Fold-0 full class1/2 specialist succeeds with ensemble 1.
- [ ] Fold-0 probe configuration succeeds with ensemble 4, or a documented official memory mode provides equivalent intended behavior.
- [ ] CUDA memory is released enough for subsequent branches.
- [ ] No validation/leakage rule was changed during debugging.

Then resume the frozen V8 successive-halving gates.

## FAIL-CLOSED: EXAONE is disabled for this run

Use this state if:

- the official runtime/checkpoint remain incompatible;
- reproducible T4 memory constraints prevent the intended specialist fit;
- package behavior cannot be pinned;
- output probability semantics cannot be validated.

Actions:

- [ ] set `RUN_EXAONE=False`;
- [ ] record exact failure taxonomy;
- [ ] keep V2 frozen;
- [ ] keep probe folds/gates frozen;
- [ ] continue TabICLv2/TabPFN-3;
- [ ] do not submit a degraded EXAONE workaround merely because it runs.

## Explicit non-exit

The following are **not** reasons to declare runtime debugging successful:

- constructor runs but `.fit` fails;
- tiny fit runs but full V8.2 features fail;
- output is generated but class-2 probability semantics are uncertain;
- fold-0 executes only after changing the validation split;
- score is good only with fold-local QWK optimization;
- score is poor and gates are manually loosened.
