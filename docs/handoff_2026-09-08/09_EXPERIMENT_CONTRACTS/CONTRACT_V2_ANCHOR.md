# V2 Anchor Contract

V2 remains the canonical raw-model anchor even though recovery is the current submission champion.

Key implementation ideas:
- 2 x 5 StratifiedGroupKFold, groups=`id`, seeds 42 and 2026;
- LGB/XGB/Cat regression core;
- raw + dictionary-semantic + nonstandard-code representations;
- fold-local feature selection;
- categorical, monotonic, multiclass-expected, cumulative-ordinal sidecars;
- continuous blending then QWK thresholds;
- one prediction per 23,666 unique test IDs.

Original results:
- core crossfit 0.479735
- final crossfit 0.481366943
- global diagnostic 0.481587476
- Public LB 0.48193

Original selected sidecars:
- `lgb_cumulative_ordinal`, alpha 0.45
- `xgb_categorical`, alpha 0.05

Caveats:
- original sidecar/alpha selection was not fully nested against all OOF;
- original headline was row-weighted, later evaluator-aligned unique-ID diagnostic ~0.48062;
- rerun variance is nontrivial.
