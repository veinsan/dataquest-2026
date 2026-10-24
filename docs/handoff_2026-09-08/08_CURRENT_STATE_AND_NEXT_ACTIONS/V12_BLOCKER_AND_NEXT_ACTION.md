# V12 Blocker and Immediate Next Action

## Latest run evidence

Dependency/setup completed enough to detect:
- GPU0 Tesla T4 14.56 GiB cc 7.5
- GPU1 Tesla T4 14.56 GiB cc 7.5
- torch 2.10.0+cu128
- numpy 2.0.2
- sklearn 1.6.1
- optuna 3.6.1

pip emitted repeated warnings that TALENT 0.1.0 required `qhoptim>=1.1.0` while it was not yet installed. The notebook dependency cell later attempted explicit qhoptim installation.

Failure then occurred in cell 3/source-integrity verification, before any data/model training:

`Official ModernNCA source drift: expected 8c0e3427..., got 02fce6a1...`

## Root-cause diagnosis reached in the conversation

The notebook installed TALENT from floating GitHub `@main` and then demanded a byte hash belonging to a prior main snapshot. This makes normal upstream source edits look like a fatal scientific implementation drift.

Subsequent source audit found current official ModernNCA mechanism/API still aligned with V12 intent. Therefore the blocker is **reproducibility/provenance implementation**, not evidence against ModernNCA.

## Immediate next action

Produce one consolidated V12 notebook revision that:
1. pins an exact TALENT Git commit instead of resolving floating main every run;
2. pins/validates compatible required dependencies (`qhoptim`, Optuna, category encoders, delu, einops, etc.);
3. records exact installed source commit and relevant file hashes;
4. validates ModernNCA constructor/forward and ModernNCAMethod preprocessing/training/predict semantics;
5. freezes the exact eight official-space HPO configs rather than depending on future utility drift;
6. performs static worker/control-flow checks;
7. executes realistic Stage R before any expensive Stage S;
8. keeps one process/fold/GPU and full traceback propagation;
9. keeps all scientific gates unchanged.

Do not run full CV or generate a submission unless Stage-S promotion is actually satisfied.
