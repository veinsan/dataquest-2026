# Runtime Debugging History and Lessons

## V8.x EXAONE

The project learned to separate install/import/checkpoint/runtime from quality. V8.3 finally verified exact runtime commit/checkpoint SHA and a T4-specific attention patch, but actual large fold inference remained resource-infeasible. This is why later designs use explicit Stage R before scientific screening.

## V9 TabICL

V8's TabICL failure was resource/offload-related, not quality. V9 upgraded to TabICL 2.2.0 with explicit disk/CPU offload and produced a clean runtime proof. Only then was the strong negative Stage-S result accepted as scientific falsification.

## V10 xRFM

See `../03_EXPERIMENT_HISTORY/V10_DEBUGGING_LINEAGE.md`. Key lessons:
- inspect actual library API rather than assume paper/config parameter names;
- preflight GPU compute capability for specialized kernels;
- understand which “memory controls” alter statistical behavior;
- derive tensor memory complexity before expensive runs;
- chunk the dimension causing the allocation, not a convenient but irrelevant axis;
- runtime workarounds such as solver replacement can introduce numerical assumptions;
- mandatory HPO search should fail fast on an unrecoverable candidate to avoid hours of wasted compute;
- clean stale worker directories and propagate underlying tracebacks.

## Current V12

The latest failure occurred before training because provenance checking was internally inconsistent:
1. install TALENT from floating `@main`;
2. compare a source file against a fixed SHA from an earlier main snapshot;
3. fail when main changes even if API/scientific mechanism is still compatible.

Actual observed source SHA: `02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f`.  
Stale expected SHA: `8c0e34271c2cbac32d5723e1d1ec5857eba63e91f5b3f3bd4ee2f23959aece6e`.

Correct fix principle:
- pin an exact audited TALENT Git commit;
- install declared dependencies first;
- verify installed commit provenance;
- verify ModernNCA signatures/behavior and official config semantics at that commit;
- preserve frozen V12 config draws;
- do not simply remove integrity checks.

No V12 training or quality evaluation has occurred yet.
