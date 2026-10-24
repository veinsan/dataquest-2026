# Failure Taxonomy

The project deliberately separates failure types because runtime failure is not scientific falsification.

## Scientific failure

A valid model run generated held-out predictions and failed preregistered quality gates. Examples: V3, V4 tested candidates, V5 B1 formulation, V6 tested corrections, V7 TabPFN formulation, V9 TabICL, V11 stacker.

## Runtime/engineering failure

The intended model quality was never validly measured. Examples: V8.1/V8.2 EXAONE setup, much of V10, current V12 source-integrity failure.

## Resource infeasibility

The model implementation is technically valid but cannot fit the available hardware under the frozen design. Example: V8.3 EXAONE large-context path on T4. This still does not imply poor quality.

## Validation/design failure

An apparent score is not honest enough to support submission. Example: V6.1 global calibration optimism.

## Artifact/provenance failure

A filename or archive does not contain what it claims. Example: `results-v10-error.zip` contains V9 artifacts. Never infer experiment state from filename alone.
