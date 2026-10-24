# Artifact Catalog and Provenance

This handoff does not embed heavy artifacts, but the following evidence existed in the working session.

## Historical results

`results.zip` contains V1 through V8.3 result directories with summary JSON, OOF/test predictions, fold metrics, feature/probe audits, and generated submissions. Use summary/OOF artifacts as actual experiment evidence; do not infer LB scores from a submission filename.

## V9

`results-v9.zip` contains:
- data/foundation manifests;
- verified TabICL checkpoint blob;
- runtime proof;
- Stage-S fold 0/2 predictions;
- Stage-S metrics/summary;
- same-run V2 anchor OOF/test.

## V10

The long runtime evidence is in Kaggle logs/executed notebooks. Important: `results-v10-error.zip` is **not V10 results**; inspection showed the same `oqdq5_v9/...` file tree as `results-v9.zip`. No V10 prediction artifact should be recovered from it.

## Recovery

`submission_cpu_recovery_v10_crisis.csv` is the LB-confirmed 0.48334 champion. It is excluded from this documentation package but its SHA/recipe are recorded.

## V11

`results-v11.zip` contains complete validation artifacts:
- crossfit thresholds
- duplicate stream audit
- family manifest
- fold metrics
- inner-C selection
- label-change matrix
- meta-feature manifest
- OOF predictions
- paired bootstrap
- provenance manifest
- summary

It intentionally contains **no test predictions/submission**, because the promotion gate failed.

## V12

Latest source notebook and log prove only dependency/provenance setup and the pre-training SHA failure. No ModernNCA quality artifact exists yet.

## Raw competition data

Available during audit but intentionally excluded. Hashes are recorded in `../01_COMPETITION_AND_DATA/DATA_PROVENANCE.md`.

## V8.3 artifact anomaly

A file named `probe_fold0_output.npz` in the V8.3 archive was observed during later audit to behave as a failure/manifest artifact rather than a normal numerical prediction NPZ. Do not trust file extensions alone; inspect content before loading or attributing model-quality evidence.
