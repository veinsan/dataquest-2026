# Source Provenance

This handoff was synthesized from the user's project history and recovered project artifacts. It intentionally does not embed notebooks or raw outputs.

## Competition source reviewed directly

- `/mnt/data/oq-dq-5.zip`
  - `train_final.csv`
  - `test_final.csv`
  - `data_dictionary.csv`
  - `sample_submission.csv`

Used only to verify structural dataset audit values. None of these raw CSVs are copied into the handoff ZIP.

## Recovered project artifacts used as evidence

### V1
- `oq_dq_5_v1.ipynb`
- Used for ordinal baseline design, fold-safe preprocessing, three-tree model portfolio, threshold protocol.

### V2
- `oq_dq_5_v2.ipynb`
- Used for repeated 2x5 grouped CV, representation-diversity design, sidecars, unique-ID submission contract, verified score anchors.

### V3
- `oq_dq_5_v3.ipynb`
- `oq_dq_5_v3_fixed.ipynb`
- `oq_dq_5_v3_fixed_v2.ipynb`
- V3 Kaggle log pasted in prior chat.
- Used to confirm structural design and failure to clear breakthrough gate.

### V4
- `oq_dq_5_v4_breakthrough_discovery.ipynb`
- Used for candidate-family discovery contract and kill/promotion design.

### V5
- `oq_dq_5_v5_boundary_breakthrough.ipynb`
- `oq_dq_5_v5_forensic_internal_signal.ipynb`
- Used for B1 TabM/RealMLP specialist design and forensic lookup path.

### V6
- `oq_dq_5_v6_error_centric_ordinal.ipynb`
- `oq_dq_5_v6_recovery_allcat.ipynb`
- Retrospective handoff markdown covering V1-V6.
- Used for V6 metrics, sidecar frequencies, calibration-overfit diagnosis, pruning decisions.

### V7
- `oq_dq_5_v7_foundation_breakthrough.ipynb`
- Later research/handoff markdown on foundation-model permission and priority.
- Used for V7 foundation-first architecture, checkpoint strategy, promotion gates, exact-V2 anchor rationale.

### V8 / V8.1 / V8.2
- `oq_dq_5_v8_conditional_foundation.ipynb`
- Used for conditional 1/2 architecture, successive-halving gates, package/checkpoint versions, V8.1 runtime setup, V8.2 fold-safe feature views, EXAONE call shape.

## Excluded on purpose

- all `.ipynb` files;
- raw Kaggle logs;
- OOF prediction CSVs;
- test prediction CSVs;
- submissions;
- raw leaderboard screenshots;
- checkpoint files;
- raw competition data;
- `tabular_template.ipynb`.

## Evidence limitations

The recovered archive provides design/code evidence for V7/V8 but does not provide a final verified foundation OOF/Public-LB result that can safely be attributed to those versions. This handoff therefore leaves those result fields open rather than guessing.
