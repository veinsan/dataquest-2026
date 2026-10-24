# Model Research Summary

This file records the project's prior model/literature conclusions. It is a handoff of earlier research, not a fresh web revalidation on 2026-09-04.

## 1. Tree families: still the anchor, not the future search space

### LightGBM + XGBoost + CatBoost

**Decision:** keep through exact V2.

Why:

- proven CV-to-LB transfer;
- strong handling of mixed survey signals and missingness;
- different enough inductive biases to support a stable ensemble;
- exact V2 implementation empirically beat later simplified rebuilds.

What not to do:

- another broad max-depth/leaf-count search expecting a 0.13 QWK jump;
- replace exact V2 with “V2-lite” for cleanliness;
- re-expand hundreds of engineered features without new evidence.

## 2. TabPFN-3

**Project priority after organizer permission:** primary foundation candidate.

Why it was selected:

- modern synthetic-pretrained tabular foundation approach;
- project research considered the ~95k x 176 dataset within its intended scale;
- classifier and regressor variants create objective diversity;
- very different prior from GBDTs;
- checkpoint can be explicitly versioned and hashed.

V7/V8 usage:

- V7: full-target classifier/regressor candidate;
- V8: repaired conditional 1/2 specialist candidate / rescue path.

Risks:

- checkpoint licensing/authentication in older V7 path;
- package/checkpoint API changes;
- T4 memory/runtime cost.

## 3. EXAONE Tabular

**Project priority:** high, with current engineering blocker.

Why it was selected:

- compact modern tabular foundation architecture in project research;
- synthetic-pretrained prior provides model-family diversity;
- classifier/regressor checkpoints are public and pin-able;
- V8 can assign it only the difficult 1/2 boundary rather than the full ordinal task.

Current V8 settings:

- checkpoint revision pinned to `96946d008279216721e951dd3ee2f55a38877864`;
- classifier/regressor safetensors explicitly named;
- EXAONE specialist on `cuda:0`;
- probe ensemble 4, full ensemble 8;
- VRAM budget 13 GiB.

Current concern:

- the Python runtime is installed from GitHub without a commit pin in V8.1;
- constructor/API/runtime drift can be confused with model failure;
- V8.3 must first establish a deterministic minimal run.

## 4. TabICLv2

**Project priority:** secondary foundation candidate.

V8 records:

- `tabicl==2.1.1`;
- explicit classifier/regressor checkpoint names;
- small batch size (2) for T4 memory;
- usually placed on `cuda:1` so it can be isolated from EXAONE on GPU 0.

Role:

- alternate foundation prior if EXAONE or TabPFN is weak/unavailable;
- candidate in the same V8 successive-halving specialist framework.

## 5. TabFM

**Project priority:** later/backup.

Why it was interesting:

- strong synthetic-pretrained tabular approach in prior research;
- class/feature counts appeared compatible.

Why it was deprioritized:

- project research expected memory/context cost to scale poorly with roughly 95k training rows on Kaggle T4;
- EXAONE/TabPFN/TabICL had clearer immediate tooling paths.

## 6. TabM

**Project priority:** train-from-scratch backup, not dead.

Important correction from V5:

V5 tested TabM primarily as a **boundary-1 binary specialist**. That negative result does not prove a full-target TabM is useless.

Why it remains conceptually suitable:

- dataset is overwhelmingly low-cardinality categorical-coded;
- modern TabM provides efficient MLP ensemble behavior and numerical embeddings;
- it would offer neural representation diversity without pretrained weights.

Why it is not V8.3's first task:

- runtime debugging EXAONE is already the identified blocker;
- introducing a new full-target model now would mix two hypotheses and reduce experimental clarity.

## 7. RealMLP

**Project priority:** train-from-scratch backup.

Like TabM, V5 only falsified the tested B1 specialist route. Full-target RealMLP was never established as a completed serious project branch.

Keep it as fallback if all permitted foundation models fail to produce new boundary signal.

## 8. Lower-priority older deep tabular families

The project explicitly deprioritized:

- TabNet;
- FT-Transformer;
- SAINT;
- generic embedding MLP / generic ResNet-style tabular nets.

Reason: if deep learning time is spent, the project already identified more modern/stronger options such as TabM, RealMLP, and foundation models.

## 9. Current model priority order

For the current V8 line:

1. **Exact V2 frozen anchor** - mandatory reference.
2. **EXAONE runtime repair** - V8.3 engineering priority, not yet a quality claim.
3. **TabPFN-3 conditional specialist** - strong foundation alternative.
4. **TabICLv2 conditional specialist** - strong alternate foundation prior.
5. **TabM / RealMLP full-target** - backup research if foundation route disappoints.
6. **TabFM** - later if hardware/runtime budget allows.
7. Older deep tabular families - low priority.

## 10. Ensemble research conclusion

V6 ended the era of direct discrete-QWK weight micro-tuning.

Preferred architecture:

`model continuous scores/probabilities -> non-negative/positive Ridge style smooth blend -> frozen continuous score -> cross-fitted QWK thresholds`

For V8 specifically, the foundation models should first improve the conditional 1/2 probability. The frozen V2 tails provide `P(y>2)` and `P(y>3)`.
