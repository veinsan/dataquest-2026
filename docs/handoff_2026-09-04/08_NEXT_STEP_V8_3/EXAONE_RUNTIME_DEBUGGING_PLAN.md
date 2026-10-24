# V8.3 - EXAONE Runtime Debugging Plan

## Goal

Turn EXAONE from an ambiguous “runtime failed” branch into one of two clean states:

1. **RUNNABLE:** deterministic install/import/load/fit/predict on Kaggle T4, ready for the frozen V8 probe gates; or
2. **FAIL-CLOSED:** a documented reproducible incompatibility, after which `RUN_EXAONE=False` and V8 proceeds with TabICLv2/TabPFN-3 without changing the statistical design.

V8.3 is **not** a model-tuning version. It should not alter V2, folds, thresholds, promotion gates, feature semantics, or leaderboard strategy.

## Stage 0 - Preserve invariants

Before touching EXAONE:

- do not reinstall PyTorch;
- keep the exact Kaggle accelerator target T4 x2;
- keep `groups=id` folds unchanged;
- keep V8.2 feature construction unchanged;
- do not run TabICL/TabPFN concurrently during EXAONE debugging;
- use EXAONE on `cuda:0` only;
- clear Python/CUDA objects between smoke stages;
- do not use any external task data.

## Stage 1 - Environment fingerprint

Record before installation:

- Python full version;
- platform/container information;
- PyTorch version;
- CUDA availability and runtime version;
- GPU count/name/total VRAM;
- `pip freeze` or at least core packages (`numpy`, `pandas`, `scikit-learn`, `torch`, `huggingface_hub`, `safetensors`);
- current `PYTORCH_CUDA_ALLOC_CONF`;
- available free VRAM before EXAONE.

Expected debug artifact in a future run: `exaone_environment_manifest.json`.

## Stage 2 - Pin the EXAONE runtime, not only the checkpoint

V8.1 pins the HF **checkpoint revision**, but installs the Python package from GitHub without a commit pin. That is a reproducibility gap.

V8.3 should:

1. install from the public repo once in an isolated smoke environment;
2. resolve the actual Git commit of the installed package;
3. rerun installation pinned to that commit, for example conceptually `git+...@<resolved_commit>`;
4. record package metadata and import path;
5. never silently upgrade that runtime inside the final V8 notebook.

Do not invent the commit in advance. The point of V8.3 is to identify a working one from the official runtime lineage and freeze it.

## Stage 3 - Import and API contract inspection

Before any checkpoint or GPU model allocation:

- import `exaonetabular`;
- print module file path;
- import classifier/regressor classes;
- inspect and save the signatures of:
  - `EXAONETabularClassifier.from_pretrained`
  - constructor if public;
  - `.fit`
  - `.predict_proba`
  - regressor equivalents.

Compare the installed runtime signature with the V8 call arguments:

- `device`
- `ensemble_count`
- `seed`
- `max_vram_bytes`
- `revision`

If an argument is absent/renamed, stop here and fix the adapter. Do not misclassify an API mismatch as a model-quality failure.

Expected debug artifact: `exaone_api_signature.txt`.

## Stage 4 - Checkpoint download and identity

Use the existing pinned checkpoint contract:

- repo: `LG-AI-Research/EXAONE-Tabular`
- HF revision: `96946d008279216721e951dd3ee2f55a38877864`
- classifier: `exaone-tabular-classifier-v1_default.safetensors`
- regressor: `exaone-tabular-regressor-v1_default.safetensors`

Actions:

1. explicitly download with `hf_hub_download`;
2. save local path;
3. compute SHA256;
4. confirm nonzero file size;
5. distinguish auth/network/cache errors from runtime errors.

This stage should succeed before `from_pretrained` is called.

## Stage 5 - Constructor-only smoke test

Start with the cheapest possible official configuration:

- `ensemble_count=1` for the smoke test;
- one GPU (`cuda:0`);
- same pinned checkpoint revision;
- a conservative VRAM cap compatible with T4.

First try the same 13 GiB cap used by V8. If the constructor fails from memory pressure, a **diagnostic-only** lower cap sweep can be used to determine whether the runtime supports offload/degraded memory mode. Do not change the production cap just to obtain a better score.

Success criteria:

- object is created;
- no checkpoint mismatch warning that invalidates the model;
- no persistent CUDA allocation leak after object deletion.

## Stage 6 - Tiny fit/predict contract

Use only competition-derived rows from classes 1 and 2, selected from one fixed training side of the frozen fold split.

Recommended smoke sequence:

1. small balanced subset of a few hundred rows;
2. full V8.2 feature set;
3. `float32` numeric matrix;
4. target encoded exactly as `class2 -> 1`, `class1 -> 0`;
5. call `.fit`;
6. call `.predict_proba` on a small held-out slice.

Validate:

- output rows match input rows;
- probabilities are finite;
- probability matrix has the expected class dimension;
- `model.classes_` can be used to identify the positive class robustly;
- no assumption that “column 1” is positive without checking class order.

## Stage 7 - Data-compatibility staircase

If tiny full-feature fit fails, isolate the data dimension systematically:

A. tiny rows + reduced feature subset;  
B. tiny rows + full features;  
C. larger rows + full features;  
D. complete fold-0 class1/2 training rows + `ensemble_count=1`;  
E. complete fold-0 class1/2 rows + probe `ensemble_count=4`.

At each stage classify the first failing transition.

Inspect specifically:

- NaN handling;
- infinite values;
- constant columns;
- feature count limits/internal selection;
- dtype conversion;
- duplicated column names;
- unexpected categorical/object dtype leakage into the numeric EXAONE matrix.

## Stage 8 - GPU isolation and memory diagnostics

During EXAONE debug:

- keep TabICL and TabPFN unloaded;
- run only on `cuda:0`;
- call `gc.collect()` and `torch.cuda.empty_cache()` between attempts;
- log allocated/reserved/free VRAM before constructor, after constructor, after fit, and after deletion;
- do not assume two visible T4 GPUs means the EXAONE runtime will split itself across them.

Only after EXAONE is stable should V8 return to the original multi-family scheduling.

## Stage 9 - Full fold-0 specialist smoke

Run the exact V8 specialist semantics on fold 0:

- train side = all rows outside primary fold 0;
- keep only train labels 1/2 for EXAONE fit;
- validation = entire fold 0;
- compute `P(class2 | y<=2)`;
- reconstruct conditional ordinal score with frozen V2 tail probabilities;
- calculate the existing V8 metrics.

At this point runtime debugging ends. **Do not change a gate because the score is disappointing.**

## Stage 10 - Restore successive-halving evaluation

If fold 0 executes successfully, use the predeclared V8 logic:

- fold-0 survivor if delta AUC or diagnostic gain passes the frozen survive rule;
- only survivors get folds 2 and 3;
- only promoted candidates get full CV;
- only then consider secondary full-task classifier/regressor runs.

## Failure taxonomy to record

Use one primary failure code per attempt:

- `INSTALL_RESOLUTION`
- `IMPORT_ERROR`
- `API_SIGNATURE_MISMATCH`
- `HF_DOWNLOAD_OR_CACHE`
- `CHECKPOINT_RUNTIME_MISMATCH`
- `CUDA_OOM_OR_ALLOCATOR`
- `DATA_DTYPE_SHAPE_NAN`
- `FIT_INTERNAL_ERROR`
- `PREDICT_OUTPUT_CONTRACT`
- `UNKNOWN`

Save full exception text and traceback in the future debug run. This documentation ZIP intentionally does not contain raw logs.

## Fail-closed rule

If EXAONE still cannot pass the fold-0 runtime smoke after the official runtime/checkpoint contract is pinned and diagnosed:

1. set `RUN_EXAONE=False`;
2. record the failure reason and environment;
3. do **not** alter V2;
4. do **not** relax V8 gates;
5. continue the same conditional experiment with TabICLv2 and TabPFN-3;
6. revisit EXAONE only when a concrete runtime/library change provides new evidence.

## What V8.3 must not become

Do not let debugging expand into:

- hyperparameter search;
- new feature engineering;
- new cross-validation scheme;
- threshold tuning;
- a new ensemble objective;
- TabM/RealMLP experimentation;
- leaderboard submission probing.

Those are separate hypotheses. V8.3 has one job: **make EXAONE runtime status unambiguous**.
