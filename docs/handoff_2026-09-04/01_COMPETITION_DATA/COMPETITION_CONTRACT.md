# Competition and Modeling Contract

## Task

Objective Quest DATAQUEST 5.0 is treated in this project as an ordinal tabular prediction task:

- **Target:** `life_satisfaction`
- **Classes:** 1, 2, 3, 4
- **Metric:** Quadratic Weighted Kappa (QWK)
- **Primary entity:** respondent `id`
- **Important related key:** `household_id`

The target is ordinal, so the core modeling strategy throughout V1-V8 is to estimate a **continuous latent satisfaction score** and discretize it with three cut points. This is preferable to treating the labels as four unrelated classes when the metric quadratically penalizes larger ordinal errors.

## Competition files

The supplied raw competition ZIP contained only:

- `train_final.csv`
- `test_final.csv`
- `data_dictionary.csv`
- `sample_submission.csv`

These raw files are intentionally **not included** in this documentation handoff.

## Submission contract learned during V1/V2

A crucial evaluator mismatch was discovered early:

- `test_final.csv`: **23,902 physical rows**
- `sample_submission.csv`: **23,902 rows**
- unique `test_final.id`: **23,666**

The evaluator behavior validated by the V1/V2 leaderboard transfer requires **one prediction per unique test ID**. V2 therefore aggregates duplicate physical rows by `id` before final thresholding/export and produces exactly **23,666 predictions**.

This is a permanent project invariant. A later notebook should never regress to blindly mirroring the 23,902-row sample file.

## Integrity constraints carried forward

The project explicitly rejects the following:

- external respondent/source reconstruction;
- mapping competition rows back to external records to obtain labels;
- manual test-label editing or hardcoding;
- leaderboard probing as a label source;
- hidden test-label inference from submission feedback;
- pseudo-labeling driven by test leakage;
- supervised encodings built using validation labels;
- treating duplicate respondents as independent train/validation observations;
- tuning model-selection decisions on an outer validation fold that is later reported as honest OOF.

Foundation checkpoints are treated as model priors, not external task data. Project notes record that pretrained foundation models were later considered permitted by the organizer, but the final competition package should preserve the organizer clarification or rule evidence alongside the reproducibility manifest.

## Runtime target

Most later notebooks were designed for Kaggle **GPU T4 x2** with Internet enabled for public package/checkpoint retrieval. Runtime reproducibility matters because foundation packages can drift even when model checkpoints are pinned.
