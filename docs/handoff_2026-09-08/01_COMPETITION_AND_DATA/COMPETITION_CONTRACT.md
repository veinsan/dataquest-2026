# Competition Contract

## Task

Objective Quest DATAQUEST 5.0 is treated as an ordinal tabular prediction problem:
- target: `life_satisfaction`
- labels: 1, 2, 3, 4
- metric: Quadratic Weighted Kappa (QWK)
- primary modeling/evaluation entity: respondent `id`
- related identifier: `household_id`

The project has repeatedly found continuous latent-score regression followed by three ordinal cutpoints more reliable than treating classes as unrelated.

## Competition files

Canonical raw files:
- `train_final.csv`
- `test_final.csv`
- `data_dictionary.csv`
- `sample_submission.csv`

Raw data are excluded from this handoff.

## Project-recorded competition constraints

The project recorded that public/open pretrained tabular models were permitted provided the run was reproducible and did not use task-relevant external labels/data or test-label reconstruction. This permission drove V7–V9 foundation experiments. Keep the organizer clarification/rule evidence with any final competition package if available.

At the V9 planning point, the project context recorded a maximum of 3 submissions/day and roughly three days remaining. Treat those timing/limit notes as historical unless the current competition page is rechecked.

## Permanent integrity constraints

Never use:
- external respondent/source reconstruction;
- manual/hardcoded test labels;
- leaderboard feedback as pseudo-labels;
- external task-relevant labeled data;
- outer-validation labels to select preprocessing/features/model weights that are later reported as honest OOF;
- duplicate respondent rows as independent train/validation entities.

## Submission/evaluator contract

Raw test contains 23,902 physical rows but only **23,666 unique `id`** values. Current champion and V2 workflows operate on one prediction per unique respondent ID.

There was an intermediate V5.1 mapping branch that considered the 23,902-row sample order as primary. That branch is **not canonical** now. The later project contract, successful V2 submission, recovery submission, and all subsequent validation use **23,666 unique IDs**. Do not revive the V5.1 mapping interpretation without direct evaluator evidence.
