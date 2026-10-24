# Leaderboard and Submission History

## Verified own leaderboard scores

| Submission/pipeline | Public LB | Meaning |
|---|---:|---|
| V1 | **0.48003** | baseline, validated CV framework |
| V2 | **0.48193** | raw-model champion until recovery |
| CPU recovery ensemble | **0.48334** | **current champion, rank 4** |

No other version in this handoff has a verified higher Public LB attributable to it. Several result directories contain generated `submission_*.csv` files, but a generated submission file is **not** equivalent to a verified leaderboard result.

## Current leaderboard context

Current project-recorded scores above rank 4:
1. 0.61541
2. 0.56232
3. 0.48853
4. **0.48334 — this project**

Gap to rank 3: **+0.00519**.  
Gap to rank 2: **+0.07898**.  
Gap to rank 1: **+0.13207**.

This supports two separate objectives:
- near-term: plausible improvement toward 0.48853;
- qualitative breakthrough: requires a new information regime to approach 0.56–0.61.

## Submission integrity

Current canonical output has one row per **23,666 unique test respondent IDs** with labels in {1,2,3,4}.

Never submit partial/fallback predictions from an experiment that failed its promotion contract.
