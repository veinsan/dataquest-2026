# Leaderboard and Score History

## Verified team results

| Version | Honest internal score | Public LB | Interpretation |
|---|---:|---:|---|
| V1 | cross-fit QWK **0.479662** | **0.48003** | Very strong CV-to-LB transfer |
| V2 | cross-fit QWK **0.481367** | **0.48193** | Best verified project result; exact anchor |
| V3 | Did not clear breakthrough gate | Not used as trusted improvement | Exploratory only |
| V6 recovery | final cross-fit **0.478832** | No verified later team LB recovered | Below V2; calibration optimism diagnosed |
| V7/V8 | Final verified foundation metric not recovered | Not claimed | Keep status open |

## Historical leaderboard/reference targets used in project notes

These are **not team scores**. They appeared as changing strategy anchors in notebook configurations and handoff discussion:

| Reference | Role in project |
|---:|---|
| **0.48231** | near-term score/nearest-LB target in later V7 research notes |
| **0.48853** | historical strong/top reference used in V2/V8 configs |
| **0.56232** | historical Top-2 reference in V8 config |
| **0.61541** | historical Top-1 reference / aggressive breakthrough target |

Because leaderboard standings change over time, this documentation does not claim those values are the live standings on 2026-09-04.

## What CV-to-LB transfer taught us

V1 and V2 are unusually valuable because the public scores landed very close to grouped cross-fitted OOF. This implies:

1. The grouped validation contract is more trustworthy than leaderboard intuition.
2. A candidate that cannot clearly beat V2 in honest OOF should not be expected to gain massively on LB by luck.
3. Tiny calibration-only gains should not consume scarce submissions.
4. The large historical gap to 0.61541 is not plausibly closed by ordinary tree hyperparameter changes alone.

## Boundary interpretation of the gap

Project oracle-sensitivity analysis estimated that, if the easier higher boundaries stayed similar, moving QWK toward roughly 0.615 would require boundary-1 discrimination in the neighborhood of **AUC 0.893-0.896**, versus the project's roughly **0.780** region.

Treat this as a research sensitivity estimate, not a theorem. Its main message is scale: the missing signal is much larger than a tiny GBDT tuning gain.
