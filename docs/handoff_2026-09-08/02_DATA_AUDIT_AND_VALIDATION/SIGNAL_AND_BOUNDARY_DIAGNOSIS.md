# Signal and Boundary Diagnosis

## Primary bottleneck: class 1 vs 2

Classes 1 and 2 are 95.34% of data. Across V2/V6/V7/V9 and related anchors:
- AUC `y>1` repeatedly sits near ~0.78
- AUC `1 vs 2` repeatedly sits near ~0.765–0.766
- higher boundaries are easier (`y>2` often ~0.87, `y>3` ~0.81)

V2 analysis estimated 1↔2 mistakes account for roughly **63% of squared disagreement mass**.

The project therefore uses first-boundary AUC as the main early diagnostic for new representations.

## Saturation pattern

Accurate global tree/ordinal variants have extremely correlated predictions/residuals. More orthogonal models frequently become weaker:
- V4 retrieval residual corr ~0.9475 but no accuracy gain
- V5 neural/specialist residual corr roughly 0.25–0.56, but no stable QWK/boundary breakthrough
- V9 TabICL residual corr ~0.9884 and clearly worse ranking
- V11 residual corr ~0.9988 and no ranking gain

Canonical inference: the project needs **independent accurate ranking signal**, not diversity for its own sake.

## Breakthrough scale

V5 sensitivity work suggested the high-LB regime would require a qualitatively larger first-boundary improvement than normal tuning. A rough target used in that research was ~0.893–0.896 first-boundary AUC, around +0.11 absolute versus the current ~0.78 region. Treat this as sensitivity/inference, not a proven mapping from AUC to LB.

This is why the rank-3 goal (0.48853, gap +0.00519 from champion) and rank-1/2 goals (0.56232–0.61541) are considered different regimes.
