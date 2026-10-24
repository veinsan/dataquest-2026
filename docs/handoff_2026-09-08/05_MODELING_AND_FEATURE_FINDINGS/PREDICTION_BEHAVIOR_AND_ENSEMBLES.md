# Prediction Behavior, Correlation, and Ensemble Complementarity

## Accurate models are highly correlated

V2 and later anchors/variants commonly have residual correlations >0.99. This explains why adding another accurate global model usually changes QWK only at noise scale.

## Diverse models often lose accuracy

Examples:
- V4 retrieval mean residual corr ~0.9475, mean QWK gain negative.
- V5 TabM/RealMLP B1 specialists residual correlations ~0.25–0.56, but no stable boundary/QWK improvement.
- V9 TabICL raw score corr ~0.968 and residual corr ~0.988, but ranking/QWK were strongly worse.

## Recovery ensemble

The successful recovery ensemble is not highly orthogonal:
- V2 vs V5.1 continuous corr ~0.9921
- V2 vs V6.1 ~0.9950
- V2 vs V7 ~0.9994
- residual correlations roughly ~0.997–0.9997

Yet tiny averaging-induced changes improved LB. This supports a variance-stabilization explanation.

## V11 falsifies learned disagreement signal in tested form

V11 vs recovery:
- prediction corr `0.9971579761`
- residual corr `0.9988344995`
- label-change rate 4.292% (4,063 respondents)
- change matrix largest shifts: recovery 1 -> V11 2 = 2,260; recovery 3 -> V11 2 = 1,138

Despite more label changes, first-boundary AUC was essentially unchanged/slightly worse and QWK fell on all five folds. Therefore historical disagreement features did not produce new ranking information.
