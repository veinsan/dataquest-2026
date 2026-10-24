# Updated Hypothesis Space After V11

## Facts

1. Recovery ensemble 0.48334 is the current LB-confirmed champion.
2. V11 runtime was valid but prediction-space stacking was strongly negative on QWK, every fold, bootstrap, and first-boundary AUC.
3. V9 showed a generic in-context foundation prior can be strongly worse on the exact hard boundary.
4. V10 did not produce quality evidence and must not be counted as a scientific failure.
5. Current raw/global model family predictions are highly correlated.

## Inference

The next useful improvement likely requires changing **respondent representation/ranking**, not how existing predictions are weighted or thresholded.

The near-term bottleneck is first-boundary ranking around 1↔2. A candidate that only changes calibration without increasing AUC is unlikely to produce robust next-step value.

## Open hypotheses with independent-signal potential

### 1. ModernNCA learned neighborhood geometry — selected V12

Fixed/raw retrieval failed but produced more independent residuals. ModernNCA directly learns an embedding where target-similar respondents become neighbors, testing whether the failure was the geometry rather than the local-learning concept.

### 2. TabR learned retrieval — secondary

Potentially powerful, but mixes learned representation with retrieval/label-interaction complexity. More difficult to interpret if it fails. It should follow, not precede, a ModernNCA test.

### 3. Psychometric/latent wellbeing representation — open research

Survey variables may reflect a latent wellbeing construct. This could be qualitatively different from global GBDT representation, but internal evidence is currently weaker than for the learned-geometry hypothesis.

### 4. Full-target TabM/RealMLP — technically open, lower posterior

V5 only falsified B1-specialist versions, not full-target training. However the main desired first-boundary signal was already weak in V5, so expected information gain is lower than ModernNCA.

## Closed/deprioritized hypotheses

Prediction stacking, calibration-only tuning, household/quasi-key lookup, naive retrieval, generic feature expansion, generic foundation cycling, and B1-specialist neural routes should not be repeated without materially new evidence.
