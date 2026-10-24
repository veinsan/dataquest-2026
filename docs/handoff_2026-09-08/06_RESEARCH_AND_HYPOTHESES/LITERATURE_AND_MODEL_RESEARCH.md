# Literature and Model Research That Materially Changed Decisions

This is a decision-oriented bibliography, not an exhaustive survey.

## TabPFN-3 / tabular foundation models

Foundation priors motivated V7 after organizer permission for open pretrained weights. Internal evidence ultimately dominated literature averages: TabPFN-3 and TabICLv2 did not improve DATAQUEST first-boundary ranking.

## TabICLv2

Repository/model used in V9: `jingang/TabICL`, package `tabicl==2.2.0`, regressor checkpoint `tabicl-regressor-v2-20260212.ckpt`, verified SHA256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`.

## EXAONE-Tabular

V8.3 pinned runtime commit `8638e07d09fad154249bd75ba4786181491f7025`, HF revision `093ad1c2613e1e95d26df8610451d2e02ed53cbe`, classifier checkpoint SHA `2aa723e024e5c56c778d6c79cc3cee7675d1b6a03c06fa08c80e620e228b224d`. Runtime feasibility, not predictive quality, blocked the route.

## xRFM

Paper: **xRFM** (2025), arXiv `2508.10053`. The learned local feature metric + recursive partition idea directly motivated V10 after global/foundation saturation. Runtime complexity and T4 compatibility made the experiment operationally expensive; no scientific quality conclusion was reached.

## TabM

Paper: **TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling**, arXiv `2410.24210`. V5 tested only the B1-specialist form; full-target TabM remains technically open but lower priority.

## TabR

Paper: **TabR: Tabular Deep Learning Meets Nearest Neighbors**, arXiv `2307.14338`. Kept as a learned-retrieval follow-up if ModernNCA validates local geometry.

## ModernNCA

Paper: **ModernNCA: Renewing Old Wisdoms for Tabular Data**, arXiv `2407.03257` / ICLR 2025. It predicts from neighbors in a learned embedding, with stochastic candidate sampling during training. This is the active V12 hypothesis because it directly tests whether V4's fixed retrieval failed due to the similarity geometry.

Official TALENT source audited during latest V12 debugging:
- repository: `LAMDA-Tabular/TALENT`
- current project installation docs recommend Git main; for competition reproducibility V12 must pin an exact audited commit.
- current `pyproject.toml` dependencies include `qhoptim>=1.1.0` and `torch>=2.8.0`.
- `ModernNCA.forward`: learned encoder, optional PLR embeddings, stochastic candidate sample during train, `torch.cdist`, softmax(-distance), candidate-label weighted prediction.
- `ModernNCAMethod`: `cat_policy='tabr_ohe'`, `num_policy='none'`, AdamW, full training support during validation/predict, regression mean/std de-normalization.

## TabArena / TabRepo

TabArena (`arXiv:2506.16791`) reinforced that GBDTs remain very strong and model-family rank is dataset-dependent. TabRepo/PMLR work on stored prediction libraries helped justify the CPU recovery experiment and V11. Internal LB/OOF evidence is still higher priority than benchmark averages.
