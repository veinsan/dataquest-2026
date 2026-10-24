# Version Status Matrix

| Version | Main question | Status | Keep / discard |
|---|---|---|---|
| V1 | Can ordinal GBDT + grouped CV transfer? | **Confirmed positive baseline** | Keep as historical baseline |
| V2 | Can representation/objective diversity improve V1? | **Confirmed champion** | **Freeze exact executable anchor** |
| V3 | Do ontology/skip/TE structural features break ceiling? | **Negative** | Do not revisit same representation stack |
| V4 | Do all-cat, fingerprint, retrieval, deep MLP produce qualitative jump? | **Negative discovery** | Keep only ideas with new evidence |
| V5A | Can TabM/RealMLP/all-cat solve boundary 1 directly? | **Negative B1 specialist result** | Does not invalidate full-target TabM/RealMLP |
| V5B | Is there legal internal quasi-key/lookup signal? | **Negative / leakage-risk** | Freeze forensic lookup |
| V6A | Can distillation/difficulty/CORN solve ambiguous labels? | **Negative incremental** | Freeze techniques |
| V6B | Can recovery + sidecars restore/improve V2? | **Negative but diagnostic** | Keep cumulative ordinal only as optional stable branch |
| V7 | Can modern foundation models break 0.48? | **Design/runtime recovered; final score open** | Foundation family remains strategic |
| V8 | Can a conditional 1/2 specialist improve V2 without relearning tails? | **Current architecture** | **Keep** |
| V8.1 | Can foundation dependencies/checkpoints be made reproducible? | **Implemented, EXAONE runtime still needs isolation** | Harden runtime pinning |
| V8.2 | Can foundation views be made fold-safe? | **Implemented** | Keep |
| V8.3 | Can EXAONE execute deterministically on T4 before quality evaluation? | **NEXT** | Runtime-debug only, no architecture drift |
