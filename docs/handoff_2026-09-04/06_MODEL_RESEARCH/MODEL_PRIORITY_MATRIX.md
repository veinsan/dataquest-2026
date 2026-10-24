# Model Priority Matrix

| Family | Intended role | Evidence so far | Runtime risk | Current action |
|---|---|---|---|---|
| Exact V2 LGB/XGB/Cat + useful sidecars | Anchor | **Best verified: 0.481367 CF / 0.48193 LB** | Medium compute | Freeze exactly |
| EXAONE Tabular | V8 1/2 specialist | Design recovered, final quality open | **Current blocker** | **Debug in V8.3** |
| TabPFN-3 | V8 1/2 specialist / rescue | Foundation design established, final metric open | Checkpoint/runtime memory | Keep candidate |
| TabICLv2 | V8 1/2 specialist | Foundation design established, final metric open | Memory/batching | Keep candidate |
| TabM | Full-target backup | B1 specialist route did not jump | Training cost | Park, not dead |
| RealMLP | Full-target backup | B1 specialist route did not jump | Training cost | Park, not dead |
| TabFM | Later foundation backup | Research-interest only in this project | High row-context memory | Deprioritize |
| All-cat CatBoost sidecar | Generic sidecar | Selected only 1/5 V6 folds | Low | Drop as generic branch |
| XGB categorical sidecar | Generic sidecar | Selected only 1/5 V6 folds | Low-medium | Drop as generic branch |
| Cumulative ordinal | Stable sidecar | Selected 3/5 V6 folds | Low | Keep optional |
| CORN | Error-centric sidecar | ~tiny gain only | Medium | Freeze |
| Forensic lookup | Quasi-key expert | No reliable OOF gain | Leakage risk | Dead/frozen |
