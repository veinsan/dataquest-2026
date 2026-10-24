# Data Provenance and Integrity Fingerprints

Latest executed logs verified:

| File | SHA256 |
|---|---|
| train_final.csv | `0ccc7f7aa3d9e7520d1edd0cae30c1ac119fcb4c508fdd7c9ebc871ceaee2056` |
| test_final.csv | `c7027da38aba915e6a9952d21c6cd960b22c172e0ccf3f9f570d63d07e008740` |
| data_dictionary.csv | `4ee1269e4e10fd62225916bb8eff1cd51930a211cb8488d6c97a5b4817e4f631` |
| sample_submission.csv | `3b27035c9910f24c7a0cf29f85c1382be423f58cda0524bbd7b0a24ab9e65abf` |

Shapes:
- train: 95,609 x 179
- test: 23,902 x 178
- unique train IDs: 94,663
- unique test IDs: 23,666
- dictionary: 187 x 3 in the original audit

Use these hashes to detect accidental dataset drift before comparing a new run to historical OOF/LB evidence.
