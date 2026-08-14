# Result records

This directory contains the 16 JSON records underlying the empirical results
reported in the paper. The controlled records include raw timings, solver
configurations, data and model digests, and relevant software and hardware
metadata. The exploratory records are less complete and are labeled
accordingly in the paper.

The `test_acc` values in `real_data_covtype.json` are invalid and are not used
in the paper; the reported held-out metric is `test_mse`. The current driver
computes classification accuracy correctly for newly generated records.

See [`REPRODUCING.md`](../REPRODUCING.md) for the artifact-to-command map.

These records and this documentation are licensed under
[CC BY 4.0](../LICENSE-CONTENT.md).
