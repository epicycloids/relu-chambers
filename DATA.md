# External datasets

The repository includes generated planted problems and small JSON result
records, but it does not redistribute the three real datasets. The experiment
driver stores downloads under the ignored `data_cache/` directory.

## California housing

`python -m experiments.real_data california` uses scikit-learn's
`fetch_california_housing` loader. Scikit-learn downloads the 20,640-row
dataset from [Figshare](https://figshare.com/articles/dataset/cal_housing_tgz/3829992),
where the archive is published under CC BY 4.0. The standard reference is
R. Kelley Pace and Ronald Barry, “Sparse Spatial Autoregressions,” 1997.

## Covertype

`python -m experiments.real_data covtype` uses scikit-learn's
`fetch_covtype` loader. The [UCI Covertype record](https://doi.org/10.24432/C50K5N)
contains 581,012 rows, identifies Jock Blackard as the creator, and licenses
the data under CC BY 4.0.

## YearPredictionMSD

Prepare the cache before running the experiment:

```bash
python -m experiments.prepare_data msd
```

Pass the same `--cache-dir PATH` option to this command and to
`experiments.real_data` when using a cache outside the repository.

The downloader obtains the archive from the
[UCI YearPredictionMSD record](https://doi.org/10.24432/C50K61), verifies the
archive and source-text SHA-256 digests, and writes `data_cache/msd.npz`.
UCI identifies Thierry Bertin-Mahieux as the creator and licenses the data
under CC BY 4.0.

The UCI record recommends its first 463,715 rows as training data and its last
51,630 rows as test data to keep a producer's songs in one split. The paper's
reported experiment instead uses the explicitly documented seeded 75/25
random split implemented in `experiments/real_data.py`. Reproducing the paper
requires that split; it should not be mistaken for UCI's artist-separated
protocol.
