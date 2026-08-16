# Residual-Adaptive Column Generation for Two-Layer ReLU Fitting

[![Paper CI](https://github.com/epicycloids/relu-chambers/actions/workflows/paper.yml/badge.svg)](https://github.com/epicycloids/relu-chambers/actions/workflows/paper.yml)

This repository contains the paper, implementation, experiment drivers, and
result records for scalar-output, fully connected two-layer ReLU fitting by
residual-adaptive column generation (RCG).

**[Paper (PDF)](https://github.com/epicycloids/relu-chambers/releases/latest/download/relu-chambers.pdf)**
is available with its [LaTeX source](paper/main.tex).

## Problem and implementation status

The code fits

$$
\min_{P,\alpha,u_j}
\frac12\left\lVert\sum_{j=1}^P \alpha_j(Xu_j)_+-y\right\rVert_2^2
+\lambda\sum_{j=1}^P|\alpha_j|,
\qquad \lVert u_j\rVert_2\le 1.
$$

The signed ReLU atoms form a convex atomic-gauge problem. The paper's ideal
column-generation analysis assumes an exact restricted-master solve and exact
global pricing of both signs. Exact pricing is NP-hard in general.

`relu_chambers.solve_rcg` is the main implementation of a practical
approximation:

- an active-set lasso solves each finite working set;
- multistart pricing searches a row sketch and rescans finalists on all rows;
- a gated group-lasso proposes joint direction changes, accepted only after a
  full-data finite-master refit does not increase the computed objective; and
- pruning, deduplication, and a column cap bound the working set.

The returned model is feasible for the displayed objective, but heuristic
pricing can miss an atom. `gap_heur` and `gap_cert` are numerical estimates,
not rigorous global-optimality certificates; the result records this explicitly
as `gap_cert_is_rigorous == False`.

## Install

Python 3.10 or newer is required. The core NumPy solver is installed with:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Install the paper and experiment dependencies with:

```bash
python -m pip install -e '.[experiments]'
```

For a lockfile-based environment, use `uv sync --extra experiments`.
PyTorch is optional and deliberately not pinned by the project because CUDA,
ROCm, and CPU installations require platform-specific builds.

## Minimal usage

```python
import numpy as np

from relu_chambers import solve_rcg
from relu_chambers.synthetic_data import make_planted

problem = make_planted(
    n=10_000,
    d=16,
    k=3,
    rng=np.random.default_rng(0),
)
solution = solve_rcg(
    problem["X"],
    problem["y"],
    lam=0.1,
    device="cpu",
    max_iter=100,
    rng_seed=0,
)

print(solution["obj"], solution["n_active"], solution["stop_reason"])
print(solution["gap_cert"], solution["gap_cert_is_rigorous"])
```

Recompute the objective from `solution["U"]` and `solution["beta"]` when
comparing externally supplied runs. The controlled benchmark harness does this
independently.

## Hardware paths

The default is NumPy with float64 data:

```python
solution = solve_rcg(X, y, lam, device="cpu")
```

Logical row shards can share one host through a thread pool:

```python
solution = solve_rcg(
    X,
    y,
    lam,
    devices=["cpu", "cpu", "cpu", "cpu"],
    threads=True,
    dtype_data="float64",
)
```

With an appropriate PyTorch build, use `device="cuda"` for one CUDA or ROCm
device, or `devices=["cuda:0", "cuda:1"]` for multiple devices. The engine is
multi-device aware, but the current implementation still gathers an `O(n)`
residual and some chamber paths transfer `O(nP)` masks. The paper does not
claim distributed or measured multi-GPU scaling.

## Verify the code

```bash
python -m tests.test_solver
python -m tests.test_data_engine
python -m tests.test_pricing_bounds
python -m experiments.controlled_benchmark --self-test
python -m experiments.candidate_order --self-check-only
```

The benchmark default is a smoke test. Evidence runs require at least three
measured repeats, one warm-up, and `--final-verify`; runs above 10,000 rows also
require `--large`.

## Paper and experiments

The Paper CI workflow validates the code, regenerates the figures, and builds
the PDF on pushes to `main` and on pull requests; each run retains
`relu-chambers.pdf` as a workflow artifact. Pushing a tag matching `v*` also
publishes that exact PDF as a GitHub Release asset.

Build the paper with Tectonic:

```bash
tectonic paper/main.tex
```

Regenerate its figures with:

```bash
python -m experiments.paper_figures
```

[`REPRODUCING.md`](REPRODUCING.md) maps each paper-cited result to its driver
and records the exact controlled commands. [`DATA.md`](DATA.md) documents the
external datasets, licenses, checksums, and preprocessing protocol.

## Repository map

```text
relu_chambers/
  solver.py              RCG loop and solve_rcg entry point
  data_engine.py         row-sharded NumPy/Torch data engine
  restricted_master.py  active-set restricted lasso
  adaptive_pricing.py   sketched two-sign search and full-row rescoring
  consolidation.py      joint frozen-mask proposal
  pricing_bounds.py     numerical evaluation of exact-arithmetic bounds
  synthetic_data.py     planted problems and chamber statistics
  reference/            small-problem and comparison implementations
experiments/
  baselines.py               neural-network comparison methods
  controlled_benchmark.py    controlled repeated benchmark harness
  candidate_order.py         candidate-order component study
  feature_reuse.py           feature-reuse component study
  exploratory_benchmarks.py planted, scaling, and optimizer comparisons
  synthetic_studies.py      enumerable and pricing studies
  real_data.py              real-data comparisons
  prepare_data.py           YearPredictionMSD preparation
  large_scale.py            standalone million-row runs
  paper_figures.py          all five manuscript figures
tests/                 focused correctness checks
results/               the 16 JSON records named by the paper
paper/                 manuscript, bibliography, five used figures, and PDF
```

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

The software and packaging files are licensed under the
[MIT License](LICENSE). The paper, figures, result records, and repository
documentation are licensed under
[CC BY 4.0](LICENSE-CONTENT.md). Downloaded datasets and third-party
dependencies retain their own licenses.
