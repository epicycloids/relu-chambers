# Reproducing the paper artifacts

The included JSON files are the records used by the manuscript. Runtime and
floating-point details will vary across machines, so reproduction means rerun
the recorded protocol and compare objectives, selected models, and qualitative
decisions before comparing wall-clock time.

Install the experiment dependencies first:

```bash
uv sync --extra experiments
```

Install the appropriate PyTorch build separately for baseline-network or
accelerator experiments.

## Fast validation

```bash
uv run python -m tests.test_solver
uv run python -m tests.test_data_engine
uv run python -m tests.test_pricing_bounds
uv run python -m experiments.controlled_benchmark --self-test
uv run python -m experiments.candidate_order --self-check-only
```

## Controlled CPU comparison

The following command reconstructs the configuration stored in
`results/controlled_cpu_variants_n10000_d16_tolerance1e-4.json`:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
uv run python -m experiments.controlled_benchmark \
  --case 'baseline=cpu;ascent=both;order=correlation' \
  --case 'damped=cpu;ascent=damped;order=correlation' \
  --case 'damped_delta=cpu;ascent=damped;order=decrease' \
  --n 10000 --d 16 --k 3 --lam 0.1 --noise 0 \
  --seed 0 --solver-seed 0 --order-seed 1729 \
  --max-iter 60 --eps-rel 1e-4 --eps-heur 1e-4 --price-tol 1e-3 \
  --add-per-round 8 --max-cols 256 \
  --certify-every 5 --cert-steps 15 \
  --gated-every 1 --gated-iters 100 --sketch-m 0 \
  --stall-rtol 1e-7 --patience 6 \
  --repeats 3 --warmups 1 --quality-rtol 1e-4 --final-verify \
  --output results/controlled_cpu_variants_n10000_d16_tolerance1e-4.json
```

## Controlled accelerator-placement comparison

The recorded ROCm machine exposed its accelerator through PyTorch's `cuda`
spelling and required the shown architecture override. Omit the override on
hardware that does not need it.

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
uv run python -m experiments.controlled_benchmark \
  --case 'host_search=rocm;ascent=damped;order=decrease;search=host' \
  --case 'device_search=rocm;ascent=damped;order=decrease;search=device' \
  --n 10000 --d 16 --k 3 --lam 0.1 --noise 0 \
  --seed 0 --solver-seed 0 --order-seed 1729 \
  --max-iter 40 --eps-rel 1e-4 --eps-heur 1e-4 --price-tol 1e-3 \
  --add-per-round 8 --max-cols 256 \
  --certify-every 5 --cert-steps 15 \
  --gated-every 1 --gated-iters 100 --sketch-m 0 \
  --stall-rtol 1e-7 --patience 6 \
  --repeats 3 --warmups 1 --quality-rtol 1e-4 --final-verify \
  --output results/controlled_rocm_placement_n10000_d16.json
```

## Component studies

```bash
uv run python -m experiments.candidate_order \
  --states-per-cell 5 --repeats 6 --timing-calls-per-repeat 64 \
  --output results/candidate_order_study.json

uv run python -m experiments.feature_reuse \
  --output results/feature_reuse_study.json
```

## Exploratory experiments

These runs range from minutes to hours and some need substantial RAM or an
accelerator. Their outputs are single-run exploratory records, not controlled
performance evidence.

```bash
# Planted validation, bound comparison, scale sweep, and tuned baselines.
uv run python -m experiments.exploratory_benchmarks planted
uv run python -m experiments.exploratory_benchmarks bounds
uv run python -m experiments.exploratory_benchmarks scaling
uv run python -m experiments.exploratory_benchmarks optimizers

# Small enumerable, sampling/dimension, and hard-pricing experiments.
uv run python -m experiments.synthetic_studies exact-enumeration
uv run python -m experiments.synthetic_studies dimension-sampling
uv run python -m experiments.synthetic_studies pricing-methods
```

The real-data records use a 30-round RCG cap. California housing and Covertype
download through scikit-learn; YearPredictionMSD needs one preparation step.

```bash
uv run python -m experiments.prepare_data msd
uv run python -m experiments.real_data california --max-iter 30
uv run python -m experiments.real_data covtype --max-iter 30
uv run python -m experiments.real_data msd --max-iter 30
```

The two standalone large-scale artifacts have dedicated drivers:

```bash
uv run python -m experiments.large_scale single \
  --n 2000000 --device cpu --max-iter 60 \
  --output results/scaling_n2000000.json

HSA_OVERRIDE_GFX_VERSION=10.3.0 \
uv run python -m experiments.large_scale compare \
  --n 1000000 --device cuda --device cpu --max-iter 60 \
  --output results/scaling_device_comparison_n1000000.json
```

## Figures and paper

The figure scripts read the included result records. The chamber and
nontermination diagrams are self-contained.

```bash
uv run python -m experiments.paper_figures
tectonic paper/main.tex
```

The source uses `\date{\today}`, so a build on another date will intentionally
change the title-page date and PDF checksum.

## Artifact map

| Manuscript evidence | Artifact | Driver |
| --- | --- | --- |
| Planted validation | `planted_validation.json` | `experiments.exploratory_benchmarks planted` |
| Planted optimizer comparison | `planted_optimizer_comparison.json` | `experiments.exploratory_benchmarks optimizers` |
| Exact enumerable instance | `exact_enumerable_instance.json` | `experiments.synthetic_studies exact-enumeration` |
| Sampling and dimension | `dimension_sampling.json` | `experiments.synthetic_studies dimension-sampling` |
| Bound and pricing observations | `pricing_bound_comparison.json`, `pricing_method_comparison.json` | `experiments.exploratory_benchmarks bounds`, `experiments.synthetic_studies pricing-methods` |
| Real data | three `real_data_*.json` files | `experiments.real_data` |
| Scale | `scaling_sweep.json`, `scaling_n2000000.json`, `scaling_device_comparison_n1000000.json` | `experiments.exploratory_benchmarks scaling`, `experiments.large_scale` |
| CPU variants | `controlled_cpu_variants_*.json` | `experiments.controlled_benchmark` |
| Device placement | `controlled_rocm_placement_*.json` | `experiments.controlled_benchmark` |
| Candidate ordering and reuse | `candidate_order_study.json`, `feature_reuse_study.json` | `experiments.candidate_order`, `experiments.feature_reuse` |

Several drivers write directly to the result paths. Run expensive
reproductions in a separate clone or preserve the included JSON files before
starting.
