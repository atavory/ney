# AUTHORITATIVE SOURCE OF TRUTH: datasets, experts, and repair API

**Status:** binding as of 2026-08-14.

This is the sole source of truth for the experimental dataset registry, expert
registry, and public repair interface. Launch manifests, analysis scripts,
tables, and prose must conform to this file. If another document, manifest, or
implementation disagrees, execution and aggregation fail closed until the
discrepancy is resolved here. New datasets, experts, or function parameters
must be added here before launch; they may not be inferred from whatever files
happen to exist.

## One public function

Every analysis and eventual experiment must enter through:

```python
repair(expert, cv_experts, dataset, params)
```

`expert` is the full-data fitted-value snapshot. `cv_experts` is the sequence
of 20 whole-pipeline repeated-crossfit snapshots of that same expert on that
same realized dataset. `dataset` contains only observed `x`, `y`, response,
and the analysis mask. `params` contains every selection, stability,
shrinkage, and move-bound choice. The function receives no expert name,
dataset name, or simulation truth.

The implementation is `scripts/cv_expert_repair.py`. It is a new analysis
module and deliberately does not modify the frozen source used by the live
bank.

The only truth-free inspection entry point is:

```python
characterize(expert, cv_experts, dataset, params)
```

No analysis may branch on the expert name or dataset name inside either
function.

## Experts

1. `aipw`: augmented inverse-probability-weighted MAR mean expert.
2. `tmle`: ordinary TMLE MAR mean expert with the frozen fixed propensity
   floor used by the experiment.
3. `ctmle`: collaborative TMLE MAR mean expert with data-adaptive propensity
   truncation/selection.
4. `cui_selective_ml`: Cui--Tchetgen-style selective doubly robust MAR mean
   expert.
5. `ma_dr_bc`: Ma-style doubly robust bias-corrected expert adapted to the
   common MAR mean estimand.

These are expert constructors. None contains a special repair rule; every one
is passed to the same public function above.

## Datasets

### Kang--Schafer: 8 cells

`cc`, `ci`, `ic`, and `ii` cross correct/incorrect propensity and outcome
models. Each is run at `n=200` and `n=1000`.

### Controlled alignment stress tests: 10 cells

- `alignment_aligned`, `n=3000`, strength `0,3,5,8`.
- `alignment_partial`, `n=3000`, strength `3,5,8`.
- `alignment_disjoint`, `n=3000`, strength `3,5,8`.

These place the outcome-model defect inside, partly inside, or outside the
low-response region.

### Real-covariate semi-synthetic tests: 12 cells

- Breast-cancer covariates, aligned and misaligned.
- Handwritten-digits covariates, aligned and misaligned.
- Every design uses `n=6000` and strength `0,1,2`.

Outcomes and missingness are simulated so truth is known; covariates come from
the named real datasets.

### Regional-shift anchor: 4 cells

`regional_shift`, `n=3000`, strength `0,3,5,8`. This is a controlled nonlinear
MAR defect used to test whether repair responds to a regional shift.

### Cui--Tchetgen published benchmarks: 8 cells

Cui--Tchetgen Section-7 scenarios 1 and 2 map the published `E[Y(1)]`
experiment to the same MAR-mean estimand:

- `cui_published_scenario1` and `cui_published_scenario2`.
- Each at `n=250,500,1000,2000`.
- All five experts, the same full-plus-20-CV schema, and the same repair API.

### Ma published DiD benchmarks

Ma's published DGP2 and DGP3 target an ATT-like difference-in-differences
estimand. They are a separately labeled dataset family. The same repair API is
estimand-agnostic and consumes Ma's full and CV expert snapshots.

## Function registry

| Responsibility | File | Public/operative function |
|---|---|---|
| Uniform repair | `scripts/cv_expert_repair.py` | `repair(expert, cv_experts, dataset, params)` |
| Truth-free profiling | `scripts/cv_expert_repair.py` | `characterize(expert, cv_experts, dataset, params)` |
| Frozen-entry dataset adapter | `scripts/cv_expert_repair.py` | `dataset_from_entry(entry)` |
| Frozen artifact schema and loader | `scripts/frozen_expert_bank.py` | `FrozenExpertBankEntry`, `load_entry(path)` |
| Full plus CV artifact construction | `scripts/build_frozen_expert_entry.py` | `build(args)` |
| Shared MAR data and expert fitting | `scripts/validated_reference_transfer.py` | `make_data(...)`, `_crossfit_selected(...)` |
| Alignment/real dataset adapters | `scripts/section4_breadth_experiments.py` | `_install_adapter(module)` |
| Cui published dataset adapter | `scripts/section4_cui_published_experiments.py` | `_install_adapter(module)` |
| Ma published DiD construction | `scripts/ma_published_did_projection.py` | frozen Ma expert construction |

The dataset adapters and expert constructors run before the public repair
function. They may construct different experts and datasets; they may not
change the repair logic. Every resulting full/CV expert bundle is passed
through the same `repair(expert, cv_experts, dataset, params)` entry point.
